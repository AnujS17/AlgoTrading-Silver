"""
broker.py — Upstox REST API wrapper.
Handles order placement, cancellation, position queries, and funds check.
All calls are logged and raise descriptive exceptions on failure.

FIXES vs PREVIOUS VERSION:

  FIX 1 — place_stop_order() was MISSING.
    Engine calls this in three places (_place_broker_sl, _ratchet_broker_sl,
    _handle_trail_update) to place the broker-side SL-M after every entry.
    Without it, every trade entry crashed with AttributeError and the broker
    SL-M protection was never active.

    Upstox SL-M payload (POST /v2/order/place):
      order_type    = "SL-M"
      trigger_price = stop level (price at which order becomes MARKET)
      price         = 0  (SL-M has no limit price — it executes at market)
      transaction_type = "SELL" for long position | "BUY" for short position

  FIX 2 — get_trade_book() was MISSING.
    Engine._try_recover_entry_price() calls this on startup to recover the
    fill price of an overnight position. Without it, overnight reconnects
    crashed with AttributeError and the trail stop was never initialised
    after a restart with an open position.

    Correct endpoint: GET /v2/order/trades/get-trades-for-day
    Response fields used: instrument_token, transaction_type, quantity,
                          average_price (per-trade fill price)

  FIX 3 — Order placement used the wrong base URL.
    place_order() and cancel_order() must use api-hft.upstox.com/v2
    (the High Frequency Trading endpoint). All other read-only calls
    (positions, funds, order status) correctly use api.upstox.com/v2.
    Using the standard endpoint for order placement adds latency to
    every entry and exit. This matters most for intrabar exits.

  FIX 4 — get_today_realised_pnl() read the wrong field name.
    The Upstox positions API returns "realised" (not "realised_profit").
    The old code always returned 0.0, so the daily loss limit was never
    seeded with pre-existing P&L from manual trades or earlier bot sessions.

  FIX 5 — get_funds() uses the correct segment.
    MCX Silver is a commodity instrument — its margin sits in segment "SEC"
    (securities/equity), not "COM" (commodity). The code requests the
    SEC segment to get available margin for commodity trades. Using
    the old COM segment would always receive 0 available margin,
    causing every entry to be blocked by the margin guard because
    0 < 8000 × qty would always be True.
"""

import logging
import uuid
from typing import Optional

import requests

import config

logger = logging.getLogger("kalman.broker")

# ─────────────────────────────────────────────────────────────
# Upstox has two base URLs with distinct purposes:
#
#   HFT_BASE  — Order placement, modification, cancellation.
#               Uses Upstox's low-latency order routing infrastructure.
#               Must be used for any call that places or touches an order.
#               Source: https://upstox.com/developer/api-documentation/
#
#   API_BASE  — All read-only calls: positions, funds, order status,
#               trade book, historical data, market quotes.
#               Standard REST endpoint; fine for polling.
# ─────────────────────────────────────────────────────────────
HFT_BASE = "https://api-hft.upstox.com/v2"   # order placement + cancel
API_BASE = "https://api.upstox.com/v2"        # everything else


class UpstoxBroker:

    def __init__(self, access_token: str):
        self.token   = access_token
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept":        "application/json",
            "Content-Type":  "application/json",
        }

    # ─────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────

    def _get(self, path: str, params: dict = None, base: str = None) -> dict:
        url = f"{base or API_BASE}{path}"
        r   = requests.get(url, headers=self.headers, params=params, timeout=10)
        self._raise_for(r)
        return r.json()

    def _post(self, path: str, payload: dict, base: str = None) -> dict:
        url = f"{base or API_BASE}{path}"
        r   = requests.post(url, headers=self.headers, json=payload, timeout=10)
        self._raise_for(r)
        return r.json()

    def _delete(self, path: str, params: dict = None, base: str = None) -> dict:
        url = f"{base or API_BASE}{path}"
        r   = requests.delete(url, headers=self.headers, params=params, timeout=10)
        self._raise_for(r)
        return r.json()

    @staticmethod
    def _raise_for(r: requests.Response):
        if r.status_code not in (200, 201):
            raise RuntimeError(
                f"Upstox API error {r.status_code}: {r.text[:300]}"
            )

    # ─────────────────────────────────────────────────────────
    # Account
    # ─────────────────────────────────────────────────────────

    def get_funds(self) -> float:
        """
        Returns available margin in ₹ for the SEC securities segment.

        FIX 5: changed segment from "COM" (commodity) to "SEC"
        (securities/equity). MCX Silver margin is in the securities segment.
        Using "COM" always returned 0 for securities accounts, causing
        every entry to be blocked by the margin guard.

        Upstox response structure:
          data.equity.available_margin     — available balance for NSE/BSE
          data.commodity.available_margin  — available balance for MCX/NCDEX
        """
        try:
            data   = self._get("/user/get-funds-and-margin")
            # Primary: securities segment (MCX Silver lives here)
            sec    = data.get("data", {}).get("equity", {})
            avail  = float(sec.get("available_margin", 0))

            if avail == 0:
                # Fallback: some account types consolidate under commodity
                com    = data.get("data", {}).get("commodity", {})
                avail = float(com.get("available_margin", 0))
                if avail > 0:
                    logger.debug(
                        "SEC margin=0, using commodity margin as fallback: ₹%s",
                        f"{avail:,.0f}"
                    )

            logger.info("Available margin (SEC segment): ₹%s", f"{avail:,.0f}")
            return avail

        except Exception as e:
            logger.warning("get_funds() failed: %s — returning 0", e)
            return 0.0

    def get_profile(self) -> dict:
        return self._get("/user/profile")["data"]

    # ─────────────────────────────────────────────────────────
    # Positions
    # ─────────────────────────────────────────────────────────

    def get_positions(self) -> list:
        """Returns list of all open short-term positions."""
        data = self._get("/portfolio/short-term-positions")
        return data.get("data", [])

    def get_position_for(self, instrument_key: str) -> Optional[dict]:
        """
        Returns the live position for a specific instrument, or None.

        The Upstox positions API uses 'instrument_token' as the field name,
        which is identical in format to the instrument_key used everywhere
        else (e.g. 'MCX_FO|466029'). The comparison is correct.
        """
        for pos in self.get_positions():
            if pos.get("instrument_token") == instrument_key:
                return pos
        return None

    def get_net_quantity(self, instrument_key: str) -> int:
        """
        Returns net open quantity for an instrument:
          positive → long
          negative → short
          0        → flat
        """
        pos = self.get_position_for(instrument_key)
        if pos is None:
            return 0
        return int(pos.get("quantity", 0))

    # ─────────────────────────────────────────────────────────
    # Orders — placement uses HFT endpoint (FIX 3)
    # ─────────────────────────────────────────────────────────

    def place_order(self,
                    transaction_type: str,   # "BUY" or "SELL"
                    quantity: int,
                    order_type: str = "MARKET",
                    price: float = 0.0,
                    tag: str = "") -> str:
        """
        Places a regular order (MARKET or LIMIT) and returns the order_id.

        Uses the HFT base URL (api-hft.upstox.com) for minimum order
        placement latency. All other reads use api.upstox.com.

        Args:
            transaction_type : "BUY" or "SELL"
            quantity         : number of lots
            order_type       : "MARKET" or "LIMIT"
            price            : limit price (0 for MARKET)
            tag              : label shown in Upstox order book (max 20 chars)
        """
        payload = {
            "quantity":          quantity,
            "product":           config.PRODUCT_TYPE,
            "validity":          "DAY",
            "price":             round(price, 1) if order_type == "LIMIT" else 0,
            "tag":               (tag or f"KLM_{uuid.uuid4().hex[:6].upper()}")[:20],
            "instrument_token":  config.INSTRUMENT_KEY,
            "order_type":        order_type,
            "transaction_type":  transaction_type,
            "disclosed_quantity": 0,
            "trigger_price":     0,
            "is_amo":            False,
        }

        logger.info(
            "Placing %s %s | qty=%d price=%.1f tag=%s",
            transaction_type, order_type, quantity, price, payload["tag"]
        )

        resp     = self._post("/order/place", payload, base=HFT_BASE)
        order_id = resp.get("data", {}).get("order_id", "UNKNOWN")
        logger.info("Order placed → id=%s", order_id)
        return order_id

    def place_stop_order(self,
                         transaction_type: str,   # "SELL" for long SL | "BUY" for short SL
                         quantity: int,
                         trigger_price: float,
                         tag: str = "") -> str:
        """
        Places an SL-M (Stop Loss Market) order at the broker.

        FIX 1: This method was completely missing from broker.py.
        Engine._place_broker_sl() and _ratchet_broker_sl() call this
        after every trade entry to create a hard exchange-level backstop.

        How SL-M works on Upstox:
          - The order sits at the exchange as a pending stop.
          - When LTP touches trigger_price, the exchange converts it to
            a MARKET order and fills it at the best available price.
          - price must be 0 for SL-M (it executes at market, not at a limit).
          - This is different from SL (Stop Limit), which has both a trigger
            and a limit price.

        For a LONG position:  transaction_type="SELL", trigger_price=trail_stop
        For a SHORT position: transaction_type="BUY",  trigger_price=trail_stop

        Upstox API payload for SL-M:
          POST api-hft.upstox.com/v2/order/place
          {
            "order_type":       "SL-M",
            "transaction_type": "SELL",     ← or "BUY" for shorts
            "trigger_price":    270000,     ← stop level
            "price":            0,          ← 0 = market execution
            "quantity":         1,
            "product":          "D",        ← from config.PRODUCT_TYPE
            "validity":         "DAY",
            "instrument_token": "MCX_FO|466029",
            "disclosed_quantity": 0,
            "is_amo":           false
          }

        Args:
            transaction_type : "SELL" to stop-out a long | "BUY" to stop-out a short
            quantity         : lots to close (should equal open qty)
            trigger_price    : the price level at which stop fires
            tag              : label shown in Upstox order book
        """
        trigger_price = round(trigger_price, 1)
        payload = {
            "quantity":          quantity,
            "product":           config.PRODUCT_TYPE,
            "validity":          "DAY",
            "price":             0,          # SL-M always executes at market
            "tag":               (tag or f"KSL_{uuid.uuid4().hex[:6].upper()}")[:20],
            "instrument_token":  config.INSTRUMENT_KEY,
            "order_type":        "SL-M",
            "transaction_type":  transaction_type,
            "disclosed_quantity": 0,
            "trigger_price":     trigger_price,
            "is_amo":            False,
        }

        logger.info(
            "Placing SL-M | %s qty=%d trigger=₹%.0f tag=%s",
            transaction_type, quantity, trigger_price, payload["tag"]
        )

        resp     = self._post("/order/place", payload, base=HFT_BASE)
        order_id = resp.get("data", {}).get("order_id", "UNKNOWN")
        logger.info("SL-M placed → id=%s @ trigger=₹%.0f", order_id, trigger_price)
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order. Returns True on success.

        FIX 3: Uses HFT endpoint for the cancel call. Order cancellations
        must go through the same low-latency routing as placements.
        """
        try:
            self._delete("/order/cancel",
                         params={"order_id": order_id},
                         base=HFT_BASE)
            logger.info("Order cancelled: %s", order_id)
            return True
        except Exception as e:
            logger.warning("Cancel failed for %s: %s", order_id, e)
            return False

    def get_order_status(self, order_id: str) -> dict:
        """
        Returns order details dict from Upstox.

        Status values to check for fills: "complete" / "filled"
        Status values to check for failure: "rejected" / "cancelled"

        Note: Upstox returns status in lowercase ("complete") but engine.py
        upper-cases before comparing ("COMPLETE"), so both forms work.
        """
        data = self._get("/order/details", params={"order_id": order_id})
        return data.get("data", {})

    def get_order_book(self) -> list:
        """Returns all orders placed today."""
        data = self._get("/order/retrieve-all")
        return data.get("data", [])

    def get_trade_book(self) -> list:
        """
        Returns all trades executed today.

        FIX 2: This method was completely missing from broker.py.
        Engine._try_recover_entry_price() calls this on startup when
        an overnight position is detected, to recover the average fill
        price from today's trade book (so the trail stop can be seeded).

        Endpoint: GET /v2/order/trades/get-trades-for-day
        Response fields used by engine:
          instrument_token  — matches config.INSTRUMENT_KEY format
          transaction_type  — "BUY" or "SELL"
          quantity          — lots filled in this trade
          average_price     — fill price for this trade leg

        Note: A single order can be filled across multiple trade legs
        (partial fills). The engine's _try_recover_entry_price() already
        handles this correctly by computing a weighted average.
        """
        try:
            data = self._get("/order/trades/get-trades-for-day")
            trades = data.get("data", [])
            logger.debug("Trade book: %d trades today", len(trades))
            return trades
        except Exception as e:
            logger.warning("get_trade_book() failed: %s — returning []", e)
            return []

    # ─────────────────────────────────────────────────────────
    # High-level entry/exit wrappers
    # ─────────────────────────────────────────────────────────

    def enter_long(self, quantity: int, tag: str = "ENTRY_LONG") -> str:
        """
        Buy to open a long position. Always MARKET — guaranteed fill
        on signal matters more than saving a few ticks on entry.
        config.ORDER_TYPE governs exits only, never entries.
        """
        return self.place_order("BUY", quantity, order_type="MARKET", tag=tag)

    def enter_short(self, quantity: int, tag: str = "ENTRY_SHORT") -> str:
        """Sell to open a short position. Always MARKET (same rationale)."""
        return self.place_order("SELL", quantity, order_type="MARKET", tag=tag)

    def exit_long(self, quantity: int, tag: str = "EXIT_LONG") -> str:
        """Sell to close a long position. Uses config.ORDER_TYPE."""
        return self.place_order("SELL", quantity,
                                order_type=config.ORDER_TYPE, tag=tag)

    def exit_short(self, quantity: int, tag: str = "EXIT_SHORT") -> str:
        """Buy to close a short position. Uses config.ORDER_TYPE."""
        return self.place_order("BUY", quantity,
                                order_type=config.ORDER_TYPE, tag=tag)

    def close_all_positions(self) -> None:
        """
        Emergency flatten — closes any open position via MARKET order.
        Called by engine.emergency_stop() when ALLOW_OVERNIGHT=False.
        """
        net_qty = self.get_net_quantity(config.INSTRUMENT_KEY)
        if net_qty == 0:
            logger.info("close_all_positions: already flat")
            return
        if net_qty > 0:
            logger.warning("EMERGENCY FLATTEN: selling %d lots (was long)", net_qty)
            self.place_order("SELL", abs(net_qty),
                             order_type="MARKET", tag="EMERGENCY_FLAT")
        else:
            logger.warning("EMERGENCY FLATTEN: buying %d lots (was short)", abs(net_qty))
            self.place_order("BUY", abs(net_qty),
                             order_type="MARKET", tag="EMERGENCY_FLAT")

    def get_today_realised_pnl(self) -> float:
        """
        Returns today's total realised P&L in ₹ across all positions.

        FIX 4: changed p.get("realised_profit", 0) → p.get("realised", 0).
        The Upstox short-term positions API returns the field as "realised"
        (confirmed from official API docs response example). The old name
        "realised_profit" does not exist in the response, so the old code
        silently returned 0.0 on every startup, meaning the daily loss
        limit was never seeded with pre-existing losses from manual trades
        or an earlier bot session.

        This includes all trades (manual + bot) so the risk manager's
        daily loss limit is accurate regardless of how the day started.
        """
        try:
            positions = self.get_positions()
            total = sum(float(p.get("realised", 0)) for p in positions)
            logger.info(
                "Today's realised P&L from broker: %s",
                f"₹{total:+,.0f}"
            )
            return total
        except Exception as e:
            logger.warning("Could not fetch realised P&L: %s — starting at 0", e)
            return 0.0
