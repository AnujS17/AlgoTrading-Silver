"""
broker.py — Upstox REST API wrapper.
Handles order placement, cancellation, position queries, and funds check.
All calls are logged and raise descriptive exceptions on failure.
"""

import logging
import uuid
from typing import Optional

import requests

import config

logger = logging.getLogger("kalman.broker")


class UpstoxBroker:
    BASE = "https://api.upstox.com/v2"

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
    def _get(self, path: str, params: dict = None) -> dict:
        r = requests.get(f"{self.BASE}{path}", headers=self.headers,
                         params=params, timeout=10)
        self._raise_for(r)
        return r.json()

    def _post(self, path: str, payload: dict) -> dict:
        r = requests.post(f"{self.BASE}{path}", headers=self.headers,
                          json=payload, timeout=10)
        self._raise_for(r)
        return r.json()

    def _delete(self, path: str, params: dict = None) -> dict:
        r = requests.delete(f"{self.BASE}{path}", headers=self.headers,
                            params=params, timeout=10)
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
        """Returns available margin in ₹. MCX futures margin shows under SEC segment."""
        data  = self._get("/user/get-funds-and-margin", params={"segment": "SEC"})
        used  = data["data"].get("equity", {})
        avail = float(used.get("available_margin", 0))
        logger.info(f"Available margin: ₹{avail:,.0f}")
        return avail

    def get_profile(self) -> dict:
        return self._get("/user/profile")["data"]

    # ─────────────────────────────────────────────────────────
    # Positions
    # ─────────────────────────────────────────────────────────
    def get_positions(self) -> list:
        """Returns list of open intraday positions."""
        data = self._get("/portfolio/short-term-positions")
        return data.get("data", [])

    def get_position_for(self, instrument_key: str) -> Optional[dict]:
        """Returns the live position for a specific instrument, or None."""
        for pos in self.get_positions():
            if pos.get("instrument_token") == instrument_key:
                return pos
        return None

    def get_net_quantity(self, instrument_key: str) -> int:
        """Returns net open quantity: positive=long, negative=short, 0=flat."""
        pos = self.get_position_for(instrument_key)
        if pos is None:
            return 0
        return int(pos.get("quantity", 0))

    def get_entry_price(self, instrument_key: str) -> float:
        """
        Returns the average entry price of the current open position.
        Used on engine restart to recover _entry_px so P&L is correct.
        Returns 0.0 if no position or price unavailable.
        """
        pos = self.get_position_for(instrument_key)
        if pos is None:
            return 0.0
        qty = int(pos.get("quantity", 0))
        if qty > 0:
            return float(pos.get("buy_average_price", 0) or
                         pos.get("average_price", 0))
        elif qty < 0:
            return float(pos.get("sell_average_price", 0) or
                         pos.get("average_price", 0))
        return 0.0

    # ─────────────────────────────────────────────────────────
    # Orders
    # ─────────────────────────────────────────────────────────
    def place_order(self,
                    transaction_type: str,   # "BUY" or "SELL"
                    quantity: int,
                    order_type: str = "MARKET",
                    price: float = 0.0,
                    tag: str = "") -> str:
        """
        Places an order and returns the order_id.

        transaction_type : "BUY" or "SELL"
        quantity         : number of lots
        order_type       : "MARKET" or "LIMIT"
        price            : limit price (ignored for MARKET)
        tag              : optional label shown in Upstox order book
        """
        payload = {
            "quantity":         quantity,
            "product":          config.PRODUCT_TYPE,     # INTRADAY
            "validity":         "DAY",
            "price":            round(price, 1) if order_type == "LIMIT" else 0,
            "tag":              tag or f"KALMAN_{uuid.uuid4().hex[:6].upper()}",
            "instrument_token": config.INSTRUMENT_KEY,
            "order_type":       order_type,
            "transaction_type": transaction_type,
            "disclosed_quantity": 0,
            "trigger_price":    0,
            "is_amo":           False,
        }

        logger.info(f"Placing {transaction_type} {order_type} order | "
                    f"qty={quantity} | price={price:.1f} | tag={payload['tag']}")

        resp     = self._post("/order/place", payload)
        order_id = resp.get("data", {}).get("order_id", "UNKNOWN")
        logger.info(f"Order placed → order_id={order_id}")
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True on success."""
        try:
            self._delete("/order/cancel", params={"order_id": order_id})
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.warning(f"Cancel failed for {order_id}: {e}")
            return False

    def get_order_status(self, order_id: str) -> dict:
        """Returns order details dict from Upstox order book."""
        data = self._get("/order/details", params={"order_id": order_id})
        return data.get("data", {})

    def get_order_book(self) -> list:
        data = self._get("/order/retrieve-all")
        return data.get("data", [])

    # ─────────────────────────────────────────────────────────
    # High-level entry/exit wrappers
    # ─────────────────────────────────────────────────────────
    def enter_long(self, quantity: int, tag: str = "ENTRY_LONG") -> str:
        """Buy to open long position."""
        return self.place_order("BUY", quantity,
                                order_type=config.ORDER_TYPE, tag=tag)

    def enter_short(self, quantity: int, tag: str = "ENTRY_SHORT") -> str:
        """Sell to open short position."""
        return self.place_order("SELL", quantity,
                                order_type=config.ORDER_TYPE, tag=tag)

    def exit_long(self, quantity: int, tag: str = "EXIT_LONG") -> str:
        """Sell to close long position."""
        return self.place_order("SELL", quantity,
                                order_type=config.ORDER_TYPE, tag=tag)

    def exit_short(self, quantity: int, tag: str = "EXIT_SHORT") -> str:
        """Buy to close short position."""
        return self.place_order("BUY", quantity,
                                order_type=config.ORDER_TYPE, tag=tag)

    def exit_smart(self, transaction_type: str, quantity: int,
                   ltp: float, tag: str = "SMART_EXIT") -> tuple:
        """
        Smart exit: try LIMIT at LTP first, fall back to MARKET.

        During low-volatility MCX hours spreads can be 50-200 pts wide.
        A LIMIT at LTP usually fills in 1-3 seconds, saving the spread.
        Falls back to MARKET after LIMIT_ORDER_TIMEOUT_SECS if unfilled.

        Returns (order_id: str, was_limit: bool)
        """
        import time as _time
        timeout  = getattr(config, "LIMIT_ORDER_TIMEOUT_SECS", 7)
        limit_px = round(ltp, 1)
        logger.info("Smart exit: LIMIT %s @ ₹%.1f (fallback MARKET in %ds)",
                    transaction_type, limit_px, timeout)
        order_id = self.place_order(transaction_type, quantity,
                                    order_type="LIMIT", price=limit_px, tag=tag)
        deadline = _time.time() + timeout
        while _time.time() < deadline:
            try:
                order  = self.get_order_status(order_id)
                status = order.get("status", "").upper()
                if status in ("COMPLETE", "FILLED"):
                    logger.info("Smart exit: LIMIT filled @ ₹%.0f",
                                float(order.get("average_price", limit_px)))
                    return order_id, True
                if status in ("REJECTED", "CANCELLED"):
                    logger.warning("Smart exit: LIMIT %s — falling to MARKET", status)
                    break
            except Exception as e:
                logger.debug("Smart exit poll: %s", e)
            _time.sleep(0.5)
        logger.warning("Smart exit: LIMIT unfilled after %ds — placing MARKET", timeout)
        self.cancel_order(order_id)
        _time.sleep(0.2)
        mkt_id = self.place_order(transaction_type, quantity,
                                  order_type="MARKET", tag=tag + "_MKT")
        return mkt_id, False

    def close_all_positions(self) -> None:
        """
        Emergency flatten — closes any open position for the instrument.
        Uses MARKET order to guarantee fill.
        """
        net_qty = self.get_net_quantity(config.INSTRUMENT_KEY)
        if net_qty == 0:
            logger.info("close_all_positions: already flat")
            return

        if net_qty > 0:
            logger.warning(f"EMERGENCY FLATTEN: selling {net_qty} lots (long)")
            self.place_order("SELL", abs(net_qty),
                             order_type="MARKET", tag="EMERGENCY_FLAT")
        else:
            logger.warning(f"EMERGENCY FLATTEN: buying {abs(net_qty)} lots (short)")
            self.place_order("BUY", abs(net_qty),
                             order_type="MARKET", tag="EMERGENCY_FLAT")

    def get_today_realised_pnl(self) -> float:
        """
        Reads today's realised P&L from Upstox short-term positions.
        Includes ALL trades (manual + bot) so the daily loss limit is
        always accurate regardless of what happened before the bot started.

        Upstox position fields used:
          realised_profit  — closed P&L for intraday trades today
        """
        try:
            positions = self.get_positions()
            total = sum(float(p.get("realised_profit", 0)) for p in positions)
            logger.info(f"Today's realised P&L from broker: Rs.{total:+,.0f}")
            return total
        except Exception as e:
            logger.warning(f"Could not fetch realised P&L: {e} — starting at 0")
            return 0.0
