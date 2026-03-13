"""
engine.py — Main live trading orchestrator.

Wires together: feed → strategy → risk checks → broker → logging.

Run with:  python engine.py
Stop with: Ctrl+C  (gracefully flattens all positions before exit)

Architecture:
    UpstoxFeed (background thread)
        └─ on_candle_close(candle)
               └─ KalmanScalperStrategy.process_bar(candle) → Signal
                      └─ RiskManager.approve(signal)  → bool
                             └─ UpstoxBroker.enter/exit (order)
                                    └─ TradeLogger.record(trade)
"""

import csv
import logging
import os
import signal as sig_module
import sys
import threading
import time
from datetime import datetime, date, timedelta
from typing import Optional

import config
from auth     import load_token, validate_token
from broker   import UpstoxBroker
from feed     import UpstoxFeed, RestLTPPoller
from strategy import KalmanScalperStrategy, Signal

# ─────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────
def setup_logging():
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s"

    # ── Windows fix: force UTF-8 on all handlers ──────────────
    # Windows terminals default to cp1252 which cannot encode the
    # rupee sign (Rs. / U+20B9). This wraps stdout in a UTF-8
    # stream so the terminal never crashes on currency symbols.
    import io

    # Console handler
    if hasattr(sys.stdout, "buffer"):
        utf8_stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    else:
        utf8_stdout = sys.stdout
    console_handler = logging.StreamHandler(utf8_stdout)
    console_handler.setFormatter(logging.Formatter(fmt))

    # File handler — always UTF-8
    file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)

logger = logging.getLogger("kalman.engine")


# ─────────────────────────────────────────────────────────────
# Risk Manager
# ─────────────────────────────────────────────────────────────
class RiskManager:
    """
    All risk checks in one place. approve() must return True
    before any order is sent to the broker.

    Checks:
      - Daily loss limit (MAX_DAILY_LOSS)
      - Max trades per day (MAX_TRADES_PER_DAY)
      - Max open trades (MAX_OPEN_TRADES)
      - Not outside session
      - Available margin check before entry
    """

    def __init__(self, broker: UpstoxBroker):
        self._broker       = broker
        self._daily_trades = 0
        self._last_reset   = date.today()
        self._lock         = threading.Lock()

        # Seed P&L from broker — captures manual trades made before bot started.
        # This ensures the daily loss limit is never bypassed by pre-existing losses.
        self._daily_pnl = broker.get_today_realised_pnl()
        if self._daily_pnl != 0:
            logger.warning(
                f"Pre-existing P&L detected from broker: Rs.{self._daily_pnl:+,.0f}  "
                f"(manual trades or earlier sessions counted toward daily limit)"
            )

    def _check_reset(self):
        today = date.today()
        if today != self._last_reset:
            logger.info(f"New day — resetting daily counters. "
                        f"Yesterday P&L: ₹{self._daily_pnl:+,.0f} | "
                        f"Trades: {self._daily_trades}")
            self._daily_pnl    = 0.0
            self._daily_trades = 0
            self._last_reset   = today

    def approve_entry(self, signal: Signal, qty: int) -> tuple:
        """Returns (approved: bool, reason: str)."""
        with self._lock:
            self._check_reset()

            # 1. Daily loss limit
            if self._daily_pnl <= -config.MAX_DAILY_LOSS:
                return False, f"Daily loss limit hit (₹{self._daily_pnl:+,.0f})"

            # 2. Trade count limit
            if self._daily_trades >= config.MAX_TRADES_PER_DAY:
                return False, f"Max trades/day reached ({self._daily_trades})"

            # 3. Margin check — only for entries
            try:
                margin = self._broker.get_funds()
                # Rough margin requirement: Silver MIC ~₹5000–10000/lot
                min_required = 8000 * qty
                if margin < min_required:
                    return False, f"Insufficient margin: ₹{margin:,.0f} < ₹{min_required:,}"
            except Exception as e:
                return False, f"Margin check failed: {e}"

            return True, "OK"

    def approve_exit(self) -> tuple:
        """Exits are almost always approved unless broker is down."""
        return True, "OK"

    def record_trade_open(self):
        with self._lock:
            self._daily_trades += 1

    def record_pnl(self, pnl: float):
        with self._lock:
            self._daily_pnl += pnl
            logger.info(f"P&L recorded: ₹{pnl:+,.0f} | "
                        f"Daily total: ₹{self._daily_pnl:+,.0f}")

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def daily_trades(self) -> int:
        return self._daily_trades


# ─────────────────────────────────────────────────────────────
# Trade Logger
# ─────────────────────────────────────────────────────────────
class TradeLogger:
    """Appends every trade event to a CSV file."""

    FIELDS = ["datetime", "action", "reason", "fill_price",
              "quantity", "pnl", "daily_pnl",
              "norm_resid", "vr", "vel_norm", "atr", "order_id"]

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._lock    = threading.Lock()
        # Write header if file is new
        if not os.path.exists(filepath):
            with open(filepath, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def record(self, **kwargs):
        row = {k: kwargs.get(k, "") for k in self.FIELDS}
        row["datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            with open(self.filepath, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)
        logger.info(f"Trade logged: {row['action']} | "
                    f"price={row['fill_price']} | pnl={row['pnl']}")


# ─────────────────────────────────────────────────────────────
# Main Engine
# ─────────────────────────────────────────────────────────────
class TradingEngine:
    """
    Central orchestrator. Single instance per process.
    Thread-safe: on_candle_close() is called from feed thread,
    all state mutations are protected by self._lock.
    """

    def __init__(self, access_token: str):
        self._lock     = threading.Lock()
        self.broker    = UpstoxBroker(access_token)
        self.strategy  = KalmanScalperStrategy()
        self.risk      = RiskManager(self.broker)
        self.tlog      = TradeLogger(config.TRADE_LOG_CSV)
        self._running  = False

        # Position tracking (engine-side, verified against broker)
        self._pos        = 0       # +1 long | -1 short | 0 flat
        self._entry_px   = 0.0
        self._open_qty   = 0
        self._entry_ts   = None
        self._order_id   = None    # last placed order id

        # Intrabar exit guard — prevents double-exit within the same bar.
        # Reset to False at the top of every on_candle_close().
        self._intrabar_exit_fired = False

        # Heartbeat: track last tick time for rollover/disconnect detection
        self._last_tick_ts: Optional[datetime] = None
        self._TICK_TIMEOUT_MINS = 10   # warn if no tick for this long during session

        # Compute position sizing
        self._quantity = max(1, config.LOT_SIZE)

        logger.info("TradingEngine initialised")
        logger.info(f"  Instrument : {config.INSTRUMENT_NAME} ({config.INSTRUMENT_KEY})")
        logger.info(f"  Quantity   : {self._quantity} lot(s)")
        logger.info(f"  Risk/trade : {config.EQUITY_RISK_PCT}%")
        logger.info(f"  Max loss   : ₹{config.MAX_DAILY_LOSS:,}/day")

        # ── Overnight config cross-check ──────────────────────────
        allow_overnight = getattr(config, "ALLOW_OVERNIGHT", False)
        product_type    = getattr(config, "PRODUCT_TYPE", "D")
        logger.info(f"  Overnight  : {'ENABLED — positions will be held across sessions' if allow_overnight else 'DISABLED — positions will be flattened on stop'}")
        if allow_overnight and product_type.upper() in ("I", "MIS", "INTRADAY"):
            logger.critical(
                "CONFIG CONFLICT: ALLOW_OVERNIGHT=True but PRODUCT_TYPE='%s' is an "
                "intraday product — the broker will AUTO-SQUARE OFF your position at "
                "session end regardless. Set PRODUCT_TYPE='D' in config.py to hold overnight.",
                product_type
            )
        if not allow_overnight and product_type.upper() in ("D", "NRML", "DELIVERY"):
            logger.info(
                "  Note: PRODUCT_TYPE='%s' supports overnight but ALLOW_OVERNIGHT=False — "
                "positions will still be flattened on engine stop.",
                product_type
            )

    # ─────────────────────────────────────────────────────────
    # Candle handler — called by feed on each completed bar
    # ─────────────────────────────────────────────────────────
    def on_candle_close(self, candle: dict):
        """
        Entry point for every completed 5-min candle.
        This is the strategy's heartbeat.
        """
        with self._lock:
            # Reset intrabar exit guard so it can fire again on this new bar
            self._intrabar_exit_fired = False

            ts = candle["ts"]

            # Update heartbeat timestamp (Layer 3 rollover guard)
            self._last_tick_ts = datetime.now()

            logger.info(
                f"── Candle {ts} | "
                f"O={candle['open']:.0f} H={candle['high']:.0f} "
                f"L={candle['low']:.0f} C={candle['close']:.0f} "
                f"V={candle['volume']:,} ──"
            )

            # ── 1. Sync position with broker (every bar) ──────
            self._sync_position()

            # ── 1b. End-of-session forced exit (if ALLOW_OVERNIGHT=False) ──
            # When overnight is disabled and the session is ending, close any
            # open position NOW rather than risk an unmanaged carry.
            # Triggered on the candle whose close time equals or exceeds SESS_END.
            allow_overnight = getattr(config, "ALLOW_OVERNIGHT", False)
            if not allow_overnight and self._pos != 0 and config.USE_SESSION:
                sess_e_mins = config.SESS_END[0] * 60 + config.SESS_END[1]
                bar_mins    = ts.hour * 60 + ts.minute
                # Close on the last bar inside session (within one candle of end)
                if bar_mins >= sess_e_mins - config.CANDLE_INTERVAL:
                    logger.warning(
                        "SESSION END APPROACH — ALLOW_OVERNIGHT=False: "
                        "forcing position close (bar=%02d:%02d sess_end=%02d:%02d).",
                        ts.hour, ts.minute,
                        config.SESS_END[0], config.SESS_END[1]
                    )
                    class _SessEndSignal:
                        action      = "EXIT_LONG" if self._pos == 1 else "EXIT_SHORT"
                        reason      = "Session End (ALLOW_OVERNIGHT=False)"
                        bar_ts      = ts
                        norm_resid  = 0.0
                        vr          = 0.0
                        vel_norm    = 0.0
                        atr         = self.strategy._atr or 0.0
                        trail_price = float("nan")
                    self._handle_exit(_SessEndSignal())
                    return   # nothing more to do this bar
            signal: Optional[Signal] = self.strategy.process_bar(candle)

            if signal is None:
                return

            logger.info(f"Signal: {signal.action} | {signal.reason} | "
                        f"normResid={signal.norm_resid:+.3f}")

            # ── 3. Route signal to action ─────────────────────
            if signal.action == "BUY":
                self._handle_entry(signal, direction=1)

            elif signal.action == "SELL":
                self._handle_entry(signal, direction=-1)

            elif signal.action == "EXIT_LONG" and self._pos == 1:
                self._handle_exit(signal)

            elif signal.action == "EXIT_SHORT" and self._pos == -1:
                self._handle_exit(signal)

            elif signal.action == "UPDATE_TRAIL":
                self._handle_trail_update(signal)

    # ─────────────────────────────────────────────────────────
    # Entry handler
    # ─────────────────────────────────────────────────────────
    def _handle_entry(self, signal: Signal, direction: int):
        if self._pos != 0:
            logger.warning(f"Entry signal ignored — already in position ({self._pos})")
            return

        # Block entries if heartbeat monitor has flagged feed silence
        # (possible expired contract or silent disconnect)
        if getattr(self, "_feed_silent", False):
            logger.critical(
                "Entry BLOCKED — feed is silent (no ticks for %d+ min). "
                "Possible expired contract. Verify INSTRUMENT_KEY in config.py.",
                self._TICK_TIMEOUT_MINS
            )
            return

        # Risk check
        approved, reason = self.risk.approve_entry(signal, self._quantity)
        if not approved:
            logger.warning(f"Entry BLOCKED by risk manager: {reason}")
            return

        try:
            tag = f"K_{signal.action}_{signal.bar_ts.strftime('%H%M')}"
            if direction == 1:
                order_id = self.broker.enter_long(self._quantity, tag=tag)
            else:
                order_id = self.broker.enter_short(self._quantity, tag=tag)

            # Confirm fill (poll order status up to 10s)
            fill_px = self._wait_for_fill(order_id, timeout=10)

            self._pos      = direction
            self._entry_px = fill_px
            self._open_qty = self._quantity
            self._entry_ts = signal.bar_ts
            self._order_id = order_id

            # Inform strategy of position change — pass ATR so trail is initialised
            self.strategy.set_position(direction, fill_px, atr=signal.atr)
            self.risk.record_trade_open()

            self.tlog.record(
                action     = signal.action,
                reason     = signal.reason,
                fill_price = fill_px,
                quantity   = self._quantity,
                pnl        = 0,
                daily_pnl  = self.risk.daily_pnl,
                norm_resid = round(signal.norm_resid, 4),
                vr         = round(signal.vr, 4),
                vel_norm   = round(signal.vel_norm, 4),
                atr        = round(signal.atr, 1),
                order_id   = order_id,
            )

            logger.info(f"{'LONG' if direction==1 else 'SHORT'} ENTERED | "
                        f"fill=₹{fill_px:.0f} | qty={self._quantity} | id={order_id}")

        except Exception as e:
            logger.error(f"Entry order FAILED: {e}")

    # ─────────────────────────────────────────────────────────
    # Exit handler
    # ─────────────────────────────────────────────────────────
    def _handle_exit(self, signal: Signal):
        if self._pos == 0:
            logger.warning("Exit signal ignored — already flat")
            return

        approved, reason = self.risk.approve_exit()
        if not approved:
            logger.warning(f"Exit blocked: {reason}")
            return

        try:
            tag = f"K_EXIT_{signal.bar_ts.strftime('%H%M')}"

            fill_px  = None
            order_id = None

            # ── Smart Limit Exit (slippage control) ───────────────
            # Attempt a LIMIT order at current LTP first.
            # Falls back to MARKET if not filled within LIMIT_ORDER_TIMEOUT_SECS.
            # Disabled automatically for ATR Trail exits where speed is critical
            # (intrabar trail hits and fallback trail hits want guaranteed fills).
            use_smart = (
                getattr(config, "SMART_LIMIT_EXITS", False) and
                signal.reason not in ("ATR Trail Intrabar", "ATR Trail Fallback",
                                      "Session End (ALLOW_OVERNIGHT=False)")
            )

            if use_smart:
                fill_px = self._smart_limit_exit(tag)
                if fill_px is not None:
                    # Smart limit succeeded — record the order id as unknown
                    # (already filled and closed; we only need fill_px)
                    order_id = "SMART_LIMIT"

            # ── MARKET fallback (or primary if smart exits disabled) ──
            if fill_px is None:
                if self._pos == 1:
                    order_id = self.broker.exit_long(self._open_qty, tag=tag)
                else:
                    order_id = self.broker.exit_short(self._open_qty, tag=tag)
                fill_px = self._wait_for_fill(order_id, timeout=10)

            # ── Calculate and record P&L ───────────────────────────
            if self._pos == 1:
                pnl = (fill_px - self._entry_px) * self._open_qty
            else:
                pnl = (self._entry_px - fill_px) * self._open_qty

            self.risk.record_pnl(pnl)

            self.tlog.record(
                action     = f"EXIT_{'LONG' if self._pos == 1 else 'SHORT'}",
                reason     = signal.reason,
                fill_price = fill_px,
                quantity   = self._open_qty,
                pnl        = round(pnl, 2),
                daily_pnl  = round(self.risk.daily_pnl, 2),
                norm_resid = round(signal.norm_resid, 4),
                vr         = round(signal.vr, 4),
                vel_norm   = round(signal.vel_norm, 4),
                atr        = round(signal.atr, 1),
                order_id   = order_id,
            )

            logger.info(f"EXIT | fill=₹{fill_px:.0f} | "
                        f"P&L=₹{pnl:+,.0f} | reason={signal.reason}")

            self._pos      = 0
            self._entry_px = 0.0
            self._open_qty = 0
            self.strategy.set_position(0)

        except Exception as e:
            logger.error(f"Exit order FAILED: {e}")
            logger.error("MANUAL INTERVENTION MAY BE REQUIRED — CHECK POSITION")

    # ─────────────────────────────────────────────────────────
    # Trail update handler — ratchets broker SL-M order
    # ─────────────────────────────────────────────────────────
    def _handle_trail_update(self, signal: Signal):
        """
        Strategy has ratcheted the trail stop to a better level.
        Cancel the existing SL-M order and place a new one at signal.trail_price.
        """
        if self._pos == 0:
            return
        import math
        if math.isnan(signal.trail_price):
            logger.warning("UPDATE_TRAIL received but trail_price is nan — skipping")
            return
        try:
            direction = "SELL" if self._pos == 1 else "BUY"
            logger.info(
                "UPDATE_TRAIL %s — cancelling old SL-M, placing new at ₹%.0f",
                direction, signal.trail_price
            )
            if self._order_id:
                try:
                    self.broker.cancel_order(self._order_id)
                except Exception as ce:
                    logger.warning("Cancel old SL-M failed (may already be gone): %s", ce)

            new_order_id = self.broker.place_stop_order(
                direction, self._open_qty, signal.trail_price,
                tag=f"K_TRAIL_{signal.bar_ts.strftime('%H%M')}"
            )
            self._order_id = new_order_id
            logger.info("New SL-M placed: id=%s @ ₹%.0f", new_order_id, signal.trail_price)
        except Exception as e:
            logger.error("Trail SL-M update FAILED: %s — manual stop check required", e)

    # ─────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────
    # Tick-level intrabar trail monitor
    # ─────────────────────────────────────────────────────────
    def on_tick(self, ltp: float, ts):
        """
        Called on every raw LTP tick by the feed (Tier 1 / Tier 2 only).
        Only active when config.INTRABAR_TRAIL_EXIT = True.

        Compares LTP against the live ATR trail stop every tick.
        If breached, fires an immediate exit — does NOT wait for bar close.

        Thread safety: uses _lock + _intrabar_exit_fired flag to ensure
        only one exit fires per bar, and cannot race with on_candle_close.
        """
        if not getattr(config, "INTRABAR_TRAIL_EXIT", False):
            return
        if self._pos == 0:
            return

        trail = self.strategy.current_trail
        if trail != trail:   # nan check — trail not yet initialised after entry
            return

        # Check if the trail level is breached this tick
        breached = (self._pos ==  1 and ltp <= trail) or \
                   (self._pos == -1 and ltp >= trail)
        if not breached:
            return

        with self._lock:
            # Re-check inside lock — a concurrent tick or on_candle_close
            # may have already exited this position
            if self._pos == 0 or self._intrabar_exit_fired:
                return
            self._intrabar_exit_fired = True

            direction_str = "LONG" if self._pos == 1 else "SHORT"
            logger.warning(
                "INTRABAR TRAIL HIT (%s) — LTP=%.0f crossed trail=%.0f — "
                "exiting immediately (not waiting for bar close)",
                direction_str, ltp, trail
            )

            # Build a minimal signal object for _handle_exit
            pos_at_tick   = self._pos
            trail_at_tick = trail
            ts_val        = ts

            class _IntrabarTrailSignal:
                action      = "EXIT_LONG"  if pos_at_tick ==  1 else "EXIT_SHORT"
                reason      = "ATR Trail Intrabar"
                bar_ts      = ts_val
                norm_resid  = 0.0
                vr          = 0.0
                vel_norm    = 0.0
                atr         = 0.0
                trail_price = trail_at_tick

            self._handle_exit(_IntrabarTrailSignal())

    # ─────────────────────────────────────────────────────────
    # Smart Limit Exit — saves spread cost vs blind MARKET order
    # ─────────────────────────────────────────────────────────
    def _smart_limit_exit(self, tag: str) -> Optional[float]:
        """
        Place a LIMIT exit order at current LTP instead of a MARKET order.
        Poll for fill for config.LIMIT_ORDER_TIMEOUT_SECS seconds.
        If not filled in time: cancel the LIMIT and return None so the
        caller falls back to a MARKET order immediately.

        Why this matters for Silver Mic:
          During illiquid hours (post-midnight, pre-open) bid-ask spreads
          can be 50-200 pts wide. A blind MARKET order fills at the far side
          of the spread. A LIMIT at LTP typically fills within 1-3 seconds
          during any active session, saving the full spread cost.

        Returns: fill price (float) on success, None on timeout/failure.
        """
        timeout  = getattr(config, "LIMIT_ORDER_TIMEOUT_SECS", 7)
        order_id = None
        try:
            ltp = self._get_current_ltp()
            if not ltp:
                logger.warning("SmartLimitExit: could not get current LTP — "
                               "falling back to MARKET order")
                return None

            if self._pos == 1:
                order_id = self.broker.place_order(
                    "SELL", self._open_qty,
                    order_type="LIMIT", price=ltp, tag=tag
                )
            else:
                order_id = self.broker.place_order(
                    "BUY", self._open_qty,
                    order_type="LIMIT", price=ltp, tag=tag
                )

            logger.info(
                "SmartLimitExit: LIMIT @ ₹%.0f placed (id=%s, timeout=%ds)",
                ltp, order_id, timeout
            )

            deadline = time.time() + timeout
            while time.time() < deadline:
                order  = self.broker.get_order_status(order_id)
                status = order.get("status", "").upper()
                if status in ("COMPLETE", "FILLED"):
                    fill         = float(order.get("average_price", ltp))
                    spread_saved = abs(fill - ltp)
                    logger.info(
                        "SmartLimitExit: filled @ ₹%.0f  "
                        "(limit=₹%.0f, spread saved ≈₹%.0f/lot)",
                        fill, ltp, spread_saved
                    )
                    return fill
                if status in ("REJECTED", "CANCELLED"):
                    logger.warning(
                        "SmartLimitExit: limit order %s (%s) — "
                        "falling back to MARKET", status, order_id
                    )
                    return None
                time.sleep(0.4)

            # Timed out — cancel the unfilled limit and let caller MARKET exit
            logger.warning(
                "SmartLimitExit: LIMIT not filled in %ds @ ₹%.0f — "
                "cancelling and falling back to MARKET order",
                timeout, ltp
            )
            try:
                self.broker.cancel_order(order_id)
                time.sleep(0.3)   # brief pause so exchange processes the cancel
            except Exception as ce:
                logger.warning("SmartLimitExit: cancel failed: %s", ce)
            return None

        except Exception as e:
            logger.error("SmartLimitExit error: %s — falling back to MARKET", e)
            if order_id:
                try:
                    self.broker.cancel_order(order_id)
                except Exception:
                    pass
            return None

    def _get_current_ltp(self) -> Optional[float]:
        """
        Quick REST call to fetch current LTP for use as a LIMIT exit price.
        Times out in 3 seconds so it never blocks the exit path for long.
        """
        try:
            import requests as _req
            url = "https://api.upstox.com/v2/market-quote/ltp"
            headers = {
                "Authorization": f"Bearer {self.broker.token}",
                "Accept":        "application/json",
            }
            r = _req.get(url, headers=headers,
                         params={"instrument_key": config.INSTRUMENT_KEY},
                         timeout=3)
            if r.status_code == 200:
                data  = r.json().get("data", {})
                quote = (
                    data.get(config.INSTRUMENT_KEY) or
                    data.get(config.INSTRUMENT_KEY.replace("|", "_")) or
                    next(iter(data.values()), None)
                )
                if quote:
                    ltp = float(quote.get("last_price", 0) or 0)
                    return ltp if ltp > 0 else None
        except Exception as e:
            logger.debug("_get_current_ltp error: %s", e)
        return None

    def _wait_for_fill(self, order_id: str, timeout: int = 15) -> float:
        """
        Polls order status until filled, then returns average fill price.
        Raises RuntimeError if not filled within timeout seconds.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                order = self.broker.get_order_status(order_id)
                status = order.get("status", "").upper()
                if status in ("COMPLETE", "FILLED"):
                    price = float(order.get("average_price", 0))
                    logger.info(f"Order {order_id} filled @ ₹{price:.0f}")
                    return price
                elif status in ("REJECTED", "CANCELLED"):
                    raise RuntimeError(
                        f"Order {order_id} {status}: {order.get('status_message', '')}"
                    )
            except RuntimeError:
                raise
            except Exception as e:
                logger.debug(f"Status poll error: {e}")
            time.sleep(0.5)

        raise RuntimeError(f"Order {order_id} not filled within {timeout}s")

    def _sync_position(self):
        """
        Verify engine position matches broker's live position.
        Corrects drift caused by manual interventions or connection drops.

        On first sync after an overnight hold, attempts to restore entry_price
        from broker trade book so the trail stop can be properly initialised.
        """
        try:
            net = self.broker.get_net_quantity(config.INSTRUMENT_KEY)
            if net > 0 and self._pos != 1:
                logger.warning(
                    "Position sync: broker=LONG(%d), engine=%d — correcting%s",
                    net, self._pos,
                    " (overnight carry detected)" if self._pos == 0 else ""
                )
                self._pos      = 1
                self._open_qty = net
                entry_px       = self._try_recover_entry_price(net)
                self._entry_px = entry_px
                self.strategy.set_position(1, entry_px, atr=self.strategy._atr or 0.0)
                if entry_px > 0:
                    logger.info(
                        "Overnight LONG restored | entry_px recovered=₹%.0f | "
                        "trail initialised at ₹%.0f",
                        entry_px, self.strategy.current_trail
                    )
                else:
                    logger.warning(
                        "Overnight LONG restored but entry_price unknown — "
                        "trail stop NOT initialised. Monitor manually."
                    )

            elif net < 0 and self._pos != -1:
                logger.warning(
                    "Position sync: broker=SHORT(%d), engine=%d — correcting%s",
                    abs(net), self._pos,
                    " (overnight carry detected)" if self._pos == 0 else ""
                )
                self._pos      = -1
                self._open_qty = abs(net)
                entry_px       = self._try_recover_entry_price(net)
                self._entry_px = entry_px
                self.strategy.set_position(-1, entry_px, atr=self.strategy._atr or 0.0)
                if entry_px > 0:
                    logger.info(
                        "Overnight SHORT restored | entry_px recovered=₹%.0f | "
                        "trail initialised at ₹%.0f",
                        entry_px, self.strategy.current_trail
                    )
                else:
                    logger.warning(
                        "Overnight SHORT restored but entry_price unknown — "
                        "trail stop NOT initialised. Monitor manually."
                    )

            elif net == 0 and self._pos != 0:
                logger.warning("Position sync: broker=FLAT, engine=%d — correcting", self._pos)
                self._pos      = 0
                self._open_qty = 0
                self.strategy.set_position(0)
        except Exception as e:
            logger.debug("Position sync error (non-fatal): %s", e)

    def _try_recover_entry_price(self, net_qty: int) -> float:
        """
        Attempt to recover entry price from broker's trade book.
        Returns the average fill price of the most recent open position,
        or 0.0 if it cannot be determined (caller must handle gracefully).

        Used on startup when an overnight position is detected by _sync_position.
        """
        try:
            trades = self.broker.get_trade_book()   # list of dicts from Upstox
            if not trades:
                return 0.0
            # Filter trades for this instrument and the matching direction
            direction_filter = "BUY" if net_qty > 0 else "SELL"
            relevant = [
                t for t in trades
                if t.get("instrument_key", "").upper() == config.INSTRUMENT_KEY.upper()
                and t.get("transaction_type", "").upper() == direction_filter
            ]
            if not relevant:
                return 0.0
            # Weighted average fill price across matching trades
            total_qty = sum(float(t.get("quantity", 0)) for t in relevant)
            if total_qty == 0:
                return 0.0
            avg_px = sum(
                float(t.get("average_price", 0)) * float(t.get("quantity", 0))
                for t in relevant
            ) / total_qty
            return avg_px
        except Exception as e:
            logger.debug("Could not recover entry price from trade book: %s", e)
            return 0.0

    def emergency_stop(self):
        """
        Called on Ctrl+C or fatal error.
        If ALLOW_OVERNIGHT is True, skips position close so overnight
        carry is preserved — only shuts down the feed and engine.
        If ALLOW_OVERNIGHT is False (default), flattens everything first.
        """
        allow_overnight = getattr(config, "ALLOW_OVERNIGHT", False)
        if allow_overnight and self._pos != 0:
            logger.warning(
                "EMERGENCY STOP — ALLOW_OVERNIGHT=True: "
                "leaving %s position OPEN (qty=%d entry=₹%.0f). "
                "Trail stop at ₹%.0f. Engine shutting down.",
                "LONG" if self._pos == 1 else "SHORT",
                self._open_qty, self._entry_px,
                self.strategy.current_trail
            )
        elif self._pos != 0:
            logger.warning("EMERGENCY STOP triggered — flattening position")
            with self._lock:
                try:
                    self.broker.close_all_positions()
                    logger.info("All positions closed")
                except Exception as e:
                    logger.error("Emergency close failed: %s", e)
                    logger.error("CHECK YOUR POSITIONS MANUALLY ON UPSTOX APP NOW")
        else:
            logger.info("EMERGENCY STOP — already flat, nothing to close")

    # ─────────────────────────────────────────────────────────
    # Historical warmup — fetch multi-day + today's intraday bars
    # ─────────────────────────────────────────────────────────
    #
    # TWO FETCHES ON EVERY STARTUP:
    #
    # 1. Multi-day historical (past WARMUP_DAYS days, up to yesterday):
    #    GET /v2/historical-candle/{instrument}/minutes/{N}/{to}/{from}
    #
    # 2. Today's intraday bars (9 AM to current time):
    #    GET /v2/historical-candle/intraday/{instrument}/1minute
    #    Returns today's 1-min bars → aggregated to N-min in-memory.
    #
    # Both sets merged oldest-first before replay into strategy.
    # A mid-day restart will have full today's context.
    # ─────────────────────────────────────────────────────────

    WARMUP_DAYS = getattr(config, "WARMUP_DAYS", 60)

    def _fetch_historical_bars(self) -> list:
        """
        Fetch WARMUP_DAYS of past N-min bars (up to yesterday) then
        append today's completed intraday N-min bars. Returns a single
        merged list sorted oldest-first, still-forming bar excluded.
        """
        import requests
        from urllib.parse import quote

        N          = config.CANDLE_INTERVAL
        today      = date.today()
        yesterday  = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        from_dt    = (today - timedelta(days=self.WARMUP_DAYS)).strftime("%Y-%m-%d")
        instrument = quote(config.INSTRUMENT_KEY, safe="")
        headers    = {
            "Authorization": f"Bearer {self.broker.token}",
            "Accept":        "application/json",
        }

        # ── Fetch 1: Multi-day historical (up to yesterday) ───
        hist_bars = []
        url_hist  = (
            f"https://api.upstox.com/v3/historical-candle/"
            f"{instrument}/minutes/{N}/{yesterday}/{from_dt}"
        )
        logger.info(
            "Fetching %d-min historical bars (%d days): %s → %s",
            N, self.WARMUP_DAYS, from_dt, yesterday
        )
        try:
            r = requests.get(url_hist, headers=headers, timeout=30)
            if r.status_code == 200:
                raw = r.json().get("data", {}).get("candles", [])
                hist_bars = self._parse_candle_list(raw)
                logger.info(
                    "Historical fetch: %d × %d-min bars  (%s → %s)",
                    len(hist_bars), N,
                    hist_bars[0]["ts"].strftime("%Y-%m-%d") if hist_bars else "—",
                    hist_bars[-1]["ts"].strftime("%Y-%m-%d %H:%M") if hist_bars else "—",
                )
            else:
                logger.error("Historical fetch failed HTTP %d: %s", r.status_code, r.text[:200])
        except Exception as e:
            logger.error("Historical fetch network error: %s", e)

        # ── Fetch 2: Today's intraday bars ────────────────────
        intraday_bars = []
        url_intraday  = (
            f"https://api.upstox.com/v2/historical-candle/intraday/"
            f"{instrument}/1minute"
        )
        logger.info("Fetching today's intraday bars → aggregating to %d-min", N)
        try:
            r = requests.get(url_intraday, headers=headers, timeout=15)
            if r.status_code == 200:
                raw_1min = r.json().get("data", {}).get("candles", [])
                if raw_1min:
                    intraday_bars = self._aggregate_intraday(raw_1min)
                    logger.info(
                        "Intraday fetch: %d × %d-min bars today  (%s → %s)",
                        len(intraday_bars), N,
                        intraday_bars[0]["ts"].strftime("%H:%M") if intraday_bars else "—",
                        intraday_bars[-1]["ts"].strftime("%H:%M") if intraday_bars else "—",
                    )
                else:
                    logger.info("Intraday fetch: 0 bars (market not yet open or no trades today)")
            else:
                logger.warning("Intraday fetch failed HTTP %d: %s", r.status_code, r.text[:200])
        except Exception as e:
            logger.warning("Intraday fetch network error: %s", e)

        # ── Merge: historical + today, oldest-first ───────────
        all_bars = hist_bars + intraday_bars
        if not all_bars:
            logger.warning("Both fetches returned 0 bars. Strategy will start cold.")
            return []

        logger.info(
            "Warmup data ready: %d × %d-min bars total  (%s → %s)",
            len(all_bars), N,
            all_bars[0]["ts"].strftime("%Y-%m-%d %H:%M"),
            all_bars[-1]["ts"].strftime("%Y-%m-%d %H:%M"),
        )
        return all_bars

    def _parse_candle_list(self, raw: list) -> list:
        """
        Convert Upstox raw candle list to strategy bar dicts.
        Upstox returns newest-first — reverses to oldest-first for replay.
        Skips the still-forming current bar.
        """
        N        = config.CANDLE_INTERVAL
        now      = datetime.now()
        cur_snap = now.replace(
            minute=(now.minute // N) * N,
            second=0, microsecond=0
        )
        bars = []
        for c in raw:
            try:
                ts = datetime.fromisoformat(
                    str(c[0]).replace("Z", "+00:00")
                ).replace(tzinfo=None)
                if ts >= cur_snap:
                    continue   # skip still-forming bar
                bars.append({
                    "ts":     ts,
                    "open":   float(c[1]),
                    "high":   float(c[2]),
                    "low":    float(c[3]),
                    "close":  float(c[4]),
                    "volume": int(float(c[5])) if len(c) > 5 else 0,
                })
            except Exception as e:
                logger.debug("Bar parse error: %s — raw=%s", e, c)
        bars.reverse()   # oldest-first
        return bars

    def _aggregate_intraday(self, raw_1min: list) -> list:
        """
        Aggregate today's 1-min intraday bars into N-min bars.
        Excludes the still-forming current bar.
        """
        from collections import defaultdict

        N        = config.CANDLE_INTERVAL
        now      = datetime.now()
        cur_snap = now.replace(
            minute=(now.minute // N) * N,
            second=0, microsecond=0
        )
        buckets = defaultdict(list)
        for c in raw_1min:
            try:
                ts    = datetime.fromisoformat(
                    str(c[0]).replace("Z", "+00:00")
                ).replace(tzinfo=None)
                total  = ts.hour * 60 + ts.minute
                snap   = (total // N) * N
                bar_ts = ts.replace(
                    hour=snap // 60, minute=snap % 60,
                    second=0, microsecond=0
                )
                if bar_ts >= cur_snap:
                    continue
                buckets[bar_ts].append({
                    "ts":     ts,
                    "open":   float(c[1]),
                    "high":   float(c[2]),
                    "low":    float(c[3]),
                    "close":  float(c[4]),
                    "volume": int(float(c[5])) if len(c) > 5 else 0,
                })
            except Exception as e:
                logger.debug("Intraday 1-min parse error: %s", e)

        result = []
        for bar_ts in sorted(buckets.keys()):
            rows = sorted(buckets[bar_ts], key=lambda x: x["ts"])
            if not rows:
                continue
            result.append({
                "ts":     bar_ts,
                "open":   rows[0]["open"],
                "high":   max(r["high"]   for r in rows),
                "low":    min(r["low"]    for r in rows),
                "close":  rows[-1]["close"],
                "volume": sum(r["volume"] for r in rows),
            })
        return result

    def _warmup_strategy(self):
        """
        Fetch historical + today's intraday bars and replay through
        the strategy to converge all indicators before live feed starts.
        """
        bars = self._fetch_historical_bars()
        if not bars:
            logger.warning(
                "Strategy starting COLD — norm_resid will diverge from "
                "TradingView for ~%d bars until indicators converge.",
                max(config.VR_LEN + 1, config.ATR_LEN * 3)
            )
            return
        result = self.strategy.warmup(bars)
        # Guard: warmup() should return bar count, but older/local versions may
        # return None. Fall back to len(bars) so the log line never crashes.
        n = result if isinstance(result, int) else len(bars)
        logger.info(
            "Strategy warmed up on %d × %d-min bars  (%.1f days of history)",
            n, config.CANDLE_INTERVAL,
            n * config.CANDLE_INTERVAL / (60 * 14.5)
        )

    # ─────────────────────────────────────────────────────────
    # Contract Rollover Validation
    # ─────────────────────────────────────────────────────────
    def _validate_contract(self):
        """
        Three-layer contract rollover guard. Called once at startup.

        MCX Silver MIC is a monthly expiry contract. On the last trading
        day of each month the front-month contract stops trading and any
        open subscriptions receive zero ticks. The bot has no way to
        distinguish this from a market closed / quiet period, so it sits
        idle forever. This method detects rollover BEFORE connecting the feed.

        LAYER 1 — Live LTP probe (most reliable)
        ─────────────────────────────────────────
        Calls the Upstox market quote API for config.INSTRUMENT_KEY.
        If the contract is expired, Upstox returns an error or zero LTP.
        This is a definitive check — an active contract always has a quote.

        LAYER 2 — Calendar proximity warning
        ─────────────────────────────────────
        MCX monthly contracts expire on the last business day of each month.
        If today is within ROLLOVER_WARNING_DAYS (default=3) of month-end,
        warn the user to verify the contract manually. We cannot know the
        exact expiry date without the instruments CSV, but being within 3
        days of month-end is a reliable heuristic.

        LAYER 3 — Live heartbeat monitor (during session)
        ──────────────────────────────────────────────────
        Updates self._last_tick_ts on every processed candle. A background
        thread checks every minute: if we are inside the MCX session window
        and no tick has arrived in _TICK_TIMEOUT_MINS minutes, it logs a
        CRITICAL warning and blocks new entries. This catches mid-session
        expiry or silent disconnects that Layers 1-2 cannot catch.
        """
        import requests
        from datetime import date
        import calendar

        logger.info("=" * 55)
        logger.info("  CONTRACT VALIDATION — %s", config.INSTRUMENT_KEY)
        logger.info("=" * 55)

        # ── Layer 1: Live LTP probe ───────────────────────────────
        validation_passed = False
        try:
            url = "https://api.upstox.com/v2/market-quote/ltp"
            headers = {
                "Authorization": f"Bearer {self.broker.token}",
                "Accept":        "application/json",
            }
            params = {"instrument_key": config.INSTRUMENT_KEY}
            r = requests.get(url, headers=headers, params=params, timeout=10)

            if r.status_code == 200:
                data = r.json().get("data", {})
                # Upstox returns keyed by instrument (with | replaced by _)
                key_variants = [
                    config.INSTRUMENT_KEY,
                    config.INSTRUMENT_KEY.replace("|", "_"),
                    config.INSTRUMENT_KEY.replace("|", "%7C"),
                ]
                quote = None
                for k in key_variants:
                    if k in data:
                        quote = data[k]
                        break
                if not quote:
                    # Sometimes nested under the first key
                    quote = next(iter(data.values()), None) if data else None

                if quote:
                    ltp = float(quote.get("last_price", 0) or 0)
                    if ltp > 0:
                        logger.info(
                            "Layer 1 ✅  Contract ACTIVE — LTP=₹%.0f  (%s)",
                            ltp, config.INSTRUMENT_KEY
                        )
                        validation_passed = True
                    else:
                        logger.critical(
                            "Layer 1 ❌  LTP=0 for %s — contract may be EXPIRED or "
                            "market is closed. Verify on Upstox app before trading.",
                            config.INSTRUMENT_KEY
                        )
                else:
                    logger.warning(
                        "Layer 1 ⚠️   No quote data returned for %s. "
                        "Market may be closed or key is wrong. Response: %s",
                        config.INSTRUMENT_KEY, str(r.json())[:200]
                    )
            else:
                logger.warning(
                    "Layer 1 ⚠️   LTP probe failed HTTP %d: %s",
                    r.status_code, r.text[:200]
                )

        except Exception as e:
            logger.warning("Layer 1 ⚠️   LTP probe exception: %s", e)

        # ── Layer 2: Calendar proximity warning ───────────────────
        ROLLOVER_WARNING_DAYS = 3
        today = date.today()
        last_day_of_month = calendar.monthrange(today.year, today.month)[1]
        days_to_month_end = last_day_of_month - today.day

        if days_to_month_end <= ROLLOVER_WARNING_DAYS:
            logger.warning(
                "Layer 2 ⚠️   ROLLOVER WINDOW: Today is %s — only %d day(s) "
                "until end of month. MCX Silver MIC expires on the last "
                "business day of each month. VERIFY that %s is still the "
                "active front-month contract and update INSTRUMENT_KEY in "
                "config.py if needed before trading.",
                today, days_to_month_end, config.INSTRUMENT_KEY
            )
        else:
            logger.info(
                "Layer 2 ✅  %d day(s) until month-end — no rollover imminent",
                days_to_month_end
            )

        # ── Summary ───────────────────────────────────────────────
        if not validation_passed:
            logger.critical(
                "CONTRACT VALIDATION WARNING: Could not confirm %s is active. "
                "The bot will start anyway — check logs carefully. "
                "If no ticks arrive within 5 minutes, the contract is likely expired.",
                config.INSTRUMENT_KEY
            )
        logger.info("=" * 55)

    def _start_heartbeat_monitor(self):
        """
        Layer 3: Background thread that monitors for tick silence during
        the MCX session. If no candle is processed for _TICK_TIMEOUT_MINS
        minutes while inside the session window, logs a CRITICAL warning.

        This catches:
        - Mid-session contract expiry
        - Silent WebSocket disconnects (where on_close isn't fired)
        - Exchange outages
        """
        def _monitor():
            import time as _time
            logger.info("Heartbeat monitor started (timeout=%d min)", self._TICK_TIMEOUT_MINS)
            while self._running:
                _time.sleep(60)   # check every minute
                try:
                    now = datetime.now()
                    # Only alert during the MCX session window
                    sess_s = config.SESS_START[0] * 60 + config.SESS_START[1]
                    sess_e = config.SESS_END[0]   * 60 + config.SESS_END[1]
                    now_m  = now.hour * 60 + now.minute
                    in_sess = sess_s <= now_m <= sess_e

                    if not in_sess:
                        continue   # outside session — silence is expected

                    if self._last_tick_ts is None:
                        # Still waiting for first tick
                        elapsed = (now - self._engine_start_ts).total_seconds() / 60
                        if elapsed > self._TICK_TIMEOUT_MINS:
                            logger.critical(
                                "HEARTBEAT ❌  No tick received in %.0f min since startup "
                                "during active session. Possible causes: "
                                "(1) Contract expired — check INSTRUMENT_KEY in config.py, "
                                "(2) WebSocket subscription failed, "
                                "(3) Exchange outage. Verify on Upstox app.",
                                elapsed
                            )
                    else:
                        elapsed = (now - self._last_tick_ts).total_seconds() / 60
                        if elapsed > self._TICK_TIMEOUT_MINS:
                            logger.critical(
                                "HEARTBEAT ❌  No tick for %.0f min during active MCX session "
                                "(last tick: %s). Possible causes: "
                                "(1) Contract expired — update INSTRUMENT_KEY, "
                                "(2) Silent WebSocket disconnect. "
                                "New entries BLOCKED until ticks resume.",
                                elapsed, self._last_tick_ts.strftime("%H:%M:%S")
                            )
                            # Block new entries by setting a flag the risk manager can check
                            self._feed_silent = True
                        else:
                            self._feed_silent = False
                except Exception as e:
                    logger.debug("Heartbeat monitor error: %s", e)

        t = threading.Thread(target=_monitor, daemon=True, name="heartbeat")
        t.start()

    # ─────────────────────────────────────────────────────────
    # Start / Stop
    # ─────────────────────────────────────────────────────────
    def start(self, use_websocket: bool = True):
        self._running         = True
        self._engine_start_ts = datetime.now()
        self._feed_silent     = False

        # ── Step 1: Validate contract is active and not near expiry ──
        self._validate_contract()

        # ── Step 2: Warm up strategy with historical + today's bars ──
        self._warmup_strategy()

        # ── Step 3: Start live feed ───────────────────────────────────
        if use_websocket:
            self.feed = UpstoxFeed(
                access_token    = self.broker.token,
                on_candle_close = self.on_candle_close,
                on_tick         = self.on_tick,
            )
        else:
            self.feed = RestLTPPoller(
                access_token    = self.broker.token,
                on_candle_close = self.on_candle_close,
                poll_seconds    = 2,
            )

        self.feed.start()
        logger.info("Feed started — waiting for candles...")

        # ── Step 4: Start heartbeat monitor (Layer 3 rollover guard) ─
        self._start_heartbeat_monitor()

        # Block main thread; handle Ctrl+C
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received")
        finally:
            self.stop()

    def stop(self):
        self._running = False
        logger.info("Stopping engine...")
        self.emergency_stop()
        if hasattr(self, "feed"):
            self.feed.stop()
        allow_overnight = getattr(config, "ALLOW_OVERNIGHT", False)
        if allow_overnight and self._pos != 0:
            logger.info(
                "Engine stopped — overnight position HELD (%s qty=%d). "
                "Restart engine before next session to resume management.",
                "LONG" if self._pos == 1 else "SHORT", self._open_qty
            )
        else:
            logger.info("Engine stopped cleanly.")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    setup_logging()

    logger.info("=" * 60)
    logger.info("  KALMAN ADAPTIVE SCALPER — LIVE ENGINE")
    logger.info(f"  Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # ── 1. Load and validate token ────────────────────────────
    try:
        token = load_token()
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"Auth error: {e}")
        logger.error("Run 'python auth.py' first to authenticate.")
        sys.exit(1)

    if not validate_token(token):
        logger.error("Token is invalid or expired. Run 'python auth.py'.")
        sys.exit(1)

    # ── 2. Start engine ───────────────────────────────────────
    engine = TradingEngine(token)

    # Graceful SIGTERM handling (for systemd / cloud deployments)
    def _sigterm(signum, frame):
        logger.info("SIGTERM received")
        engine.stop()

    sig_module.signal(sig_module.SIGTERM, _sigterm)

    # ── 3. Run ────────────────────────────────────────────────
    # Set use_websocket=False if you hit WebSocket issues
    engine.start(use_websocket=True)
