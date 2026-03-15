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

FIXES vs PREVIOUS VERSION:
  BUG 1 — strategy.set_position() missing atr= parameter.
           Engine called set_position(direction, fill_px, atr=signal.atr)
           which crashed with TypeError on every trade entry.
           Fixed in strategy.py — engine calls remain unchanged.

  BUG 2 — strategy.current_trail property did not exist.
           on_tick(), emergency_stop(), _sync_position(), and the new
           _ratchet_broker_sl() all reference self.strategy.current_trail.
           Previously this raised AttributeError on every tick while in
           a position, making intrabar trail exits completely non-functional.
           Fixed in strategy.py — engine references remain unchanged.

  BUG 3 — RestLTPPoller instantiated with unsupported poll_seconds= kwarg.
           RESTCandleFeed.__init__ only accepts (access_token, on_candle_close).
           The extra argument crashed Tier 3 fallback with TypeError.
           Fixed: poll_seconds=2 removed from the RestLTPPoller call.

  BUG 4 — No broker-level stop loss order (process crash = unprotected position).
           Implemented: SL-M order placed at broker immediately after every
           entry fill. Trail is ratcheted on each bar close. SL is cancelled
           before every Python-initiated exit (avoids double-close). SL is
           preserved (not cancelled) on emergency stop with ALLOW_OVERNIGHT=True.

  BUG 5 — on_reconnect() read _last_bar_ts before acquiring _lock.
           A concurrent on_candle_close() could update _last_bar_ts between
           the read and the lock, causing one already-processed bar to be
           replayed through strategy.process_bar(), advancing indicators by
           one phantom bar.
           Fixed: _last_bar_ts snapshot taken inside the lock.

  BUG 6 — emergency_stop() did not reset engine position state after
           broker.close_all_positions(). If stop() somehow called
           emergency_stop() twice, the second call would attempt to close
           an already-flat position.
           Fixed: _pos, _open_qty, strategy.set_position(0), _sl_order_id
           all reset after a successful flat-on-stop close.

  BUG 7 — logger.info(f"P&L recorded: ₹{pnl:+,.0f} ...") used ,:+,.0f
           Python's %-style logging formatter doesn't support the comma
           thousands-separator — raises ValueError on every trade close.
           Fixed: f-string formatted numbers passed as %s arguments.
"""

import csv
import logging
import math
import os
import signal as sig_module
import sys
import threading
import time
from datetime import datetime, date, timedelta
from typing import Optional

import numpy as np

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

    # Windows fix: force UTF-8 on all handlers.
    # Windows terminals default to cp1252 which cannot encode the rupee sign.
    import io
    if hasattr(sys.stdout, "buffer"):
        utf8_stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    else:
        utf8_stdout = sys.stdout

    console_handler = logging.StreamHandler(utf8_stdout)
    console_handler.setFormatter(logging.Formatter(fmt))

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
                "Pre-existing P&L from broker: Rs.%s  "
                "(manual trades or earlier sessions counted toward daily limit)",
                f"{self._daily_pnl:+,.0f}"
            )

    def _check_reset(self):
        today = date.today()
        if today != self._last_reset:
            logger.info(
                "New day — resetting daily counters. "
                "Yesterday P&L: %s | Trades: %d",
                f"₹{self._daily_pnl:+,.0f}", self._daily_trades
            )
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
                margin       = self._broker.get_funds()
                min_required = 8000 * qty   # rough Silver MIC margin estimate
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
            # FIX BUG 7: was logger.info(f"...₹{pnl:+,.0f}...") which uses
            # %-style logging internally and chokes on the ',' formatter flag.
            # Fixed: pre-format numbers as strings and pass via %s.
            logger.info(
                "P&L recorded: %s | Daily total: %s",
                f"₹{pnl:+,.0f}", f"₹{self._daily_pnl:+,.0f}"
            )

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
              "norm_resid", "vr", "vel_norm", "atr", "order_id", "sl_order_id"]

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._lock    = threading.Lock()
        if not os.path.exists(filepath):
            with open(filepath, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def record(self, **kwargs):
        row = {k: kwargs.get(k, "") for k in self.FIELDS}
        row["datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            with open(self.filepath, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)
        logger.info(
            "Trade logged: %s | price=%s | pnl=%s",
            row["action"], row["fill_price"], row["pnl"]
        )


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
        self._lock    = threading.Lock()
        self.broker   = UpstoxBroker(access_token)
        self.strategy = KalmanScalperStrategy()
        self.risk     = RiskManager(self.broker)
        self.tlog     = TradeLogger(config.TRADE_LOG_CSV)
        self._running = False

        # Position tracking (engine-side, verified against broker each bar)
        self._pos        = 0       # +1 long | -1 short | 0 flat
        self._entry_px   = 0.0
        self._open_qty   = 0
        self._entry_ts   = None
        self._order_id   = None    # last entry/exit order id

        # Broker-side stop loss tracking (BUG 4 fix)
        # An SL-M order is placed at the broker immediately after every entry.
        # It is cancelled before every Python-initiated exit (reversion / intrabar).
        # It is NOT cancelled when ALLOW_OVERNIGHT=True on engine stop — it
        # acts as the overnight protection if the process is not running.
        self._sl_order_id    = None    # active broker SL-M order id
        self._sl_trail_level = np.nan  # price at which broker SL-M was last placed

        # Intrabar exit guard — prevents double-exit within the same bar.
        # Reset to False at the top of every on_candle_close().
        self._intrabar_exit_fired = False

        # Timestamp of the last bar successfully processed by on_candle_close.
        # Used by on_reconnect() to identify and replay only missed bars.
        self._last_bar_ts: Optional[datetime] = None

        # Heartbeat: track last tick time for rollover/disconnect detection
        self._last_tick_ts: Optional[datetime]  = None
        self._TICK_TIMEOUT_MINS                 = 10

        # Compute position sizing
        self._quantity = max(1, config.LOT_SIZE)

        logger.info("TradingEngine initialised")
        logger.info("  Instrument : %s (%s)", config.INSTRUMENT_NAME, config.INSTRUMENT_KEY)
        logger.info("  Quantity   : %d lot(s)", self._quantity)
        logger.info("  Risk/trade : %s%%", config.EQUITY_RISK_PCT)
        logger.info("  Max loss   : ₹%s/day", f"{config.MAX_DAILY_LOSS:,}")

        # Overnight config cross-check
        allow_overnight = getattr(config, "ALLOW_OVERNIGHT", False)
        product_type    = getattr(config, "PRODUCT_TYPE", "D")
        logger.info(
            "  Overnight  : %s",
            "ENABLED — positions will be held across sessions"
            if allow_overnight else
            "DISABLED — positions will be flattened on stop"
        )
        if allow_overnight and product_type.upper() in ("I", "MIS", "INTRADAY"):
            logger.critical(
                "CONFIG CONFLICT: ALLOW_OVERNIGHT=True but PRODUCT_TYPE='%s' is an "
                "intraday product — the broker will AUTO-SQUARE OFF your position at "
                "session end regardless. Set PRODUCT_TYPE='D' in config.py.",
                product_type
            )
        if not allow_overnight and product_type.upper() in ("D", "NRML", "DELIVERY"):
            logger.info(
                "  Note: PRODUCT_TYPE='%s' supports overnight but ALLOW_OVERNIGHT=False"
                " — positions will still be flattened on engine stop.",
                product_type
            )

    # ─────────────────────────────────────────────────────────
    # Candle handler — called by feed on each completed bar
    # ─────────────────────────────────────────────────────────
    def on_candle_close(self, candle: dict):
        """
        Entry point for every completed N-min candle.
        This is the strategy's heartbeat.
        """
        with self._lock:
            # Reset intrabar exit guard so it can fire again on this new bar
            self._intrabar_exit_fired = False

            ts = candle["ts"]

            # Update heartbeat (Layer 3 rollover guard)
            self._last_tick_ts = datetime.now()

            logger.info(
                "── Candle %s | O=%.0f H=%.0f L=%.0f C=%.0f V=%s ──",
                ts,
                candle["open"], candle["high"], candle["low"], candle["close"],
                f"{candle['volume']:,}"
            )

            # 1. Sync position with broker (every bar)
            self._sync_position()

            # 1b. End-of-session forced exit (if ALLOW_OVERNIGHT=False)
            allow_overnight = getattr(config, "ALLOW_OVERNIGHT", False)
            if not allow_overnight and self._pos != 0 and config.USE_SESSION:
                sess_e_mins = config.SESS_END[0] * 60 + config.SESS_END[1]
                bar_mins    = ts.hour * 60 + ts.minute
                if bar_mins >= sess_e_mins - config.CANDLE_INTERVAL:
                    logger.warning(
                        "SESSION END APPROACH — ALLOW_OVERNIGHT=False: "
                        "forcing position close (bar=%02d:%02d sess_end=%02d:%02d).",
                        ts.hour, ts.minute, config.SESS_END[0], config.SESS_END[1]
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
                    self._last_bar_ts = ts
                    return

            # 2. Run strategy
            signal: Optional[Signal] = self.strategy.process_bar(candle)

            # 2b. If still in position after bar processing (no exit signal),
            #     ratchet the broker SL-M to the updated Python trail level.
            if signal is None and self._pos != 0:
                self._ratchet_broker_sl()

            if signal is None:
                self._last_bar_ts = ts
                return

            logger.info(
                "Signal: %s | %s | normResid=%+.3f",
                signal.action, signal.reason, signal.norm_resid
            )

            # 3. Route signal to action
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

            self._last_bar_ts = ts

    # ─────────────────────────────────────────────────────────
    # Reconnect catch-up
    # ─────────────────────────────────────────────────────────
    def on_reconnect(self):
        """
        Called by the feed whenever the WebSocket reconnects after a drop.
        Fetches only today's intraday N-min bars and replays just the bars
        that are newer than _last_bar_ts to keep indicators in sync.
        """
        import requests as _req
        from urllib.parse import quote

        logger.info("=" * 55)
        logger.info("  RECONNECT CATCH-UP — fetching missed bars")
        logger.info(
            "  Last processed bar: %s",
            self._last_bar_ts.strftime("%H:%M:%S") if self._last_bar_ts else "none"
        )
        logger.info("=" * 55)

        N          = config.CANDLE_INTERVAL
        instrument = quote(config.INSTRUMENT_KEY, safe="")
        headers    = {
            "Authorization": f"Bearer {self.broker.token}",
            "Accept":        "application/json",
        }
        url = (
            f"https://api.upstox.com/v2/historical-candle/intraday/"
            f"{instrument}/1minute"
        )

        try:
            r = _req.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                logger.warning(
                    "Catch-up fetch failed HTTP %d — resuming without replay. "
                    "Indicators may be stale for up to %d bars.",
                    r.status_code, N
                )
                return

            raw_1min = r.json().get("data", {}).get("candles", [])
            if not raw_1min:
                logger.info("Catch-up: no intraday bars returned (market closed?)")
                return

            today_bars = self._aggregate_intraday(raw_1min)

            # FIX BUG 5: _last_bar_ts snapshot + missed-bar filter both happen
            # inside the lock, eliminating the race with on_candle_close().
            with self._lock:
                cutoff = self._last_bar_ts
                missed = (
                    [b for b in today_bars if b["ts"] > cutoff]
                    if cutoff is not None
                    else today_bars
                )

                if not missed:
                    logger.info("Catch-up: no missed bars detected — indicators are current")
                    return

                logger.info(
                    "Catch-up: replaying %d missed bar(s) [%s → %s]",
                    len(missed),
                    missed[0]["ts"].strftime("%H:%M"),
                    missed[-1]["ts"].strftime("%H:%M"),
                )

                self.strategy._warming_up = True
                for bar in missed:
                    try:
                        self.strategy.process_bar(bar)
                        self._last_bar_ts = bar["ts"]
                    except Exception as e:
                        logger.error("Catch-up bar error: %s", e)
                self.strategy._warming_up = False

            logger.info("Catch-up complete — strategy indicators are up to date")

        except Exception as e:
            logger.error(
                "Catch-up exception: %s — resuming without replay. "
                "Indicators may be slightly stale.", e
            )

    # ─────────────────────────────────────────────────────────
    # Entry handler
    # ─────────────────────────────────────────────────────────
    def _handle_entry(self, signal: Signal, direction: int):
        if self._pos != 0:
            logger.warning("Entry signal ignored — already in position (%d)", self._pos)
            return

        if getattr(self, "_feed_silent", False):
            logger.critical(
                "Entry BLOCKED — feed is silent (no ticks for %d+ min). "
                "Possible expired contract. Verify INSTRUMENT_KEY in config.py.",
                self._TICK_TIMEOUT_MINS
            )
            return

        approved, reason = self.risk.approve_entry(signal, self._quantity)
        if not approved:
            logger.warning("Entry BLOCKED by risk manager: %s", reason)
            return

        try:
            tag      = f"K_{signal.action}_{signal.bar_ts.strftime('%H%M')}"
            order_id = (
                self.broker.enter_long(self._quantity, tag=tag)
                if direction == 1
                else self.broker.enter_short(self._quantity, tag=tag)
            )

            fill_px = self._wait_for_fill(order_id, timeout=10)

            self._pos      = direction
            self._entry_px = fill_px
            self._open_qty = self._quantity
            self._entry_ts = signal.bar_ts
            self._order_id = order_id

            # Inform strategy — seeds the ATR trail immediately from fill price
            self.strategy.set_position(direction, fill_px, atr=signal.atr)
            self.risk.record_trade_open()

            # ── BUG 4 FIX: Place a broker-side SL-M order immediately ────────
            # This protects the position even if the Python process crashes.
            # The SL-M price = current Python trail (seeded just above by
            # set_position). It is ratcheted upward on each subsequent bar close.
            self._place_broker_sl(signal.atr)

            self.tlog.record(
                action      = signal.action,
                reason      = signal.reason,
                fill_price  = fill_px,
                quantity    = self._quantity,
                pnl         = 0,
                daily_pnl   = self.risk.daily_pnl,
                norm_resid  = round(signal.norm_resid, 4),
                vr          = round(signal.vr, 4),
                vel_norm    = round(signal.vel_norm, 4),
                atr         = round(signal.atr, 1),
                order_id    = order_id,
                sl_order_id = self._sl_order_id or "",
            )

            logger.info(
                "%s ENTERED | fill=₹%.0f | qty=%d | entry_id=%s | sl_id=%s | trail=₹%.0f",
                "LONG" if direction == 1 else "SHORT",
                fill_px, self._quantity, order_id,
                self._sl_order_id or "none",
                self.strategy.current_trail
            )

        except Exception as e:
            logger.error("Entry order FAILED: %s", e)

    # ─────────────────────────────────────────────────────────
    # Exit handler
    # ─────────────────────────────────────────────────────────
    def _handle_exit(self, signal: Signal):
        if self._pos == 0:
            logger.warning("Exit signal ignored — already flat")
            return

        approved, reason = self.risk.approve_exit()
        if not approved:
            logger.warning("Exit blocked: %s", reason)
            return

        try:
            tag = f"K_EXIT_{signal.bar_ts.strftime('%H%M')}"

            # ── BUG 4 FIX: Cancel broker SL-M before placing the exit order ──
            # This prevents the SL-M from triggering after we've already closed.
            # We don't cancel for ATR Trail exits that were themselves triggered
            # by the broker SL-M (in that case the SL-M is already consumed).
            if signal.reason not in ("ATR Trail Broker",):
                self._cancel_broker_sl(reason="pre-exit cancel")

            fill_px  = None
            order_id = None

            # Smart Limit Exit (slippage control) — attempt LIMIT at LTP first.
            # Disabled for ATR Trail and Session End exits where speed matters.
            use_smart = (
                getattr(config, "SMART_LIMIT_EXITS", False) and
                signal.reason not in (
                    "ATR Trail Intrabar",
                    "ATR Trail Fallback",
                    "ATR Trail",
                    "Session End (ALLOW_OVERNIGHT=False)",
                )
            )

            if use_smart:
                fill_px = self._smart_limit_exit(tag)
                if fill_px is not None:
                    order_id = "SMART_LIMIT"

            # MARKET fallback (or primary if smart exits disabled)
            if fill_px is None:
                if self._pos == 1:
                    order_id = self.broker.exit_long(self._open_qty, tag=tag)
                else:
                    order_id = self.broker.exit_short(self._open_qty, tag=tag)
                fill_px = self._wait_for_fill(order_id, timeout=10)

            # Calculate P&L
            if self._pos == 1:
                pnl = (fill_px - self._entry_px) * self._open_qty
            else:
                pnl = (self._entry_px - fill_px) * self._open_qty

            self.risk.record_pnl(pnl)

            self.tlog.record(
                action      = f"EXIT_{'LONG' if self._pos == 1 else 'SHORT'}",
                reason      = signal.reason,
                fill_price  = fill_px,
                quantity    = self._open_qty,
                pnl         = round(pnl, 2),
                daily_pnl   = round(self.risk.daily_pnl, 2),
                norm_resid  = round(signal.norm_resid, 4),
                vr          = round(signal.vr, 4),
                vel_norm    = round(signal.vel_norm, 4),
                atr         = round(signal.atr, 1),
                order_id    = order_id,
                sl_order_id = "",
            )

            logger.info(
                "EXIT | fill=₹%.0f | P&L=%s | reason=%s",
                fill_px, f"₹{pnl:+,.0f}", signal.reason
            )

            self._pos            = 0
            self._entry_px       = 0.0
            self._open_qty       = 0
            self._sl_order_id    = None
            self._sl_trail_level = np.nan
            self.strategy.set_position(0)

        except Exception as e:
            logger.error("Exit order FAILED: %s", e)
            logger.error("MANUAL INTERVENTION MAY BE REQUIRED — CHECK POSITION")

    # ─────────────────────────────────────────────────────────
    # Broker SL-M management — BUG 4 FIX
    # ─────────────────────────────────────────────────────────
    def _place_broker_sl(self, atr: float):
        """
        Place a SL-M (Stop Loss Market) order at the broker at the current
        Python trail level, immediately after a new position is entered.

        Upstox SL-M order via POST /v2/order/place:
          order_type    = "SL-M"
          trigger_price = trail stop level
          price         = 0  (market execution when trigger is hit)
          transaction_type = "SELL" for long | "BUY" for short
          quantity      = open lot count
          product       = config.PRODUCT_TYPE  ("D" = NRML, "I" = MIS)
          validity      = "DAY"
          tag           = order tag for identification

        The SL-M acts as a hard backstop:
          - If the Python process crashes, network dies, or the machine
            reboots, the broker SL-M still protects the position.
          - If price gaps through the trigger at open, the SL-M becomes
            a market order and exits at the best available price.
          - The Python trail ratchets this level upward on each bar close
            via _ratchet_broker_sl().

        Note: DAY validity means the SL-M expires at market close.
        For overnight positions (ALLOW_OVERNIGHT=True), the SL-M is
        NOT cancelled on engine stop so it covers the overnight gap risk.
        The engine re-places it at startup via _sync_position() → set_position().
        """
        trail = self.strategy.current_trail
        if math.isnan(trail):
            logger.warning(
                "Could not place broker SL-M — trail is NaN immediately after entry. "
                "This means signal.atr was 0. Python trail will be active from next bar."
            )
            return

        direction = "SELL" if self._pos == 1 else "BUY"
        tag       = f"K_SL_{datetime.now().strftime('%H%M%S')}"

        try:
            sl_id = self.broker.place_stop_order(
                direction, self._open_qty, trail, tag=tag
            )
            self._sl_order_id    = sl_id
            self._sl_trail_level = trail
            logger.info(
                "Broker SL-M placed | direction=%s | trigger=₹%.0f | id=%s",
                direction, trail, sl_id
            )
        except Exception as e:
            logger.error(
                "Broker SL-M placement FAILED: %s — Python trail still active, "
                "but no exchange-level protection. Monitor manually.",
                e
            )

    def _ratchet_broker_sl(self):
        """
        Called after each bar close while in a position and no exit signal fired.

        Reads the updated Python trail (which process_bar() already ratcheted)
        and, if it has moved, cancels the old SL-M and places a new one at the
        higher (for longs) or lower (for shorts) level.

        Skips the cancel/replace if the trail hasn't moved, to avoid unnecessary
        API calls (Silver MIC generates ~180 bars/day; many bars won't move the
        trail if price is choppy within the existing range).
        """
        if self._pos == 0:
            return

        new_trail = self.strategy.current_trail
        if math.isnan(new_trail):
            return

        # Skip if trail hasn't moved since last broker SL placement
        if not math.isnan(self._sl_trail_level) and new_trail == self._sl_trail_level:
            return

        direction = "SELL" if self._pos == 1 else "BUY"
        tag       = f"K_SL_{datetime.now().strftime('%H%M%S')}"

        # Cancel the old SL-M first
        if self._sl_order_id:
            try:
                self.broker.cancel_order(self._sl_order_id)
                logger.debug("Old SL-M %s cancelled (ratchet)", self._sl_order_id)
            except Exception as ce:
                # Non-fatal: old order may have already been filled/expired
                logger.debug("SL cancel during ratchet (may be gone): %s", ce)

        # Place new SL-M at updated trail level
        try:
            new_sl_id = self.broker.place_stop_order(
                direction, self._open_qty, new_trail, tag=tag
            )
            self._sl_order_id    = new_sl_id
            self._sl_trail_level = new_trail
            logger.info(
                "Broker SL-M ratcheted to ₹%.0f | id=%s",
                new_trail, new_sl_id
            )
        except Exception as e:
            logger.warning(
                "Broker SL-M ratchet failed: %s — old level was ₹%.0f, "
                "Python trail is now ₹%.0f. Will retry next bar.",
                e, self._sl_trail_level, new_trail
            )

    def _cancel_broker_sl(self, reason: str = ""):
        """
        Cancel the active broker SL-M order.
        Called before every Python-initiated exit to avoid a double-close.
        Failures are non-fatal (logged as warning) so the exit still proceeds.
        """
        if not self._sl_order_id:
            return
        try:
            self.broker.cancel_order(self._sl_order_id)
            logger.info(
                "Broker SL-M %s cancelled%s",
                self._sl_order_id,
                f" ({reason})" if reason else ""
            )
        except Exception as e:
            logger.warning(
                "Broker SL-M cancel failed (id=%s, reason=%s): %s — "
                "may already be triggered or expired. Proceeding with exit.",
                self._sl_order_id, reason, e
            )
        finally:
            self._sl_order_id    = None
            self._sl_trail_level = np.nan

    # ─────────────────────────────────────────────────────────
    # Trail update handler (kept for compatibility)
    # ─────────────────────────────────────────────────────────
    def _handle_trail_update(self, signal: Signal):
        """
        Handles UPDATE_TRAIL signals if strategy.py is modified to emit them.
        Currently unused — trail ratcheting is handled by _ratchet_broker_sl()
        which is called every bar close from on_candle_close().
        """
        if self._pos == 0:
            return
        if math.isnan(getattr(signal, "trail_price", float("nan"))):
            logger.warning("UPDATE_TRAIL received but trail_price is nan — skipping")
            return
        logger.info(
            "UPDATE_TRAIL: deferring to _ratchet_broker_sl() on next bar close. "
            "New level will be %.0f", signal.trail_price
        )

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

        # Fast path: read trail before acquiring lock (nan-safe property from BUG 2 fix)
        trail = self.strategy.current_trail
        if math.isnan(trail):
            return

        breached = (self._pos ==  1 and ltp <= trail) or \
                   (self._pos == -1 and ltp >= trail)
        if not breached:
            return

        with self._lock:
            # Re-check inside lock — concurrent tick or on_candle_close may have exited
            if self._pos == 0 or self._intrabar_exit_fired:
                return
            self._intrabar_exit_fired = True

            direction_str = "LONG" if self._pos == 1 else "SHORT"
            logger.warning(
                "INTRABAR TRAIL HIT (%s) — LTP=%.0f crossed trail=%.0f — "
                "exiting immediately (not waiting for bar close)",
                direction_str, ltp, trail
            )

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
    # Smart Limit Exit
    # ─────────────────────────────────────────────────────────
    def _smart_limit_exit(self, tag: str) -> Optional[float]:
        """
        Place a LIMIT exit order at current LTP instead of a MARKET order.
        Poll for fill for config.LIMIT_ORDER_TIMEOUT_SECS seconds.
        If not filled in time: cancel the LIMIT and return None so the
        caller falls back to a MARKET order immediately.
        """
        timeout  = getattr(config, "LIMIT_ORDER_TIMEOUT_SECS", 7)
        order_id = None
        try:
            ltp = self._get_current_ltp()
            if not ltp:
                logger.warning(
                    "SmartLimitExit: could not get current LTP — falling back to MARKET"
                )
                return None

            order_id = (
                self.broker.place_order("SELL", self._open_qty, order_type="LIMIT",
                                        price=ltp, tag=tag)
                if self._pos == 1
                else self.broker.place_order("BUY",  self._open_qty, order_type="LIMIT",
                                             price=ltp, tag=tag)
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
                        "SmartLimitExit: filled @ ₹%.0f (limit=₹%.0f, spread saved ≈₹%.0f/lot)",
                        fill, ltp, spread_saved
                    )
                    return fill
                if status in ("REJECTED", "CANCELLED"):
                    logger.warning(
                        "SmartLimitExit: limit order %s (%s) — falling back to MARKET",
                        status, order_id
                    )
                    return None
                time.sleep(0.4)

            logger.warning(
                "SmartLimitExit: LIMIT not filled in %ds @ ₹%.0f — "
                "cancelling and falling back to MARKET",
                timeout, ltp
            )
            try:
                self.broker.cancel_order(order_id)
                time.sleep(0.3)
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
        """Quick REST call to fetch current LTP for use as a LIMIT exit price."""
        try:
            import requests as _req
            url     = "https://api.upstox.com/v2/market-quote/ltp"
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
                order  = self.broker.get_order_status(order_id)
                status = order.get("status", "").upper()
                if status in ("COMPLETE", "FILLED"):
                    price = float(order.get("average_price", 0))
                    logger.info("Order %s filled @ ₹%.0f", order_id, price)
                    return price
                elif status in ("REJECTED", "CANCELLED"):
                    raise RuntimeError(
                        f"Order {order_id} {status}: {order.get('status_message', '')}"
                    )
            except RuntimeError:
                raise
            except Exception as e:
                logger.debug("Status poll error: %s", e)
            time.sleep(0.5)
        raise RuntimeError(f"Order {order_id} not filled within {timeout}s")

    # ─────────────────────────────────────────────────────────
    # Position sync with broker
    # ─────────────────────────────────────────────────────────
    def _sync_position(self):
        """
        Verify engine position matches broker's live position.
        Corrects drift caused by manual interventions or connection drops.
        On overnight startup, restores entry price and re-places the broker SL-M.
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
                atr            = self.strategy._atr or 0.0
                self.strategy.set_position(1, entry_px, atr=atr)

                if entry_px > 0:
                    logger.info(
                        "Overnight LONG restored | entry_px=₹%.0f | trail=₹%.0f",
                        entry_px, self.strategy.current_trail
                    )
                    # Re-place broker SL-M since it expired overnight (DAY validity)
                    self._place_broker_sl(atr)
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
                atr            = self.strategy._atr or 0.0
                self.strategy.set_position(-1, entry_px, atr=atr)

                if entry_px > 0:
                    logger.info(
                        "Overnight SHORT restored | entry_px=₹%.0f | trail=₹%.0f",
                        entry_px, self.strategy.current_trail
                    )
                    self._place_broker_sl(atr)
                else:
                    logger.warning(
                        "Overnight SHORT restored but entry_price unknown — "
                        "trail stop NOT initialised. Monitor manually."
                    )

            elif net == 0 and self._pos != 0:
                logger.warning(
                    "Position sync: broker=FLAT, engine=%d — correcting", self._pos
                )
                self._pos            = 0
                self._open_qty       = 0
                self._sl_order_id    = None
                self._sl_trail_level = np.nan
                self.strategy.set_position(0)

        except Exception as e:
            logger.debug("Position sync error (non-fatal): %s", e)

    def _try_recover_entry_price(self, net_qty: int) -> float:
        """
        Attempt to recover entry price from broker's trade book.
        Returns weighted average fill price or 0.0 if unavailable.
        """
        try:
            trades = self.broker.get_trade_book()
            if not trades:
                return 0.0
            direction_filter = "BUY" if net_qty > 0 else "SELL"
            relevant = [
                t for t in trades
                if t.get("instrument_key", "").upper() == config.INSTRUMENT_KEY.upper()
                and t.get("transaction_type", "").upper() == direction_filter
            ]
            if not relevant:
                return 0.0
            total_qty = sum(float(t.get("quantity", 0)) for t in relevant)
            if total_qty == 0:
                return 0.0
            return sum(
                float(t.get("average_price", 0)) * float(t.get("quantity", 0))
                for t in relevant
            ) / total_qty
        except Exception as e:
            logger.debug("Could not recover entry price: %s", e)
            return 0.0

    # ─────────────────────────────────────────────────────────
    # Emergency stop
    # ─────────────────────────────────────────────────────────
    def emergency_stop(self):
        """
        Called on Ctrl+C or fatal error.

        ALLOW_OVERNIGHT=True:  leaves the position open AND keeps the broker
            SL-M in place (it's the only protection while the engine is off).
        ALLOW_OVERNIGHT=False: cancels the broker SL-M, then flattens via
            broker.close_all_positions().

        BUG 6 FIX: resets engine position state after close_all_positions()
            so that a second emergency_stop() call (or the log in stop())
            does not try to close an already-flat position.
        """
        allow_overnight = getattr(config, "ALLOW_OVERNIGHT", False)

        if allow_overnight and self._pos != 0:
            logger.warning(
                "EMERGENCY STOP — ALLOW_OVERNIGHT=True: "
                "leaving %s position OPEN (qty=%d entry=₹%.0f trail=₹%.0f). "
                "Broker SL-M id=%s is PRESERVED as overnight protection. "
                "Engine shutting down.",
                "LONG" if self._pos == 1 else "SHORT",
                self._open_qty, self._entry_px,
                self.strategy.current_trail,
                self._sl_order_id or "none"
            )
            # Deliberately NOT cancelling _sl_order_id — it protects overnight.

        elif self._pos != 0:
            logger.warning("EMERGENCY STOP triggered — flattening position")
            with self._lock:
                # Cancel broker SL-M first so it doesn't race with the flat close
                self._cancel_broker_sl(reason="emergency stop")
                try:
                    self.broker.close_all_positions()
                    logger.info("All positions closed via close_all_positions()")
                    # BUG 6 FIX: reset state so a repeat call is idempotent
                    self._pos            = 0
                    self._entry_px       = 0.0
                    self._open_qty       = 0
                    self._sl_order_id    = None
                    self._sl_trail_level = np.nan
                    self.strategy.set_position(0)
                except Exception as e:
                    logger.error("Emergency close failed: %s", e)
                    logger.error("CHECK YOUR POSITIONS MANUALLY ON UPSTOX APP NOW")

        else:
            logger.info("EMERGENCY STOP — already flat, nothing to close")

    # ─────────────────────────────────────────────────────────
    # Historical warmup
    # ─────────────────────────────────────────────────────────
    WARMUP_DAYS = getattr(config, "WARMUP_DAYS", 60)

    def _fetch_historical_bars(self) -> list:
        """
        Fetch WARMUP_DAYS of past N-min bars (up to yesterday) then
        append today's completed intraday N-min bars.
        Returns a single merged list sorted oldest-first.
        """
        import requests
        from urllib.parse import quote as url_quote

        N          = config.CANDLE_INTERVAL
        today      = date.today()
        yesterday  = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        from_dt    = (today - timedelta(days=self.WARMUP_DAYS)).strftime("%Y-%m-%d")
        instrument = url_quote(config.INSTRUMENT_KEY, safe="")
        headers    = {
            "Authorization": f"Bearer {self.broker.token}",
            "Accept":        "application/json",
        }

        # Fetch 1: Multi-day historical (up to yesterday)
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
                logger.error("Historical fetch failed HTTP %d: %s",
                             r.status_code, r.text[:200])
        except Exception as e:
            logger.error("Historical fetch network error: %s", e)

        # Fetch 2: Today's intraday bars
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
                        "Intraday fetch: %d × %d-min bars today (%s → %s)",
                        len(intraday_bars), N,
                        intraday_bars[0]["ts"].strftime("%H:%M") if intraday_bars else "—",
                        intraday_bars[-1]["ts"].strftime("%H:%M") if intraday_bars else "—",
                    )
                else:
                    logger.info(
                        "Intraday fetch: 0 bars (market not yet open or no trades today)"
                    )
            else:
                logger.warning("Intraday fetch failed HTTP %d: %s",
                               r.status_code, r.text[:200])
        except Exception as e:
            logger.warning("Intraday fetch network error: %s", e)

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
        """Convert Upstox raw candle list to strategy bar dicts, oldest-first."""
        N        = config.CANDLE_INTERVAL
        now      = datetime.now()
        cur_snap = now.replace(minute=(now.minute // N) * N, second=0, microsecond=0)
        bars     = []
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
        bars.reverse()
        return bars

    def _aggregate_intraday(self, raw_1min: list) -> list:
        """Aggregate today's 1-min bars into N-min bars, oldest-first."""
        from collections import defaultdict
        N        = config.CANDLE_INTERVAL
        now      = datetime.now()
        cur_snap = now.replace(minute=(now.minute // N) * N, second=0, microsecond=0)
        buckets  = defaultdict(list)
        for c in raw_1min:
            try:
                ts     = datetime.fromisoformat(
                    str(c[0]).replace("Z", "+00:00")
                ).replace(tzinfo=None)
                total  = ts.hour * 60 + ts.minute
                snap   = (total // N) * N
                bar_ts = ts.replace(hour=snap // 60, minute=snap % 60,
                                    second=0, microsecond=0)
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
        """Fetch historical + today's bars and replay through strategy."""
        bars = self._fetch_historical_bars()
        if not bars:
            logger.warning(
                "Strategy starting COLD — norm_resid will diverge from "
                "TradingView for ~%d bars until indicators converge.",
                max(config.VR_LEN + 1, config.ATR_LEN * 3)
            )
            return
        result = self.strategy.warmup(bars)
        n      = result if isinstance(result, int) else len(bars)
        logger.info(
            "Strategy warmed up on %d × %d-min bars  (%.1f days of history)",
            n, config.CANDLE_INTERVAL,
            n * config.CANDLE_INTERVAL / (60 * 14.5)
        )

    # ─────────────────────────────────────────────────────────
    # Contract validation
    # ─────────────────────────────────────────────────────────
    def _validate_contract(self):
        """
        Three-layer contract rollover guard. Called once at startup.
        Layer 1: Live LTP probe via Upstox API.
        Layer 2: Calendar proximity warning (within 3 days of month-end).
        Layer 3: Live heartbeat monitor (started via _start_heartbeat_monitor).
        """
        import requests
        import calendar as cal_mod
        from datetime import date as _date

        logger.info("=" * 55)
        logger.info("  CONTRACT VALIDATION — %s", config.INSTRUMENT_KEY)
        logger.info("=" * 55)

        # Layer 1
        validation_passed = False
        try:
            url     = "https://api.upstox.com/v2/market-quote/ltp"
            headers = {
                "Authorization": f"Bearer {self.broker.token}",
                "Accept":        "application/json",
            }
            r = requests.get(url, headers=headers,
                             params={"instrument_key": config.INSTRUMENT_KEY},
                             timeout=10)
            if r.status_code == 200:
                data = r.json().get("data", {})
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
                    quote = next(iter(data.values()), None) if data else None

                if quote:
                    ltp = float(quote.get("last_price", 0) or 0)
                    if ltp > 0:
                        logger.info(
                            "Layer 1 OK  Contract ACTIVE — LTP=%.0f  (%s)",
                            ltp, config.INSTRUMENT_KEY
                        )
                        validation_passed = True
                    else:
                        logger.critical(
                            "Layer 1 FAIL  LTP=0 for %s — contract may be EXPIRED or "
                            "market is closed. Verify on Upstox app before trading.",
                            config.INSTRUMENT_KEY
                        )
                else:
                    logger.warning(
                        "Layer 1 WARN  No quote data returned for %s. "
                        "Market may be closed or key is wrong.",
                        config.INSTRUMENT_KEY
                    )
            else:
                logger.warning(
                    "Layer 1 WARN  LTP probe failed HTTP %d: %s",
                    r.status_code, r.text[:200]
                )
        except Exception as e:
            logger.warning("Layer 1 WARN  LTP probe exception: %s", e)

        # Layer 2
        ROLLOVER_WARNING_DAYS = 3
        today              = _date.today()
        last_day_of_month  = cal_mod.monthrange(today.year, today.month)[1]
        days_to_month_end  = last_day_of_month - today.day
        if days_to_month_end <= ROLLOVER_WARNING_DAYS:
            logger.warning(
                "Layer 2 WARN  ROLLOVER WINDOW: Today is %s — only %d day(s) until "
                "end of month. MCX Silver MIC expires on the last business day of "
                "each month. VERIFY that %s is still active and update INSTRUMENT_KEY "
                "in config.py if needed before trading.",
                today, days_to_month_end, config.INSTRUMENT_KEY
            )
        else:
            logger.info(
                "Layer 2 OK  %d day(s) until month-end — no rollover imminent",
                days_to_month_end
            )

        if not validation_passed:
            logger.critical(
                "CONTRACT VALIDATION WARNING: Could not confirm %s is active. "
                "The bot will start anyway — check logs carefully.",
                config.INSTRUMENT_KEY
            )
        logger.info("=" * 55)

    # ─────────────────────────────────────────────────────────
    # Heartbeat monitor (Layer 3)
    # ─────────────────────────────────────────────────────────
    def _start_heartbeat_monitor(self):
        """
        Background thread that monitors for tick silence during the MCX session.
        Flags self._feed_silent = True to block new entries if no candle arrives
        within _TICK_TIMEOUT_MINS.
        """
        def _monitor():
            import time as _time
            logger.info("Heartbeat monitor started (timeout=%d min)", self._TICK_TIMEOUT_MINS)
            while self._running:
                _time.sleep(60)
                try:
                    now     = datetime.now()
                    sess_s  = config.SESS_START[0] * 60 + config.SESS_START[1]
                    sess_e  = config.SESS_END[0]   * 60 + config.SESS_END[1]
                    now_m   = now.hour * 60 + now.minute
                    in_sess = sess_s <= now_m <= sess_e
                    if not in_sess:
                        continue

                    if self._last_tick_ts is None:
                        elapsed = (now - self._engine_start_ts).total_seconds() / 60
                        if elapsed > self._TICK_TIMEOUT_MINS:
                            logger.critical(
                                "HEARTBEAT FAIL  No tick received in %.0f min since startup "
                                "during active session. Possible: (1) contract expired, "
                                "(2) WebSocket failed, (3) exchange outage.",
                                elapsed
                            )
                    else:
                        elapsed = (now - self._last_tick_ts).total_seconds() / 60
                        if elapsed > self._TICK_TIMEOUT_MINS:
                            logger.critical(
                                "HEARTBEAT FAIL  No tick for %.0f min during active MCX session "
                                "(last tick: %s). New entries BLOCKED.",
                                elapsed, self._last_tick_ts.strftime("%H:%M:%S")
                            )
                            self._feed_silent = True
                        else:
                            self._feed_silent = False
                except Exception as e:
                    logger.debug("Heartbeat monitor error: %s", e)

        threading.Thread(target=_monitor, daemon=True, name="heartbeat").start()

    # ─────────────────────────────────────────────────────────
    # Start / Stop
    # ─────────────────────────────────────────────────────────
    def start(self, use_websocket: bool = True):
        self._running         = True
        self._engine_start_ts = datetime.now()
        self._feed_silent     = False

        self._validate_contract()
        self._warmup_strategy()

        if use_websocket:
            self.feed = UpstoxFeed(
                access_token    = self.broker.token,
                on_candle_close = self.on_candle_close,
                on_tick         = self.on_tick,
                on_reconnect    = self.on_reconnect,
            )
        else:
            # FIX BUG 3: removed poll_seconds=2 — RESTCandleFeed.__init__ does not
            # accept that keyword argument; it hardcodes 30-second polling internally.
            self.feed = RestLTPPoller(
                access_token    = self.broker.token,
                on_candle_close = self.on_candle_close,
            )

        self.feed.start()
        logger.info("Feed started — waiting for candles...")

        self._start_heartbeat_monitor()

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
                "Broker SL-M id=%s is active. "
                "Restart engine before next session to resume management.",
                "LONG" if self._pos == 1 else "SHORT",
                self._open_qty,
                self._sl_order_id or "none"
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
    logger.info("  Starting at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    try:
        token = load_token()
    except (FileNotFoundError, RuntimeError) as e:
        logger.error("Auth error: %s", e)
        logger.error("Run 'python auth.py' first to authenticate.")
        sys.exit(1)

    if not validate_token(token):
        logger.error("Token is invalid or expired. Run 'python auth.py'.")
        sys.exit(1)

    engine = TradingEngine(token)

    def _sigterm(signum, frame):
        logger.info("SIGTERM received")
        engine.stop()

    sig_module.signal(sig_module.SIGTERM, _sigterm)

    engine.start(use_websocket=True)
