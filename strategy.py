"""
strategy.py — Live Kalman Adaptive Scalper signal engine.

Maintains all indicator state bar-by-bar, mirrors the PineScript logic,
and emits trading signals: "BUY", "SELL", "EXIT_LONG", "EXIT_SHORT", or None.

Designed to receive completed candles from feed.py one at a time.

FIXES vs PREVIOUS VERSION:
  1. set_position() now accepts atr= keyword argument.
     Engine calls set_position(direction, fill_px, atr=signal.atr) to
     pre-initialise the ATR trail immediately at entry, so on_tick()
     can start monitoring the trail from the very first tick of the new
     bar. Without this fix, trail was NaN until the next bar close and
     engine crashed with TypeError on every trade entry.

  2. current_trail property added.
     Engine references self.strategy.current_trail in on_tick(),
     emergency_stop(), _sync_position(), and _ratchet_broker_sl().
     Previously these all raised AttributeError — only the private
     _trail_l / _trail_s floats existed, not a public property.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

import config

logger = logging.getLogger("kalman.strategy")


@dataclass
class Signal:
    action:  str           # "BUY" | "SELL" | "EXIT_LONG" | "EXIT_SHORT"
    reason:  str           # human-readable explanation
    bar_ts:  object        # timestamp of the triggering bar
    norm_resid: float = 0.0
    vr:         float = 0.0
    vel_norm:   float = 0.0
    atr:        float = 0.0


class KalmanScalperStrategy:
    """
    Stateful live strategy engine.

    Call process_bar(candle) on each completed candle.
    Returns a Signal or None.

    State is persistent across bars (Kalman x/v/p, velocity EMA,
    cooldown counter, ATR RMA, etc.) — mirrors PineScript var declarations.
    """

    def __init__(self):
        # Kalman state (var float in PineScript)
        self._kx   = None    # Initialised on first bar with close price
        self._kv   = 0.0
        self._kp   = 1.0

        # ATR RMA state
        self._atr  = None    # Wilder's RMA; None until first bar

        # Velocity EMA
        self._vel_ema   = None
        self._alpha_vel = 2.0 / (config.VEL_LEN + 1)

        # Variance Ratio buffers
        self._closes = deque(maxlen=2 * config.VR_LEN + 1)

        # Cooldown counter (var int in PineScript)
        # Pine: var int barsSince = cooldown + 1  (start cooled down)
        self._bars_since = config.COOLDOWN + 1

        # ATR trail stops for live exit tracking.
        # NaN when flat. Long: _trail_l is the stop below price.
        # Short: _trail_s is the stop above price.
        self._trail_l = np.nan
        self._trail_s = np.nan

        # Current position tracking (strategy-side mirror of engine)
        self.position    = 0       # +1 long | -1 short | 0 flat
        self.entry_price = 0.0

        # Bar counter and warmup flag
        self._bar_count  = 0
        self._warming_up = False   # True during historical replay

        logger.info("KalmanScalperStrategy initialised")

    # -----------------------------------------------------------------
    # Public properties
    # -----------------------------------------------------------------

    @property
    def current_trail(self) -> float:
        """
        Active ATR trail stop price, or NaN when flat / not yet initialised.

        FIX: This property was missing in the previous version, causing
        AttributeError crashes in:
          - engine.on_tick()              (every tick while in position)
          - engine.emergency_stop()       (on Ctrl+C with open position)
          - engine._sync_position()       (overnight restore)
          - engine._ratchet_broker_sl()   (broker SL update on each bar)
        """
        if self.position == 1:
            return self._trail_l
        if self.position == -1:
            return self._trail_s
        return np.nan

    # -----------------------------------------------------------------
    # Core bar-by-bar processing
    # -----------------------------------------------------------------

    def process_bar(self, candle: dict) -> Optional[Signal]:
        """
        Feed one completed OHLCV candle.
        Returns a Signal if an action is required, else None.
        Always returns None during warmup (signals suppressed).

        candle keys: ts, open, high, low, close, volume
        """
        ts    = candle["ts"]
        close = float(candle["close"])
        high  = float(candle["high"])
        low   = float(candle["low"])

        self._bar_count += 1

        # 1. Kalman Filter update
        if self._kx is None:
            self._kx = close    # Pine: var float kalmX = close

        xp    = self._kx + self._kv
        pp    = self._kp + config.KALM_Q
        k     = pp / (pp + config.KALM_R)
        innov = close - xp

        self._kx = xp + k * innov
        self._kv = self._kv + k * innov * 0.1
        self._kp = (1.0 - k) * pp

        kalm_x = self._kx
        kalm_v = self._kv

        # 2. ATR — Wilder's RMA
        if len(self._closes) > 0:
            prev_close = self._closes[-1]
            tr = max(high - low,
                     abs(high - prev_close),
                     abs(low  - prev_close))
        else:
            tr = high - low

        if self._atr is None:
            self._atr = tr
        else:
            self._atr = self._atr + (1.0 / config.ATR_LEN) * (tr - self._atr)

        atr = self._atr
        self._closes.append(close)

        # 3. ATR-normalised residual
        residual   = close - kalm_x
        norm_resid = residual / atr if atr > 0 else 0.0

        # 4. Velocity EMA (normalised)
        vel_abs = abs(kalm_v)
        if self._vel_ema is None:
            self._vel_ema = vel_abs
        else:
            self._vel_ema = self._vel_ema + self._alpha_vel * (vel_abs - self._vel_ema)
        vel_norm = self._vel_ema / atr if atr > 0 else 0.0

        # 5. Variance Ratio (Hurst proxy)
        vr = 1.0   # default = random walk (no trade)
        if len(self._closes) >= 2 * config.VR_LEN + 1:
            arr       = np.array(self._closes)
            r1        = np.diff(arr)
            var1      = np.var(r1[-config.VR_LEN:], ddof=0)
            rN_series = arr[config.VR_LEN:] - arr[:len(arr) - config.VR_LEN]
            varN      = np.var(rN_series, ddof=0) if len(rN_series) >= 2 else 0.0
            vr        = varN / (config.VR_LEN * var1) if var1 > 0 else 1.0

        # 6. Regime & velocity filters
        vr_mr  = vr  < config.VR_THRESHOLD
        vel_ok = vel_norm < config.MAX_VEL_MULT

        # 7. Session filter
        hhmm    = ts.hour * 100 + ts.minute
        sess_s  = config.SESS_START[0] * 100 + config.SESS_START[1]
        sess_e  = config.SESS_END[0]   * 100 + config.SESS_END[1]
        in_sess = (not config.USE_SESSION) or (sess_s <= hhmm <= sess_e)

        # 8. Cooldown counter
        if self.position != 0:
            self._bars_since = 0
        else:
            self._bars_since = min(self._bars_since + 1, config.COOLDOWN + 1)
        cooled = self._bars_since >= config.COOLDOWN

        logger.debug(
            "Bar %d | %s | close=%.0f | kalmX=%.0f | resid=%+.3f | "
            "VR=%.3f(%s) | vel=%.3f(%s) | pos=%d | cooled=%s",
            self._bar_count, ts, close, kalm_x, norm_resid,
            vr, "MR" if vr_mr else "TR",
            vel_norm, "OK" if vel_ok else "FAST",
            self.position, cooled
        )

        # Suppress all signals during warmup
        if self._warming_up:
            return None

        # -----------------------------------------------------------------
        # EXIT LOGIC (checked before entry)
        # -----------------------------------------------------------------

        # ATR trail ratchet — only ever moves in the favourable direction
        if self.position == 1:
            new_trail = close - config.ATR_MULT * atr
            self._trail_l = max(
                self._trail_l if not np.isnan(self._trail_l) else (low - config.ATR_MULT * atr),
                new_trail
            )
            # Trail hit check (using bar low, consistent with Pine)
            if low <= self._trail_l:
                logger.info("EXIT LONG — ATR Trail hit at %.0f", self._trail_l)
                self._reset_trails()
                return Signal("EXIT_LONG", "ATR Trail", ts,
                              norm_resid, vr, vel_norm, atr)

        elif self.position == -1:
            new_trail = close + config.ATR_MULT * atr
            self._trail_s = min(
                self._trail_s if not np.isnan(self._trail_s) else (high + config.ATR_MULT * atr),
                new_trail
            )
            if high >= self._trail_s:
                logger.info("EXIT SHORT — ATR Trail hit at %.0f", self._trail_s)
                self._reset_trails()
                return Signal("EXIT_SHORT", "ATR Trail", ts,
                              norm_resid, vr, vel_norm, atr)

        # Mean-reversion exit
        if self.position == 1 and norm_resid >= -config.EXIT_THRESH:
            logger.info("EXIT LONG — Mean reversion (normResid=%+.3f)", norm_resid)
            self._reset_trails()
            return Signal("EXIT_LONG", "Revert", ts,
                          norm_resid, vr, vel_norm, atr)

        if self.position == -1 and norm_resid <= config.EXIT_THRESH:
            logger.info("EXIT SHORT — Mean reversion (normResid=%+.3f)", norm_resid)
            self._reset_trails()
            return Signal("EXIT_SHORT", "Revert", ts,
                          norm_resid, vr, vel_norm, atr)

        # -----------------------------------------------------------------
        # ENTRY LOGIC
        # -----------------------------------------------------------------
        if self.position == 0 and in_sess and cooled and vr_mr and vel_ok:
            go_long  = (norm_resid < -config.ENTRY_THRESH
                        and config.DIRECTION in ("Long Only", "Both"))
            go_short = (norm_resid >  config.ENTRY_THRESH
                        and config.DIRECTION in ("Short Only", "Both"))

            if go_long:
                logger.info(
                    "BUY signal | normResid=%+.3f VR=%.3f vel=%.3f",
                    norm_resid, vr, vel_norm
                )
                return Signal("BUY", "Kalman Long Entry", ts,
                              norm_resid, vr, vel_norm, atr)

            if go_short:
                logger.info(
                    "SELL signal | normResid=%+.3f VR=%.3f vel=%.3f",
                    norm_resid, vr, vel_norm
                )
                return Signal("SELL", "Kalman Short Entry", ts,
                              norm_resid, vr, vel_norm, atr)

        return None

    # -----------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------

    def set_position(self, pos: int, entry_price: float = 0.0, atr: float = 0.0):
        """
        Called by the engine after a confirmed order fill (or overnight restore).

        FIX: Added atr= parameter.  When non-zero, the ATR trail is seeded
        immediately from the fill price so on_tick() can start checking the
        trail from the very first tick — not just after the next bar close.

        Without this, every call to engine.set_position(direction, fill_px,
        atr=signal.atr) raised TypeError because the old signature only
        accepted (pos, entry_price).

        Args:
            pos:          +1 long | -1 short | 0 flat
            entry_price:  fill price of the entry order (0.0 when going flat)
            atr:          current ATR — seeds the initial trail from fill price.
                          Pass 0.0 or omit if ATR is unknown (trail stays NaN
                          until the next bar close updates it via process_bar).
        """
        self.position    = pos
        self.entry_price = entry_price

        if pos == 0:
            self._reset_trails()
        elif atr > 0:
            if pos == 1:
                self._trail_l = entry_price - config.ATR_MULT * atr
                logger.info(
                    "Strategy LONG @ %.0f | trail seeded at %.0f (%.1f × ATR %.1f)",
                    entry_price, self._trail_l, config.ATR_MULT, atr
                )
            else:
                self._trail_s = entry_price + config.ATR_MULT * atr
                logger.info(
                    "Strategy SHORT @ %.0f | trail seeded at %.0f (%.1f × ATR %.1f)",
                    entry_price, self._trail_s, config.ATR_MULT, atr
                )
        else:
            logger.info(
                "Strategy position set: %d @ %.0f (ATR=0 — trail stays NaN until next bar)",
                pos, entry_price
            )

    def _reset_trails(self):
        self._trail_l = np.nan
        self._trail_s = np.nan

    def warmup(self, bars: list):
        """
        Replay historical OHLCV bars to initialise all indicator state
        (Kalman, ATR, VEL EMA, VR buffers) before the live feed starts.

        Signals are suppressed during warmup — no trades are generated.

        bars: list of dicts {ts, open, high, low, close, volume}, oldest-first.
        """
        self._warming_up = True
        logger.info("Strategy warmup: replaying %d historical bars ...", len(bars))
        for i, bar in enumerate(bars):
            self.process_bar(bar)
            if (i + 1) % 500 == 0:
                logger.info("  Warmup progress: %d / %d bars", i + 1, len(bars))
        self._warming_up = False
        logger.info(
            "Warmup complete. Kalman x=%.1f v=%.4f | ATR=%.1f | VR buffer=%d bars",
            self._kx or 0, self._kv, self._atr or 0, len(self._closes)
        )
