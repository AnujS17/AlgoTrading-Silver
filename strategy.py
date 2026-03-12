"""
strategy.py — Live Kalman Adaptive Scalper signal engine.

Maintains all indicator state bar-by-bar, mirrors the PineScript logic,
and emits trading signals to engine.py.

Signal actions:
  "BUY"               — enter long next bar's open
  "SELL"              — enter short next bar's open
  "EXIT_LONG"         — close long position
  "EXIT_SHORT"        — close short position
  "UPDATE_TRAIL"      — ATR trail ratcheted; engine must cancel+replace SL-M order
                        at broker with signal.trail_price

ATR Trail Stop — broker SL-M model
─────────────────────────────────────────────────────────────
  On confirmed entry fill:
    1. Engine calls strategy.set_position(pos, fill_price, atr=signal.atr)
    2. Strategy initialises _trail_l / _trail_s from fill_price
    3. Engine reads strategy.current_trail and places SL-M order at that level

  Each subsequent bar while in position:
    - Strategy ratchets the trail (only moves in profit direction)
    - If trail moved → emits UPDATE_TRAIL(trail_price=new_level)
    - Engine cancels old SL-M, places new SL-M at trail_price

  If broker SL-M fires (broker callback):
    - Engine calls strategy.set_position(0) to clear state

  Fallback safety (broker SL missed / rejected):
    - Strategy detects low <= trail_l (or high >= trail_s) at bar close
    - Emits EXIT_LONG/EXIT_SHORT with reason="ATR Trail Fallback"
    - Engine should log CRITICAL, cancel any pending SL-M, market-exit immediately

  Reversion exit (always market order — cancel SL-M first):
    - Emits EXIT_LONG/EXIT_SHORT with reason="Revert"
    - Engine should cancel active SL-M order, then place market exit
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
    action:      str       # "BUY"|"SELL"|"EXIT_LONG"|"EXIT_SHORT"|"UPDATE_TRAIL"
    reason:      str       # human-readable explanation
    bar_ts:      object    # timestamp of the triggering bar
    norm_resid:  float = 0.0
    vr:          float = 0.0
    vel_norm:    float = 0.0
    atr:         float = 0.0
    trail_price: float = float("nan")  # populated for UPDATE_TRAIL signals


class KalmanScalperStrategy:
    """
    Stateful live strategy engine.

    Call process_bar(candle) on each completed candle.
    Returns a Signal or None.

    State is persistent across bars (Kalman x/v/p, velocity EMA,
    cooldown counter, ATR RMA, etc.) — mirrors PineScript var declarations.

    ATR Trail Stop — engine.py responsibilities:
      After BUY/SELL fill confirmed:
        strategy.set_position(pos, fill_price, atr=signal.atr)
        broker.place_stop_order(strategy.current_trail)

      On UPDATE_TRAIL signal:
        broker.cancel_order(active_stop_order_id)
        broker.place_stop_order(signal.trail_price)

      When broker stop order triggers:
        strategy.set_position(0)

      On EXIT_LONG/SHORT reason="Revert":
        broker.cancel_order(active_stop_order_id)
        broker.market_exit()

      On EXIT_LONG/SHORT reason="ATR Trail Fallback":
        broker.cancel_order(active_stop_order_id)  # in case it wasn't filled
        broker.market_exit()
        logger.critical("Broker SL-M may have failed — forced fallback exit")
    """

    def __init__(self):
        # ── Kalman state (var float in PineScript) ────────────
        self._kx = None    # Initialised on first bar with close price
        self._kv = 0.0
        self._kp = 1.0

        # ── ATR RMA state ─────────────────────────────────────
        self._atr = None   # Wilder's RMA; None until first bar

        # ── Velocity EMA ──────────────────────────────────────
        self._vel_ema   = None
        self._alpha_vel = 2.0 / (config.VEL_LEN + 1)

        # ── Variance Ratio buffers ────────────────────────────
        # ROOT CAUSE FIX: deque must hold 2*VR_LEN+1 closes, not VR_LEN+1.
        #
        # The VR formula builds rN_series = closes[VR_LEN:] - closes[:n-VR_LEN]
        # With maxlen=VR_LEN+1, arr always has exactly VR_LEN+1 elements, so
        # rN_series has exactly 1 element. np.var([single_value]) = 0.0 always,
        # making varN=0 and VR=0 permanently — the filter is disabled forever.
        #
        # With maxlen=2*VR_LEN+1, rN_series grows to VR_LEN+1 elements,
        # giving a meaningful variance that correctly gates entry signals.
        self._closes = deque(maxlen=2 * config.VR_LEN + 1)

        # ── Cooldown counter (var int in PineScript) ──────────
        # Pine: var int barsSince = cooldown + 1  (start cooled)
        self._bars_since = config.COOLDOWN + 1

        # ── ATR trail stop levels ─────────────────────────────
        # Initialised by set_position() on confirmed fill.
        # Ratcheted each bar; never moves against the trade.
        self._trail_l = np.nan   # long stop — only moves UP
        self._trail_s = np.nan   # short stop — only moves DOWN

        # ── Current position tracking (strategy-side) ─────────
        self.position    = 0     # +1 long | -1 short | 0 flat
        self.entry_price = 0.0

        # ── Bar counter ───────────────────────────────────────
        self._bar_count = 0

        # ── Warmup flag — suppresses signals during historical replay ──
        self._warming_up = False

        logger.info("KalmanScalperStrategy initialised")

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    @property
    def current_trail(self) -> float:
        """
        Active trail stop price that should be sitting as a SL-M order
        at the broker.

        Returns nan when flat — engine should assert this is nan when not
        in a position.

        Read by engine.py after set_position() to place the initial SL-M.
        """
        if self.position == 1:
            return self._trail_l
        if self.position == -1:
            return self._trail_s
        return float("nan")

    def process_bar(self, candle: dict) -> Optional[Signal]:
        """
        Feed a completed OHLCV candle.
        Returns a Signal if action required, else None.

        candle keys: ts, open, high, low, close, volume
        """
        ts    = candle["ts"]
        close = float(candle["close"])
        high  = float(candle["high"])
        low   = float(candle["low"])

        self._bar_count += 1

        # ── 1. Kalman Filter update ────────────────────────────
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

        # ── 2. ATR — Wilder's RMA ─────────────────────────────
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

        # ── 3. ATR-normalised residual ─────────────────────────
        residual   = close - kalm_x
        norm_resid = residual / atr if atr > 0 else 0.0

        # ── 4. Velocity EMA (normalised) ──────────────────────
        vel_abs = abs(kalm_v)
        if self._vel_ema is None:
            self._vel_ema = vel_abs
        else:
            self._vel_ema = self._vel_ema + self._alpha_vel * (vel_abs - self._vel_ema)
        vel_norm = self._vel_ema / atr if atr > 0 else 0.0

        # ── 5. Variance Ratio ─────────────────────────────────
        vr = 1.0   # default = random walk (no trade) until buffer is full
        if len(self._closes) >= 2 * config.VR_LEN + 1:
            arr  = np.array(self._closes)
            r1   = np.diff(arr)
            var1 = np.var(r1[-config.VR_LEN:], ddof=0)
            rN_series = arr[config.VR_LEN:] - arr[:len(arr) - config.VR_LEN]
            varN = np.var(rN_series[-config.VR_LEN:], ddof=0)
            vr   = varN / (config.VR_LEN * var1) if var1 > 0 else 1.0

        # ── 6. Regime & velocity filters ──────────────────────
        vr_mr  = vr       < config.VR_THRESHOLD
        vel_ok = vel_norm < config.MAX_VEL_MULT

        # ── 7. Session filter ─────────────────────────────────
        hhmm   = ts.hour * 100 + ts.minute
        sess_s = config.SESS_START[0] * 100 + config.SESS_START[1]
        sess_e = config.SESS_END[0]   * 100 + config.SESS_END[1]
        in_sess = (not config.USE_SESSION) or (sess_s <= hhmm <= sess_e)

        # ── 8. Cooldown counter ───────────────────────────────
        # Pine: barsSince := position != 0 ? 0 : min(barsSince+1, cooldown+1)
        if self.position != 0:
            self._bars_since = 0
        else:
            self._bars_since = min(self._bars_since + 1, config.COOLDOWN + 1)
        cooled = self._bars_since >= config.COOLDOWN

        # ── Log indicator snapshot ────────────────────────────
        logger.debug(
            "Bar %d | %s | close=%.0f | kalmX=%.0f | resid=%+.3f | "
            "VR=%.3f(%s) | vel=%.3f(%s) | pos=%d | trail=%.0f | cooled=%s",
            self._bar_count, ts, close, kalm_x, norm_resid,
            vr, "MR" if vr_mr else "TR",
            vel_norm, "OK" if vel_ok else "FAST",
            self.position, self.current_trail, cooled
        )

        # ══════════════════════════════════════════════════════
        # EXIT LOGIC — checked before entry on every bar
        # Priority: Fallback Trail Hit > Reversion Exit > UPDATE_TRAIL
        # ══════════════════════════════════════════════════════

        if self.position == 1:
            # ── Ratchet: trail only moves UP ──────────────────
            new_trail  = close - config.ATR_MULT * atr
            prev_trail = self._trail_l
            self._trail_l = max(
                self._trail_l if not np.isnan(self._trail_l) else new_trail,
                new_trail
            )

            # ── (a) FALLBACK trail hit ─────────────────────────
            # Fires only if broker's SL-M order was missed / rejected.
            # Primary path: broker's resting SL-M handles this intrabar.
            if low <= self._trail_l:
                logger.warning(
                    "EXIT LONG — ATR Trail FALLBACK (broker SL-M may have failed) "
                    "trail=%.0f low=%.0f", self._trail_l, low
                )
                self._reset_trails()
                if self._warming_up:
                    return None
                return Signal("EXIT_LONG", "ATR Trail Fallback", ts,
                              norm_resid, vr, vel_norm, atr,
                              trail_price=self._trail_l)

            # ── (b) Reversion exit ────────────────────────────
            # Engine: cancel active SL-M, then market-exit.
            if norm_resid >= -config.EXIT_THRESH:
                logger.info(
                    "EXIT LONG — Mean reversion (normResid=%+.3f)", norm_resid
                )
                self._reset_trails()
                if self._warming_up:
                    return None
                return Signal("EXIT_LONG", "Revert", ts,
                              norm_resid, vr, vel_norm, atr)

            # ── (c) Trail ratcheted — update broker SL-M ──────
            # Engine: cancel old SL-M, place new SL-M at trail_price.
            trail_moved = (np.isnan(prev_trail) or
                           self._trail_l > prev_trail)
            if trail_moved and not self._warming_up:
                logger.info(
                    "UPDATE_TRAIL LONG  %.0f → %.0f",
                    prev_trail if not np.isnan(prev_trail) else 0,
                    self._trail_l
                )
                return Signal("UPDATE_TRAIL", "Trail Ratchet", ts,
                              norm_resid, vr, vel_norm, atr,
                              trail_price=self._trail_l)

        elif self.position == -1:
            # ── Ratchet: trail only moves DOWN ────────────────
            new_trail  = close + config.ATR_MULT * atr
            prev_trail = self._trail_s
            self._trail_s = min(
                self._trail_s if not np.isnan(self._trail_s) else new_trail,
                new_trail
            )

            # ── (a) FALLBACK trail hit ─────────────────────────
            if high >= self._trail_s:
                logger.warning(
                    "EXIT SHORT — ATR Trail FALLBACK (broker SL-M may have failed) "
                    "trail=%.0f high=%.0f", self._trail_s, high
                )
                self._reset_trails()
                if self._warming_up:
                    return None
                return Signal("EXIT_SHORT", "ATR Trail Fallback", ts,
                              norm_resid, vr, vel_norm, atr,
                              trail_price=self._trail_s)

            # ── (b) Reversion exit ────────────────────────────
            if norm_resid <= config.EXIT_THRESH:
                logger.info(
                    "EXIT SHORT — Mean reversion (normResid=%+.3f)", norm_resid
                )
                self._reset_trails()
                if self._warming_up:
                    return None
                return Signal("EXIT_SHORT", "Revert", ts,
                              norm_resid, vr, vel_norm, atr)

            # ── (c) Trail ratcheted — update broker SL-M ──────
            trail_moved = (np.isnan(prev_trail) or
                           self._trail_s < prev_trail)
            if trail_moved and not self._warming_up:
                logger.info(
                    "UPDATE_TRAIL SHORT  %.0f → %.0f",
                    prev_trail if not np.isnan(prev_trail) else 0,
                    self._trail_s
                )
                return Signal("UPDATE_TRAIL", "Trail Ratchet", ts,
                              norm_resid, vr, vel_norm, atr,
                              trail_price=self._trail_s)

        # ══════════════════════════════════════════════════════
        # ENTRY LOGIC
        # ══════════════════════════════════════════════════════
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
                if self._warming_up:
                    return None
                return Signal("BUY", "Kalman Long Entry", ts,
                              norm_resid, vr, vel_norm, atr)

            if go_short:
                logger.info(
                    "SELL signal | normResid=%+.3f VR=%.3f vel=%.3f",
                    norm_resid, vr, vel_norm
                )
                if self._warming_up:
                    return None
                return Signal("SELL", "Kalman Short Entry", ts,
                              norm_resid, vr, vel_norm, atr)

        return None

    # ─────────────────────────────────────────────────────────
    # Warmup
    # ─────────────────────────────────────────────────────────
    def warmup(self, historical_bars: list) -> int:
        """
        Replay historical completed bars to converge all indicator state
        before going live. All signals are suppressed — nothing is returned.

        Call from engine BEFORE starting the live feed.
        historical_bars: list of {ts, open, high, low, close, volume}, oldest-first.
        """
        if not historical_bars:
            logger.warning("Warmup: no historical bars provided — indicators start cold")
            return 0

        logger.info("=" * 55)
        logger.info("  STRATEGY WARMUP — replaying %d historical bars",
                    len(historical_bars))
        logger.info("  From : %s", historical_bars[0]["ts"])
        logger.info("  To   : %s", historical_bars[-1]["ts"])
        logger.info("=" * 55)

        self._warming_up = True
        for bar in historical_bars:
            self.process_bar(bar)
        self._warming_up = False

        logger.info(
            "Warmup complete | kalmX=%.2f | ATR=%.2f | VR_bars=%d | bars_since=%d",
            self._kx or 0.0, self._atr or 0.0,
            len(self._closes), self._bars_since
        )
        logger.info("=" * 55)
        return len(historical_bars)

    # ─────────────────────────────────────────────────────────
    # Position management — called by engine after confirmed fills
    # ─────────────────────────────────────────────────────────
    def set_position(self, pos: int,
                     entry_price: float = 0.0,
                     atr: float = 0.0):
        """
        Called by engine.py after a confirmed order fill (entry or exit).

        pos          : +1 long | -1 short | 0 flat
        entry_price  : actual fill price (0.0 on exit)
        atr          : ATR at time of entry signal — used to initialise
                       the trail stop level; pass signal.atr on entry calls

        After calling this on entry, engine should read strategy.current_trail
        and place a SL-M order at that price with the broker.

        Example (engine.py):
            strategy.set_position(1, fill_price, atr=entry_signal.atr)
            stop_order_id = broker.place_stop_order(
                "SELL", qty, strategy.current_trail
            )
        """
        prev_pos = self.position
        self.position    = pos
        self.entry_price = entry_price

        if pos == 0:
            # Exiting — clear all trail state
            self._reset_trails()
            logger.info("Position closed (was %+d)", prev_pos)

        elif pos == 1:
            # Entering long — initialise trail at fill_price - ATR_MULT * atr
            if atr > 0:
                self._trail_l = entry_price - config.ATR_MULT * atr
                logger.info(
                    "Position LONG @ %.0f | initial trail stop = %.0f  "
                    "(ATR=%.1f × mult=%.1f)",
                    entry_price, self._trail_l, atr, config.ATR_MULT
                )
            else:
                logger.warning(
                    "set_position(LONG) called with atr=0 — "
                    "trail not initialised, SL-M cannot be placed"
                )

        elif pos == -1:
            # Entering short — initialise trail at fill_price + ATR_MULT * atr
            if atr > 0:
                self._trail_s = entry_price + config.ATR_MULT * atr
                logger.info(
                    "Position SHORT @ %.0f | initial trail stop = %.0f  "
                    "(ATR=%.1f × mult=%.1f)",
                    entry_price, self._trail_s, atr, config.ATR_MULT
                )
            else:
                logger.warning(
                    "set_position(SHORT) called with atr=0 — "
                    "trail not initialised, SL-M cannot be placed"
                )

    # ─────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────
    def _reset_trails(self):
        self._trail_l = np.nan
        self._trail_s = np.nan
