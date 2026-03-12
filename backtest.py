"""
backtest.py — Kalman Adaptive Scalper backtester.

Uses the EXACT same strategy.py logic as the live engine — zero divergence.
Fetches data from the Upstox v3 historical API, feeds bars one-by-one through
KalmanScalperStrategy, simulates fills, and produces a full performance
report + trade log CSV + equity curve PNG.

Kalman strategy coefficients (KALM_Q, KALM_R, ENTRY_THRESH, EXIT_THRESH,
VR_LEN, VR_THRESHOLD, VEL_LEN, MAX_VEL_MULT, ATR_LEN, ATR_MULT, COOLDOWN)
are read from config.py — all other parameters are set in the
CONFIGURABLE PARAMETERS block below.

Usage:
    python backtest.py                       # run with defaults below
    python backtest.py --days 90             # override date range
    python backtest.py --csv mydata.csv      # use local OHLCV CSV
    python backtest.py --days 60 --slip 2.0  # with slippage
    python backtest.py --no-plot             # skip chart

CSV format (if using --csv):
    datetime,open,high,low,close,volume
    2025-01-02 09:15:00,92100,92350,91980,92200,1200

Dependencies: pip install requests pandas numpy matplotlib
"""

import argparse
import csv
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, date, timedelta
from typing import Optional
from urllib.parse import quote

import numpy as np
import pandas as pd

import config
from strategy import KalmanScalperStrategy, Signal


# ═════════════════════════════════════════════════════════════
#  CONFIGURABLE PARAMETERS — edit everything in this block
# ═════════════════════════════════════════════════════════════

# ── Instrument ────────────────────────────────────────────────
INSTRUMENT_KEY  = "MCX_FO|466029"      # Upstox instrument key
INSTRUMENT_NAME = "SILVERMIC"          # Display label only

# ── Date range ────────────────────────────────────────────────
# Option A: rolling window — fetch BACKTEST_DAYS calendar days back from today
# Option B: fixed window   — set USE_FIXED_DATES = True and fill FIXED_START/END
USE_FIXED_DATES = False
FIXED_START     = "2025-10-01"         # "YYYY-MM-DD", used only when USE_FIXED_DATES = True
FIXED_END       = "2026-03-10"         # "YYYY-MM-DD", used only when USE_FIXED_DATES = True
BACKTEST_DAYS   = 30                   # calendar days back from today (Option A default)

# ── Candle interval ────────────────────────────────────────────
CANDLE_INTERVAL = 5                    # minutes

# ── Session filter ────────────────────────────────────────────
USE_SESSION  = True
SESS_START   = (9,  0)                 # (hour, minute) IST
SESS_END     = (23, 55)                # (hour, minute) IST

# ── Direction ─────────────────────────────────────────────────
DIRECTION    = "Both"                  # "Long Only" | "Short Only" | "Both"

# ── Position sizing ────────────────────────────────────────────
LOT_SIZE     = 2                       # kg per lot (Silver Micro = 1 kg/lot)
TRADE_QTY    = 1                       # number of lots per trade

# ── Risk limits ────────────────────────────────────────────────
MAX_DAILY_LOSS     = 10000             # ₹ — no new entries after this daily loss
MAX_TRADES_PER_DAY = 10               # circuit breaker on over-trading

# ── Fill / cost model ─────────────────────────────────────────
SLIPPAGE_PTS       = 0.0              # one-way slippage in price points
COMMISSION_PER_LOT = 5.0             # ₹ per lot per side (one-way)

# ── Output ────────────────────────────────────────────────────
OUT_DIR       = "."                    # directory for CSV + PNG outputs
GENERATE_PLOT = True                   # set False on headless servers

# ═════════════════════════════════════════════════════════════
#  END OF CONFIGURABLE PARAMETERS
# ═════════════════════════════════════════════════════════════


# Push session / direction / interval into config so strategy.py reads them
config.USE_SESSION     = USE_SESSION
config.SESS_START      = SESS_START
config.SESS_END        = SESS_END
config.DIRECTION       = DIRECTION
config.CANDLE_INTERVAL = CANDLE_INTERVAL


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-8s | %(name)-14s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("backtest.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger("backtest")
logger.setLevel(logging.INFO)


# ─────────────────────────────────────────────────────────────
# Token loader
# ─────────────────────────────────────────────────────────────
def _load_token() -> Optional[str]:
    import json
    path = config.ACCESS_TOKEN_FILE
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    token    = data.get("access_token", "")
    saved_at = data.get("saved_at", "")
    if saved_at:
        saved_date = datetime.fromisoformat(saved_at).date()
        if saved_date < date.today():
            logger.warning(
                "Token saved on %s is expired (today=%s). Re-run auth.py.",
                saved_date, date.today()
            )
            return None
    return token or None


# ─────────────────────────────────────────────────────────────
# Data fetching — Upstox v3 historical candle API
# ─────────────────────────────────────────────────────────────
def fetch_historical_bars(from_dt: str, to_dt: str, access_token: str) -> list:
    """
    Fetch CANDLE_INTERVAL-minute bars from Upstox v3.

    Primary:  /v3/historical-candle/{instrument}/minutes/{N}/{from}/{to}
    Fallback: if the primary returns an error or empty data, fetches 1-minute
              bars and aggregates them up to CANDLE_INTERVAL minutes in-memory.

    Returns list of dicts {ts, open, high, low, close, volume}, oldest-first.
    """
    import requests as req

    N          = CANDLE_INTERVAL
    instrument = quote(INSTRUMENT_KEY, safe="")
    headers    = {
        "Authorization": f"Bearer {access_token}",
        "Accept":        "application/json",
    }

    # ── Primary: N-minute bars ────────────────────────────────
    url = (
        f"https://api.upstox.com/v3/historical-candle/"
        f"{instrument}/minutes/{N}/{from_dt}/{to_dt}"
    )
    logger.info(
        "Fetching %d-min bars from Upstox v3: %s → %s", N, from_dt, to_dt
    )
    logger.info("Request URL: %s", url)
    try:
        r = req.get(url, headers=headers, timeout=30)
        if r.status_code == 200:
            raw = r.json().get("data", {}).get("candles", [])
            if raw:
                bars = _parse_upstox_candles(raw)
                logger.info(
                    "Primary fetch OK: %d × %d-min bars  (%s → %s)",
                    len(bars), N,
                    bars[0]["ts"].strftime("%Y-%m-%d"),
                    bars[-1]["ts"].strftime("%Y-%m-%d"),
                )
                return bars
            logger.warning(
                "Primary fetch returned 0 candles for %d-min — "
                "falling back to 1-min aggregation.", N
            )
        else:
            logger.warning(
                "Primary fetch HTTP %d: %s — falling back to 1-min aggregation.",
                r.status_code, r.text[:200]
            )
    except Exception as e:
        logger.warning("Primary fetch error: %s — falling back to 1-min.", e)

    # ── Fallback: 1-minute bars → aggregate ──────────────────
    logger.info(
        "Fallback: fetching 1-min bars and aggregating to %d-min...", N
    )
    url_1min = (
        f"https://api.upstox.com/v3/historical-candle/"
        f"{instrument}/minutes/1/{from_dt}/{to_dt}"
    )
    logger.info("Request URL (1-min fallback): %s", url_1min)
    try:
        r = req.get(url_1min, headers=headers, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(
                f"1-min fallback HTTP {r.status_code}: {r.text[:300]}\n"
                "Check INSTRUMENT_KEY and ensure the token is valid."
            )
        raw_1min = r.json().get("data", {}).get("candles", [])
        if not raw_1min:
            raise RuntimeError(
                "Both N-min and 1-min fetches returned 0 candles. "
                "The contract may be expired — check INSTRUMENT_KEY."
            )
        bars_1min = _parse_upstox_candles(raw_1min)
        bars      = _aggregate_to_nmin(bars_1min, N)
        logger.info(
            "Fallback OK: %d 1-min bars → %d × %d-min bars  (%s → %s)",
            len(bars_1min), len(bars), N,
            bars[0]["ts"].strftime("%Y-%m-%d") if bars else "—",
            bars[-1]["ts"].strftime("%Y-%m-%d") if bars else "—",
        )
        return bars
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(str(e)) from e


def _parse_upstox_candles(raw: list) -> list:
    """Convert Upstox raw candle list (newest-first) to oldest-first dicts."""
    bars = []
    for c in raw:
        try:
            ts = datetime.fromisoformat(
                str(c[0]).replace("Z", "+00:00")
            ).replace(tzinfo=None)
            bars.append({
                "ts":     ts,
                "open":   float(c[1]),
                "high":   float(c[2]),
                "low":    float(c[3]),
                "close":  float(c[4]),
                "volume": int(float(c[5])) if len(c) > 5 else 0,
            })
        except Exception as e:
            logger.debug("Candle parse error: %s — raw=%s", e, c)
    bars.reverse()
    return bars


def _aggregate_to_nmin(bars_1min: list, N: int) -> list:
    """Aggregate 1-minute bars into N-minute OHLCV bars, oldest-first."""
    buckets = defaultdict(list)
    for b in bars_1min:
        ts     = b["ts"]
        total  = ts.hour * 60 + ts.minute
        snap   = (total // N) * N
        bar_ts = ts.replace(hour=snap // 60, minute=snap % 60,
                            second=0, microsecond=0)
        buckets[bar_ts].append(b)

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


def load_csv_bars(filepath: str) -> list:
    """
    Load OHLCV bars from a local CSV file.
    Expected columns: datetime, open, high, low, close, volume
    datetime format : YYYY-MM-DD HH:MM:SS
    """
    bars = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                bars.append({
                    "ts":     datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S"),
                    "open":   float(row["open"]),
                    "high":   float(row["high"]),
                    "low":    float(row["low"]),
                    "close":  float(row["close"]),
                    "volume": int(float(row.get("volume", 0))),
                })
            except Exception as e:
                logger.debug("CSV row parse error: %s — row=%s", e, row)
    bars.sort(key=lambda x: x["ts"])
    logger.info("Loaded %d bars from %s", len(bars), filepath)
    return bars


# ─────────────────────────────────────────────────────────────
# Backtest Engine
# ─────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Runs the live strategy bar-by-bar over historical data.

    Fill model:
      - Signal fires on bar close; fill simulated at NEXT bar's open.
      - Slippage applied directionally: buy +slip, sell -slip.
      - Commission charged round-trip (open + close).

    Risk model mirrors engine.py:
      - One position at a time
      - Daily loss limit (MAX_DAILY_LOSS) enforced
      - Trade count limit (MAX_TRADES_PER_DAY) enforced
    """

    def __init__(self,
                 slip_pts: float = SLIPPAGE_PTS,
                 commission_per_lot: float = COMMISSION_PER_LOT):
        self.slip     = slip_pts
        self.comm     = commission_per_lot
        self.lot_size = LOT_SIZE
        self.qty      = TRADE_QTY

        self.strategy = KalmanScalperStrategy()

        self._pos          = 0
        self._entry_px     = 0.0
        self._entry_ts     = None

        self._current_day  = None
        self._daily_pnl    = 0.0
        self._daily_trades = 0

        self.trades   = []
        self._equity  = []
        self._cum_pnl = 0.0

        logger.info(
            "BacktestEngine | slip=%.1f pts | comm=₹%.0f/lot/side | qty=%d lot(s)",
            slip_pts, commission_per_lot, self.qty
        )

    def _check_day_reset(self, ts: datetime):
        day = ts.date()
        if day != self._current_day:
            self._current_day  = day
            self._daily_pnl    = 0.0
            self._daily_trades = 0

    def _entry_allowed(self) -> tuple:
        if self._pos != 0:
            return False, "already_in_position"
        if self._daily_pnl <= -MAX_DAILY_LOSS:
            return False, "daily_loss_limit"
        if self._daily_trades >= MAX_TRADES_PER_DAY:
            return False, "max_trades_reached"
        return True, "OK"

    def run(self, bars: list) -> None:
        if len(bars) < 2:
            raise ValueError("Need at least 2 bars to backtest.")

        logger.info("=" * 60)
        logger.info("  BACKTEST START — %d bars  (%s → %s)",
                    len(bars),
                    bars[0]["ts"].strftime("%Y-%m-%d %H:%M"),
                    bars[-1]["ts"].strftime("%Y-%m-%d %H:%M"))
        logger.info("=" * 60)

        pending_signal: Optional[Signal] = None

        for i, bar in enumerate(bars):
            ts = bar["ts"]
            self._check_day_reset(ts)

            # Fill pending signal at this bar's open
            if pending_signal is not None:
                self._execute_fill(pending_signal, bar)
                pending_signal = None

            signal = self.strategy.process_bar(bar)

            unreal = self._unrealised_pnl(bar["close"])
            self._equity.append((ts, self._cum_pnl + unreal))

            if signal is None:
                continue

            if signal.action in ("BUY", "SELL"):
                ok, reason = self._entry_allowed()
                if ok:
                    if i + 1 < len(bars):
                        pending_signal = signal
                else:
                    logger.debug("Entry blocked: %s", reason)

            elif signal.action in ("EXIT_LONG", "EXIT_SHORT"):
                if self._pos != 0:
                    if i + 1 < len(bars):
                        pending_signal = signal
                    else:
                        self._execute_fill(signal, bar, force_close=True)

        # Force-close any open position at end of data
        if self._pos != 0 and bars:
            logger.warning("Open position at end of data — closing at last close.")
            last      = bars[-1]
            direction = -self._pos
            px        = last["close"] + direction * self.slip
            self._close_position(px, last["ts"], "EndOfData")

        logger.info("Backtest complete — %d trades", len(self.trades))

    def _execute_fill(self, signal: Signal, fill_bar: dict, force_close: bool = False):
        px = fill_bar["close"] if force_close else fill_bar["open"]

        if signal.action == "BUY":
            self._open_position(px + self.slip, 1, signal)
        elif signal.action == "SELL":
            self._open_position(px - self.slip, -1, signal)
        elif signal.action == "EXIT_LONG" and self._pos == 1:
            self._close_position(px - self.slip, fill_bar["ts"], signal.reason)
        elif signal.action == "EXIT_SHORT" and self._pos == -1:
            self._close_position(px + self.slip, fill_bar["ts"], signal.reason)

    def _open_position(self, fill_px: float, direction: int, signal: Signal):
        self._pos           = direction
        self._entry_px      = fill_px
        self._entry_ts      = signal.bar_ts
        self._daily_trades += 1
        self.strategy.set_position(direction, fill_px)
        logger.info(
            "OPEN %s @ %.0f  |  normResid=%+.3f  VR=%.3f  vel=%.3f",
            "LONG" if direction == 1 else "SHORT",
            fill_px, signal.norm_resid, signal.vr, signal.vel_norm
        )

    def _close_position(self, fill_px: float, exit_ts: datetime, reason: str):
        direction = self._pos
        if direction == 1:
            gross_pnl = (fill_px - self._entry_px) * self.qty * self.lot_size
        else:
            gross_pnl = (self._entry_px - fill_px) * self.qty * self.lot_size

        total_comm    = self.comm * self.qty * 2   # open + close
        net_pnl       = gross_pnl - total_comm
        self._cum_pnl    += net_pnl
        self._daily_pnl  += net_pnl

        self.trades.append({
            "entry_ts":   self._entry_ts,
            "exit_ts":    exit_ts,
            "direction":  "LONG" if direction == 1 else "SHORT",
            "entry_px":   round(self._entry_px, 1),
            "exit_px":    round(fill_px, 1),
            "qty":        self.qty,
            "gross_pnl":  round(gross_pnl, 2),
            "commission": round(total_comm, 2),
            "net_pnl":    round(net_pnl, 2),
            "cum_pnl":    round(self._cum_pnl, 2),
            "reason":     reason,
        })

        logger.info(
            "CLOSE %s @ %.0f  |  net P&L=₹%+,.0f  |  cum=₹%+,.0f  |  reason=%s",
            "LONG" if direction == 1 else "SHORT",
            fill_px, net_pnl, self._cum_pnl, reason
        )

        self._pos      = 0
        self._entry_px = 0.0
        self.strategy.set_position(0)

    def _unrealised_pnl(self, current_price: float) -> float:
        if self._pos == 0:
            return 0.0
        if self._pos == 1:
            return (current_price - self._entry_px) * self.qty * self.lot_size
        return (self._entry_px - current_price) * self.qty * self.lot_size


# ─────────────────────────────────────────────────────────────
# Performance Report
# ─────────────────────────────────────────────────────────────

def compute_metrics(trades: list, equity_curve: list) -> dict:
    if not trades:
        return {"error": "No trades to analyse"}

    df = pd.DataFrame(trades)

    total_trades  = len(df)
    winners       = df[df["net_pnl"] > 0]
    losers        = df[df["net_pnl"] < 0]
    win_rate      = len(winners) / total_trades * 100

    gross_profit  = winners["net_pnl"].sum() if len(winners) else 0.0
    gross_loss    = losers["net_pnl"].sum()  if len(losers)  else 0.0
    net_pnl       = df["net_pnl"].sum()
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    avg_win  = winners["net_pnl"].mean() if len(winners) else 0.0
    avg_loss = losers["net_pnl"].mean()  if len(losers)  else 0.0
    avg_rr   = abs(avg_win / avg_loss)   if avg_loss != 0 else float("inf")

    eq_vals    = np.array([e[1] for e in equity_curve])
    peak       = np.maximum.accumulate(eq_vals)
    dd         = eq_vals - peak
    max_dd     = float(dd.min())
    dd_denom   = peak[np.argmin(dd)]
    max_dd_pct = (max_dd / dd_denom * 100) if dd_denom != 0 else 0.0
    calmar     = net_pnl / abs(max_dd) if max_dd != 0 else float("inf")

    wins_streak = losses_streak = cur_w = cur_l = 0
    for pnl in df["net_pnl"]:
        if pnl > 0:
            cur_w += 1; cur_l = 0
        elif pnl < 0:
            cur_l += 1; cur_w = 0
        else:
            cur_w = cur_l = 0
        wins_streak   = max(wins_streak,   cur_w)
        losses_streak = max(losses_streak, cur_l)

    df["exit_date"] = pd.to_datetime(df["exit_ts"]).dt.date
    daily_pnl       = df.groupby("exit_date")["net_pnl"].sum()
    trading_days    = len(daily_pnl)

    df["hold_mins"] = (
        pd.to_datetime(df["exit_ts"]) - pd.to_datetime(df["entry_ts"])
    ).dt.total_seconds() / 60

    return {
        "total_trades":    total_trades,
        "win_rate":        round(win_rate, 1),
        "net_pnl":         round(net_pnl, 0),
        "gross_profit":    round(gross_profit, 0),
        "gross_loss":      round(gross_loss, 0),
        "profit_factor":   round(profit_factor, 2),
        "avg_win":         round(avg_win, 0),
        "avg_loss":        round(avg_loss, 0),
        "avg_rr":          round(avg_rr, 2),
        "max_winner":      round(df["net_pnl"].max(), 0),
        "max_loser":       round(df["net_pnl"].min(), 0),
        "max_drawdown":    round(max_dd, 0),
        "max_drawdown_pct":round(max_dd_pct, 2),
        "calmar_ratio":    round(calmar, 2),
        "wins_streak":     wins_streak,
        "losses_streak":   losses_streak,
        "trading_days":    trading_days,
        "avg_daily_pnl":   round(daily_pnl.mean(), 0),
        "best_day":        round(daily_pnl.max(), 0),
        "worst_day":       round(daily_pnl.min(), 0),
        "positive_days":   int((daily_pnl > 0).sum()),
        "avg_hold_mins":   round(df["hold_mins"].mean(), 1),
        "max_hold_mins":   round(df["hold_mins"].max(), 1),
        "exit_reasons":    df["reason"].value_counts().to_dict(),
        "long_trades":     int((df["direction"] == "LONG").sum()),
        "short_trades":    int((df["direction"] == "SHORT").sum()),
    }


def print_report(m: dict, bars: list, slip: float, comm: float):
    sep   = "─" * 58
    start = bars[0]["ts"].strftime("%Y-%m-%d")
    end   = bars[-1]["ts"].strftime("%Y-%m-%d")

    print(f"\n{'═'*58}")
    print(f"  KALMAN SCALPER — BACKTEST REPORT")
    print(f"  {INSTRUMENT_NAME} ({INSTRUMENT_KEY})")
    print(f"  Period  : {start}  →  {end}  ({m['trading_days']} trading days)")
    print(f"  Interval: {CANDLE_INTERVAL} min  |  "
          f"Slip: {slip} pts  |  Comm: ₹{comm}/lot/side")
    print(f"{'═'*58}")

    print(f"\n  TRADE SUMMARY")
    print(sep)
    print(f"  Total trades      : {m['total_trades']}")
    print(f"    Long trades      : {m['long_trades']}")
    print(f"    Short trades     : {m['short_trades']}")
    print(f"  Win rate          : {m['win_rate']}%")
    print(f"  Avg hold time     : {m['avg_hold_mins']} min  (max: {m['max_hold_mins']} min)")
    print(f"  Exit reasons      : {m['exit_reasons']}")

    print(f"\n  P&L")
    print(sep)
    print(f"  Net P&L           : ₹{m['net_pnl']:+,.0f}")
    print(f"  Gross profit      : ₹{m['gross_profit']:+,.0f}")
    print(f"  Gross loss        : ₹{m['gross_loss']:+,.0f}")
    print(f"  Profit factor     : {m['profit_factor']}")
    print(f"  Avg daily P&L     : ₹{m['avg_daily_pnl']:+,.0f}")
    print(f"  Best day          : ₹{m['best_day']:+,.0f}")
    print(f"  Worst day         : ₹{m['worst_day']:+,.0f}")
    print(f"  Positive days     : {m['positive_days']} / {m['trading_days']}")

    print(f"\n  PER-TRADE")
    print(sep)
    print(f"  Avg win           : ₹{m['avg_win']:+,.0f}")
    print(f"  Avg loss          : ₹{m['avg_loss']:+,.0f}")
    print(f"  Avg R:R           : {m['avg_rr']}")
    print(f"  Largest win       : ₹{m['max_winner']:+,.0f}")
    print(f"  Largest loss      : ₹{m['max_loser']:+,.0f}")
    print(f"  Max consec. wins  : {m['wins_streak']}")
    print(f"  Max consec. losses: {m['losses_streak']}")

    print(f"\n  RISK")
    print(sep)
    print(f"  Max drawdown      : ₹{m['max_drawdown']:+,.0f}  ({m['max_drawdown_pct']}%)")
    print(f"  Calmar ratio      : {m['calmar_ratio']}")
    print(f"\n{'═'*58}\n")


def save_trade_log(trades: list, filepath: str):
    if not trades:
        return
    fields = [
        "entry_ts", "exit_ts", "direction", "entry_px", "exit_px",
        "qty", "gross_pnl", "commission", "net_pnl", "cum_pnl", "reason"
    ]
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(trades)
    print(f"  Trade log      → {filepath}")


def save_equity_csv(equity_curve: list, filepath: str):
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["datetime", "equity"])
        for ts, eq in equity_curve:
            w.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), round(eq, 2)])
    print(f"  Equity curve   → {filepath}")


def plot_equity(equity_curve: list, trades: list, filepath: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("  matplotlib not installed — skipping chart. pip install matplotlib")
        return

    timestamps = [e[0] for e in equity_curve]
    equity     = np.array([e[1] for e in equity_curve])
    peak       = np.maximum.accumulate(equity)
    drawdown   = equity - peak

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#c9d1d9", labelsize=9)
        ax.spines[:].set_color("#30363d")
        ax.yaxis.label.set_color("#c9d1d9")

    ax1.plot(timestamps, equity, color="#58a6ff", linewidth=1.2, label="Equity")
    ax1.fill_between(timestamps, equity, alpha=0.08, color="#58a6ff")
    ax1.axhline(0, color="#30363d", linewidth=0.7, linestyle="--")
    for t in trades:
        color = "#3fb950" if t["net_pnl"] > 0 else "#f85149"
        ax1.axvline(t["exit_ts"], color=color, alpha=0.25, linewidth=0.6)
    ax1.set_ylabel("Cumulative Net P&L (₹)", fontsize=10, color="#c9d1d9")
    ax1.set_title(
        f"Kalman Adaptive Scalper — {INSTRUMENT_NAME}  "
        f"({equity_curve[0][0].strftime('%Y-%m-%d')} → "
        f"{equity_curve[-1][0].strftime('%Y-%m-%d')})",
        color="#e6edf3", fontsize=12, fontweight="bold", pad=12
    )
    ax1.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9")

    ax2.fill_between(timestamps, drawdown, color="#f85149", alpha=0.5)
    ax2.plot(timestamps, drawdown, color="#f85149", linewidth=0.8)
    ax2.axhline(0, color="#30363d", linewidth=0.7)
    ax2.set_ylabel("Drawdown (₹)", fontsize=10, color="#c9d1d9")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=30, color="#c9d1d9")
    plt.tight_layout(pad=1.5)
    fig.savefig(filepath, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Equity chart   → {filepath}")


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Backtest the Kalman Adaptive Scalper."
    )
    p.add_argument(
        "--days", type=int, default=None,
        help=f"Calendar days back from today (default: {BACKTEST_DAYS})"
    )
    p.add_argument(
        "--csv", type=str, default=None,
        help="Path to local OHLCV CSV file (skips API fetch)"
    )
    p.add_argument(
        "--slip", type=float, default=SLIPPAGE_PTS,
        help=f"One-way slippage in price points (default: {SLIPPAGE_PTS})"
    )
    p.add_argument(
        "--comm", type=float, default=COMMISSION_PER_LOT,
        help=f"Commission per lot per side in ₹ (default: {COMMISSION_PER_LOT})"
    )
    p.add_argument(
        "--no-plot", action="store_true",
        help="Skip equity chart generation"
    )
    p.add_argument(
        "--out-dir", type=str, default=OUT_DIR,
        help=f"Directory for output files (default: {OUT_DIR!r})"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Resolve date range ─────────────────────────────────
    if USE_FIXED_DATES:
        from_dt = FIXED_START
        to_dt   = FIXED_END
    else:
        days    = args.days if args.days is not None else BACKTEST_DAYS
        today   = date.today()
        to_dt   = today.strftime("%Y-%m-%d")
        from_dt = (today - timedelta(days=days)).strftime("%Y-%m-%d")

    # ── 2. Load bars ──────────────────────────────────────────
    if args.csv:
        if not os.path.exists(args.csv):
            print(f"Error: CSV file not found: {args.csv}")
            sys.exit(1)
        bars = load_csv_bars(args.csv)
    else:
        token = _load_token()
        if not token:
            print(
                "\nError: No valid Upstox token found.\n"
                "Run 'python auth.py' first, or pass --csv <file>.\n"
            )
            sys.exit(1)
        try:
            bars = fetch_historical_bars(from_dt, to_dt, token)
        except RuntimeError as e:
            print(f"\nFetch error: {e}\n")
            sys.exit(1)

    if len(bars) < 50:
        print(f"Only {len(bars)} bars — need at least 50 for meaningful results.")
        sys.exit(1)

    # ── 3. Run backtest ───────────────────────────────────────
    engine = BacktestEngine(slip_pts=args.slip, commission_per_lot=args.comm)
    engine.run(bars)

    if not engine.trades:
        print("\nNo trades generated. Check strategy parameters in config.py.\n")
        sys.exit(0)

    # ── 4. Report ─────────────────────────────────────────────
    metrics = compute_metrics(engine.trades, engine._equity)
    print_report(metrics, bars, args.slip, args.comm)

    # ── 5. Save outputs ───────────────────────────────────────
    print("  Output files:")
    save_trade_log(
        engine.trades,
        os.path.join(args.out_dir, "backtest_trades.csv")
    )
    save_equity_csv(
        engine._equity,
        os.path.join(args.out_dir, "backtest_equity.csv")
    )
    if GENERATE_PLOT and not args.no_plot:
        plot_equity(
            engine._equity,
            engine.trades,
            os.path.join(args.out_dir, "backtest_equity.png")
        )
    print()
