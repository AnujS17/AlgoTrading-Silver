# ─────────────────────────────────────────────────────────────
# config.py  —  ALL user-configurable settings in one place
# Edit this file before running the live engine.
# ─────────────────────────────────────────────────────────────

# ── Upstox Credentials ────────────────────────────────────────
# Get these from https://developer.upstox.com → Your Apps
API_KEY     = ""          # from Upstox developer portal
API_SECRET  = ""       # from Upstox developer portal
REDIRECT_URI = "http://127.0.0.1:8765/callback"   # must match app settings

# Access token — refreshed daily via auth flow (see auth.py)
# On first run, leave blank; the auth flow will populate this.
ACCESS_TOKEN_FILE = "upstox_token.json"

# ── Instrument ────────────────────────────────────────────────
# MCX Silver Micro front-month futures
# Find the current contract key at: https://assets.upstox.com/market-quote/instruments/exchange/MCX.csv.gz
INSTRUMENT_KEY  = "MCX_FO|466029"    # update to current front-month before each expiry
INSTRUMENT_NAME = "SILVERMIC"
EXCHANGE        = "MCX_FO"
TRADING_SYMBOL  = "SILVERMIC25APRFUT"  # update monthly

# ── Capital & Risk ────────────────────────────────────────────
INITIAL_CAPITAL   = 1000000        # ₹ — used only for position sizing reference
EQUITY_RISK_PCT   = 24.0              # % of capital per trade
LOT_SIZE          = 2                # 1 lot = 1 kg Silver Micro
MAX_DAILY_LOSS    = 6500             # ₹ — hard kill switch for the day

MAX_OPEN_TRADES   = 1                # never hold more than 1 position simultaneously
MAX_TRADES_PER_DAY = 10             # circuit breaker on over-trading

# ── Strategy Parameters (Kalman Scalper) ──────────────────────
KALM_Q          = 0.013
KALM_R          = 1.0
ENTRY_THRESH    = 2.1
EXIT_THRESH     = 0.2
VR_LEN          = 25
VR_THRESHOLD    = 1.0
VEL_LEN         = 5
MAX_VEL_MULT    = 1.5
ATR_LEN         = 10
ATR_MULT        = 1.5
COOLDOWN        = 6
DIRECTION       = "Both"            # "Long Only" | "Short Only" | "Both"
CANDLE_INTERVAL = 5                 # minutes — must match your strategy

# How many calendar days of config.CANDLE_INTERVAL-min bars to fetch
# at startup for indicator warmup. More = closer match to TradingView.
# 60 days gives near-perfect convergence of Kalman/ATR/VR/Velocity.
# Only exact-interval bars are used — never a different resolution.
WARMUP_DAYS     = 20                # calendar days of history to replay

# ── Session Filter ────────────────────────────────────────────
USE_SESSION  = True
SESS_START   = (9,  00)             # (hour, minute) — IST
SESS_END     = (23,  55)

# ── Overnight Position Holding ───────────────────────────────
# Set True  → engine leaves open positions untouched on Ctrl+C / stop,
#             and reconnects to manage them on next day's startup.
# Set False → engine flattens ALL positions before shutting down (safe default).
#
# ⚠️  REQUIRES PRODUCT_TYPE = "D" (NRML/Delivery).
#     If PRODUCT_TYPE = "I" (Intraday / MIS), the broker will auto-square off
#     your position at session end regardless of this flag.
#
# ⚠️  MCX Silver MIC is a monthly contract.
#     Do NOT hold overnight on the contract's last trading day (expiry).
#     Verify INSTRUMENT_KEY is the next month's contract before doing so.
ALLOW_OVERNIGHT = False             # True = hold positions overnight | False = always flatten on exit

# ── Order Settings ────────────────────────────────────────────
ORDER_TYPE      = "LIMIT"          # MARKET or LIMIT
PRODUCT_TYPE    = "D"               # "D" = NRML/Delivery (supports overnight) | "I" = Intraday/MIS (auto-squared off at session end)
SLIPPAGE_GUARD  = 90                # pts — cancel limit order if market moves this far

# ── Logging ───────────────────────────────────────────────────
LOG_FILE        = "kalman_live.log"
TRADE_LOG_CSV   = "live_trades.csv"
