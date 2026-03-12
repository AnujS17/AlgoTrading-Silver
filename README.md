# Kalman Adaptive Scalper — Live Trading Engine
## MCX Silver Micro | Upstox | Python

---

## ⚠️ IMPORTANT DISCLAIMER
This software places REAL orders with REAL money.
Test in paper trading mode first. Never risk more than you can afford to lose.
The authors are not responsible for trading losses.

It is a fully automatic algo trading setup that fires entry and exit signals as per your strategy and configurations setup in the project. Just upload your strategy logic and the project will handle automatic authentication, signal generation, market entry, slippage etc. I have used a strategy involving Kalman filters for my trading setup.
---

## File Structure

```
live_trader/
├── config.py       ← ALL settings — edit this first
├── auth.py         ← Run once each morning to get Upstox token
├── broker.py       ← Upstox REST API (orders, positions, funds)
├── feed.py         ← WebSocket feed + candle assembler
├── strategy.py     ← Kalman signal engine (stateful, bar-by-bar)
├── engine.py       ← Main orchestrator — run this to trade live
├── requirements.txt
└── README.md
```

---

## Setup (One-Time)

### 1. Create a Upstox Developer App
1. Go to https://developer.upstox.com
2. Create an app → note your **API Key** and **API Secret**
3. Set Redirect URI to: `http://127.0.0.1:5000/callback`

### 2. Edit config.py
```python
API_KEY      = "your_api_key_here"
API_SECRET   = "your_api_secret_here"
REDIRECT_URI = "http://127.0.0.1:5000/callback"
```
Also update `INSTRUMENT_KEY` and `TRADING_SYMBOL` to the
current Silver MIC front-month contract before each expiry.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Daily Startup Procedure

### Step 1 — Authenticate (run every morning before market open)
```bash
python auth.py
```
- Your browser opens to Upstox login
- Log in → token saved to `upstox_token.json`
- Takes ~30 seconds

### Step 2 — Start the live engine
```bash
python engine.py
```
The engine will:
- Validate your token
- Connect to the Upstox WebSocket feed
- Subscribe to Silver MIC tick data
- Assemble 5-minute candles in real time
- Run the Kalman strategy on each completed candle
- Place BUY / SELL orders automatically
- Log everything to `kalman_live.log` and `live_trades.csv`

### Step 3 — Stop
Press `Ctrl+C` — the engine will automatically **close all open positions** before exiting.

---

## Risk Controls (in config.py)

| Parameter | Default | Description |
|---|---|---|
| `MAX_DAILY_LOSS` | ₹3,000 | Hard kill switch — no new trades after this loss |
| `MAX_TRADES_PER_DAY` | 10 | Prevents over-trading |
| `MAX_OPEN_TRADES` | 1 | Only 1 position at a time |
| `EQUITY_RISK_PCT` | 2.0% | Capital risked per trade |
| `LOT_SIZE` | 1 | Number of Silver MIC lots |

**Start with LOT_SIZE=1 and MAX_DAILY_LOSS=₹2000 until you're confident.**

---

## Instrument Key Update (Monthly)

Silver MIC is a monthly expiry contract. You must update
`INSTRUMENT_KEY` and `TRADING_SYMBOL` in config.py before
each contract rollover (usually last week of the month).

Find the new key:
1. Download: https://assets.upstox.com/market-quote/instruments/exchange/MCX.csv.gz
2. Search for `SILVERMIC` with the upcoming expiry date
3. Copy the `instrument_key` value into config.py

---

## Monitoring

### Live log
```bash
tail -f kalman_live.log
```

### Trade history
```bash
# View today's trades
python -c "
import pandas as pd
df = pd.read_csv('live_trades.csv')
print(df.tail(20).to_string())
print(f'Daily P&L: ₹{df[df.action.str.contains(\"EXIT\")][\"pnl\"].sum():,.0f}')
"
```

---

## Troubleshooting

**"Token file not found"**
→ Run `python auth.py` first

**"Token expired"**
→ Upstox tokens expire at midnight IST. Re-run `python auth.py` each morning.

**WebSocket not connecting**
→ In engine.py, change `use_websocket=True` to `use_websocket=False`
  This uses REST polling as a fallback (slightly less real-time)

**Orders rejected**
→ Check available margin on Upstox app
→ Ensure TRADING_SYMBOL matches the current front-month contract
→ Check `kalman_live.log` for the rejection reason

**Position mismatch warning**
→ The engine syncs with broker every bar — it will self-correct
→ If it persists, check your Upstox order book manually

---

## Paper Trading First!

Before going live, test the engine by:
1. Setting `ORDER_TYPE = "LIMIT"` with an unreachable price
2. Watching logs to confirm signals fire correctly
3. Then switch to `ORDER_TYPE = "MARKET"` for live execution
