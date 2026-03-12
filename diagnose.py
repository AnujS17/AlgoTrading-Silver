"""
diagnose.py v2 — Deep feed diagnostic for Upstox Silver MIC.

Prints RAW responses at every layer so you can pinpoint exactly
where the chain breaks. No assumptions — everything verified live.

Run: python diagnose.py
"""

import json, sys, threading, time, struct
from datetime import datetime, date
import requests

sys.path.insert(0, ".")
import config
from auth import load_token

SEP  = "=" * 62
SEP2 = "-" * 62
PASS = "  [PASS]"
FAIL = "  [FAIL]"
SKIP = "  [SKIP]"
INFO = "  [INFO]"

print("\n" + SEP)
print("  UPSTOX FEED DIAGNOSTICS  v2")
print("  Run at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"))
print(SEP)

# ── Load token ────────────────────────────────────────────────
try:
    TOKEN = load_token()
    print(PASS, "Token loaded:", TOKEN[:20] + "..." + TOKEN[-8:])
except Exception as e:
    print(FAIL, "Token error:", e)
    print("       Run: python auth.py")
    sys.exit(1)

H  = {"Authorization": "Bearer " + TOKEN, "Accept": "application/json"}
IK = config.INSTRUMENT_KEY
IV = config.CANDLE_INTERVAL
print(INFO, "Instrument key :", IK)
print(INFO, "Candle interval:", IV, "min")
print(INFO, "Today's date   :", date.today())

# ═════════════════════════════════════════════════════════════
# T1 — Profile / token validity
# ═════════════════════════════════════════════════════════════
print("\n" + SEP2)
print("  T1 | Profile API (token validity check)")
print(SEP2)
try:
    r = requests.get("https://api.upstox.com/v2/user/profile", headers=H, timeout=8)
    print("  HTTP:", r.status_code)
    if r.status_code == 200:
        d = r.json().get("data", {})
        print(PASS, "User:", d.get("user_name"), "| Email:", d.get("email"))
    else:
        print(FAIL, "Response:", r.text[:200])
        print("  Token is invalid. Run: python auth.py")
        sys.exit(1)
except Exception as e:
    print(FAIL, e)
    sys.exit(1)

# ═════════════════════════════════════════════════════════════
# T2 — Market status (is MCX open right now?)
# ═════════════════════════════════════════════════════════════
print("\n" + SEP2)
print("  T2 | Market Status")
print(SEP2)
try:
    r = requests.get("https://api.upstox.com/v2/market/status/NSE", headers=H, timeout=8)
    print("  NSE HTTP:", r.status_code, "->", r.json().get("data", {}).get("market_status", r.text[:80]))
except Exception as e:
    print(INFO, "NSE status error:", e)

try:
    r = requests.get("https://api.upstox.com/v2/market/status/MCX", headers=H, timeout=8)
    print("  MCX HTTP:", r.status_code, "->", r.json().get("data", {}).get("market_status", r.text[:80]))
    mcx_status = r.json().get("data", {}).get("market_status", "")
    if "close" in mcx_status.lower():
        print("  *** MCX IS CLOSED — candles will only arrive during market hours ***")
except Exception as e:
    print(INFO, "MCX status error:", e)

# ═════════════════════════════════════════════════════════════
# T3 — Instrument key validation
# ═════════════════════════════════════════════════════════════
print("\n" + SEP2)
print("  T3 | Instrument Key Validation — LTP Quote")
print(SEP2)

quote_urls = [
    ("v2 ltp",    "https://api.upstox.com/v2/market-quote/ltp",    {"instrument_key": IK}),
    ("v2 quotes", "https://api.upstox.com/v2/market-quote/quotes",  {"instrument_key": IK}),
    ("v2 ohlc",   "https://api.upstox.com/v2/market-quote/ohlc",    {"instrument_key": IK, "interval": "I%d" % IV}),
]

ltp_found = False
for label, url, params in quote_urls:
    try:
        r = requests.get(url, headers=H, params=params, timeout=8)
        print("  %s | HTTP %s" % (label, r.status_code))
        if r.status_code == 200:
            raw = r.json()
            data = raw.get("data", {})
            print("       data keys:", list(data.keys())[:6])
            for k, v in data.items():
                ltp = v.get("last_price") or v.get("ltp") or v.get("close")
                print("       %s -> LTP/close = %s" % (k, ltp))
                if ltp:
                    ltp_found = True
        else:
            print("       Error:", r.text[:120])
    except Exception as e:
        print("  %s | ERROR: %s" % (label, e))

if ltp_found:
    print(PASS, "Instrument key is valid — LTP received")
else:
    print(FAIL, "No LTP returned for instrument key:", IK)
    print("       If MCX is closed this is normal — try during market hours.")
    print("       Otherwise update INSTRUMENT_KEY in config.py")

# ═════════════════════════════════════════════════════════════
# T4 — Historical candle API (all URL variants)
# ═════════════════════════════════════════════════════════════
print("\n" + SEP2)
print("  T4 | Historical Candle API — all URL variants")
print(SEP2)

today     = date.today().strftime("%Y-%m-%d")
yesterday = date.today().strftime("%Y-%m-%d")  # same day for intraday

candle_urls = [
    ("v2 intraday 1min",  "https://api.upstox.com/v2/historical-candle/intraday/%s/1minute" % IK),
    ("v2 intraday 30min", "https://api.upstox.com/v2/historical-candle/intraday/%s/30minute" % IK),
    ("v2 intraday 1min",  "https://api.upstox.com/v2/historical-candle/intraday/%s/1minute" % IK),
    ("v2 historical 5m",  "https://api.upstox.com/v2/historical-candle/%s/minutes/%s/%s/%s" % (IK, IV, today, today)),
    ("v2 historical 1m",  "https://api.upstox.com/v2/historical-candle/%s/1minute/%s/%s"    % (IK, today, today)),
]

working_candle_url = None
for label, url in candle_urls:
    try:
        r = requests.get(url, headers=H, timeout=10)
        candles = []
        if r.status_code == 200:
            body = r.json()
            candles = (body.get("data", {}).get("candles", [])
                       or body.get("data", {}).get("candles", []))
            print("  %s | HTTP 200 | candles: %d" % (label, len(candles)))
            if candles:
                c = candles[0]
                print("       Latest bar: ts=%s O=%.0f H=%.0f L=%.0f C=%.0f" % (
                    c[0], float(c[1]), float(c[2]), float(c[3]), float(c[4])))
                if working_candle_url is None:
                    working_candle_url = (label, url)
            else:
                print("       Empty candles array (market closed or wrong key?)")
                print("       Full response:", json.dumps(body)[:200])
        else:
            body_text = r.text[:150]
            print("  %s | HTTP %s | %s" % (label, r.status_code, body_text))
    except Exception as e:
        print("  %s | ERROR: %s" % (label, e))

if working_candle_url:
    print(PASS, "Working candle URL: %s" % working_candle_url[0])
    print("       URL:", working_candle_url[1])
else:
    print(FAIL, "No candle URL returned data")
    print("       If market is open, check INSTRUMENT_KEY in config.py")

# ═════════════════════════════════════════════════════════════
# T5 — WebSocket authorized URL
# ═════════════════════════════════════════════════════════════
print("\n" + SEP2)
print("  T5 | WebSocket Authorized URL")
print(SEP2)

ws_url = None
# feed.py uses the v3 authorize endpoint — test it first.
# The v2 endpoint is kept as a fallback reference.
ws_auth_urls = [
    "https://api.upstox.com/v3/feed/market-data-feed/authorize",   # used by feed.py
    "https://api.upstox.com/v2/feed/market-data-feed/authorize",   # legacy reference
]
for url in ws_auth_urls:
    try:
        r = requests.get(url, headers=H, timeout=10)
        print("  %s | HTTP %s" % (url.split("/")[-2] + "/authorize", r.status_code))
        if r.status_code == 200:
            ws_url = r.json()["data"]["authorizedRedirectUri"]
            print(PASS, "Got WS URL:", ws_url[:70] + "...")
            break
        else:
            print("       Error:", r.text[:150])
    except Exception as e:
        print("  ERROR:", e)

# ═════════════════════════════════════════════════════════════
# T6 — WebSocket connection + raw message dump
# ═════════════════════════════════════════════════════════════
print("\n" + SEP2)
print("  T6 | WebSocket Connection + Raw Message Dump (15s)")
print(SEP2)

if not ws_url:
    print(SKIP, "No WS URL — skipping WebSocket test")
else:
    try:
        import websocket
        ws_ok = True
    except ImportError:
        ws_ok = False
        print(SKIP, "websocket-client not installed (pip install websocket-client)")

    if ws_ok:
        msgs   = []
        ws_evt = threading.Event()

        def on_open(ws):
            print(PASS, "WebSocket connected!")
            sub = json.dumps({
                "guid": "diag",
                "method": "sub",
                "data": {"mode": "ltpc", "instrumentKeys": [IK]}
            })
            ws.send(sub)
            print(INFO, "Subscribe sent for", IK, "in ltpc mode")

        def on_message(ws, message):
            idx = len(msgs) + 1
            if isinstance(message, bytes):
                msgs.append(message)
                print(INFO, "MSG #%d: bytes len=%d  hex_preview=%s" % (
                    idx, len(message), message[:16].hex()))
                # Try to extract LTP from raw bytes using struct
                # Upstox proto: field 1 = ltp (double, tag=0x09)
                for i in range(len(message) - 8):
                    if message[i] == 0x09:  # field 1, wire type 1 (64-bit)
                        try:
                            val = struct.unpack_from('<d', message, i+1)[0]
                            if 50000 < val < 200000:  # Silver price range
                                print(INFO, "  Possible LTP at offset %d: %.2f" % (i, val))
                        except:
                            pass
            else:
                msgs.append(message)
                print(INFO, "MSG #%d: string len=%d  preview=%s" % (
                    idx, len(message), str(message)[:100]))
            if idx >= 5:
                ws.close()
                ws_evt.set()

        def on_error(ws, error):
            print(FAIL, "WS error:", str(error)[:200])
            ws_evt.set()

        def on_close(ws, code, msg):
            print(INFO, "WS closed code=%s" % code)
            ws_evt.set()

        try:
            ws_app = websocket.WebSocketApp(
                ws_url,
                on_open=on_open, on_message=on_message,
                on_error=on_error, on_close=on_close
            )
            threading.Thread(target=ws_app.run_forever, daemon=True).start()
            ws_evt.wait(timeout=20)

            if msgs:
                print(PASS, "%d WS messages received" % len(msgs))
                if isinstance(msgs[0], bytes):
                    print(INFO, "Messages are binary (protobuf) — SDK needed to decode LTP")
                    # Try SDK decode
                    try:
                        from upstox_client.utils.market_data_feed_helper import decode_data
                        decoded = decode_data(msgs[0])
                        print(PASS, "SDK decode_data worked:", str(decoded)[:200])
                    except Exception as e:
                        print(FAIL, "SDK decode_data failed:", e)
                        print("       Try: from upstox_client.feeder.market_data_feeder import decode_proto")
                        try:
                            from upstox_client.feeder.market_data_feeder import decode_proto
                            decoded = decode_proto(msgs[0])
                            print(PASS, "decode_proto worked:", str(decoded)[:200])
                        except Exception as e2:
                            print(FAIL, "decode_proto also failed:", e2)
                            # List all upstox_client submodules to find decoder
                            try:
                                import upstox_client, pkgutil
                                mods = [m.name for m in pkgutil.walk_packages(
                                    upstox_client.__path__,
                                    upstox_client.__name__ + "."
                                )]
                                feed_mods = [m for m in mods if any(
                                    k in m.lower() for k in ["feed", "stream", "decode", "market"]
                                )]
                                print(INFO, "Relevant SDK modules:", feed_mods)
                            except Exception:
                                pass
            else:
                print(FAIL, "No WS messages in 20s")
                print("       If MCX is closed, no ticks will arrive")
                print("       Try during market hours (09:00 - 23:30 IST)")
        except Exception as e:
            print(FAIL, "WS test error:", e)

# ═════════════════════════════════════════════════════════════
# T7 — SDK module map (find the correct decoder path)
# ═════════════════════════════════════════════════════════════
print("\n" + SEP2)
print("  T7 | upstox-python-sdk Module Map")
print(SEP2)
try:
    import upstox_client, pkgutil
    ver = getattr(upstox_client, "__version__", "unknown")
    print(PASS, "SDK installed, version:", ver)
    all_mods = [m.name for m in pkgutil.walk_packages(
        upstox_client.__path__, upstox_client.__name__ + ".")]
    print(INFO, "Total modules:", len(all_mods))
    feed_mods = [m for m in all_mods if any(
        k in m.lower() for k in ["feed", "stream", "decode", "socket", "market"])]
    print(INFO, "Feed/stream modules:")
    for m in feed_mods:
        print("       ", m)
    # Check top-level classes
    classes = [x for x in dir(upstox_client) if not x.startswith("_") and x[0].isupper()]
    stream_classes = [x for x in classes if any(
        k in x.lower() for k in ["stream", "feed", "socket", "market"])]
    print(INFO, "Top-level streamer classes:", stream_classes)
except ImportError:
    print(FAIL, "upstox-python-sdk not installed (pip install upstox-python-sdk)")

# ═════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════
print("\n" + SEP)
print("  DIAGNOSIS COMPLETE")
print(SEP)
print()
print("  Checklist:")
print("  1. MCX open 09:00-23:30 IST? Check T2 output above")
print("  2. Instrument key valid?    Check T3 (LTP received?)")
print("  3. Candle API working?      Check T4 (got candles?)")
print("  4. WebSocket connecting?    Check T6 (msgs received?)")
print("  5. Protobuf decoding?       Check T6 (LTP extracted?)")
print()
if working_candle_url:
    print("  ACTION: Update RESTCandleFeed.URL in feed.py to use:")
    print("  " + working_candle_url[1].replace(IK, "{instrument}").replace(str(IV), "{interval}"))
print(SEP + "\n")
