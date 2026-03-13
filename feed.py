"""
feed.py — Upstox real-time data feed for Kalman Scalper.

TIER SELECTION (auto):
  Tier 1 — upstox_client.MarketDataStreamer (SDK v3, recommended)
            Needs: pip install upstox-python-sdk
  Tier 2 — Raw WebSocket + SDK protobuf decoder (fallback if Tier 1 fails)
            Needs: pip install websocket-client upstox-python-sdk
  Tier 3 — REST 1-min poller + N-min aggregator (no extra deps, ~30-60s delay)

BUGS FIXED vs PREVIOUS VERSION:
  1. upstox-python-sdk was optional (commented out in requirements.txt).
     Without it, both Tier 1 and Tier 2 are unavailable, falling to REST
     which logged "no data" at DEBUG level — invisible to the user.
     FIX: SDK is now required. Also added Tier 1 using MarketDataStreamer.

  2. _extract_ltp_ltt() silently dropped ALL ticks even when SDK was present.
     The SDK wraps its message as {"type": "live_feed", "feeds": {instr: ...}}.
     The old extractor iterated decoded.items() and got ("type", "live_feed")
     as the first entry — never drilling into "feeds" — so LTP was never found.
     FIX: Unwrap the SDK envelope before extracting LTP.

  3. REST feed logged "no bars returned" at DEBUG — invisible at INFO level.
     FIX: Elevated to WARNING; also prints a market-hours reminder.

  4. int(ltt) crashed if ltt was a string ISO timestamp or proto Timestamp.
     FIX: Robust _parse_ltt() handles int ms, string ISO, and proto Timestamp.

  5. diagnose.py only tested the v2 WS auth endpoint; _get_authorized_ws_url()
     uses v3. A v3 auth failure was invisible in diagnostics.
     FIX: diagnose.py updated to test both v2 and v3 authorize URLs.
"""

import json
import logging
import threading
import time
from datetime import datetime
from typing import Callable, Optional

import requests
import config

logger = logging.getLogger("kalman.feed")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _new_candle(ts, price, vol):
    return {"ts": ts, "open": price, "high": price,
            "low": price, "close": price, "volume": vol}


def _get_authorized_ws_url(access_token: str) -> str:
    """Fetch a signed WebSocket URL from the Upstox v3 authorize endpoint."""
    url = "https://api.upstox.com/v3/feed/market-data-feed/authorize"
    headers = {"Authorization": "Bearer " + access_token, "Accept": "application/json"}
    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code != 200:
        raise RuntimeError("WS v3 auth failed (%d): %s" % (r.status_code, r.text[:200]))
    data = r.json().get("data", {})
    # Upstox returns snake_case: "authorized_redirect_uri"
    ws_url = data.get("authorized_redirect_uri") or data.get("authorizedRedirectUri")
    if not ws_url:
        raise RuntimeError("No WS URL in auth response: %s" % r.text[:200])
    return ws_url


def _parse_ltt(ltt) -> datetime:
    """
    Convert Upstox ltt (last traded time) to a datetime.

    Handles three formats that come back depending on SDK version:
      - int / float  : Unix milliseconds (most common)
      - str          : ISO-8601 string e.g. "2025-03-09T10:30:00+05:30"
      - proto object : has .seconds and .nanos attributes
    Falls back to datetime.now() if conversion fails.
    """
    if ltt is None:
        return datetime.now()
    try:
        # Proto Timestamp (has .seconds attribute)
        if hasattr(ltt, "seconds"):
            return datetime.fromtimestamp(ltt.seconds + ltt.nanos / 1e9)
        # Numeric — treat as milliseconds
        if isinstance(ltt, (int, float)):
            return datetime.fromtimestamp(int(ltt) / 1000)
        # String — ISO-8601
        if isinstance(ltt, str):
            return datetime.fromisoformat(ltt.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception as e:
        logger.debug("ltt parse failed (%s) for value %r — using now()", e, ltt)
    return datetime.now()


# ─────────────────────────────────────────────────────────────
# Candle Assembler  —  tick -> OHLCV bar
# ─────────────────────────────────────────────────────────────
class CandleAssembler:
    def __init__(self, interval_minutes: int,
                 on_candle_close: Callable,
                 on_tick: Optional[Callable] = None):
        self.interval    = interval_minutes
        self.callback    = on_candle_close
        self._on_tick    = on_tick   # optional per-tick callback (ltp, ts) → None
        self._candle     = None
        self._lock       = threading.Lock()
        self._tick_count = 0

    def _bar_start(self, ts: datetime) -> datetime:
        total   = ts.hour * 60 + ts.minute
        snapped = (total // self.interval) * self.interval
        return ts.replace(hour=snapped // 60, minute=snapped % 60,
                          second=0, microsecond=0)

    def process_tick(self, ltp: float, volume: int, ts: datetime):
        with self._lock:
            self._tick_count += 1
            if self._tick_count % 100 == 1:
                logger.debug("Tick #%d: %.0f @ %s", self._tick_count, ltp, ts)

            bar_ts = self._bar_start(ts)

            if self._candle is None:
                self._candle = _new_candle(bar_ts, ltp, volume)
                logger.info("First tick received  LTP=%.0f  bar=%s", ltp, bar_ts)
                if self._on_tick:
                    try:
                        self._on_tick(ltp, ts)
                    except Exception as e:
                        logger.debug("on_tick error: %s", e)
                return

            if bar_ts == self._candle["ts"]:
                c = self._candle
                c["high"]    = max(c["high"],  ltp)
                c["low"]     = min(c["low"],   ltp)
                c["close"]   = ltp
                c["volume"] += volume
                # fire on_tick every tick while bar is open (intrabar trail monitor)
                if self._on_tick:
                    try:
                        self._on_tick(ltp, ts)
                    except Exception as e:
                        logger.debug("on_tick error: %s", e)
            else:
                done = dict(self._candle)
                logger.info(
                    "CANDLE CLOSED  %s  O=%.0f  H=%.0f  L=%.0f  C=%.0f",
                    done["ts"], done["open"], done["high"], done["low"], done["close"]
                )
                try:
                    self.callback(done)
                except Exception as e:
                    logger.error("Strategy callback error: %s", e, exc_info=True)
                self._candle = _new_candle(bar_ts, ltp, volume)
                # also fire on_tick for the first tick of the new bar
                if self._on_tick:
                    try:
                        self._on_tick(ltp, ts)
                    except Exception as e:
                        logger.debug("on_tick error: %s", e)

    def get_current(self) -> Optional[dict]:
        with self._lock:
            return dict(self._candle) if self._candle else None


# ─────────────────────────────────────────────────────────────
# LTP extractor  —  handles SDK envelope and raw proto formats
# ─────────────────────────────────────────────────────────────
def _extract_ltp_ltt(decoded) -> Optional[tuple]:
    """
    Extract (ltp: float, ltt: raw) from the decoded feed message.

    Handles three SDK response shapes:

    Shape A — SDK MarketDataStreamer envelope (most common):
      {"type": "live_feed", "feeds": {instr: {ff: {marketFF: {ltpc: {ltp, ltt}}}}}}

    Shape B — Flat dict (some SDK versions / raw proto decode):
      {instr: {ltpc: {ltp, ltt}}}
      {instr: {ff: {marketFF: {ltpc: {ltp, ltt}}}}}

    Shape C — Proto message object (direct pb2 parse):
      decoded.feeds[instr].ff.market_ff.ltpc.ltp
    """
    if decoded is None:
        return None

    if isinstance(decoded, dict):
        # ── Shape A: unwrap SDK envelope ──────────────────────
        # {"type": "live_feed", "feeds": {instr: ...}}
        # IMPORTANT: must unwrap BEFORE iterating, otherwise "type"
        # and "feeds" are treated as instrument keys and LTP is never found.
        if "feeds" in decoded and isinstance(decoded["feeds"], dict):
            decoded = decoded["feeds"]   # now looks like Shape B

        # ── Shape B: flat instrument-keyed dict ───────────────
        for key, feed in decoded.items():
            if not isinstance(feed, dict):
                continue

            # B1: {ltpc: {ltp, ltt}}
            ltpc = feed.get("ltpc")
            if isinstance(ltpc, dict):
                ltp = ltpc.get("ltp")
                if ltp:
                    return float(ltp), ltpc.get("ltt")

            # B2: {ff: {marketFF: {ltpc: {ltp, ltt}}}}  (older SDK shape)
            ff   = feed.get("ff", {})
            mff  = ff.get("marketFF", {}) if isinstance(ff, dict) else {}
            ltpc = mff.get("ltpc", {})    if isinstance(mff, dict) else {}
            if isinstance(ltpc, dict):
                ltp = ltpc.get("ltp")
                if ltp:
                    return float(ltp), ltpc.get("ltt")

            # B3: {fullFeed: {marketFF: {ltpc: {ltp, ltt}}}}  (v3 full mode)
            full_feed = feed.get("fullFeed", {})
            mff2      = full_feed.get("marketFF", {}) if isinstance(full_feed, dict) else {}
            ltpc2     = mff2.get("ltpc", {})          if isinstance(mff2, dict) else {}
            if isinstance(ltpc2, dict):
                ltp = ltpc2.get("ltp")
                if ltp:
                    return float(ltp), ltpc2.get("ltt")

    # ── Shape C: proto message object ─────────────────────────
    try:
        for key, feed in decoded.feeds.items():
            ltp = feed.ff.market_ff.ltpc.ltp
            ltt = feed.ff.market_ff.ltpc.ltt
            if ltp:
                return float(ltp), ltt
    except AttributeError:
        pass

    return None


# ─────────────────────────────────────────────────────────────
# Proto decoder  —  for raw WebSocket bytes (Tier 2 fallback)
# ─────────────────────────────────────────────────────────────
def _decode_proto_message(raw: bytes) -> Optional[dict]:
    """
    Decode Upstox v3 protobuf binary frame.

    Tries SDK paths first, then falls back to a pure-Python wire-format
    parser that needs zero external dependencies.

    Wire-format path to LTP in Upstox v3 FeedResponse:
      FeedResponse  { map<string, Feed> feeds = 2 }
      Feed          { LTPC ltpc = 1  |  FullFeed full_feed = 2 }
      LTPC          { double ltp = 1;  Timestamp ltt = 2 }
      FullFeed      { MarketFullFeed marketFF = 1 }
      MarketFullFeed{ LTPC ltpc = 1 }
    """
    # ── Path 1: SDK utils helper (SDK >= 2.x) ─────────────────
    try:
        from upstox_client.utils.market_data_feed_helper import decode_data
        return decode_data(raw)
    except (ImportError, AttributeError):
        pass
    except Exception as e:
        logger.debug("utils.market_data_feed_helper.decode_data error: %s", e)

    # ── Path 2: SDK feeder v3 decode_data ─────────────────────
    try:
        from upstox_client.feeder.market_data_feeder_v3 import decode_data
        return decode_data(raw)
    except (ImportError, AttributeError):
        pass
    except Exception as e:
        logger.debug("market_data_feeder_v3.decode_data error: %s", e)

    # ── Path 3: SDK feeder v3 decode_proto ────────────────────
    try:
        from upstox_client.feeder.market_data_feeder_v3 import decode_proto
        return decode_proto(raw)
    except (ImportError, AttributeError):
        pass

    # ── Path 4: Direct pb2 parse ──────────────────────────────
    try:
        from upstox_client.feeder.proto import MarketDataFeedV3_pb2 as pb
        msg = pb.FeedResponse()
        msg.ParseFromString(raw)
        out = {}
        for key, feed in msg.feeds.items():
            ltp = feed.ff.market_ff.ltpc.ltp
            ltt = feed.ff.market_ff.ltpc.ltt
            if ltp:
                out[key] = {"ltpc": {"ltp": ltp, "ltt": ltt}}
        return out if out else None
    except (ImportError, AttributeError):
        pass
    except Exception as e:
        logger.debug("pb2 direct parse error: %s", e)

    # ── Path 5: Dynamic discovery in SDK ──────────────────────
    try:
        import upstox_client.feeder.market_data_feeder_v3 as m
        for fname in dir(m):
            if "decode" in fname.lower() and callable(getattr(m, fname)):
                logger.info("Found decoder via discovery: %s.%s", m.__name__, fname)
                return getattr(m, fname)(raw)
    except Exception as e:
        logger.debug("Dynamic decoder search error: %s", e)

    # ── Path 6: Pure-Python protobuf wire-format parser ───────
    # No SDK, no compiled proto files needed.
    # Parses the binary directly using the protobuf wire format spec.
    result = _decode_proto_pure_python(raw)
    if result:
        return result

    logger.warning("All proto decoder paths failed. hex_prefix=%s", raw[:20].hex())
    return None


def _decode_proto_pure_python(raw: bytes) -> Optional[dict]:
    """
    Pure-Python protobuf wire-format decoder for Upstox v3 FeedResponse.

    Handles two message shapes that come off the wire:
      Shape A — ltpc  : Feed.ltpc (field 1) -> LTPC.ltp (field 1, double)
      Shape B — full  : Feed.full_feed (field 2) -> MarketFullFeed (field 1) -> LTPC (field 1) -> ltp

    Returns a dict compatible with _extract_ltp_ltt():
      {"feeds": {"MCX_FO|466029": {"ltpc": {"ltp": 92500.0, "ltt": 1710000000000}}}}
    """
    import struct

    # ── Wire-format helpers ────────────────────────────────────
    def read_varint(buf, pos):
        result, shift = 0, 0
        while pos < len(buf):
            b = buf[pos]; pos += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        return result, pos

    def read_field(buf, pos):
        """Returns (field_number, value, new_pos). value type depends on wire_type."""
        if pos >= len(buf):
            return None, None, pos
        tag, pos = read_varint(buf, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7
        try:
            if wire_type == 0:   # varint
                val, pos = read_varint(buf, pos)
                return field_num, val, pos
            elif wire_type == 1:  # 64-bit (double / fixed64)
                val = struct.unpack_from('<Q', buf, pos)[0]
                return field_num, val, pos + 8
            elif wire_type == 2:  # length-delimited (bytes / string / sub-message)
                length, pos = read_varint(buf, pos)
                val = buf[pos: pos + length]
                return field_num, val, pos + length
            elif wire_type == 5:  # 32-bit (float / fixed32)
                val = struct.unpack_from('<I', buf, pos)[0]
                return field_num, val, pos + 4
            else:
                return None, None, pos + 1   # unknown wire type — skip 1 byte
        except struct.error:
            return None, None, len(buf)

    def parse_ltpc(buf):
        """Parse LTPC message -> (ltp: float, ltt_ms: int|None)."""
        pos, ltp, ltt_ms = 0, None, None
        while pos < len(buf):
            fn, val, pos = read_field(buf, pos)
            if fn is None:
                break
            if fn == 1 and isinstance(val, int):
                # field 1 of LTPC is `double ltp` → stored as wire_type 1 (64-bit)
                candidate = struct.unpack('<d', struct.pack('<Q', val))[0]
                if 1000 < candidate < 10_000_000:   # broad sanity: covers all MCX prices
                    ltp = candidate
            elif fn == 2 and isinstance(val, bytes):
                # field 2 is google.protobuf.Timestamp {int64 seconds=1; int32 nanos=2}
                ts_pos = 0
                while ts_pos < len(val):
                    tf, tv, ts_pos = read_field(val, ts_pos)
                    if tf is None:
                        break
                    if tf == 1 and isinstance(tv, int):
                        ltt_ms = tv * 1000   # seconds → milliseconds
            elif fn == 3 and isinstance(val, bytes):
                # Some SDK versions send ltt as string milliseconds in field 3
                try:
                    ltt_ms = int(val.decode())
                except (ValueError, UnicodeDecodeError):
                    pass
        return ltp, ltt_ms

    def parse_feed(buf):
        """
        Parse Feed message. Tries ltpc (field 1) first, then full_feed (field 2).
        Returns (ltp, ltt_ms) or (None, None).
        """
        pos = 0
        while pos < len(buf):
            fn, val, pos = read_field(buf, pos)
            if fn is None:
                break
            if fn == 1 and isinstance(val, bytes):
                # Shape A: Feed.ltpc = LTPC
                ltp, ltt = parse_ltpc(val)
                if ltp:
                    return ltp, ltt
            elif fn == 2 and isinstance(val, bytes):
                # Shape B: Feed.full_feed = FullFeed { MarketFullFeed marketFF = 1 }
                inner_pos = 0
                while inner_pos < len(val):
                    fn2, val2, inner_pos = read_field(val, inner_pos)
                    if fn2 is None:
                        break
                    if fn2 == 1 and isinstance(val2, bytes):
                        # MarketFullFeed { LTPC ltpc = 1 }
                        inner2_pos = 0
                        while inner2_pos < len(val2):
                            fn3, val3, inner2_pos = read_field(val2, inner2_pos)
                            if fn3 is None:
                                break
                            if fn3 == 1 and isinstance(val3, bytes):
                                ltp, ltt = parse_ltpc(val3)
                                if ltp:
                                    return ltp, ltt
        return None, None

    # ── Top-level FeedResponse parse ──────────────────────────
    try:
        pos = 0
        out = {}
        while pos < len(raw):
            fn, val, pos = read_field(raw, pos)
            if fn is None:
                break
            if fn == 2 and isinstance(val, bytes):
                # FeedResponse.feeds is map<string, Feed>
                # Each map entry is a sub-message: { string key=1; Feed value=2 }
                entry_pos = 0
                map_key, feed_bytes = None, None
                while entry_pos < len(val):
                    ef, ev, entry_pos = read_field(val, entry_pos)
                    if ef is None:
                        break
                    if ef == 1 and isinstance(ev, bytes):
                        map_key = ev.decode("utf-8", errors="ignore")
                    elif ef == 2 and isinstance(ev, bytes):
                        feed_bytes = ev

                if map_key and feed_bytes:
                    ltp, ltt_ms = parse_feed(feed_bytes)
                    if ltp:
                        out[map_key] = {"ltpc": {"ltp": ltp, "ltt": ltt_ms}}

        if out:
            logger.debug("Pure-Python decoder extracted LTP for: %s", list(out.keys()))
            return {"feeds": out}

    except Exception as e:
        logger.debug("Pure-Python proto decoder error: %s", e)

    return None


# ─────────────────────────────────────────────────────────────
# Tier 1 — MarketDataStreamer (official SDK, recommended)
# ─────────────────────────────────────────────────────────────
class MarketDataStreamerFeed:
    """
    Uses upstox_client.MarketDataStreamer — the SDK's high-level v3 streamer.

    The SDK internally:
      - Fetches the v3 authorize URL
      - Connects the WebSocket
      - Decodes protobuf frames
      - Calls on("message", ...) with a decoded Python dict

    Message dict delivered to _handle_message():
      {
        "type": "live_feed",
        "feeds": {
          "MCX_FO|466029": {
            "ff": {
              "marketFF": {
                "ltpc": {"ltp": 92500.0, "ltt": 1678900000000, ...}
              }
            }
          }
        }
      }
    """

    @staticmethod
    def is_available() -> bool:
        try:
            import upstox_client
            return hasattr(upstox_client, "MarketDataStreamer")
        except ImportError:
            return False

    def __init__(self, access_token: str, on_candle_close: Callable,
                 on_tick: Optional[Callable] = None):
        self.access_token = access_token
        self.assembler    = CandleAssembler(config.CANDLE_INTERVAL, on_candle_close, on_tick)
        self._stop        = threading.Event()
        self._streamer    = None

    def _handle_open(self):
        logger.info("MarketDataStreamer connected — subscribed to %s",
                    config.INSTRUMENT_KEY)

    def _handle_message(self, message: dict):
        """SDK delivers a fully-decoded dict. Extract LTP and feed assembler."""
        result = _extract_ltp_ltt(message)
        if result:
            ltp, ltt = result
            self.assembler.process_tick(ltp, 0, _parse_ltt(ltt))
        else:
            logger.debug(
                "MarketDataStreamer: message arrived but LTP not extracted "
                "(type=%s)", message.get("type", "?") if isinstance(message, dict) else "?"
            )

    def _handle_error(self, error):
        logger.error("MarketDataStreamer error: %s", str(error)[:200])

    def _handle_close(self, code, msg):
        if not self._stop.is_set():
            logger.warning("MarketDataStreamer closed (code=%s) — SDK will reconnect", code)

    def start(self):
        import upstox_client

        cfg              = upstox_client.Configuration()
        cfg.access_token = self.access_token
        api_client       = upstox_client.ApiClient(cfg)

        self._streamer = upstox_client.MarketDataStreamer(
            api_client,
            [config.INSTRUMENT_KEY],
            "ltpc",
        )
        self._streamer.on("open",    self._handle_open)
        self._streamer.on("message", self._handle_message)
        self._streamer.on("error",   self._handle_error)
        self._streamer.on("close",   self._handle_close)

        def _run():
            try:
                self._streamer.connect()
            except Exception as e:
                if not self._stop.is_set():
                    logger.error("MarketDataStreamer connect error: %s", e)

        threading.Thread(target=_run, name="mds-feed", daemon=True).start()
        logger.info("Tier 1 — MarketDataStreamer (SDK v3) started")

    def stop(self):
        self._stop.set()
        if self._streamer:
            try:
                self._streamer.disconnect()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────
# Tier 2 — Raw WebSocket + proto decoder (fallback)
# ─────────────────────────────────────────────────────────────
class RawWebSocketFeed:
    """
    Manual two-step v3 WebSocket:
      Step 1: GET /v3/feed/market-data-feed/authorize -> signed URL
      Step 2: Connect WebSocket to that URL
      Step 3: Decode protobuf via upstox_client.feeder.market_data_feeder_v3

    Use only if MarketDataStreamerFeed is unavailable.
    """

    RECONNECT_DELAY = 5

    @staticmethod
    def is_available() -> bool:
        try:
            import websocket  # noqa: F401
            import upstox_client.feeder.market_data_feeder_v3  # noqa: F401
            return True
        except ImportError:
            return False

    def __init__(self, access_token: str, on_candle_close: Callable,
                 on_tick: Optional[Callable] = None):
        self.access_token = access_token
        self.assembler    = CandleAssembler(config.CANDLE_INTERVAL, on_candle_close, on_tick)
        self._ws          = None
        self._stop        = threading.Event()

    def _subscribe_msg(self) -> str:
        return json.dumps({
            "guid":   "kalman_feed",
            "method": "sub",
            "data": {
                "mode":           "ltpc",
                "instrumentKeys": [config.INSTRUMENT_KEY],
            },
        })

    def _on_open(self, ws):
        import websocket as _ws_lib
        logger.info("RawWebSocket connected — subscribing to %s", config.INSTRUMENT_KEY)
        # Upstox v3 REQUIRES the subscription message as a BINARY WebSocket frame.
        ws.send(self._subscribe_msg().encode("utf-8"), opcode=_ws_lib.ABNF.OPCODE_BINARY)

    def _on_message(self, ws, message):
        if not isinstance(message, bytes):
            return
        decoded = _decode_proto_message(message)
        result  = _extract_ltp_ltt(decoded)
        if result:
            ltp, ltt = result
            self.assembler.process_tick(ltp, 0, _parse_ltt(ltt))
        else:
            logger.debug("RawWS tick received but LTP not extracted (len=%d)", len(message))

    def _on_error(self, ws, error):
        logger.error("RawWebSocket error: %s", str(error)[:200])

    def _on_close(self, ws, code, msg):
        logger.warning("RawWebSocket closed (code=%s)", code)
        if not self._stop.is_set():
            logger.info("Reconnecting in %ds...", self.RECONNECT_DELAY)
            time.sleep(self.RECONNECT_DELAY)
            self._connect()

    def _connect(self):
        import websocket as ws_lib
        try:
            logger.info("Fetching v3 WebSocket authorized URL...")
            ws_url   = _get_authorized_ws_url(self.access_token)
            self._ws = ws_lib.WebSocketApp(
                ws_url,
                on_open    = self._on_open,
                on_message = self._on_message,
                on_error   = self._on_error,
                on_close   = self._on_close,
            )
            self._ws.run_forever(ping_interval=20, ping_timeout=8)
        except Exception as e:
            logger.error("RawWS connect error: %s", e)
            if not self._stop.is_set():
                time.sleep(self.RECONNECT_DELAY)
                self._connect()

    def start(self):
        self._stop.clear()
        threading.Thread(target=self._connect, name="raw-ws-feed", daemon=True).start()
        logger.info("Tier 2 — RawWebSocket (manual v3) started")

    def stop(self):
        self._stop.set()
        if self._ws:
            self._ws.close()


# ─────────────────────────────────────────────────────────────
# Tier 3 — REST 1-minute poller + N-minute aggregator
# ─────────────────────────────────────────────────────────────
class RESTCandleFeed:
    """
    Fetches 1-minute intraday bars from Upstox v2 REST API every 30 s.
    Aggregates into N-minute candles matching config.CANDLE_INTERVAL.

    Why 1-minute: v2 API only accepts '1minute' or '30minute'.
    We fetch 1-min and group into N-min buckets ourselves.

    Delay: ~30-60 s behind real-time.
    """

    POLL_SECONDS = 30
    URL_1MIN = (
        "https://api.upstox.com/v2/historical-candle/intraday/{instrument}/1minute"
    )

    def __init__(self, access_token: str, on_candle_close: Callable):
        self.headers         = {
            "Authorization": "Bearer " + access_token,
            "Accept":        "application/json",
        }
        self.callback        = on_candle_close
        self._stop           = threading.Event()
        self._delivered      = set()
        self._no_data_warned = False

    def _fetch_1min_bars(self) -> list:
        url = self.URL_1MIN.format(instrument=config.INSTRUMENT_KEY)
        try:
            r = requests.get(url, headers=self.headers, timeout=10)
            if r.status_code == 200:
                bars = r.json().get("data", {}).get("candles", [])
                logger.debug("Fetched %d 1-min bars", len(bars))
                return bars
            logger.warning("REST 1min fetch HTTP %s: %s", r.status_code, r.text[:150])
        except Exception as e:
            logger.warning("REST fetch error: %s", e)
        return []

    def _aggregate_to_nmin(self, bars_1min: list) -> list:
        from collections import defaultdict
        N       = config.CANDLE_INTERVAL
        buckets = defaultdict(list)

        for raw in bars_1min:
            try:
                ts_str = str(raw[0]).replace("Z", "+00:00")
                ts     = datetime.fromisoformat(ts_str).replace(tzinfo=None)
                total  = ts.hour * 60 + ts.minute
                snap   = (total // N) * N
                bar_ts = ts.replace(hour=snap // 60, minute=snap % 60,
                                    second=0, microsecond=0)
                buckets[bar_ts].append({
                    "ts":     ts,
                    "open":   float(raw[1]),
                    "high":   float(raw[2]),
                    "low":    float(raw[3]),
                    "close":  float(raw[4]),
                    "volume": int(float(raw[5])),
                })
            except Exception as e:
                logger.debug("1min bar parse error: %s  raw=%s", e, raw)

        if not buckets:
            return []

        latest = max(buckets.keys())
        result = []
        for bar_ts in sorted(buckets.keys()):
            if bar_ts == latest:
                continue   # skip still-forming bar
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

    def _poll(self):
        logger.warning("=" * 55)
        logger.warning("  FEED: REST Poller active (Tier 3 fallback)")
        logger.warning("  Delay : ~30-60 s behind real-time")
        logger.warning("  Reason: upstox-python-sdk not installed")
        logger.warning("  Fix   : pip install upstox-python-sdk")
        logger.warning("=" * 55)

        while not self._stop.is_set():
            bars_1min = self._fetch_1min_bars()

            if bars_1min:
                self._no_data_warned = False
                candles = self._aggregate_to_nmin(bars_1min)
                for candle in candles:
                    key = str(candle["ts"])
                    if key not in self._delivered:
                        self._delivered.add(key)
                        logger.info(
                            "CANDLE DELIVERED  %s  O=%.0f  H=%.0f  L=%.0f  C=%.0f  V=%d",
                            candle["ts"], candle["open"], candle["high"],
                            candle["low"], candle["close"], candle["volume"],
                        )
                        try:
                            self.callback(candle)
                        except Exception as e:
                            logger.error("Callback error: %s", e, exc_info=True)
            else:
                # Elevated from DEBUG → WARNING so the user can see it
                if not self._no_data_warned:
                    logger.warning(
                        "REST poll: no 1-min bars returned for %s. "
                        "MCX hours: 09:00-23:30 IST. "
                        "If market is open, check INSTRUMENT_KEY in config.py.",
                        config.INSTRUMENT_KEY,
                    )
                    self._no_data_warned = True
                else:
                    logger.info("REST poll: still no data (market closed?)")

            self._stop.wait(self.POLL_SECONDS)

    def start(self):
        self._stop.clear()
        threading.Thread(target=self._poll, name="rest-feed", daemon=True).start()

    def stop(self):
        self._stop.set()


# ─────────────────────────────────────────────────────────────
# UpstoxFeed — auto-selects tier
# ─────────────────────────────────────────────────────────────
class UpstoxFeed:
    """
    Auto-selects the best available feed tier:

      Tier 1 — MarketDataStreamer (SDK v3, real-time)   <- recommended
      Tier 2 — RawWebSocket + proto decoder (real-time) <- fallback
      Tier 3 — REST 1-min poller (~30-60 s delay)       <- last resort

    Install upstox-python-sdk to enable Tiers 1 and 2:
      pip install upstox-python-sdk
    """

    def __init__(self, access_token: str, on_candle_close: Callable,
                 on_tick: Optional[Callable] = None):
        self.access_token = access_token
        self._callback    = on_candle_close
        self._on_tick     = on_tick
        self._feed        = None

    def start(self):
        # ── Tier 1: MarketDataStreamer ─────────────────────────
        if MarketDataStreamerFeed.is_available():
            logger.info("Tier 1 selected: MarketDataStreamer (SDK v3)")
            try:
                self._feed = MarketDataStreamerFeed(self.access_token, self._callback,
                                                    self._on_tick)
                self._feed.start()
                return
            except Exception as e:
                logger.error("Tier 1 failed: %s — trying Tier 2", e)

        # ── Tier 2: RawWebSocket ───────────────────────────────
        if RawWebSocketFeed.is_available():
            logger.info("Tier 2 selected: RawWebSocket + proto decoder")
            try:
                _get_authorized_ws_url(self.access_token)   # pre-flight check
                self._feed = RawWebSocketFeed(self.access_token, self._callback,
                                              self._on_tick)
                self._feed.start()
                return
            except Exception as e:
                logger.error("Tier 2 failed: %s — falling to Tier 3", e)

        # ── Tier 3: REST poller ────────────────────────────────
        # NOTE: REST feed has no per-tick data — on_tick is not fired in Tier 3.
        # Intrabar trail exits will NOT be active when running on Tier 3.
        logger.warning(
            "Tier 3 selected: REST 1-min poller. "
            "Install upstox-python-sdk for real-time data."
        )
        if self._on_tick:
            logger.warning(
                "INTRABAR_TRAIL_EXIT is enabled but Tier 3 (REST) has no tick data. "
                "Trail exits will only fire at bar close while on Tier 3."
            )
        self._feed = RESTCandleFeed(self.access_token, self._callback)
        self._feed.start()

    def stop(self):
        if self._feed:
            self._feed.stop()


# backward-compat alias used by engine.py
RestLTPPoller = RESTCandleFeed
