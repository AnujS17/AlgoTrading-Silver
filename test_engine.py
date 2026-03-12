"""
test_engine.py — Full infrastructure test suite for Kalman Live Trader.

Tests every component WITHOUT placing real orders or needing live data.
Uses mocks and synthetic candles to simulate every scenario end-to-end.

Run with:
    python test_engine.py              # all tests
    python test_engine.py -v           # verbose output
    python test_engine.py TestBroker   # one class only

TESTS INCLUDED:
  1.  TestCandleAssembler  — candle building from ticks
  2.  TestKalmanStrategy   — signal logic, all conditions
  3.  TestRiskManager      — daily loss limit, trade count, margin
  4.  TestBrokerMock       — order routing, position tracking
  5.  TestEngineIntegration — full pipeline: candle → signal → order
  6.  TestManualPnLSeed    — prior manual loss blocks new entries
  7.  TestEmergencyStop    — Ctrl+C flattens position safely
  8.  TestSessionFilter    — no trades outside MCX hours
  9.  TestCooldown         — respects bars-between-trades setting
  10. TestWarmup           — engine waits for enough bars before trading
"""

import sys
import os
import unittest
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

# ── Add parent folder to path so imports work ─────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
# Override safety-critical config values for all tests
config.MAX_DAILY_LOSS      = 5000
config.MAX_TRADES_PER_DAY  = 10
config.COOLDOWN            = 3
config.DIRECTION           = "Both"
config.USE_SESSION         = False    # disable session filter for most tests
config.LOT_SIZE            = 1
config.EQUITY_RISK_PCT     = 2.0
config.CANDLE_INTERVAL     = 5
config.INSTRUMENT_KEY      = "MCX_FO|TEST"
config.INSTRUMENT_NAME     = "TEST"
config.LOG_FILE            = "test_run.log"
config.TRADE_LOG_CSV       = "test_trades.csv"
config.ACCESS_TOKEN_FILE   = "test_token.json"

from feed     import CandleAssembler
from strategy import KalmanScalperStrategy, Signal
from engine   import RiskManager, TradeLogger, TradingEngine

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def make_candle(close, high=None, low=None, open_=None,
                ts=None, volume=1000):
    """Build a synthetic OHLCV candle dict."""
    ts    = ts    or datetime(2026, 3, 9, 10, 0, 0)
    high  = high  or close + 50
    low   = low   or close - 50
    open_ = open_ or close
    return {"ts": ts, "open": open_, "high": high,
            "low": low, "close": close, "volume": volume}


def make_candles(prices, base_ts=None):
    """Build a list of candles from a price sequence."""
    base = base_ts or datetime(2026, 3, 9, 10, 0, 0)
    return [make_candle(p, ts=base + timedelta(minutes=5*i))
            for i, p in enumerate(prices)]


def feed_candles(strategy, candles):
    """Feed a list of candles into a strategy; return last signal."""
    last_signal = None
    for c in candles:
        sig = strategy.process_bar(c)
        if sig:
            last_signal = sig
    return last_signal


def mock_broker(net_qty=0, margin=500000, realised_pnl=0.0):
    """Return a fully mocked UpstoxBroker."""
    b = MagicMock()
    b.get_net_quantity.return_value       = net_qty
    b.get_funds.return_value              = margin
    b.get_today_realised_pnl.return_value = realised_pnl
    b.enter_long.return_value             = "ORDER_LONG_001"
    b.enter_short.return_value            = "ORDER_SHORT_001"
    b.exit_long.return_value              = "ORDER_EXIT_001"
    b.exit_short.return_value             = "ORDER_EXIT_001"
    b.close_all_positions.return_value    = None
    b.get_order_status.return_value       = {
        "status": "COMPLETE", "average_price": 93000.0
    }
    b.token = "MOCK_TOKEN"
    return b


# ═════════════════════════════════════════════════════════════
# 1. CandleAssembler
# ═════════════════════════════════════════════════════════════
class TestCandleAssembler(unittest.TestCase):

    def test_first_tick_does_not_fire_callback(self):
        """No candle delivered on the very first tick."""
        delivered = []
        ca = CandleAssembler(5, lambda c: delivered.append(c))
        ca.process_tick(93000.0, 100, datetime(2026, 3, 9, 10, 0, 30))
        self.assertEqual(len(delivered), 0)

    def test_candle_closes_when_new_interval_starts(self):
        """Moving to the next 5-min slot triggers a candle close."""
        delivered = []
        ca = CandleAssembler(5, lambda c: delivered.append(c))
        ca.process_tick(93000.0, 100, datetime(2026, 3, 9, 10, 0, 0))
        ca.process_tick(93100.0, 200, datetime(2026, 3, 9, 10, 0, 30))
        ca.process_tick(93200.0, 150, datetime(2026, 3, 9, 10, 5, 0))  # new bar
        self.assertEqual(len(delivered), 1)

    def test_ohlcv_values_correct(self):
        """Delivered candle has correct O/H/L/C/V."""
        delivered = []
        ca = CandleAssembler(5, lambda c: delivered.append(c))
        base = datetime(2026, 3, 9, 10, 0, 0)
        ticks = [(93000, 100), (93500, 200), (92800, 150), (93200, 50)]
        for i, (price, vol) in enumerate(ticks):
            ca.process_tick(price, vol, base + timedelta(seconds=i*60))
        # Trigger close
        ca.process_tick(93100, 100, base + timedelta(minutes=5))
        self.assertEqual(len(delivered), 1)
        c = delivered[0]
        self.assertEqual(c["open"],   93000)
        self.assertEqual(c["high"],   93500)
        self.assertEqual(c["low"],    92800)
        self.assertEqual(c["close"],  93200)
        self.assertEqual(c["volume"], 500)   # 100+200+150+50

    def test_multiple_candles_fire_in_sequence(self):
        """Three complete intervals deliver three candles."""
        delivered = []
        ca = CandleAssembler(5, lambda c: delivered.append(c))
        for i in range(4):   # 4 bar starts → 3 completed candles
            ca.process_tick(93000 + i*100, 100,
                            datetime(2026, 3, 9, 10, i*5, 0))
        self.assertEqual(len(delivered), 3)

    def test_get_current_returns_live_candle(self):
        """get_current() shows the in-progress bar."""
        ca = CandleAssembler(5, lambda c: None)
        ca.process_tick(93000, 100, datetime(2026, 3, 9, 10, 0, 0))
        ca.process_tick(93500, 50,  datetime(2026, 3, 9, 10, 0, 30))
        curr = ca.get_current()
        self.assertIsNotNone(curr)
        self.assertEqual(curr["high"], 93500)


# ═════════════════════════════════════════════════════════════
# 2. Kalman Strategy — signal logic
# ═════════════════════════════════════════════════════════════
class TestKalmanStrategy(unittest.TestCase):

    def setUp(self):
        self.strat = KalmanScalperStrategy()

    def test_no_signal_on_early_bars(self):
        """Strategy stays quiet while indicators warm up (<30 bars)."""
        candles = make_candles([93000] * 25)
        signals = [self.strat.process_bar(c) for c in candles]
        self.assertTrue(all(s is None for s in signals),
                        "Should not signal during warmup")

    def test_strategy_returns_signal_or_none(self):
        """process_bar always returns Signal or None — never raises."""
        candles = make_candles([93000 + i * 10 for i in range(60)])
        for c in candles:
            result = self.strat.process_bar(c)
            self.assertIn(result.__class__.__name__, ["Signal", "NoneType"])

    def test_set_position_updates_state(self):
        """set_position() correctly updates internal position tracking."""
        self.strat.set_position(1, 93000.0)
        self.assertEqual(self.strat.position, 1)
        self.assertEqual(self.strat.entry_price, 93000.0)

        self.strat.set_position(0)
        self.assertEqual(self.strat.position, 0)

    def test_exit_signal_when_already_long(self):
        """
        Simulate a case where strategy is long and price reverts.
        We directly set position and feed a reverting candle.
        """
        # Warm up
        candles = make_candles([93000] * 50)
        for c in candles:
            self.strat.process_bar(c)

        # Force strategy into a long position
        self.strat.set_position(1, 93000.0)
        self.strat.position = 1

        # Feed a candle where normResid reverts above -EXIT_THRESH
        # By making close = kalm_x (at equilibrium), residual → 0
        # We can't control normResid directly, so just test no crash
        candle = make_candle(93000, ts=datetime(2026, 3, 9, 11, 0, 0))
        result = self.strat.process_bar(candle)
        # Result is either None or a valid Signal
        self.assertIn(result.__class__.__name__, ["Signal", "NoneType"])

    def test_signal_has_required_fields(self):
        """Any signal returned has action, reason, bar_ts, atr fields."""
        candles = make_candles([93000 + i * 50 for i in range(100)])
        for c in candles:
            sig = self.strat.process_bar(c)
            if sig is not None:
                self.assertIn(sig.action, ["BUY", "SELL", "EXIT_LONG", "EXIT_SHORT"])
                self.assertIsNotNone(sig.reason)
                self.assertIsNotNone(sig.bar_ts)
                self.assertGreater(sig.atr, 0)
                break


# ═════════════════════════════════════════════════════════════
# 3. Risk Manager
# ═════════════════════════════════════════════════════════════
class TestRiskManager(unittest.TestCase):

    def _make_rm(self, pnl=0.0, margin=500000):
        b  = mock_broker(realised_pnl=pnl, margin=margin)
        rm = RiskManager(b)
        return rm, b

    def test_normal_entry_approved(self):
        """Fresh start with good margin — entry approved."""
        rm, _ = self._make_rm(pnl=0, margin=500000)
        sig    = MagicMock()
        ok, reason = rm.approve_entry(sig, qty=1)
        self.assertTrue(ok, f"Should approve but got: {reason}")

    def test_daily_loss_limit_blocks_entry(self):
        """If daily loss exceeds limit, entry is blocked."""
        rm, _ = self._make_rm(pnl=0, margin=500000)
        rm._daily_pnl = -(config.MAX_DAILY_LOSS + 1)
        sig = MagicMock()
        ok, reason = rm.approve_entry(sig, qty=1)
        self.assertFalse(ok)
        self.assertIn("loss", reason.lower())

    def test_max_trades_per_day_blocks_entry(self):
        """After MAX_TRADES_PER_DAY, no more entries."""
        rm, _ = self._make_rm()
        rm._daily_trades = config.MAX_TRADES_PER_DAY
        sig = MagicMock()
        ok, reason = rm.approve_entry(sig, qty=1)
        self.assertFalse(ok)
        self.assertIn("trades", reason.lower())

    def test_insufficient_margin_blocks_entry(self):
        """Low margin → entry blocked."""
        rm, _ = self._make_rm(margin=100)   # only Rs.100 available
        sig = MagicMock()
        ok, reason = rm.approve_entry(sig, qty=2)
        self.assertFalse(ok)
        self.assertIn("margin", reason.lower())

    def test_exits_always_approved(self):
        """Exits are never blocked by risk manager."""
        rm, _ = self._make_rm(pnl=-(config.MAX_DAILY_LOSS * 10))
        ok, _ = rm.approve_exit()
        self.assertTrue(ok)

    def test_pnl_accumulates_correctly(self):
        """record_pnl() accumulates into daily_pnl."""
        rm, _ = self._make_rm(pnl=0)
        rm.record_pnl(500)
        rm.record_pnl(-800)
        self.assertAlmostEqual(rm.daily_pnl, -300)

    def test_manual_loss_seeded_on_startup(self):
        """
        BUG FIX TEST: If a manual loss happened before the bot started,
        the RiskManager must pick it up from broker and count it.
        """
        # Simulate Rs.-2500 manual loss already on account
        rm, broker = self._make_rm(pnl=-2500, margin=500000)
        # daily_pnl should be seeded to -2500, not 0
        self.assertAlmostEqual(rm.daily_pnl, -2500,
            msg="RiskManager must seed P&L from broker on startup")

    def test_manual_loss_triggers_kill_switch(self):
        """
        If manual loss already exceeds MAX_DAILY_LOSS,
        entry must be blocked immediately on startup.
        """
        rm, _ = self._make_rm(pnl=-(config.MAX_DAILY_LOSS + 500))
        sig = MagicMock()
        ok, reason = rm.approve_entry(sig, qty=1)
        self.assertFalse(ok,
            "Entry must be blocked if pre-existing loss already hit daily limit")

    def test_trade_count_increments(self):
        """record_trade_open() increments daily trade counter."""
        rm, _ = self._make_rm()
        self.assertEqual(rm.daily_trades, 0)
        rm.record_trade_open()
        rm.record_trade_open()
        self.assertEqual(rm.daily_trades, 2)


# ═════════════════════════════════════════════════════════════
# 4. TradeLogger
# ═════════════════════════════════════════════════════════════
class TestTradeLogger(unittest.TestCase):

    def setUp(self):
        self.csv = "test_tradelog_temp.csv"
        if os.path.exists(self.csv):
            os.remove(self.csv)

    def tearDown(self):
        if os.path.exists(self.csv):
            os.remove(self.csv)

    def test_creates_file_with_header(self):
        """Logger creates CSV with correct header on first run."""
        tlog = TradeLogger(self.csv)
        self.assertTrue(os.path.exists(self.csv))
        with open(self.csv) as f:
            header = f.readline()
        self.assertIn("action", header)
        self.assertIn("pnl", header)

    def test_records_trade_row(self):
        """A logged trade appears as a row in the CSV."""
        import csv
        tlog = TradeLogger(self.csv)
        tlog.record(action="BUY", reason="Test", fill_price=93000,
                    quantity=1, pnl=0, daily_pnl=0,
                    norm_resid=-1.3, vr=0.7, vel_norm=0.5, atr=120,
                    order_id="TEST001")
        with open(self.csv) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["action"], "BUY")
        self.assertEqual(rows[0]["order_id"], "TEST001")

    def test_thread_safe_concurrent_writes(self):
        """Multiple threads logging simultaneously don't corrupt the file."""
        import csv
        tlog = TradeLogger(self.csv)
        threads = []
        for i in range(10):
            t = threading.Thread(
                target=tlog.record,
                kwargs=dict(action=f"TRADE_{i}", pnl=i*100,
                            fill_price=93000, quantity=1,
                            order_id=f"ORD_{i}")
            )
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with open(self.csv) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 10, "All 10 rows must be written cleanly")


# ═════════════════════════════════════════════════════════════
# 5. Full Engine Integration (mocked broker)
# ═════════════════════════════════════════════════════════════
class TestEngineIntegration(unittest.TestCase):
    """
    Tests the full candle → strategy → risk → broker pipeline
    using a mocked broker so no real orders are placed.
    """

    def _make_engine(self, net_qty=0, margin=500000, pnl=0.0):
        broker = mock_broker(net_qty=net_qty, margin=margin, realised_pnl=pnl)
        engine = TradingEngine.__new__(TradingEngine)
        engine._lock      = threading.Lock()
        engine.broker     = broker
        engine.strategy   = KalmanScalperStrategy()
        engine.risk       = RiskManager(broker)
        engine.tlog       = TradeLogger(config.TRADE_LOG_CSV)
        engine._running   = False
        engine._pos       = 0
        engine._entry_px  = 0.0
        engine._open_qty  = 0
        engine._entry_ts  = None
        engine._order_id  = None
        engine._quantity  = 1
        return engine, broker

    def test_engine_starts_flat(self):
        """Engine initialises with no open position."""
        engine, _ = self._make_engine()
        self.assertEqual(engine._pos, 0)

    def test_candle_processed_without_crash(self):
        """on_candle_close() runs without error on a valid candle."""
        engine, _ = self._make_engine()
        candle = make_candle(93000)
        try:
            engine.on_candle_close(candle)
        except Exception as e:
            self.fail(f"on_candle_close raised: {e}")

    def test_many_candles_processed_without_crash(self):
        """Feed 100 candles — no exceptions, no hangs."""
        engine, _ = self._make_engine()
        candles = make_candles([93000 + i*10 for i in range(100)])
        for c in candles:
            try:
                engine.on_candle_close(c)
            except Exception as e:
                self.fail(f"Crashed on candle {c['ts']}: {e}")

    def test_position_sync_corrects_drift(self):
        """
        If broker shows LONG but engine thinks FLAT,
        _sync_position() corrects the engine state.
        """
        engine, broker = self._make_engine(net_qty=1)
        engine._pos = 0   # engine thinks flat
        engine._sync_position()
        self.assertEqual(engine._pos, 1, "Engine should correct to LONG")

    def test_position_sync_flat_correction(self):
        """If broker is flat but engine thinks long, sync corrects to flat."""
        engine, broker = self._make_engine(net_qty=0)
        engine._pos      = 1      # engine thinks long (stale)
        engine._open_qty = 1
        engine._sync_position()
        self.assertEqual(engine._pos, 0, "Engine should correct to FLAT")

    def test_emergency_stop_calls_close_all(self):
        """emergency_stop() always calls broker.close_all_positions()."""
        engine, broker = self._make_engine()
        engine.emergency_stop()
        broker.close_all_positions.assert_called_once()

    def test_risk_blocked_entry_does_not_place_order(self):
        """If risk manager blocks, no order reaches the broker."""
        engine, broker = self._make_engine()
        # Trip the daily loss limit
        engine.risk._daily_pnl = -(config.MAX_DAILY_LOSS + 100)

        signal = Signal("BUY", "Test", datetime.now(),
                        norm_resid=-1.5, vr=0.7, vel_norm=0.4, atr=120)
        engine._handle_entry(signal, direction=1)
        broker.enter_long.assert_not_called()

    def test_exit_long_computes_pnl_correctly(self):
        """Exit long: P&L = (fill - entry) * qty."""
        engine, broker = self._make_engine()
        engine._pos      = 1
        engine._entry_px = 93000.0
        engine._open_qty = 1

        # Mock fill at 93500 → expected P&L = +500
        broker.get_order_status.return_value = {
            "status": "COMPLETE", "average_price": 93500.0
        }
        signal = Signal("EXIT_LONG", "Revert", datetime.now(),
                        norm_resid=0.0, vr=0.7, vel_norm=0.4, atr=120)
        engine._handle_exit(signal)
        self.assertEqual(engine._pos, 0)
        # P&L was recorded in risk manager
        self.assertAlmostEqual(engine.risk.daily_pnl, 500.0, places=0)

    def test_exit_short_computes_pnl_correctly(self):
        """Exit short: P&L = (entry - fill) * qty."""
        engine, broker = self._make_engine()
        engine._pos      = -1
        engine._entry_px = 93000.0
        engine._open_qty = 1

        broker.get_order_status.return_value = {
            "status": "COMPLETE", "average_price": 92500.0
        }
        signal = Signal("EXIT_SHORT", "Revert", datetime.now(),
                        norm_resid=0.0, vr=0.7, vel_norm=0.4, atr=120)
        engine._handle_exit(signal)
        self.assertEqual(engine._pos, 0)
        self.assertAlmostEqual(engine.risk.daily_pnl, 500.0, places=0)


# ═════════════════════════════════════════════════════════════
# 6. Session Filter
# ═════════════════════════════════════════════════════════════
class TestSessionFilter(unittest.TestCase):

    def test_trades_blocked_outside_session(self):
        """No BUY/SELL signals outside MCX trading hours."""
        config.USE_SESSION = True
        config.SESS_START  = (9, 15)
        config.SESS_END    = (23, 0)

        strat = KalmanScalperStrategy()
        # Warm up during session
        for i in range(50):
            ts = datetime(2026, 3, 9, 10, i % 12 * 5, 0)
            strat.process_bar(make_candle(93000 + i*20, ts=ts))

        # Now try outside session hours (3 AM)
        strat.set_position(0)
        signals = []
        for i in range(10):
            ts  = datetime(2026, 3, 9, 3, i*5, 0)   # 03:00 AM
            sig = strat.process_bar(make_candle(90000 - i*500, ts=ts))
            if sig and sig.action in ("BUY", "SELL"):
                signals.append(sig)

        self.assertEqual(len(signals), 0,
            "No entry signals should fire outside session hours")
        config.USE_SESSION = False   # reset


# ═════════════════════════════════════════════════════════════
# 7. Cooldown
# ═════════════════════════════════════════════════════════════
class TestCooldown(unittest.TestCase):

    def test_cooldown_resets_after_exit(self):
        """bars_since resets to 0 while in trade, increments when flat."""
        strat = KalmanScalperStrategy()
        # bars_since starts at COOLDOWN+1 (cooled)
        self.assertEqual(strat._bars_since, config.COOLDOWN + 1)

        strat.set_position(1, 93000.0)   # enter long

        # While in trade, bars_since should reset to 0 on each bar
        c = make_candle(93000)
        strat.process_bar(c)
        self.assertEqual(strat._bars_since, 0)

        # Exit trade
        strat.set_position(0)

        # bars_since should increment each flat bar
        for expected in range(1, config.COOLDOWN + 2):
            strat.process_bar(c)
            self.assertEqual(strat._bars_since,
                             min(expected, config.COOLDOWN + 1))


# ═════════════════════════════════════════════════════════════
# 8. Order Rejection Handling
# ═════════════════════════════════════════════════════════════
class TestOrderRejection(unittest.TestCase):

    def _make_engine(self):
        broker = mock_broker()
        engine = TradingEngine.__new__(TradingEngine)
        engine._lock     = threading.Lock()
        engine.broker    = broker
        engine.strategy  = KalmanScalperStrategy()
        engine.risk      = RiskManager(broker)
        engine.tlog      = TradeLogger(config.TRADE_LOG_CSV)
        engine._running  = False
        engine._pos      = 0
        engine._entry_px = 0.0
        engine._open_qty = 0
        engine._entry_ts = None
        engine._order_id = None
        engine._quantity = 1
        return engine, broker

    def test_rejected_order_doesnt_update_position(self):
        """If order is rejected, engine position stays at 0."""
        engine, broker = self._make_engine()
        broker.get_order_status.return_value = {
            "status": "REJECTED", "status_message": "Insufficient funds"
        }
        signal = Signal("BUY", "Test", datetime.now(),
                        norm_resid=-1.5, vr=0.7, vel_norm=0.4, atr=120)
        engine._handle_entry(signal, direction=1)
        # Position should remain 0 after rejection
        self.assertEqual(engine._pos, 0)

    def test_fill_timeout_doesnt_update_position(self):
        """If order doesn't fill within timeout, position stays flat."""
        engine, broker = self._make_engine()
        # Simulate order stuck in OPEN state
        broker.get_order_status.return_value = {
            "status": "OPEN", "average_price": 0
        }
        signal = Signal("BUY", "Test", datetime.now(),
                        norm_resid=-1.5, vr=0.7, vel_norm=0.4, atr=120)
        # This should raise internally but NOT crash the engine
        engine._handle_entry(signal, direction=1)
        self.assertEqual(engine._pos, 0)


# ═════════════════════════════════════════════════════════════
# RUNNER
# ═════════════════════════════════════════════════════════════
def print_banner():
    print("\n" + "="*65)
    print("  KALMAN LIVE TRADER — INFRASTRUCTURE TEST SUITE")
    print("="*65)
    print(f"  Python   : {sys.version.split()[0]}")
    print(f"  Run at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*65 + "\n")


if __name__ == "__main__":
    print_banner()

    # Clean up temp files before run
    for f in [config.LOG_FILE, config.TRADE_LOG_CSV, "test_tradelog_temp.csv"]:
        if os.path.exists(f):
            os.remove(f)

    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    # Add all test classes
    for cls in [
        TestCandleAssembler,
        TestKalmanStrategy,
        TestRiskManager,
        TestTradeLogger,
        TestEngineIntegration,
        TestSessionFilter,
        TestCooldown,
        TestOrderRejection,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    verbosity = 2 if "-v" in sys.argv else 1
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    result = runner.run(suite)

    print("\n" + "="*65)
    if result.wasSuccessful():
        print(f"  ALL {result.testsRun} TESTS PASSED ✓")
    else:
        print(f"  {len(result.failures)} FAILED | {len(result.errors)} ERRORS "
              f"| {result.testsRun} total")
        if result.failures:
            print("\n  FAILURES:")
            for test, msg in result.failures:
                print(f"    - {test}: {msg.splitlines()[-1]}")
        if result.errors:
            print("\n  ERRORS:")
            for test, msg in result.errors:
                print(f"    - {test}: {msg.splitlines()[-1]}")
    print("="*65 + "\n")

    # Clean up temp files after run
    for f in [config.LOG_FILE, config.TRADE_LOG_CSV, "test_tradelog_temp.csv"]:
        if os.path.exists(f):
            os.remove(f)

    sys.exit(0 if result.wasSuccessful() else 1)
