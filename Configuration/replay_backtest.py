#!/usr/bin/env python3
"""Replay validation / shadow backtest: simulate trades on historical quotes
using CURRENT calibrated params, then compare with actual OPERATIONAL trades.

Usage:
    python3 replay_backtest.py --since 2026-04-14 --until 2026-04-15
    python3 replay_backtest.py --since 2026-04-14 --until 2026-04-15 --minWinRate 0.69
    python3 replay_backtest.py --since 2026-04-14 --until 2026-04-15 --MIN_CALIBRATION_SCORE
    python3 replay_backtest.py --since 2026-04-14 --until 2026-04-15 --onlySecurity AMD
    python3 replay_backtest.py --since 2026-04-14T15:30 --until 2026-04-14T21:00 --minWinRate 0.69
"""
import sys
import os
import datetime as dt
import logging

# Setup path and config BEFORE other imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

# Force TEST_OFFLINE mode so _simulate_profit uses simulation constraints
os.environ['DOLPH_MODE_OVERRIDE'] = 'TEST_OFFLINE'

import pandas as pd
import numpy as np
import psycopg2
from TradingPlatforms.InteractiveBrokers.ib_fees import ib_commission_per_side

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("replay_backtest")

DB = dict(host='127.0.0.1', port=4713, user='dolph_user',
          password='dolph_password', dbname='dolph_db', sslmode='disable')

# ---------- CLI args ----------
since = None
until = None
min_calibration_score = None
min_calibration_wr = None
only_security = None

for i, arg in enumerate(sys.argv[1:], 1):
    if arg == '--since' and i < len(sys.argv) - 1:
        since = sys.argv[i + 1]
    if arg == '--until' and i < len(sys.argv) - 1:
        until = sys.argv[i + 1]
    if arg == '--MIN_CALIBRATION_SCORE':
        min_calibration_score = 0.0
    if arg == '--MIN_CALIBRATION_WIN_RATE':
        min_calibration_wr = 0.69
    if arg == '--minWinRate' and i < len(sys.argv) - 1:
        min_calibration_wr = float(sys.argv[i + 1])
    if arg == '--onlySecurity' and i < len(sys.argv) - 1:
        only_security = sys.argv[i + 1]

if not since or not until:
    print("ERROR: --since and --until are required")
    print("Usage: python3 replay_backtest.py --since YYYY-MM-DD[THH:MM] --until YYYY-MM-DD[THH:MM] [--minWinRate 0.69] [--onlySecurity AMD]")
    print("  Date formats: 2026-04-14  or  2026-04-14T16:30")
    sys.exit(1)

since_dt = pd.Timestamp(since, tz='UTC')
until_dt = pd.Timestamp(until, tz='UTC')
# If only date given (no time component), include the full day
if 'T' not in until and ':' not in until:
    until_dt += pd.Timedelta(days=1)

if since_dt >= until_dt:
    print(f"ERROR: --since ({since}) must be before --until ({until})")
    sys.exit(1)

# ---------- Load securities from DB ----------
conn = psycopg2.connect(**DB)
cur = conn.cursor()
cur.execute("""
    SELECT code, currency, primary_exchange, timezone, board_lot,
           alg_parameters, id
    FROM security
    WHERE alg_parameters IS NOT NULL
""")
rows = cur.fetchall()
cur.close()
conn.close()

import json
securities = []
for r in rows:
    params = json.loads(r[5]) if isinstance(r[5], str) else r[5]
    if params is None:
        continue
    sec = {
        'seccode': r[0],
        'currency': r[1],
        'primaryExchange': r[2],
        'primary_exchange': r[2],
        'timezone': r[3],
        'board_lot': r[4] or 1,
        'id': str(r[6]),
        'params': params,
        'models': {},
        'predictions': {'1Min': []},
        'lastPositionTaken': None,
    }
    # Parse trading times from DB
    import pytz
    sec_tz = pytz.timezone(sec['timezone'] or 'America/New_York')
    # Use defaults; actual times loaded separately if needed
    sec['tradingTimes'] = (dt.time(9, 46), dt.time(15, 8))
    sec['time2close'] = dt.time(15, 53)
    securities.append(sec)

# Load trading times from DB
conn = psycopg2.connect(**DB)
cur = conn.cursor()
cur.execute("SELECT code, trading_times_start, trading_times_end, time2close FROM security")
for code, tts, tte, t2c in cur.fetchall():
    for s in securities:
        if s['seccode'] == code:
            if tts: s['tradingTimes'] = (tts, tte)
            if t2c: s['time2close'] = t2c
cur.close()
conn.close()

# ---------- Filter securities ----------
filtered = []
for s in securities:
    p = s['params']
    wr = p.get('calibration_win_rate', 0.0)
    score = p.get('calibration_score', 0.0)

    if only_security and s['seccode'] != only_security:
        continue
    if min_calibration_wr is not None and wr < min_calibration_wr:
        continue
    if min_calibration_score is not None and score < min_calibration_score:
        continue
    if score <= 0:
        continue
    filtered.append(s)

# Print filters
filters = [f"since={since}", f"until={until}"]
if min_calibration_wr is not None:
    filters.append(f"minWinRate>={min_calibration_wr:.0%}")
if min_calibration_score is not None:
    filters.append(f"MIN_CALIBRATION_SCORE>={min_calibration_score:.0f}")
if only_security:
    filters.append(f"security={only_security}")
print(f"Replay Backtest: {', '.join(filters)}")
print(f"Securities qualifying: {len(filtered)} — {[s['seccode'] for s in filtered]}")
print()

if not filtered:
    print("No securities qualify.")
    sys.exit(0)

# ---------- Load quotes for the date range ----------
# Need extra lookback for indicators (BB_WINDOW, EMA_SLOW, etc.)
# and extra lookahead for TP/SL evaluation after signals near --until
INDICATOR_LOOKBACK_DAYS = 15  # extra days for indicator warmup (EMA, BB, ATR)
LOOKAHEAD_DAYS = 3            # extra days after --until for TP/SL resolution
load_since = since_dt - pd.Timedelta(days=INDICATOR_LOOKBACK_DAYS)
load_until = until_dt + pd.Timedelta(days=LOOKAHEAD_DAYS)

conn = psycopg2.connect(**DB)
cur = conn.cursor()
cur.execute("""
    SELECT q.date_time, q.open, q.high, q.low, q.close, q.vol, s.code
    FROM quote q
    JOIN security s ON q.security_id = s.id
    WHERE q.date_time >= %s AND q.date_time < %s
    ORDER BY s.code, q.date_time
""", (load_since, load_until))
quote_rows = cur.fetchall()
cur.close()
conn.close()

df_all = pd.DataFrame(quote_rows, columns=['date_time', 'open', 'high', 'low', 'close', 'volume', 'seccode'])
df_all.set_index('date_time', inplace=True)
if df_all.index.tz is None:
    df_all.index = df_all.index.tz_localize('UTC')

# Verify data availability up to --until
if df_all.empty:
    print(f"ERROR: No quotes found in range {load_since} to {load_until}")
    sys.exit(1)

last_quote_ts = df_all.index.max()
if last_quote_ts < until_dt:
    print(f"WARNING: Quotes only available until {last_quote_ts}.")
    print(f"         Requested --until {until_dt}.")
    print(f"         Trades near the end may not have future data for TP/SL resolution.")
    print(f"         Results will be incomplete.")
    print()

# ---------- Import MinerviniClaude components ----------
# Mock TradingPlatfomSettings before importing Conf (avoids IB dependency)
import types
_mock_tps = types.ModuleType('Configuration.TradingPlatfomSettings')
_mock_tps.platform = {'name': 'replay', 'secrets': {'openaikey': ''}}
sys.modules['Configuration.TradingPlatfomSettings'] = _mock_tps

from Configuration import Conf as cm
# Override mode
cm.MODE = 'TEST_OFFLINE'
cm.simulation_net_balance = getattr(cm, 'simulation_net_balance', 5000)

from PredictionModels.MinerviniClaude import MinerviniClaude

# Create a minimal Dolph instance for compute_position_size
class MinimalDolph:
    """Minimal Dolph stub for replay — only needs compute_position_size."""
    @staticmethod
    def compute_position_size(net_balance, price, security):
        leverage = getattr(cm, 'LEVERAGE_FACTOR', 1.0)
        cash_4_position = net_balance * leverage * cm.factorPosition_Balance
        sec_currency = security.get('fx_currency', security.get('currency', 'USD'))
        fx_rates = getattr(cm, 'FX_RATES_FROM_USD', {'USD': 1.0})
        fx_rate = fx_rates.get(sec_currency, 1.0)
        cash_4_position *= fx_rate
        board_lot = security.get('board_lot', 1)
        quantity = int(cash_4_position / price) if price > 0 else 0
        quantity = (quantity // board_lot) * board_lot
        return cash_4_position, fx_rate, board_lot, quantity

    class ds:
        @staticmethod
        def searchData(since, limitResult=None):
            return {'1Min': pd.DataFrame()}

    class tp:
        @staticmethod
        def get_net_balance():
            return getattr(cm, 'simulation_net_balance', 5000)

dolph_stub = MinimalDolph()

# ---------- Run simulation per security ----------
all_trades = []

for sec in filtered:
    seccode = sec['seccode']
    params = dict(sec['params'])
    MinerviniClaude._derive_params(params)

    # Get quotes for this security, matching OPERATIONAL data pipeline:
    # 1. Apply between_time filter (remove overnight bars, same as DataServer.searchData)
    # 2. Limit to OPERATIONAL_LIMIT_BARS via .tail() up to until_dt
    # 3. Append future bars for TP/SL resolution
    LIMIT_BARS = getattr(cm, 'OPERATIONAL_LIMIT_BARS', 4000)
    df_sec_all = df_all[df_all['seccode'] == seccode][['open', 'high', 'low', 'close', 'volume']].copy()

    # between_time filter: keep only bars within trading session hours
    # OPERATIONAL uses cm.between_time (07:00-23:40 NY) per security's timezone
    sec_tz = sec.get('timezone', 'America/New_York')
    bt_start = getattr(cm, 'between_time', (dt.time(7, 0), dt.time(23, 40)))[0]
    bt_end = getattr(cm, 'between_time', (dt.time(7, 0), dt.time(23, 40)))[1]
    df_sec_local = df_sec_all.copy()
    df_sec_local.index = df_sec_local.index.tz_convert(sec_tz)
    df_sec_local = df_sec_local.between_time(bt_start, bt_end)
    df_sec_local.index = df_sec_local.index.tz_convert('UTC')

    df_sec_before = df_sec_local[df_sec_local.index <= until_dt].tail(LIMIT_BARS)
    df_sec_after = df_sec_local[df_sec_local.index > until_dt]
    df_sec = pd.concat([df_sec_before, df_sec_after])

    if len(df_sec) < params.get('CALIBRATION_MIN_ROWS', 100):
        print(f"  {seccode}: skipped ({len(df_sec)} bars, need {params.get('CALIBRATION_MIN_ROWS', 100)})")
        continue

    # Create a MinerviniClaude instance without triggering calibration
    # We use a trick: put params in cache so __init__ doesn't calibrate
    MinerviniClaude._calibration_cache[seccode] = dict(params)

    # Build minimal data dict for constructor (needs datetime index)
    dummy_df = pd.DataFrame(index=pd.DatetimeIndex([], name='date_time'))
    dummy_data = {'1Min': dummy_df}
    try:
        model = MinerviniClaude(dummy_data, sec, dolph_stub)
    except Exception as e:
        print(f"  {seccode}: init failed: {e}")
        continue

    # Run simulation with collect_trades=True
    try:
        result = model._simulate_profit(df_sec, params, collect_trades=True)
        if isinstance(result, tuple):
            profit, trades = result
        else:
            profit = result
            trades = []
    except Exception as e:
        print(f"  {seccode}: simulation failed: {e}")
        import traceback
        traceback.print_exc()
        continue

    # Filter trades to only those within the requested date range
    for t in trades:
        ts = t['open_ts']
        if hasattr(ts, 'tz_localize') and ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        elif hasattr(ts, 'tz_convert'):
            ts = ts.tz_convert('UTC')
        if ts >= since_dt and ts < until_dt:
            t['seccode'] = seccode
            t['currency'] = sec['currency']
            t['primary_exchange'] = sec.get('primaryExchange', '')
            all_trades.append(t)

    n_in_range = sum(1 for t in trades
                     if (t['open_ts'].tz_localize('UTC') if t['open_ts'].tzinfo is None else t['open_ts']) >= since_dt
                     and (t['open_ts'].tz_localize('UTC') if t['open_ts'].tzinfo is None else t['open_ts']) < until_dt)
    wins_in_range = sum(1 for t in trades
                        if ((t['open_ts'].tz_localize('UTC') if t['open_ts'].tzinfo is None else t['open_ts']) >= since_dt
                            and (t['open_ts'].tz_localize('UTC') if t['open_ts'].tzinfo is None else t['open_ts']) < until_dt
                            and t['outcome_class'] == 'WIN'))
    print(f"  {seccode}: {n_in_range} trades in range, {wins_in_range} wins, profit_score={profit:.2f}")

# Sort by timestamp
all_trades.sort(key=lambda t: t['open_ts'])

# ---------- Output in same format as extract_trades_from_db.py ----------
print()
print("=" * 182)
print(f"{'Date':20s} {'Seccode':8s} {'Dir':6s} {'Conf':6s} {'Entry':10s} {'TP':10s} {'SL':10s} {'Outcome':28s} {'Class':8s} {'Ccy':4s} {'Qty':>5s} {'PnL$':>8s} {'Mins':6s}")
print("=" * 182)

total_pnl = 0.0
total_resolved = 0
total_wins = 0
total_losses = 0
win_pnl = 0.0
loss_pnl = 0.0

for t in all_trades:
    qty = t['quantity']
    pnl_pips = t['pnl_pips']
    commission = ib_commission_per_side(qty, t['primary_exchange']) * 2
    pnl_total = qty * pnl_pips - commission

    conf_str = f"{t['confidence']:.4f}" if t.get('confidence') is not None else "N/A"
    cls = t.get('outcome_class', '')
    pnl_str = f"{pnl_total:+.2f}"

    open_ts = t['open_ts']
    close_ts = t['close_ts']
    if hasattr(open_ts, 'strftime'):
        ts_str = open_ts.strftime('%Y-%m-%d %H:%M:%S')
    else:
        ts_str = str(open_ts)[:19]

    duration_min = 0
    if open_ts is not None and close_ts is not None:
        try:
            duration_min = (close_ts - open_ts).total_seconds() / 60.0
        except Exception:
            pass
    dur_str = f"{duration_min:.0f}"

    ccy = t.get('currency', 'USD')

    print(f"{ts_str:20s} {t['seccode']:8s} {t['direction']:6s} {conf_str:6s} {t['entry_price']:10.3f} {t['tp_target']:10.3f} {t['sl_target']:10.3f} {t['outcome']:28s} {cls:8s} {ccy:4s} {qty:>5d} {pnl_str:>8s} {dur_str:>6s}")

    if cls in ('WIN', 'LOSS'):
        total_resolved += 1
        total_pnl += pnl_total
        if cls == 'WIN':
            total_wins += 1
            win_pnl += pnl_total
        else:
            total_losses += 1
            loss_pnl += pnl_total

# Summary
print()
print("=" * 80)
print("REPLAY BACKTEST SUMMARY")
print("=" * 80)
print(f"Total trades:             {len(all_trades)}")
print(f"TOTAL RESOLVED:           {total_resolved} (WIN={total_wins}, LOSS={total_losses})")
if total_resolved > 0:
    print(f"OVERALL WIN RATE:         {total_wins/total_resolved*100:.1f}%")
    print(f"TOTAL PnL$:               {total_pnl:+.2f}  (wins: {win_pnl:+.2f}, losses: {loss_pnl:+.2f})")
