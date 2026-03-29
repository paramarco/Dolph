#!/usr/bin/env python3
"""
Compare calibration params with realistic TP/SL/profit examples per security.
Computes margin faithfully: TP_MULT * max(ATR/close, BB_width)

Usage:
    python3 Dolph/Configuration/compare_calibration.py
    python3 Dolph/Configuration/compare_calibration.py --americas   # filter by region
"""
import sys
import psycopg2
import numpy as np
import pandas as pd

DB = dict(host='127.0.0.1', port=4713, user='dolph_user',
          password='dolph_password', dbname='dolph_db', sslmode='disable')

NET_BALANCE = 29000
FACTOR_POSITION = 0.18
CASH_4_POSITION = NET_BALANCE * FACTOR_POSITION  # 5220


def load_securities():
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT s.id, s.code, s.timezone, s.currency,
               (s.alg_parameters->>'TP_MULT')::numeric,
               (s.alg_parameters->>'SL_RR')::numeric,
               (s.alg_parameters->>'VOL_WINDOW')::numeric,
               (s.alg_parameters->>'BB_STD')::numeric,
               (s.alg_parameters->>'BB_WINDOW')::numeric,
               (s.alg_parameters->>'CALIBRATION_GAUSS_MU'),
               (s.alg_parameters->>'CALIBRATION_GAUSS_SIGMA'),
               (s.alg_parameters->>'calibration_score')::numeric
        FROM security s
        WHERE s.alg_parameters IS NOT NULL
        ORDER BY s.timezone, s.code
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def compute_margin(sec_id, tp_mult, sl_rr, vol_window, bb_std, bb_window):
    """Compute margin = TP_MULT * max(ATR/close, BB_width) from recent quotes."""
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    # Get enough bars for indicator warm-up
    cur.execute("""
        SELECT date_time, open, high, low, close, vol
        FROM quote
        WHERE security_id = %s AND vol > 0
        ORDER BY date_time DESC
        LIMIT 500
    """, (sec_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if len(rows) < 50:
        return None, None, None, None

    rows.reverse()
    df = pd.DataFrame(rows, columns=['dt', 'open', 'high', 'low', 'close', 'vol'])

    vol_window = int(vol_window) if vol_window else 20
    bb_std_val = float(bb_std) if bb_std else 2.0
    bb_win = int(bb_window) if bb_window else vol_window

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(vol_window).mean()

    # BB_width
    ma = df['close'].rolling(bb_win).mean()
    std = df['close'].rolling(bb_win).std()
    bb_width = (bb_std_val * std) / ma.replace(0, np.nan)

    last_close = float(df['close'].iloc[-1])
    last_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0
    last_bb_w = float(bb_width.iloc[-1]) if not np.isnan(bb_width.iloc[-1]) else 0

    atr_norm = last_atr / last_close if last_close > 0 else 0
    dominant = max(atr_norm, last_bb_w)
    margin_pct = float(tp_mult) * dominant

    return last_close, margin_pct, atr_norm, last_bb_w


def main():
    region_filter = None
    if '--americas' in sys.argv:
        region_filter = 'America/'
    elif '--europe' in sys.argv:
        region_filter = 'Europe/'
    elif '--asia' in sys.argv:
        region_filter = 'Asia/'

    securities = load_securities()

    print(f"{'Code':<7} {'Ccy':>3} {'Price':>9} {'TP_M':>5} {'Mu':>3} {'Sig':>3} "
          f"{'ATR%':>6} {'BB_w%':>6} {'Dom':>4} {'Mrg%':>6} "
          f"{'TP_Long':>9} {'SL_Long':>9} {'Qty':>5} {'Profit':>7} {'Score':>7}")
    print("-" * 115)

    for row in securities:
        sec_id, code, tz, ccy = row[0], row[1], row[2], row[3]
        tp_mult = float(row[4]) if row[4] else 1.5
        sl_rr = float(row[5]) if row[5] else 2.0
        vol_window = row[6]
        bb_std = row[7]
        bb_window = row[8]
        gauss_mu = row[9] if row[9] else '?'
        gauss_sigma = row[10] if row[10] else '?'
        score = float(row[11]) if row[11] else 0

        if region_filter and not tz.startswith(region_filter):
            continue

        last_close, margin_pct, atr_norm, bb_w = compute_margin(
            sec_id, tp_mult, sl_rr, vol_window, bb_std, bb_window)

        if last_close is None:
            print(f"{code:<7} {'':>3} {'(no data)':>9}")
            continue

        margin_abs = last_close * margin_pct
        tp_long = last_close + margin_abs
        sl_long = last_close - sl_rr * margin_abs

        # Position sizing
        fx_rates = {'USD': 1.0, 'EUR': 0.92, 'GBP': 0.79, 'HKD': 7.82, 'JPY': 149.5, 'KRW': 1350}
        fx = fx_rates.get(ccy, 1.0)
        qty = int(round(CASH_4_POSITION / (last_close / fx)))
        if qty <= 0:
            qty = 0

        # Profit = qty * margin_abs - round_trip_cost
        rtc = max(1.0, qty * 0.005) * 2 * fx
        profit = qty * margin_abs - rtc if qty > 0 else -rtc

        dom = 'ATR' if atr_norm >= bb_w else 'BB'

        print(f"{code:<7} {ccy:>3} {last_close:>9.2f} {tp_mult:>5.2f} {gauss_mu:>3} {gauss_sigma:>3} "
              f"{atr_norm*100:>5.3f}% {bb_w*100:>5.3f}% {dom:>4} {margin_pct*100:>5.3f}% "
              f"{tp_long:>9.2f} {sl_long:>9.2f} {qty:>5d} {profit:>7.2f} {score:>7.1f}")


if __name__ == '__main__':
    main()
