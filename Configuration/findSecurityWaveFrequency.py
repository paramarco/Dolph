#!/usr/bin/env python3
"""
Compute the average time (in minutes) for each security to move its price
90% of the mean peak-to-valley amplitude within hourly oscillations.

Uses 1-min quote data from the DB, filters to trading hours per security,
detects peaks/valleys via Butterworth-filtered price series, and measures
the time from each valley (peak) to the point where 90% of the swing to
the next peak (valley) is reached.

Usage:
    python3 Dolph/Configuration/findSecurityWaveFrequency.py
"""
import psycopg2
import numpy as np
import pandas as pd
import pytz
import datetime as dt
from scipy import signal
from scipy.signal import find_peaks

DB = dict(host='127.0.0.1', port=4713, user='dolph_user',
          password='dolph_password', dbname='dolph_db', sslmode='disable')

LOOKBACK_DAYS = 30       # analyze last N days of quotes
BUTTER_ORDER = 2         # Butterworth filter order
BUTTER_CUTOFF = 0.05     # low cutoff → smoother curve (0.05 = ~20-bar period)
PEAK_DISTANCE = 30       # minimum bars between peaks (30 min)
SWING_THRESHOLD = 0.90   # 90% of swing amplitude


def load_securities():
    """Load all active securities from DB with their trading hours."""
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, code, timezone, trading_times_start, trading_times_end
        FROM security
        WHERE alg_parameters IS NOT NULL
        ORDER BY timezone, code
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def load_quotes(sec_id, since):
    """Load 1-min quotes for a security from the DB."""
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT date_time, open, high, low, close, vol
        FROM quote
        WHERE security_id = %s AND date_time >= %s AND vol > 0
        ORDER BY date_time ASC
    """, (sec_id, since))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=['date_time', 'open', 'high', 'low', 'close', 'vol'])
    df['date_time'] = pd.to_datetime(df['date_time'], utc=True)
    df.set_index('date_time', inplace=True)
    return df


def filter_trading_hours(df, timezone, tt_start, tt_end):
    """Keep only bars within trading hours in the security's local timezone."""
    sec_tz = pytz.timezone(timezone)
    df.index = df.index.tz_convert(sec_tz)
    df = df.between_time(tt_start, tt_end)
    df.index = df.index.tz_convert('UTC')
    return df


def find_peaks_valleys(prices, distance=30):
    """Find peaks and valleys on a Butterworth-filtered price series."""
    if len(prices) < distance * 3:
        return np.array([]), np.array([])

    # Butterworth low-pass filter (double-pass for zero phase shift)
    b, a = signal.butter(BUTTER_ORDER, BUTTER_CUTOFF)
    try:
        y = signal.filtfilt(b, a, prices)
    except ValueError:
        return np.array([]), np.array([])

    peak_idx, _ = find_peaks(y, distance=distance)
    valley_idx, _ = find_peaks(-y, distance=distance)

    return peak_idx, valley_idx


def compute_swing_times(df, peak_idx, valley_idx, threshold=0.90):
    """Compute the time (in bars/minutes) to reach `threshold` of each swing.

    For each consecutive valley→peak and peak→valley pair, measures how many
    bars it takes from the start of the swing to reach 90% of the amplitude.

    Returns:
        list of (swing_type, amplitude, bars_to_90pct)
    """
    closes = df['close'].values
    results = []

    # Merge peaks and valleys into a single sorted sequence of turning points
    # Each entry: (bar_index, 'peak' or 'valley')
    events = []
    for idx in peak_idx:
        events.append((idx, 'peak'))
    for idx in valley_idx:
        events.append((idx, 'valley'))
    events.sort(key=lambda x: x[0])

    # Remove consecutive same-type events (keep first)
    filtered = []
    for e in events:
        if not filtered or filtered[-1][1] != e[1]:
            filtered.append(e)
    events = filtered

    for i in range(len(events) - 1):
        start_idx, start_type = events[i]
        end_idx, end_type = events[i + 1]

        start_price = closes[start_idx]
        end_price = closes[end_idx]
        amplitude = abs(end_price - start_price)

        if amplitude == 0:
            continue

        target_price = start_price + (end_price - start_price) * threshold

        # Walk forward from start to find when price crosses the 90% level
        bars_to_target = None
        for j in range(start_idx + 1, end_idx + 1):
            if start_type == 'valley':  # price rising
                if closes[j] >= target_price:
                    bars_to_target = j - start_idx
                    break
            else:  # peak → price falling
                if closes[j] <= target_price:
                    bars_to_target = j - start_idx
                    break

        if bars_to_target is not None:
            swing_type = 'up' if start_type == 'valley' else 'down'
            results.append((swing_type, amplitude, bars_to_target))

    return results


def analyze_security(sec_id, code, timezone, tt_start, tt_end):
    """Full analysis for one security. Returns dict with results."""
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=LOOKBACK_DAYS)
    df = load_quotes(sec_id, since)

    if df.empty or len(df) < 200:
        return None

    df = filter_trading_hours(df, timezone, tt_start, tt_end)

    if len(df) < 200:
        return None

    # Average price for smoother signal (like the legacy PeaksAndValleysModel)
    avg_price = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0

    peak_idx, valley_idx = find_peaks_valleys(avg_price.values, distance=PEAK_DISTANCE)

    if len(peak_idx) < 2 and len(valley_idx) < 2:
        return None

    swings = compute_swing_times(df, peak_idx, valley_idx, SWING_THRESHOLD)

    if not swings:
        return None

    bars_list = [s[2] for s in swings]
    amplitudes = [s[1] for s in swings]

    return {
        'code': code,
        'timezone': timezone,
        'num_swings': len(swings),
        'avg_amplitude': np.mean(amplitudes),
        'median_bars_90pct': np.median(bars_list),
        'mean_bars_90pct': np.mean(bars_list),
        'p25_bars': np.percentile(bars_list, 25),
        'p75_bars': np.percentile(bars_list, 75),
        'last_close': df['close'].iloc[-1],
        'avg_amplitude_pct': np.mean(amplitudes) / df['close'].mean() * 100,
    }


def main():
    securities = load_securities()
    print(f"Analyzing {len(securities)} securities (last {LOOKBACK_DAYS} days, "
          f"peak distance={PEAK_DISTANCE}min, swing threshold={SWING_THRESHOLD*100:.0f}%)\n")

    print(f"{'Code':<8} {'TZ':<18} {'Price':>8} {'Swings':>7} {'AvgAmp':>8} {'Amp%':>6} "
          f"{'Mean90':>7} {'Med90':>7} {'P25':>5} {'P75':>5}")
    print("-" * 100)

    results = []
    for sec_id, code, timezone, tt_start, tt_end in securities:
        r = analyze_security(sec_id, code, timezone, tt_start, tt_end)
        if r is None:
            print(f"{code:<8} {'(insufficient data)':>40}")
            continue
        results.append(r)
        print(f"{r['code']:<8} {r['timezone']:<18} {r['last_close']:>8.2f} {r['num_swings']:>7d} "
              f"{r['avg_amplitude']:>8.4f} {r['avg_amplitude_pct']:>5.2f}% "
              f"{r['mean_bars_90pct']:>6.1f}m {r['median_bars_90pct']:>6.1f}m "
              f"{r['p25_bars']:>5.1f} {r['p75_bars']:>5.1f}")

    if results:
        print("\n" + "=" * 100)
        print(f"\n{'Code':<8} {'Mean90 (min)':>12} {'Suggested GAUSS_MU':>20}")
        print("-" * 45)
        for r in sorted(results, key=lambda x: x['mean_bars_90pct']):
            suggested = round(r['mean_bars_90pct'])
            print(f"{r['code']:<8} {r['mean_bars_90pct']:>11.1f}m {suggested:>19d} min")


if __name__ == '__main__':
    main()
