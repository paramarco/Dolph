import re
from collections import defaultdict
from datetime import datetime

logdir = "/tmp/dolph_logs"

# Parse all data
entries = []     # (timestamp, seccode, confidence, position, entryPrice, exitTP, exitSL)
brackets = {}    # seccode -> [(timestamp, tp_id, sl_id)]
fills = []       # (timestamp, order_id)
cancels = []     # (timestamp, order_id)
closes = []      # (timestamp, seccode, duration)

with open(f"{logdir}/all_logs.log") as f:
    for line in f:
        # Entry: "sending a"
        m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*confidence=(\d+\.\d+).*sending a\s+position=(\w+) seccode=(\S+) quantity=(\d+) entryPrice=(\S+) exitTakeProfit=(\S+) exitStopLoss=(\S+)', line)
        if m:
            ts = m.group(1)
            entries.append({
                'ts': ts,
                'confidence': float(m.group(2)),
                'position': m.group(3),
                'seccode': m.group(4),
                'qty': int(m.group(5)),
                'entryPrice': float(m.group(6)),
                'exitTP': float(m.group(7)),
                'exitSL': float(m.group(8)),
            })
            continue
        
        # Bracket: "Exit bracket for X created: TP=Y, SL=Z"
        m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*Exit bracket for (\S+) created: TP=(\d+), SL=(\d+)', line)
        if m:
            ts = m.group(1)
            sec = m.group(2)
            tp_id = int(m.group(3))
            sl_id = int(m.group(4))
            if sec not in brackets:
                brackets[sec] = []
            brackets[sec].append({'ts': ts, 'tp_id': tp_id, 'sl_id': sl_id})
            continue
        
        # Exit fill: "exitOrder: X in status: Filled"
        m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*exitOrder: (\d+) in status: Filled', line)
        if m:
            fills.append({'ts': m.group(1), 'order_id': int(m.group(2))})
            continue
        
        # Exit cancel: "exitOrder: X in status: Cancelled"
        m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*exitOrder: (\d+) in status: Cancelled', line)
        if m:
            cancels.append({'ts': m.group(1), 'order_id': int(m.group(2))})
            continue
        
        # Position close: "Position closed for X after Ys"
        m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*Position closed for (\S+) after (\d+)s', line)
        if m:
            closes.append({'ts': m.group(1), 'seccode': m.group(2), 'duration': int(m.group(3))})
            continue

# Build order_id -> (seccode, type) mapping from brackets
order_map = {}  # order_id -> (seccode, 'TP' or 'SL', bracket_ts)
for sec, blist in brackets.items():
    for b in blist:
        order_map[b['tp_id']] = (sec, 'TP', b['ts'])
        order_map[b['sl_id']] = (sec, 'SL', b['ts'])

# Build fill set
fill_set = {f['order_id']: f['ts'] for f in fills}
cancel_set = {c['order_id']: c['ts'] for c in cancels}

# Now correlate: for each bracket, determine if TP or SL was filled
trades = []
for sec, blist in brackets.items():
    for b in blist:
        tp_filled = b['tp_id'] in fill_set
        sl_filled = b['sl_id'] in fill_set
        tp_cancelled = b['tp_id'] in cancel_set
        sl_cancelled = b['sl_id'] in cancel_set
        
        if tp_filled and sl_cancelled:
            outcome = 'TP_HIT'
        elif sl_filled and tp_cancelled:
            outcome = 'SL_HIT'
        elif tp_filled and sl_filled:
            outcome = 'BOTH_FILLED'
        elif not tp_filled and not sl_filled:
            outcome = 'PENDING'
        else:
            outcome = f'UNKNOWN(tp_f={tp_filled},sl_f={sl_filled},tp_c={tp_cancelled},sl_c={sl_cancelled})'
        
        # Find matching entry (closest entry for this seccode before bracket time)
        matching_entries = [e for e in entries if e['seccode'] == sec and e['ts'] <= b['ts']]
        if matching_entries:
            # Get the closest one
            entry = matching_entries[-1]
            confidence = entry['confidence']
            position = entry['position']
            entryPrice = entry['entryPrice']
            exitTP = entry['exitTP']
            exitSL = entry['exitSL']
        else:
            confidence = None
            position = '?'
            entryPrice = 0
            exitTP = 0
            exitSL = 0
        
        trades.append({
            'seccode': sec,
            'bracket_ts': b['ts'],
            'tp_id': b['tp_id'],
            'sl_id': b['sl_id'],
            'outcome': outcome,
            'confidence': confidence,
            'position': position,
            'entryPrice': entryPrice,
            'exitTP': exitTP,
            'exitSL': exitSL,
        })

# Sort by timestamp
trades.sort(key=lambda x: x['bracket_ts'])

# Print all completed trades
print("=" * 140)
print(f"{'Date':20s} {'Seccode':8s} {'Dir':6s} {'Conf':6s} {'Entry':10s} {'TP':10s} {'SL':10s} {'Outcome':12s} {'TP_ID':8s} {'SL_ID':8s}")
print("=" * 140)

completed = [t for t in trades if t['outcome'] in ('TP_HIT', 'SL_HIT')]
pending = [t for t in trades if t['outcome'] == 'PENDING']

for t in trades:
    conf_str = f"{t['confidence']:.4f}" if t['confidence'] is not None else "N/A"
    print(f"{t['bracket_ts']:20s} {t['seccode']:8s} {t['position']:6s} {conf_str:6s} {t['entryPrice']:10.3f} {t['exitTP']:10.3f} {t['exitSL']:10.3f} {t['outcome']:12s} {t['tp_id']:8d} {t['sl_id']:8d}")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"Total entries sent: {len(entries)}")
print(f"Total brackets created: {sum(len(v) for v in brackets.values())}")
print(f"Completed trades: {len(completed)}")
print(f"Pending trades: {len(pending)}")

# Win rate by confidence bucket
print("\n" + "=" * 80)
print("WIN RATE BY CONFIDENCE BUCKET")
print("=" * 80)

# Group by confidence
buckets = defaultdict(lambda: {'tp': 0, 'sl': 0, 'total': 0})
for t in completed:
    if t['confidence'] is not None:
        # Round to 2 decimals for bucketing
        bucket = round(t['confidence'], 2)
        buckets[bucket]['total'] += 1
        if t['outcome'] == 'TP_HIT':
            buckets[bucket]['tp'] += 1
        else:
            buckets[bucket]['sl'] += 1

print(f"\n{'Confidence':12s} {'Trades':8s} {'TP Hits':8s} {'SL Hits':8s} {'Win Rate':10s}")
print("-" * 50)
for conf in sorted(buckets.keys()):
    b = buckets[conf]
    wr = b['tp'] / b['total'] * 100 if b['total'] > 0 else 0
    print(f"{conf:12.4f} {b['total']:8d} {b['tp']:8d} {b['sl']:8d} {wr:9.1f}%")

# Wider buckets
print("\n\nWIDER CONFIDENCE RANGES:")
print(f"{'Range':15s} {'Trades':8s} {'TP Hits':8s} {'SL Hits':8s} {'Win Rate':10s}")
print("-" * 55)
ranges = [(0.0, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 0.95), (0.95, 1.01)]
for lo, hi in ranges:
    tp = sum(1 for t in completed if t['confidence'] is not None and lo <= t['confidence'] < hi and t['outcome'] == 'TP_HIT')
    sl = sum(1 for t in completed if t['confidence'] is not None and lo <= t['confidence'] < hi and t['outcome'] == 'SL_HIT')
    total = tp + sl
    wr = tp / total * 100 if total > 0 else 0
    print(f"[{lo:.2f}, {hi:.2f})   {total:8d} {tp:8d} {sl:8d} {wr:9.1f}%")

# Overall
tp_total = sum(1 for t in completed if t['outcome'] == 'TP_HIT')
sl_total = sum(1 for t in completed if t['outcome'] == 'SL_HIT')
print(f"\n{'OVERALL':15s} {len(completed):8d} {tp_total:8d} {sl_total:8d} {tp_total/len(completed)*100 if completed else 0:9.1f}%")

# Average P&L per trade by confidence
print("\n" + "=" * 80)
print("ESTIMATED P&L BY CONFIDENCE (using entry/TP/SL prices)")
print("=" * 80)
print(f"\n{'Confidence':12s} {'Trades':8s} {'Avg TP pips':12s} {'Avg SL pips':12s} {'Expectancy':12s}")
print("-" * 60)

for conf in sorted(buckets.keys()):
    relevant = [t for t in completed if t['confidence'] is not None and round(t['confidence'], 2) == conf]
    tp_gains = []
    sl_losses = []
    for t in relevant:
        if t['position'] == 'long':
            tp_gain = t['exitTP'] - t['entryPrice']
            sl_loss = t['entryPrice'] - t['exitSL']
        else:  # short
            tp_gain = t['entryPrice'] - t['exitTP']
            sl_loss = t['exitSL'] - t['entryPrice']
        if t['outcome'] == 'TP_HIT':
            tp_gains.append(tp_gain)
        else:
            sl_losses.append(sl_loss)
    
    avg_tp = sum(tp_gains)/len(tp_gains) if tp_gains else 0
    avg_sl = sum(sl_losses)/len(sl_losses) if sl_losses else 0
    b = buckets[conf]
    wr = b['tp'] / b['total'] if b['total'] > 0 else 0
    expectancy = wr * avg_tp - (1-wr) * avg_sl
    print(f"{conf:12.4f} {b['total']:8d} {avg_tp:12.4f} {avg_sl:12.4f} {expectancy:12.4f}")

# By direction
print("\n" + "=" * 80)
print("WIN RATE BY DIRECTION")
print("=" * 80)
for direction in ['long', 'short']:
    tp_d = sum(1 for t in completed if t['position'] == direction and t['outcome'] == 'TP_HIT')
    sl_d = sum(1 for t in completed if t['position'] == direction and t['outcome'] == 'SL_HIT')
    total_d = tp_d + sl_d
    wr = tp_d / total_d * 100 if total_d > 0 else 0
    print(f"{direction:6s}: {total_d} trades, {tp_d} TP, {sl_d} SL, Win Rate: {wr:.1f}%")

# Cross-tab: direction x confidence
print("\n" + "=" * 80)
print("WIN RATE BY DIRECTION x CONFIDENCE")
print("=" * 80)
print(f"{'Dir':6s} {'Confidence':12s} {'Trades':8s} {'TP':6s} {'SL':6s} {'WR%':8s}")
print("-" * 45)
for direction in ['long', 'short']:
    for conf in sorted(buckets.keys()):
        tp = sum(1 for t in completed if t['position'] == direction and t['confidence'] is not None and round(t['confidence'],2) == conf and t['outcome'] == 'TP_HIT')
        sl = sum(1 for t in completed if t['position'] == direction and t['confidence'] is not None and round(t['confidence'],2) == conf and t['outcome'] == 'SL_HIT')
        total = tp + sl
        if total > 0:
            wr = tp / total * 100
            print(f"{direction:6s} {conf:12.4f} {total:8d} {tp:6d} {sl:6d} {wr:7.1f}%")

