import re
import bisect
from collections import defaultdict
from datetime import datetime

logdir = "/tmp/dolph_logs"

# Parse all data
entries = []     # (timestamp, seccode, confidence, position, entryPrice, exitTP, exitSL)
brackets = {}    # seccode -> [(timestamp, tp_id, sl_id)]
fills = []       # (timestamp, order_id)
cancels = []     # (timestamp, order_id)
closes = []      # (timestamp, seccode, duration)

# Forced/timeout closes: seccode -> [(timestamp, close_order_id)]
forced_closes = {}   # seccode -> [{'ts': ..., 'reason': ..., 'close_order_id': ...}]
market_close_fills = {}  # close_order_id -> {'ts': ..., 'status': ...}

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

        # Forced close: time2close exceeded
        m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*closing position for (\w+?)(?:\s*\(no exit orders\))?: time2close exceeded', line)
        if m:
            sec = m.group(2)
            if sec not in forced_closes:
                forced_closes[sec] = []
            forced_closes[sec].append({'ts': m.group(1), 'reason': 'time2close'})
            continue

        # Forced close: after endOfTradingTimes, opposite prediction
        m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*closing position for (\w+): after endOfTradingTimes', line)
        if m:
            sec = m.group(2)
            if sec not in forced_closes:
                forced_closes[sec] = []
            forced_closes[sec].append({'ts': m.group(1), 'reason': 'endOfTradingTimes'})
            continue

        # Forced close: signal reversed
        m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*closing position for (\w+): SIGNAL REVERSED', line)
        if m:
            sec = m.group(2)
            if sec not in forced_closes:
                forced_closes[sec] = []
            forced_closes[sec].append({'ts': m.group(1), 'reason': 'signal_reversed'})
            continue

        # closeExit market order success: "exit by market successfully processed for X, close_order_id=Y"
        m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*exit by market successfully processed for (\w+), close_order_id=(\d+)', line)
        if m:
            sec = m.group(2)
            # Attach close_order_id to the most recent forced_close for this seccode
            if sec in forced_closes and forced_closes[sec]:
                forced_closes[sec][-1]['close_order_id'] = int(m.group(3))
            continue

        # Market close order filled: "market close order X finished with status: Filled"
        m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*market close order (\d+) finished with status: (\w+)', line)
        if m:
            market_close_fills[int(m.group(2))] = {'ts': m.group(1), 'status': m.group(3)}
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

# Pre-index entries by seccode for fast lookup (sorted by timestamp)
entries_by_sec = defaultdict(list)  # seccode -> [(ts, entry)]
for e in entries:
    entries_by_sec[e['seccode']].append((e['ts'], e))

# Helper: find the last entry for a seccode before a given timestamp (O(log n))
def find_entry(sec, before_ts):
    sec_entries = entries_by_sec.get(sec)
    if not sec_entries:
        return None
    idx = bisect.bisect_right(sec_entries, (before_ts, )) - 1
    # bisect_right on (before_ts, ) finds insertion point after all entries with ts <= before_ts
    # since tuples compare element by element and entry dict > empty tuple
    if idx >= 0 and sec_entries[idx][0] <= before_ts:
        return sec_entries[idx][1]
    return None

# Helper: find next entry for a seccode after a given timestamp (O(log n))
def find_next_entry(sec, after_ts):
    sec_entries = entries_by_sec.get(sec)
    if not sec_entries:
        return None
    idx = bisect.bisect_right(sec_entries, (after_ts + 'z', ))
    if idx < len(sec_entries):
        return sec_entries[idx][1]
    return None

# Now correlate: for each bracket, determine outcome
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
            # Check if there's a forced close for this seccode after bracket creation
            forced = forced_closes.get(sec, [])
            forced_match = [f for f in forced if f['ts'] >= b['ts']]
            if forced_match:
                fc = forced_match[0]
                outcome = f"FORCED_{fc['reason'].upper()}"
            else:
                outcome = 'PENDING'
        else:
            outcome = f'UNKNOWN(tp_f={tp_filled},sl_f={sl_filled},tp_c={tp_cancelled},sl_c={sl_cancelled})'

        # Find matching entry (closest entry for this seccode before bracket time)
        entry = find_entry(sec, b['ts'])
        if entry:
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

# ============================================================
# Classify FORCED closes as WIN/LOSS based on P&L direction.
# Since we don't have the exact close price from the log, we use
# the next entry's entryPrice for the same seccode as a proxy
# (the market price moments after the close).
# If no proxy is available, we classify as NEUTRAL.
# ============================================================
for i, t in enumerate(trades):
    if not t['outcome'].startswith('FORCED_'):
        continue
    sec = t['seccode']
    close_ts = t['bracket_ts']

    # Look for the next entry for this seccode after the close as price proxy
    next_entry = find_next_entry(sec, close_ts)
    if next_entry:
        proxy_price = next_entry['entryPrice']
    else:
        # No subsequent entry — try the entry itself (close ~= entry for small margins)
        proxy_price = None

    if proxy_price is not None and t['entryPrice'] > 0:
        if t['position'] == 'long':
            pnl = proxy_price - t['entryPrice']
        else:
            pnl = t['entryPrice'] - proxy_price
        t['close_price'] = proxy_price
        if pnl > 0:
            t['outcome_class'] = 'WIN'
        elif pnl < 0:
            t['outcome_class'] = 'LOSS'
        else:
            t['outcome_class'] = 'NEUTRAL'
        t['pnl_pips'] = pnl
    else:
        t['outcome_class'] = 'UNKNOWN'
        t['close_price'] = None
        t['pnl_pips'] = None

# For TP_HIT / SL_HIT, set outcome_class directly
for t in trades:
    if t['outcome'] == 'TP_HIT':
        t['outcome_class'] = 'WIN'
        if t['position'] == 'long':
            t['pnl_pips'] = t['exitTP'] - t['entryPrice']
        else:
            t['pnl_pips'] = t['entryPrice'] - t['exitTP']
    elif t['outcome'] == 'SL_HIT':
        t['outcome_class'] = 'LOSS'
        if t['position'] == 'long':
            t['pnl_pips'] = t['entryPrice'] - t['exitSL']
        else:
            t['pnl_pips'] = t['exitSL'] - t['entryPrice']

# ============================================================
# OUTPUT
# ============================================================

# Print all trades
print("=" * 160)
print(f"{'Date':20s} {'Seccode':8s} {'Dir':6s} {'Conf':6s} {'Entry':10s} {'TP':10s} {'SL':10s} {'Outcome':28s} {'Class':8s} {'PnL':10s} {'TP_ID':8s} {'SL_ID':8s}")
print("=" * 160)

for t in trades:
    conf_str = f"{t['confidence']:.4f}" if t['confidence'] is not None else "N/A"
    cls = t.get('outcome_class', '')
    pnl_str = f"{t['pnl_pips']:.4f}" if t.get('pnl_pips') is not None else "N/A"
    print(f"{t['bracket_ts']:20s} {t['seccode']:8s} {t['position']:6s} {conf_str:6s} {t['entryPrice']:10.3f} {t['exitTP']:10.3f} {t['exitSL']:10.3f} {t['outcome']:28s} {cls:8s} {pnl_str:>10s} {t['tp_id']:8d} {t['sl_id']:8d}")

# Count by outcome type
outcome_counts = defaultdict(int)
for t in trades:
    outcome_counts[t['outcome']] += 1

# Summary statistics
all_resolved = [t for t in trades if t.get('outcome_class') in ('WIN', 'LOSS')]
wins = [t for t in all_resolved if t['outcome_class'] == 'WIN']
losses = [t for t in all_resolved if t['outcome_class'] == 'LOSS']

tp_hit = [t for t in trades if t['outcome'] == 'TP_HIT']
sl_hit = [t for t in trades if t['outcome'] == 'SL_HIT']
forced_resolved = [t for t in trades if t['outcome'].startswith('FORCED_') and t.get('outcome_class') in ('WIN', 'LOSS')]
forced_win = [t for t in forced_resolved if t['outcome_class'] == 'WIN']
forced_loss = [t for t in forced_resolved if t['outcome_class'] == 'LOSS']
still_pending = [t for t in trades if t['outcome'] == 'PENDING']

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"Total entries sent:       {len(entries)}")
print(f"Total brackets created:   {sum(len(v) for v in brackets.values())}")
print(f"")
print(f"  TP_HIT (bracket):       {len(tp_hit)}")
print(f"  SL_HIT (bracket):       {len(sl_hit)}")
for oc in sorted(outcome_counts):
    if oc.startswith('FORCED_'):
        print(f"  {oc:26s} {outcome_counts[oc]}")
print(f"  Still PENDING:          {len(still_pending)}")
print(f"")
print(f"Forced closes resolved:   {len(forced_resolved)} (WIN={len(forced_win)}, LOSS={len(forced_loss)})")
print(f"")
print(f"TOTAL RESOLVED:           {len(all_resolved)} (WIN={len(wins)}, LOSS={len(losses)})")
print(f"OVERALL WIN RATE:         {len(wins)/len(all_resolved)*100:.1f}%" if all_resolved else "N/A")

# ============================================================
# WIN RATE BY CONFIDENCE BUCKET (all resolved trades)
# ============================================================
print("\n" + "=" * 80)
print("WIN RATE BY CONFIDENCE BUCKET (all resolved: TP_HIT + SL_HIT + forced closes)")
print("=" * 80)

buckets = defaultdict(lambda: {'win': 0, 'loss': 0, 'total': 0, 'tp_hit': 0, 'sl_hit': 0, 'forced_win': 0, 'forced_loss': 0})
for t in all_resolved:
    if t['confidence'] is not None:
        bucket = round(t['confidence'], 2)
        buckets[bucket]['total'] += 1
        if t['outcome_class'] == 'WIN':
            buckets[bucket]['win'] += 1
        else:
            buckets[bucket]['loss'] += 1
        if t['outcome'] == 'TP_HIT':
            buckets[bucket]['tp_hit'] += 1
        elif t['outcome'] == 'SL_HIT':
            buckets[bucket]['sl_hit'] += 1
        elif t['outcome'].startswith('FORCED_') and t['outcome_class'] == 'WIN':
            buckets[bucket]['forced_win'] += 1
        elif t['outcome'].startswith('FORCED_') and t['outcome_class'] == 'LOSS':
            buckets[bucket]['forced_loss'] += 1

print(f"\n{'Confidence':12s} {'Total':6s} {'WIN':5s} {'LOSS':5s} {'WR%':7s}  {'TP':4s} {'SL':4s} {'F.Win':5s} {'F.Loss':6s}")
print("-" * 70)
for conf in sorted(buckets.keys()):
    b = buckets[conf]
    wr = b['win'] / b['total'] * 100 if b['total'] > 0 else 0
    print(f"{conf:12.4f} {b['total']:6d} {b['win']:5d} {b['loss']:5d} {wr:6.1f}%  {b['tp_hit']:4d} {b['sl_hit']:4d} {b['forced_win']:5d} {b['forced_loss']:6d}")

# Wider buckets
print("\n\nWIDER CONFIDENCE RANGES:")
print(f"{'Range':15s} {'Total':6s} {'WIN':5s} {'LOSS':5s} {'WR%':7s}  {'TP':4s} {'SL':4s} {'F.Win':5s} {'F.Loss':6s}")
print("-" * 70)
ranges = [(0.0, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 0.95), (0.95, 1.01)]
for lo, hi in ranges:
    sub = [t for t in all_resolved if t['confidence'] is not None and lo <= t['confidence'] < hi]
    w = sum(1 for t in sub if t['outcome_class'] == 'WIN')
    l = sum(1 for t in sub if t['outcome_class'] == 'LOSS')
    tp = sum(1 for t in sub if t['outcome'] == 'TP_HIT')
    sl = sum(1 for t in sub if t['outcome'] == 'SL_HIT')
    fw = sum(1 for t in sub if t['outcome'].startswith('FORCED_') and t['outcome_class'] == 'WIN')
    fl = sum(1 for t in sub if t['outcome'].startswith('FORCED_') and t['outcome_class'] == 'LOSS')
    total = w + l
    wr = w / total * 100 if total > 0 else 0
    print(f"[{lo:.2f}, {hi:.2f})   {total:6d} {w:5d} {l:5d} {wr:6.1f}%  {tp:4d} {sl:4d} {fw:5d} {fl:6d}")

overall_w = sum(1 for t in all_resolved if t['outcome_class'] == 'WIN')
overall_l = sum(1 for t in all_resolved if t['outcome_class'] == 'LOSS')
overall_t = overall_w + overall_l
print(f"\n{'OVERALL':15s} {overall_t:6d} {overall_w:5d} {overall_l:5d} {overall_w/overall_t*100 if overall_t else 0:6.1f}%")

# ============================================================
# ESTIMATED P&L BY CONFIDENCE
# ============================================================
print("\n" + "=" * 80)
print("ESTIMATED P&L BY CONFIDENCE (all resolved trades)")
print("=" * 80)
print(f"\n{'Confidence':12s} {'Trades':8s} {'Avg Win':12s} {'Avg Loss':12s} {'Expectancy':12s}")
print("-" * 60)

for conf in sorted(buckets.keys()):
    relevant = [t for t in all_resolved if t['confidence'] is not None and round(t['confidence'], 2) == conf and t.get('pnl_pips') is not None]
    win_pnl = [t['pnl_pips'] for t in relevant if t['outcome_class'] == 'WIN']
    loss_pnl = [abs(t['pnl_pips']) for t in relevant if t['outcome_class'] == 'LOSS']

    avg_win = sum(win_pnl)/len(win_pnl) if win_pnl else 0
    avg_loss = sum(loss_pnl)/len(loss_pnl) if loss_pnl else 0
    b = buckets[conf]
    wr = b['win'] / b['total'] if b['total'] > 0 else 0
    expectancy = wr * avg_win - (1-wr) * avg_loss
    print(f"{conf:12.4f} {b['total']:8d} {avg_win:12.4f} {avg_loss:12.4f} {expectancy:12.4f}")

# ============================================================
# BY DIRECTION
# ============================================================
print("\n" + "=" * 80)
print("WIN RATE BY DIRECTION (all resolved)")
print("=" * 80)
for direction in ['long', 'short']:
    w = sum(1 for t in all_resolved if t['position'] == direction and t['outcome_class'] == 'WIN')
    l = sum(1 for t in all_resolved if t['position'] == direction and t['outcome_class'] == 'LOSS')
    total = w + l
    wr = w / total * 100 if total > 0 else 0
    print(f"{direction:6s}: {total} trades, {w} WIN, {l} LOSS, Win Rate: {wr:.1f}%")

# Cross-tab: direction x confidence
print("\n" + "=" * 80)
print("WIN RATE BY DIRECTION x CONFIDENCE (all resolved)")
print("=" * 80)
print(f"{'Dir':6s} {'Confidence':12s} {'Trades':8s} {'WIN':6s} {'LOSS':6s} {'WR%':8s}")
print("-" * 50)
for direction in ['long', 'short']:
    for conf in sorted(buckets.keys()):
        w = sum(1 for t in all_resolved if t['position'] == direction and t['confidence'] is not None and round(t['confidence'],2) == conf and t['outcome_class'] == 'WIN')
        l = sum(1 for t in all_resolved if t['position'] == direction and t['confidence'] is not None and round(t['confidence'],2) == conf and t['outcome_class'] == 'LOSS')
        total = w + l
        if total > 0:
            wr = w / total * 100
            print(f"{direction:6s} {conf:12.4f} {total:8d} {w:6d} {l:6d} {wr:7.1f}%")

# ============================================================
# FORCED CLOSE BREAKDOWN
# ============================================================
print("\n" + "=" * 80)
print("FORCED CLOSE BREAKDOWN")
print("=" * 80)
forced_trades = [t for t in trades if t['outcome'].startswith('FORCED_')]
print(f"Total forced closes: {len(forced_trades)}")
for reason in sorted(set(t['outcome'] for t in forced_trades)):
    sub = [t for t in forced_trades if t['outcome'] == reason]
    w = sum(1 for t in sub if t.get('outcome_class') == 'WIN')
    l = sum(1 for t in sub if t.get('outcome_class') == 'LOSS')
    u = sum(1 for t in sub if t.get('outcome_class') in ('UNKNOWN', 'NEUTRAL'))
    print(f"  {reason:30s}: {len(sub):4d} (WIN={w}, LOSS={l}, unknown={u})")
