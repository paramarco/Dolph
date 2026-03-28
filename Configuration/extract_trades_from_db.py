#!/usr/bin/env python3
"""Generate the same trade correlation report as extract_trades.sh + correlate_trades.py,
but reading entirely from the trade_history DB table (no log parsing needed).

Usage:
    python3 extract_trades_from_db.py               # all trades
    python3 extract_trades_from_db.py --since 2026-03-25   # trades from date
    python3 extract_trades_from_db.py --source live         # only live trades
"""
import sys
import psycopg2
from collections import defaultdict

DB = dict(host='127.0.0.1', port=4713, user='dolph_user',
          password='dolph_password', dbname='dolph_db', sslmode='disable')

# ---------- CLI args ----------
since = None
source_filter = None
for i, arg in enumerate(sys.argv[1:], 1):
    if arg == '--since' and i < len(sys.argv) - 1:
        since = sys.argv[i + 1]
    if arg == '--source' and i < len(sys.argv) - 1:
        source_filter = sys.argv[i + 1]

# ---------- Query ----------
where_clauses = []
params = []
if since:
    where_clauses.append("open_ts >= %s")
    params.append(since)
if source_filter:
    where_clauses.append("source = %s")
    params.append(source_filter)

where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

conn = psycopg2.connect(**DB)
cur = conn.cursor()
cur.execute(f"""
    SELECT open_ts, seccode, direction, confidence, entry_price, tp_target, sl_target,
           outcome, outcome_class, pnl_pips, tp_order_id, sl_order_id, source
    FROM trade_history
    {where_sql}
    ORDER BY open_ts
""", params)
rows = cur.fetchall()
cur.close()
conn.close()

if not rows:
    print("No trades found.")
    sys.exit(0)

trades = []
for r in rows:
    trades.append({
        'bracket_ts': r[0].strftime('%Y-%m-%d %H:%M:%S') if r[0] else '',
        'seccode': r[1],
        'position': r[2],
        'confidence': float(r[3]) if r[3] is not None else None,
        'entryPrice': float(r[4]),
        'exitTP': float(r[5]),
        'exitSL': float(r[6]),
        'outcome': r[7] or 'PENDING',
        'outcome_class': r[8],
        'pnl_pips': float(r[9]) if r[9] is not None else None,
        'tp_id': r[10] or 0,
        'sl_id': r[11] or 0,
        'source': r[12],
    })

# ============================================================
# OUTPUT (identical format to correlate_trades.py)
# ============================================================

print("=" * 160)
print(f"{'Date':20s} {'Seccode':8s} {'Dir':6s} {'Conf':6s} {'Entry':10s} {'TP':10s} {'SL':10s} {'Outcome':28s} {'Class':8s} {'PnL':10s} {'TP_ID':8s} {'SL_ID':8s}")
print("=" * 160)

for t in trades:
    conf_str = f"{t['confidence']:.4f}" if t['confidence'] is not None else "N/A"
    cls = t.get('outcome_class') or ''
    pnl_str = f"{t['pnl_pips']:.4f}" if t.get('pnl_pips') is not None else "N/A"
    print(f"{t['bracket_ts']:20s} {t['seccode']:8s} {t['position']:6s} {conf_str:6s} {t['entryPrice']:10.3f} {t['exitTP']:10.3f} {t['exitSL']:10.3f} {t['outcome']:28s} {cls:8s} {pnl_str:>10s} {t['tp_id']:8d} {t['sl_id']:8d}")

# Count by outcome type
outcome_counts = defaultdict(int)
for t in trades:
    outcome_counts[t['outcome']] += 1

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
print(f"Total trades:             {len(trades)}")
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
if all_resolved:
    print(f"OVERALL WIN RATE:         {len(wins)/len(all_resolved)*100:.1f}%")

# ============================================================
# WIN RATE BY CONFIDENCE BUCKET
# ============================================================
print("\n" + "=" * 80)
print("WIN RATE BY CONFIDENCE BUCKET (all resolved: TP_HIT + SL_HIT + forced closes)")
print("=" * 80)

buckets = defaultdict(lambda: {'win': 0, 'loss': 0, 'total': 0, 'tp_hit': 0, 'sl_hit': 0, 'forced_win': 0, 'forced_loss': 0})
for t in all_resolved:
    if t['confidence'] is not None:
        bucket = round(t['confidence'], 2)
        buckets[bucket]['total'] += 1
        buckets[bucket]['win' if t['outcome_class'] == 'WIN' else 'loss'] += 1
        if t['outcome'] == 'TP_HIT': buckets[bucket]['tp_hit'] += 1
        elif t['outcome'] == 'SL_HIT': buckets[bucket]['sl_hit'] += 1
        elif t['outcome'].startswith('FORCED_') and t['outcome_class'] == 'WIN': buckets[bucket]['forced_win'] += 1
        elif t['outcome'].startswith('FORCED_') and t['outcome_class'] == 'LOSS': buckets[bucket]['forced_loss'] += 1

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

ow = sum(1 for t in all_resolved if t['outcome_class'] == 'WIN')
ol = sum(1 for t in all_resolved if t['outcome_class'] == 'LOSS')
ot = ow + ol
print(f"\n{'OVERALL':15s} {ot:6d} {ow:5d} {ol:5d} {ow/ot*100 if ot else 0:6.1f}%")

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

# Cross-tab
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
            print(f"{direction:6s} {conf:12.4f} {total:8d} {w:6d} {l:6d} {w/total*100:7.1f}%")

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
    u = sum(1 for t in sub if t.get('outcome_class') in ('UNKNOWN', 'NEUTRAL', None))
    print(f"  {reason:30s}: {len(sub):4d} (WIN={w}, LOSS={l}, unknown={u})")
