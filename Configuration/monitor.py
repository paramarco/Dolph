#!/usr/bin/env python3
"""Monitor Instance 5 trading activity from log file.
Usage: python3 monitor.py [logfile]
Default logfile: /home/dolph_user/data/5/Dolph/log/Dolph.log
"""
import re
import sys
from collections import OrderedDict

log_file = sys.argv[1] if len(sys.argv) > 1 else '/home/dolph_user/data/5/Dolph/log/Dolph.log'

with open(log_file) as f:
    lines = f.readlines()

# Build a map: exit_order_id -> (seccode, order_type) from monitoredPositions reports
# e.g. exit_tp_id=5208 -> ('DKNG', 'TP'), exit_sl_id=5209 -> ('DKNG', 'SL')
exit_id_map = {}
for line in lines:
    m = re.search(
        r'position=\S+ seccode=(\w+).*exit_tp_id=(\d+) exit_sl_id=(\d+)', line
    )
    if m:
        sec, tp_id, sl_id = m.group(1), m.group(2), m.group(3)
        exit_id_map[tp_id] = (sec, 'TP')
        exit_id_map[sl_id] = (sec, 'SL')

entries = tp_count = sl_count = 0
last_entry_sec = ''
last_tp_sec = ''
last_sl_sec = ''
snapshots = []

for line in lines:
    ts_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2})', line)
    if not ts_match:
        continue

    # Entry: Dolph.takePosition sending a position
    if 'Dolph.takePosition' in line and 'sending a' in line:
        sec_match = re.search(r'seccode:(\w+)', line)
        if sec_match:
            last_entry_sec = sec_match.group(1)

    # Entry filled
    if 'Order is Filled-Monitored' in line:
        entries += 1

    # Exit order filled and removed
    if 'exitOrder:' in line and 'Filled' in line and 'deleted' in line:
        oid = re.search(r'exitOrder: (\d+)', line)
        if oid:
            oid_val = oid.group(1)
            info = exit_id_map.get(oid_val)
            if info:
                sec, kind = info
                if kind == 'TP':
                    tp_count += 1
                    last_tp_sec = sec
                else:
                    sl_count += 1
                    last_sl_sec = sec
            else:
                tp_count += 1  # fallback: count as TP if unknown

    # Position closed (cooldown log) - backup for last_tp_sec
    closed_match = re.search(r'Position closed for (\w+)', line)
    if closed_match:
        last_tp_sec = closed_match.group(1)

    # Balance snapshot
    bal_match = re.search(r'net_balance=([\d.]+)', line)
    if bal_match:
        snapshots.append((
            ts_match.group(1)[11:16], entries, tp_count, sl_count,
            bal_match.group(1), last_entry_sec, last_tp_sec, last_sl_sec
        ))

seen = OrderedDict()
for s in snapshots:
    seen[s[0]] = s

print('| Hora  | Entradas | TP | SL | Balance     | Ultima Entrada | Ultimo TP | Ultimo SL |')
print('|-------|----------|----|----|-------------|----------------|-----------|-----------|')
for minute, data in seen.items():
    _, e, t, s, b, esec, tsec, ssec = data
    print(f'| {minute} | {e:>8} | {t:>2} | {s:>2} | ${b:>10} | {esec:<14} | {tsec:<9} | {ssec:<9} |')
