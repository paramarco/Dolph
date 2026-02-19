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

entries = tp = sl = 0
last_entry_sec = ''
last_tp_sec = ''
snapshots = []

for line in lines:
    ts_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2})', line)
    if not ts_match:
        continue
    if 'triggerExitOrder' in line:
        sec_match = re.search(r"symbol='(\w+)'", line)
        if sec_match:
            last_entry_sec = sec_match.group(1)
    if 'Order is Filled-Monitored' in line:
        entries += 1
    if 'exitOrder' in line and 'Filled' in line and 'deleted' in line:
        tp += 1
        oid = re.search(r'exitOrder: (\d+)', line)
        if oid:
            oid_val = oid.group(1)
            for prev in lines:
                if oid_val in prev and 'triggerExitOrder' in prev:
                    sm = re.search(r"symbol='(\w+)'", prev)
                    if sm:
                        last_tp_sec = sm.group(1)
                        break
    if 'STP' in line and 'status=Filled' in line:
        sl += 1
    bal_match = re.search(r'net_balance=([\d.]+)', line)
    if bal_match:
        snapshots.append((ts_match.group(1)[11:16], entries, tp, sl, bal_match.group(1), last_entry_sec, last_tp_sec))

seen = OrderedDict()
for s in snapshots:
    seen[s[0]] = s

print('| Hora  | Entradas | TP | SL | Balance     | Ultima Entrada | Ultimo TP |')
print('|-------|----------|----|----|-------------|----------------|-----------|')
for minute, data in seen.items():
    _, e, t, s, b, esec, tsec = data
    print(f'| {minute} | {e:>8} | {t:>2} | {s:>2} | ${b:>10} | {esec:<14} | {tsec:<9} |')
