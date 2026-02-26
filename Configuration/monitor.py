#!/usr/bin/env python3
"""Monitor Instance 5 trading activity from log file.

Parses the operational log to track entries, exits (TP/SL), expired entries,
and balance snapshots. Uses multiple lookup strategies to resolve seccode
and TP/SL type from exit order fill events.

Usage: python3 monitor.py [logfile]
Default logfile: /home/dolph_user/data/5/Dolph/log/Dolph.log
"""
import re
import sys
from collections import OrderedDict

log_file = sys.argv[1] if len(sys.argv) > 1 else '/home/dolph_user/data/5/Dolph/log/Dolph.log'

with open(log_file) as f:
    lines = f.readlines()

# ---------------------------------------------------------------------------
# Phase 1: Build lookup maps from entire log (single pass)
# ---------------------------------------------------------------------------

# Map A: exit_tp_id/exit_sl_id → (seccode, 'TP'/'SL') from position reports
# e.g. exit_tp_id=9800 → ('BBVA', 'TP'), exit_sl_id=9801 → ('BBVA', 'SL')
exit_id_map = {}

# Map B: entry_id → seccode from position reports (works even when exit IDs are None)
entry_seccode = {}

# Map C: IB buy-side order id → (symbol, 'TP'/'SL') from OrderIB lines
# type=LMT on buy-side = TP exit, type=STP on buy-side = SL exit
ib_exit_map = {}

# Map D: entry_id → IB TP order id from triggerExitOrder
# "Exit order <entry_id> successfully in IB OrderId: <ib_tp_id>"
entry_to_ib_tp = {}

for line in lines:
    # Position reports with numeric exit IDs
    m = re.search(
        r'seccode=(\w+).*entry_id=(\d+).*exit_tp_id=(\d+) exit_sl_id=(\d+)', line
    )
    if m:
        sec = m.group(1)
        entry_seccode[m.group(2)] = sec
        exit_id_map[m.group(3)] = (sec, 'TP')
        exit_id_map[m.group(4)] = (sec, 'SL')
        continue

    # Position reports with entry_id but exit IDs may be None
    m = re.search(r'seccode=(\w+).*entry_id=(\d+)', line)
    if m:
        entry_seccode[m.group(2)] = m.group(1)

    # OrderIB buy-side exit orders (TP = LMT, SL = STP)
    m = re.search(r'OrderIB\(id=(\d+), symbol=(\w+), side=buy, type=(LMT|STP)', line)
    if m:
        kind = 'TP' if m.group(3) == 'LMT' else 'SL'
        ib_exit_map[m.group(1)] = (m.group(2), kind)

    # triggerExitOrder: entry_id → IB exit order ID (logs the TP order)
    m = re.search(r'Exit order (\d+) successfully in IB OrderId: (\d+)', line)
    if m:
        entry_to_ib_tp[m.group(1)] = m.group(2)


def _resolve_exit(oid_val):
    """Resolve seccode and TP/SL type for a filled exit order ID.

    Tries multiple lookup strategies:
    1. Direct match on exit_tp_id/exit_sl_id from position reports
    2. Match on IB order type (LMT=TP, STP=SL)
    3. Match on entry_id → seccode (with TP/SL inferred from IB order map)
    Returns (seccode, 'TP'/'SL') or None.
    """
    # Strategy 1: exact exit_tp_id / exit_sl_id match
    info = exit_id_map.get(oid_val)
    if info:
        return info

    # Strategy 2: IB order type match (buy LMT=TP, buy STP=SL)
    info = ib_exit_map.get(oid_val)
    if info:
        return info

    # Strategy 3: the filled ID is the entry_id
    sec = entry_seccode.get(oid_val)
    if sec:
        # Try to determine TP vs SL via the IB exit order map
        ib_tp_id = entry_to_ib_tp.get(oid_val)
        if ib_tp_id:
            tp_info = ib_exit_map.get(ib_tp_id)
            if tp_info:
                return (sec, tp_info[1])
        return (sec, 'TP')  # default to TP when type is unknown

    return None


# ---------------------------------------------------------------------------
# Phase 2: Process trading events (single pass)
# ---------------------------------------------------------------------------
entries = tp_count = sl_count = expired_entries = 0
last_entry_sec = ''
last_tp_sec = ''
last_sl_sec = ''
open_positions = {}   # entry_id → seccode (track currently open positions)
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
        eid = re.search(r'exitOrderRequested: (\d+)', line)
        if eid:
            sec = entry_seccode.get(eid.group(1), last_entry_sec)
            open_positions[eid.group(1)] = sec

    # Entry cancelled (timed out)
    if 'cancelTimedoutEntries' in line and 'Cancelling Order' in line:
        expired_entries += 1

    # Exit order filled and removed
    if 'exitOrder:' in line and 'Filled' in line and 'deleted' in line:
        oid = re.search(r'exitOrder: (\d+)', line)
        if oid:
            oid_val = oid.group(1)
            info = _resolve_exit(oid_val)
            if info:
                sec, kind = info
                if kind == 'TP':
                    tp_count += 1
                    last_tp_sec = sec
                else:
                    sl_count += 1
                    last_sl_sec = sec
            else:
                tp_count += 1  # complete fallback: unknown exit
            # Remove from open positions
            open_positions.pop(oid_val, None)

    # Balance snapshot
    bal_match = re.search(r'net_balance=([\d.]+)', line)
    if bal_match:
        snapshots.append((
            ts_match.group(1)[11:16], entries, tp_count, sl_count,
            bal_match.group(1), last_entry_sec, last_tp_sec, last_sl_sec,
            expired_entries
        ))

# ---------------------------------------------------------------------------
# Phase 3: Output
# ---------------------------------------------------------------------------
seen = OrderedDict()
for s in snapshots:
    seen[s[0]] = s

print('| Hora  | Entradas | TP | SL | Exp | Balance     | Ultima Entrada | Ultimo TP | Ultimo SL |')
print('|-------|----------|----|----|-----|-------------|----------------|-----------|-----------|')
for minute, data in seen.items():
    _, e, t, s, b, esec, tsec, ssec, exp = data
    print(f'| {minute} | {e:>8} | {t:>2} | {s:>2} | {exp:>3} | ${b:>10} | {esec:<14} | {tsec:<9} | {ssec:<9} |')

# Current open positions summary
if open_positions:
    print(f'\nPosiciones abiertas: {len(open_positions)}')
    for eid, sec in open_positions.items():
        print(f'  entry_id={eid}  seccode={sec}')
