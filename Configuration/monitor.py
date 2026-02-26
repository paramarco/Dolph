#!/usr/bin/env python3
"""Monitor Instance 5 trading activity from log file.

Single-pass parser that incrementally builds lookup maps to avoid
counting startup/restore artifacts as real exits. Uses the last
reportCurrentOpenPositions block as ground truth for open positions.

Usage: python3 monitor.py [logfile]
Default logfile: /home/dolph_user/data/5/Dolph/log/Dolph.log
"""
import re
import sys
from collections import OrderedDict

log_file = sys.argv[1] if len(sys.argv) > 1 else '/home/dolph_user/data/5/Dolph/log/Dolph.log'

with open(log_file) as f:
    lines = f.readlines()

# --- Incrementally built lookup maps (populated during single pass) ---
exit_id_map = {}      # exit_tp_id/sl_id → (seccode, 'TP'/'SL')
entry_seccode = {}    # entry_id → seccode
ib_exit_map = {}      # ib_order_id → (symbol, 'TP'/'SL')
entry_to_ib_tp = {}   # entry_id → ib_tp_order_id


def resolve_exit(oid_val):
    """Resolve seccode and TP/SL type for a filled exit order ID.

    Tries multiple strategies using the incrementally built maps:
    1. Direct match on exit_tp_id/exit_sl_id from position reports
    2. Match on IB order type (LMT=TP, STP=SL)
    3. Match on entry_id → seccode (with TP/SL inferred from IB order map)
    Returns (seccode, 'TP'/'SL') or None.
    """
    info = exit_id_map.get(oid_val)
    if info:
        return info
    info = ib_exit_map.get(oid_val)
    if info:
        return info
    sec = entry_seccode.get(oid_val)
    if sec:
        ib_tp_id = entry_to_ib_tp.get(oid_val)
        if ib_tp_id:
            tp_info = ib_exit_map.get(ib_tp_id)
            if tp_info:
                return (sec, tp_info[1])
        return (sec, 'TP')  # default to TP when type is unknown
    return None


# --- Event counters ---
entries = tp_count = sl_count = expired_entries = 0
last_entry_sec = last_tp_sec = last_sl_sec = ''
first_entry_seen = False

# Position report tracking (ground truth for open positions)
in_pos_report = False
report_positions = {}     # positions in the block being parsed
current_positions = {}    # last fully parsed position report

snapshots = []
current_ts = ''

for line in lines:
    # Track latest timestamp for non-timestamped continuation lines
    ts_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2})', line)
    if ts_match:
        current_ts = ts_match.group(1)[11:16]

    # --- Position report block: "monitored Positions: N" ---
    if 'monitored Positions:' in line:
        in_pos_report = True
        report_positions = {}
        continue

    if in_pos_report:
        # Position lines start with whitespace + "position="
        if re.match(r'\s+position=', line):
            m = re.search(r'seccode=(\w+).*entry_id=(\d+)', line)
            if m:
                sec, eid = m.group(1), m.group(2)
                report_positions[eid] = sec
                entry_seccode[eid] = sec
                m2 = re.search(r'exit_tp_id=(\d+) exit_sl_id=(\d+)', line)
                if m2:
                    exit_id_map[m2.group(1)] = (sec, 'TP')
                    exit_id_map[m2.group(2)] = (sec, 'SL')
            continue
        else:
            # Non-position line → end of report block
            in_pos_report = False
            current_positions = dict(report_positions)
            # Fall through to process this line normally

    # --- Build IB exit map from OrderIB buy-side lines ---
    m = re.search(r'OrderIB\(id=(\d+), symbol=(\w+), side=buy, type=(LMT|STP)', line)
    if m:
        kind = 'TP' if m.group(3) == 'LMT' else 'SL'
        ib_exit_map[m.group(1)] = (m.group(2), kind)

    # --- triggerExitOrder: entry_id → IB TP order id ---
    m = re.search(r'Exit order (\d+) successfully in IB OrderId: (\d+)', line)
    if m:
        entry_to_ib_tp[m.group(1)] = m.group(2)

    # --- Only process timestamped events below ---
    if not current_ts:
        continue

    # Entry: takePosition sending a position
    if 'Dolph.takePosition' in line and 'sending a' in line:
        sec_match = re.search(r'seccode:(\w+)', line)
        if sec_match:
            last_entry_sec = sec_match.group(1)

    # Entry filled
    if 'Order is Filled-Monitored' in line:
        entries += 1
        first_entry_seen = True

    # Entry cancelled (timed out)
    if 'cancelTimedoutEntries' in line and 'Cancelling Order' in line:
        expired_entries += 1

    # Exit order filled and removed
    if 'exitOrder:' in line and 'Filled' in line and 'deleted' in line:
        # Skip startup/restore artifacts: exit fills before first real entry
        if not first_entry_seen:
            continue
        oid = re.search(r'exitOrder: (\d+)', line)
        if oid:
            info = resolve_exit(oid.group(1))
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

    # Balance snapshot
    bal_match = re.search(r'net_balance=([\d.]+)', line)
    if bal_match:
        snapshots.append((
            current_ts, entries, tp_count, sl_count,
            bal_match.group(1), last_entry_sec, last_tp_sec, last_sl_sec,
            expired_entries
        ))

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
seen = OrderedDict()
for s in snapshots:
    seen[s[0]] = s

print('| Hora  | Entradas | TP | SL | Exp | Balance     | Ultima Entrada | Ultimo TP | Ultimo SL |')
print('|-------|----------|----|----|-----|-------------|----------------|-----------|-----------|')
for minute, data in seen.items():
    _, e, t, s, b, esec, tsec, ssec, exp = data
    print(f'| {minute} | {e:>8} | {t:>2} | {s:>2} | {exp:>3} | ${b:>10} | {esec:<14} | {tsec:<9} | {ssec:<9} |')

# Current open positions from last reportCurrentOpenPositions (ground truth)
if current_positions:
    print(f'\nPosiciones abiertas: {len(current_positions)}')
    for eid, sec in current_positions.items():
        print(f'  entry_id={eid}  seccode={sec}')
