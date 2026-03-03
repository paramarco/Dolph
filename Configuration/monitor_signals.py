#!/usr/bin/env python3
"""Monitor Instance 5: Analyze every signal that doesn't close with TP.

Tails the live log and tracks each position from entry to exit.
Captures the predict() signal at entry time (phase, confidence, volume_contexts,
margin, entry/exit prices) and classifies the close reason.

Failed signals = positions that close by:
  - SL hit (exitOrder filled where order_id matches exit_sl_id)
  - exitTimeSeconds timeout (120min)
  - End of trading session (after endOfTradingTimes)
  - Opposite prediction while open

Writes a live report to /home/dolph_user/data/5/signal_analysis.txt

Usage: python3 monitor_signals.py [hours] [logfile]
  Default: 12 hours, /home/dolph_user/data/5/Dolph/log/Dolph.log
"""
import re
import sys
import time
import os
from datetime import datetime, timedelta
from collections import OrderedDict

# Unbuffered stdout for real-time log output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 12
LOG_FILE = sys.argv[2] if len(sys.argv) > 2 else '/home/dolph_user/data/5/Dolph/log/Dolph.log'
REPORT_FILE = '/home/dolph_user/data/5/Dolph/log/signal_analysis.txt'

end_time = datetime.now() + timedelta(hours=HOURS)

# ---- State tracking ----
# Active positions: entry_id -> {seccode, direction, confidence, entry_price,
#     exit_tp_price, exit_sl_price, phase, volume_contexts, margin,
#     entry_time, exit_tp_id, exit_sl_id}
active_positions = {}

# Last predict signal per seccode (before takePosition)
last_signal = {}  # seccode -> {phase, signal, confidence, volume_contexts, margin, entry_price, exit_price, time}

# Map entry_id -> seccode (from takePosition/processPosition lines)
entry_id_to_seccode = {}
# Map exit_tp_id -> entry_id, exit_sl_id -> entry_id
exit_to_entry = {}

# Completed trades (for report)
completed_trades = []  # list of dicts

# Stats
stats = {'tp': 0, 'sl': 0, 'timeout': 0, 'eot': 0, 'opposite': 0,
         'entry_expired': 0, 'total_entries': 0}


def parse_timestamp(line):
    """Extract datetime from log line."""
    m = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
    if m:
        return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
    return None


def write_report():
    """Write current analysis to report file."""
    with open(REPORT_FILE, 'w') as f:
        f.write(f"=== Dolph Instance 5 Signal Analysis ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Monitoring until: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"--- Summary ---\n")
        f.write(f"Total entries filled: {stats['total_entries']}\n")
        f.write(f"TP hits:      {stats['tp']}\n")
        f.write(f"SL hits:      {stats['sl']}\n")
        f.write(f"Timeout:      {stats['timeout']}  (exitTimeSeconds expired)\n")
        f.write(f"End of trade: {stats['eot']}  (after endOfTradingTimes)\n")
        f.write(f"Opposite:     {stats['opposite']}  (prediction flipped)\n")
        f.write(f"Entry expired:{stats['entry_expired']}  (entry order didn't fill)\n")
        f.write(f"Active now:   {len(active_positions)}\n\n")

        # Failed trades detail
        failed = [t for t in completed_trades if t['close_reason'] != 'TP']
        if failed:
            f.write(f"--- Failed Signals ({len(failed)}) ---\n\n")
            for i, t in enumerate(failed, 1):
                f.write(f"#{i} {t['seccode']} ({t['direction'].upper()}) "
                        f"conf={t['confidence']:.4f}\n")
                f.write(f"   Entry: {t['entry_time']}  Price: {t['entry_price']}\n")
                f.write(f"   Close: {t['close_time']}  Reason: {t['close_reason']}\n")
                f.write(f"   Phase: {t.get('phase', '?')}  "
                        f"Volume: {t.get('volume_contexts', '?')}  "
                        f"Margin: {t.get('margin', '?')}\n")
                f.write(f"   TP target: {t.get('exit_tp_price', '?')}  "
                        f"SL target: {t.get('exit_sl_price', '?')}\n")
                if t.get('close_detail'):
                    f.write(f"   Detail: {t['close_detail']}\n")
                f.write(f"\n")

        # TP trades summary
        tp_trades = [t for t in completed_trades if t['close_reason'] == 'TP']
        if tp_trades:
            f.write(f"\n--- TP Hits ({len(tp_trades)}) ---\n\n")
            for i, t in enumerate(tp_trades, 1):
                dur = t.get('duration_s', '?')
                f.write(f"#{i} {t['seccode']} ({t['direction'].upper()}) "
                        f"conf={t['confidence']:.4f} "
                        f"phase={t.get('phase', '?')} "
                        f"vol={t.get('volume_contexts', '?')} "
                        f"dur={dur}s\n")

        # Active positions
        if active_positions:
            f.write(f"\n--- Currently Open ({len(active_positions)}) ---\n\n")
            for eid, p in active_positions.items():
                f.write(f"  {p['seccode']} ({p['direction'].upper()}) "
                        f"entry_id={eid} conf={p['confidence']:.4f} "
                        f"since {p.get('entry_time', '?')}\n")

    # Also print summary to stdout
    failed_count = len([t for t in completed_trades if t['close_reason'] != 'TP'])
    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
          f"Entries={stats['total_entries']} "
          f"TP={stats['tp']} SL={stats['sl']} "
          f"Timeout={stats['timeout']} EOT={stats['eot']} "
          f"Failed={failed_count} "
          f"Open={len(active_positions)}")


def close_position(entry_id, reason, close_time_str='', detail=''):
    """Close a tracked position and record it."""
    pos = active_positions.pop(entry_id, None)
    if pos is None:
        return

    # Clean up exit maps
    if pos.get('exit_tp_id'):
        exit_to_entry.pop(pos['exit_tp_id'], None)
    if pos.get('exit_sl_id'):
        exit_to_entry.pop(pos['exit_sl_id'], None)

    trade = dict(pos)
    trade['close_reason'] = reason
    trade['close_time'] = close_time_str
    trade['close_detail'] = detail

    # Duration
    entry_ts = parse_timestamp(pos.get('entry_time', ''))
    close_ts = parse_timestamp(close_time_str) if close_time_str else None
    if entry_ts and close_ts:
        trade['duration_s'] = int((close_ts - entry_ts).total_seconds())

    completed_trades.append(trade)

    if reason == 'TP':
        stats['tp'] += 1
    elif reason == 'SL':
        stats['sl'] += 1
    elif reason == 'TIMEOUT':
        stats['timeout'] += 1
    elif reason == 'EOT':
        stats['eot'] += 1
    elif reason == 'OPPOSITE':
        stats['opposite'] += 1


def process_line(line):
    """Process a single log line."""
    ts_str = ''
    ts_m = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
    if ts_m:
        ts_str = ts_m.group(1)

    # ---- 1) predict() signal log ----
    m = re.search(
        r'PredictionModel\.predict \|seccode=(\w+) phase=(\w+), '
        r'signal=(\S+), confidence=([\d.]+), '
        r'volume_contexts=\[([^\]]*)\], '
        r'margin=([\d.]+), entryPrice=([\d.]+), exitPrice=([\d.]+)',
        line)
    if m:
        sec = m.group(1)
        last_signal[sec] = {
            'phase': m.group(2),
            'signal': m.group(3),
            'confidence': float(m.group(4)),
            'volume_contexts': m.group(5),
            'margin': m.group(6),
            'entry_price': m.group(7),
            'exit_price': m.group(8),
            'time': ts_str,
        }
        return

    # ---- 2) takePosition: entry sent ----
    m = re.search(
        r'Dolph\.takePosition \|seccode:(\w+) confidence=([\d.]+) '
        r'sending a\s+position=(\w+) seccode=\w+ quantity=(\d+) '
        r'entryPrice=([\d.]+) exitTakeProfit=([\d.]+) exitStopLoss=([\d.]+)',
        line)
    if m:
        sec = m.group(1)
        sig = last_signal.get(sec, {})
        # Store pending entry (no entry_id yet)
        # We'll match it when processPosition assigns entry_id
        pos_data = {
            'seccode': sec,
            'direction': m.group(3),
            'confidence': float(m.group(2)),
            'quantity': int(m.group(4)),
            'entry_price': m.group(5),
            'exit_tp_price': m.group(6),
            'exit_sl_price': m.group(7),
            'phase': sig.get('phase', '?'),
            'volume_contexts': sig.get('volume_contexts', '?'),
            'margin': sig.get('margin', '?'),
            'entry_time': ts_str,
            'exit_tp_id': None,
            'exit_sl_id': None,
        }
        # Store as pending for this seccode (will be finalized with entry_id)
        last_signal[sec + '_pending'] = pos_data
        return

    # ---- 3) processPosition: entry_id assigned ----
    m = re.search(
        r'position=(\w+) seccode=(\w+) quantity=\d+ '
        r'entryPrice=[\d.]+ exitTakeProfit=[\d.]+ exitStopLoss=[\d.]+ '
        r'entry_id=(\d+) exit_tp_id=(\S+) exit_sl_id=(\S+)',
        line)
    if m:
        sec = m.group(2)
        entry_id = m.group(3)
        tp_id = m.group(4) if m.group(4) != 'None' else None
        sl_id = m.group(5) if m.group(5) != 'None' else None

        entry_id_to_seccode[entry_id] = sec

        # If we have a pending position for this seccode, activate it
        pending_key = sec + '_pending'
        if pending_key in last_signal:
            pos = last_signal.pop(pending_key)
            pos['exit_tp_id'] = tp_id
            pos['exit_sl_id'] = sl_id
            active_positions[entry_id] = pos
            if tp_id:
                exit_to_entry[tp_id] = entry_id
            if sl_id:
                exit_to_entry[sl_id] = entry_id
        elif entry_id not in active_positions:
            # Position was created but we missed the takePosition line;
            # update existing if we have it
            if entry_id in active_positions:
                if tp_id:
                    active_positions[entry_id]['exit_tp_id'] = tp_id
                    exit_to_entry[tp_id] = entry_id
                if sl_id:
                    active_positions[entry_id]['exit_sl_id'] = sl_id
                    exit_to_entry[sl_id] = entry_id
        else:
            # Update exit IDs for existing position
            if tp_id and active_positions[entry_id].get('exit_tp_id') is None:
                active_positions[entry_id]['exit_tp_id'] = tp_id
                exit_to_entry[tp_id] = entry_id
            if sl_id and active_positions[entry_id].get('exit_sl_id') is None:
                active_positions[entry_id]['exit_sl_id'] = sl_id
                exit_to_entry[sl_id] = entry_id
        return

    # ---- 4) Order filled (entry) ----
    if 'Order is Filled-Monitored' in line:
        m = re.search(r'exitOrderRequested: (\d+)', line)
        if m:
            eid = m.group(1)
            stats['total_entries'] += 1
            # The position should already be in active_positions from step 3
        return

    # ---- 5) Exit order filled (TP or SL) ----
    if 'exitOrder:' in line and 'Filled' in line and 'deleted' in line:
        m = re.search(r'exitOrder: (\d+)', line)
        if m:
            exit_oid = m.group(1)
            entry_id = exit_to_entry.get(exit_oid)
            if entry_id and entry_id in active_positions:
                pos = active_positions[entry_id]
                if exit_oid == pos.get('exit_tp_id'):
                    close_position(entry_id, 'TP', ts_str)
                elif exit_oid == pos.get('exit_sl_id'):
                    close_position(entry_id, 'SL', ts_str)
                else:
                    # Unknown exit - try to determine from IB order type
                    close_position(entry_id, 'TP', ts_str, 'exit_id not matched')
        return

    # ---- 6) Position closed by removeMonitoredPositionByExit ----
    m = re.search(r'Position closed for (\w+) (?:after|in) (\d+)s', line)
    if m:
        sec = m.group(1)
        duration = int(m.group(2))
        # Find matching active position
        for eid, pos in list(active_positions.items()):
            if pos['seccode'] == sec:
                # Already handled by exit order fill; just update if still active
                # This line comes right after the exitOrder fill, so position
                # should already be closed. If still here, it means the exit
                # was from a close_order (market order), not TP/SL
                pass
        return

    # ---- 7) Forced close: cancelTimedoutExits ----
    if 'closing position for' in line:
        m = re.search(r'closing position for (\w+):\s*(.*)', line)
        if m:
            sec = m.group(1)
            detail = m.group(2).strip()

            # Determine close reason from detail
            if 'time2close exceeded' in detail:
                reason = 'EOT'
            elif 'after endOfTradingTimes' in detail:
                reason = 'EOT'
            elif 'open' in detail and 'min >' in detail:
                reason = 'TIMEOUT'
            elif 'prediction=' in detail:
                # Could be timeout or EOT with prediction flip
                if 'endOfTradingTimes' in detail:
                    reason = 'EOT'
                else:
                    reason = 'TIMEOUT'
            else:
                reason = 'TIMEOUT'

            # Find the active position for this seccode
            for eid, pos in list(active_positions.items()):
                if pos['seccode'] == sec:
                    close_position(eid, reason, ts_str, detail)
                    break
        return

    # ---- 8) closeExit confirmation (market order for forced close) ----
    if 'closeExit' in line and 'exit by market' in line:
        m = re.search(r'for (\w+), close_order_id=(\d+)', line)
        if m:
            sec = m.group(1)
            # Position already closed by step 7; this is the confirmation
        return

    # ---- 9) Entry expired (cancelTimedoutEntries) ----
    if 'cancelTimedoutEntries' in line and 'Cancelling Order' in line:
        stats['entry_expired'] += 1
        m = re.search(r'symbol=(\w+)', line)
        if m:
            sec = m.group(1)
            # Remove pending position if any
            pending_key = sec + '_pending'
            last_signal.pop(pending_key, None)
        return


def tail_file(filepath, from_end=True):
    """Generator that tails a file, yielding new lines."""
    with open(filepath, 'r') as f:
        if from_end:
            f.seek(0, 2)  # Go to end of file
        while True:
            line = f.readline()
            if line:
                yield line
            else:
                time.sleep(1)
                # Check if file was rotated
                try:
                    if os.stat(filepath).st_ino != os.fstat(f.fileno()).st_ino:
                        break
                except OSError:
                    break


# ---- Main ----
print(f"=== Dolph Signal Monitor ===")
print(f"Log: {LOG_FILE}")
print(f"Report: {REPORT_FILE}")
print(f"Monitoring for {HOURS} hours until {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Tailing from end of current log...")
print()

last_report = datetime.now()
report_interval = 300  # Write report every 5 minutes

try:
    for line in tail_file(LOG_FILE):
        if datetime.now() > end_time:
            print(f"\n=== Monitoring period ended ===")
            break

        process_line(line)

        # Periodic report
        now = datetime.now()
        if (now - last_report).total_seconds() > report_interval:
            write_report()
            last_report = now

except KeyboardInterrupt:
    print(f"\n=== Interrupted ===")

# Final report
write_report()
print(f"\nFinal report written to: {REPORT_FILE}")
print(f"Total entries: {stats['total_entries']}")
print(f"TP: {stats['tp']}  SL: {stats['sl']}  "
      f"Timeout: {stats['timeout']}  EOT: {stats['eot']}")
failed = [t for t in completed_trades if t['close_reason'] != 'TP']
print(f"Failed signals: {len(failed)}")
for t in failed:
    print(f"  {t['seccode']} ({t['direction']}) conf={t['confidence']:.4f} "
          f"phase={t.get('phase', '?')} vol=[{t.get('volume_contexts', '?')}] "
          f"reason={t['close_reason']}: {t.get('close_detail', '')}")
