#!/bin/bash
# Extract operational logs and prepare them for correlate_trades.py
#
# Usage:
#   ./extract_trades.sh              # extract + run analysis
#   ./extract_trades.sh --extract    # extract only (inspect manually)
#
# Output: /tmp/dolph_logs/all_logs.log (combined log for correlate_trades.py)

LOGDIR="/home/dolph_user/data/5/Dolph/log"
TMPDIR="/tmp/dolph_logs"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$TMPDIR"
rm -f "$TMPDIR"/log_*.log "$TMPDIR/all_logs.log"

# Extract all compressed logs
for f in "$LOGDIR"/Dolph.log_*.tar.gz; do
    [ ! -f "$f" ] && continue
    date=$(echo "$f" | grep -oP '\d{4}-\d{2}-\d{2}')
    tar xzf "$f" -C "$TMPDIR" 2>/dev/null
    if [ -f "$TMPDIR/home/dolph_user/data/5/Dolph/log/Dolph.log" ]; then
        mv "$TMPDIR/home/dolph_user/data/5/Dolph/log/Dolph.log" "$TMPDIR/log_$date.log"
    fi
done

# Include current (uncompressed) log
if [ -f "$LOGDIR/Dolph.log" ]; then
    cp "$LOGDIR/Dolph.log" "$TMPDIR/log_current.log"
fi

# Combine all logs chronologically
cat "$TMPDIR"/log_*.log > "$TMPDIR/all_logs.log"

echo "=== Logs extracted to $TMPDIR/all_logs.log ==="
wc -l "$TMPDIR/all_logs.log"

if [ "$1" = "--extract" ]; then
    exit 0
fi

# Run correlation analysis
echo ""
python3 "$SCRIPT_DIR/correlate_trades.py"
