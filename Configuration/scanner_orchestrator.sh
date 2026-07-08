#!/bin/bash
# Dolph Scanner Orchestrator
# Scans US and EU markets for Three Power Moves opportunities.
# Runs every 3 hours via cron, sends email alerts for opportunities.
#
# Usage:
#   scanner_orchestrator.sh              # Full scan with email
#   scanner_orchestrator.sh --no-email   # Scan without email
#   scanner_orchestrator.sh --dry-run    # Show plan without scanning

set -e

DOLPH_DIR="/home/dolph_user/Dolph"
VENV="/opt/venv"
DRY_RUN=false
NO_EMAIL=""

for arg in "$@"; do
    [ "$arg" = "--dry-run" ] && DRY_RUN=true
    [ "$arg" = "--no-email" ] && NO_EMAIL="--no-email"
done

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | SCANNER | $*"; }

# Activate venv explicitly with full path
export PATH="${VENV}/bin:${PATH}"
export VIRTUAL_ENV="${VENV}"
export PYTHONPATH="${DOLPH_DIR}:${PYTHONPATH}"

# Verify Python works
PYTHON="${VENV}/bin/python3"
if [ ! -x "$PYTHON" ]; then
    PYTHON=$(which python3)
fi

log "=== Scanner Orchestrator START ==="
log "Python: $PYTHON"
log "Working dir: ${DOLPH_DIR}"

if [ "$DRY_RUN" = true ]; then
    log "DRY RUN — would scan ~7,500 stocks (all US + STOXX 600 EU + IB hot stocks)"
    log "=== DRY RUN COMPLETE ==="
    exit 0
fi

# ============================================================
# Run full market scan with IB scanner for microcaps
# ============================================================
log "Scanning full market (all US + STOXX 600 EU + IB hot stocks)..."
log "Estimated time: ~2-3 hours for ~7,500+ stocks"

cd "${DOLPH_DIR}"
$PYTHON ScanEngine/scanner.py \
    --full \
    --ib \
    --min-power-moves 2 \
    ${NO_EMAIL} \
    2>&1 | tee -a /home/dolph_user/data/scanner_detail.log

SCAN_EXIT=$?

# ============================================================
# Summary
# ============================================================
log "=== Scanner Orchestrator COMPLETE (exit=$SCAN_EXIT) ==="

# Query DB for scan summary
SUMMARY=$(PGPASSWORD=dolph_password psql -h 127.0.0.1 -p 4713 -U dolph_user -d dolph_db -t -A -c "
SELECT alert_type, COUNT(*)
FROM scan_opportunities
WHERE scan_ts >= NOW() - interval '3 hours'
GROUP BY alert_type
ORDER BY COUNT(*) DESC;
" 2>/dev/null || echo "DB query failed")

log "Scan results:"
echo "$SUMMARY" | while IFS='|' read -r alert_type count; do
    [ -n "$alert_type" ] && log "  ${alert_type}: ${count}"
done

log "=== DONE ==="
