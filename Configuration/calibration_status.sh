#!/bin/bash
# Dolph Calibration Capacity Report
# Shows security counts per region, required batches, and resource estimates.

BATCH_SIZE=12
RAM_PER_INSTANCE=300  # MB
RAM_DB_OS_OPER=720    # MB (DB ~200 + OPER ~220 + OS ~300)
#MAX_PARALLEL=$(nproc 2>/dev/null || echo 2)  # auto-detect CPUs
MAX_PARALLEL=3

run_sql() {
    PGPASSWORD=dolph_password psql -h 127.0.0.1 -p 4713 -U dolph_user -d dolph_db -t -A -c "$1" 2>/dev/null
}

echo "=== Dolph Calibration Capacity Report ==="
echo ""

# Get counts per region
count_am=$(run_sql "SELECT COUNT(*) FROM security s WHERE s.alg_parameters IS NOT NULL AND EXISTS (SELECT 1 FROM quote q WHERE q.security_id = s.id) AND s.timezone LIKE 'America/%'")
count_eu=$(run_sql "SELECT COUNT(*) FROM security s WHERE s.alg_parameters IS NOT NULL AND EXISTS (SELECT 1 FROM quote q WHERE q.security_id = s.id) AND s.timezone LIKE 'Europe/%'")
#count_as=$(run_sql "SELECT COUNT(*) FROM security s WHERE s.alg_parameters IS NOT NULL AND EXISTS (SELECT 1 FROM quote q WHERE q.security_id = s.id) AND s.timezone LIKE 'Asia/%'")
count_as=0

count_am=${count_am:-0}
count_eu=${count_eu:-0}
count_as=${count_as:-0}

batches_am=$(( (count_am + BATCH_SIZE - 1) / BATCH_SIZE ))
batches_eu=$(( (count_eu + BATCH_SIZE - 1) / BATCH_SIZE ))
#batches_as=$(( (count_as + BATCH_SIZE - 1) / BATCH_SIZE ))
batches_as=0
#total_sec=$((count_am + count_eu + count_as))
total_sec=$((count_am + count_eu))
#total_batches=$((batches_am + batches_eu + batches_as))
total_batches=$((batches_am + batches_eu))

# Build instance lists
inst_am="1"
inst_eu="2"
#inst_as="3"
inst_as="-"
next=6
for ((i=1; i<batches_am; i++)); do inst_am="$inst_am, $next"; next=$((next+1)); done
for ((i=1; i<batches_eu; i++)); do inst_eu="$inst_eu, $next"; next=$((next+1)); done
#for ((i=1; i<batches_as; i++)); do inst_as="$inst_as, $next"; next=$((next+1)); done

printf "%-12s %10s %8s   %-15s\n" "Region" "Securities" "Batches" "Instances"
printf "%-12s %10s %8s   %-15s\n" "----------" "----------" "-------" "---------------"
printf "%-12s %10d %8d   %-15s\n" "Americas" "$count_am" "$batches_am" "$inst_am"
printf "%-12s %10d %8d   %-15s\n" "Europe" "$count_eu" "$batches_eu" "$inst_eu"
#printf "%-12s %10d %8d   %-15s\n" "Asia" "$count_as" "$batches_as" "$inst_as"
printf "%-12s %10s %8s   %-15s\n" "Asia" "(skip)" "(skip)" "-"
printf "%-12s %10s %8s   %-15s\n" "----------" "----------" "-------" "---------------"
printf "%-12s %10d %8d\n" "TOTAL" "$total_sec" "$total_batches"

echo ""
echo "=== Resource Estimates ==="
peak_ram=$((MAX_PARALLEL * RAM_PER_INSTANCE + RAM_DB_OS_OPER))
total_ram=$(awk '/MemTotal/ {printf "%.0f", $2/1024}' /proc/meminfo)
cpus=$(nproc)

echo "RAM per calibration instance:  ~${RAM_PER_INSTANCE} MB"
echo "Max parallel instances:        ${MAX_PARALLEL} (${cpus} CPUs)"
echo "Peak RAM (${MAX_PARALLEL} parallel + DB/OPER/OS): ${peak_ram} MB / ${total_ram} MB total"
echo "Batch size:                    ${BATCH_SIZE} securities/instance"

echo ""
echo "=== Running Instances ==="
for pidfile in /proc/[0-9]*/cmdline; do
    [ ! -f "$pidfile" ] && continue
    pid=$(echo "$pidfile" | cut -d/ -f3)
    cmd=$(tr '\0' ' ' < "$pidfile" 2>/dev/null) || continue
    if echo "$cmd" | grep -q "DolphRobot.py"; then
        inst=$(echo "$cmd" | grep -oP 'data/\K[0-9]+')
        rss=$(awk '/VmRSS/ {print int($2/1024)}' /proc/$pid/status 2>/dev/null)
        echo "  Instance ${inst:-?}  PID=${pid}  RSS=${rss:-?} MB"
    fi
done

echo ""
echo "=== Estimated Calibration Time ==="
time_min=$((total_sec * 25 / MAX_PARALLEL))
time_h=$((time_min / 60))
time_m=$((time_min % 60))
echo "First pass (~25 min/security, ${MAX_PARALLEL} parallel): ~${time_h}h ${time_m}m"
echo "Subsequent passes (converged): ~${total_batches} seconds"
