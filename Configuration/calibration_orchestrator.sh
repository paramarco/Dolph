#!/bin/bash
# Dolph Calibration Orchestrator
# Dynamically partitions securities into batches of max BATCH_SIZE per instance,
# deploys and launches calibration instances with limited parallelism.
#
# Usage:
#   calibration_orchestrator.sh              # Run calibration
#   calibration_orchestrator.sh --dry-run    # Show plan without executing
#
# Instance assignment:
#   Americas → 1, 6, 7, ...
#   Europe   → 2, 8, 9, ...
#   Asia     → 3, 10, 11, ...
#   (4 = INIT_DB, 5 = OPERATIONAL — never touched)

set -e

BATCH_SIZE=12
MAX_PARALLEL=$(nproc 2>/dev/null || echo 2)  # auto-detect CPUs
DRY_RUN=false
DOLPH_DIR="/home/dolph_user/Dolph"
DATA_DIR="/home/dolph_user/data"
CONF_TEMPLATE="${DOLPH_DIR}/Configuration/Conf.py"
DEPLOY_SCRIPT="${DOLPH_DIR}/Configuration/deploy.sh"
STOP_SCRIPT="${DOLPH_DIR}/Configuration/stop.sh"

[ "$1" = "--dry-run" ] && DRY_RUN=true

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | ORCHESTRATOR | $*"; }

run_sql() {
    PGPASSWORD=dolph_password psql -h 127.0.0.1 -p 4713 -U dolph_user -d dolph_db -t -A -c "$1" 2>/dev/null
}

# Activate venv
source /opt/venv/bin/activate 2>/dev/null || true

log "=== Calibration Orchestrator START ==="
log "BATCH_SIZE=${BATCH_SIZE}, MAX_PARALLEL=${MAX_PARALLEL}, DRY_RUN=${DRY_RUN}"

# ============================================================
# Step 1: Query DB for securities per region
# ============================================================
declare -A TZ_PREFIXES=( ["Americas"]="America/" ["Europe"]="Europe/" ["Asia"]="Asia/" )
declare -A FIRST_INST=( ["Americas"]=1 ["Europe"]=2 ["Asia"]=3 )
declare -A REGION_CODES

for region in Americas Europe Asia; do
    prefix="${TZ_PREFIXES[$region]}"
    codes=$(run_sql "
        SELECT s.code FROM security s
        WHERE s.alg_parameters IS NOT NULL
          AND EXISTS (SELECT 1 FROM quote q WHERE q.security_id = s.id)
          AND s.timezone LIKE '${prefix}%'
        ORDER BY s.code;
    " | tr '\n' ' ')
    REGION_CODES[$region]="$codes"
    count=$(echo "$codes" | wc -w)
    log "${region}: ${count} securities"
done

# ============================================================
# Step 2: Partition into batches and assign instance numbers
# ============================================================
declare -a ALL_INSTANCES=()  # (instance_num region tz_prefix "code1 code2 ...")
next_inst=6

generate_conf() {
    local inst=$1 tz_filter=$2 codes_str=$3
    local conf_file="${DOLPH_DIR}/Configuration/Conf-${inst}.py"

    # Copy base template
    cp "${CONF_TEMPLATE}" "${conf_file}"

    # Add SECURITY_TZ_FILTER and SECURITY_CODES_FILTER after 'platform = tps.platform'
    local codes_py=$(echo "$codes_str" | tr ' ' '\n' | sed "s/.*/'&'/" | tr '\n' ',' | sed 's/,$//')
    sed -i "/^platform = tps.platform/a\\
\\
SECURITY_TZ_FILTER = '${tz_filter}'\\
SECURITY_CODES_FILTER = [${codes_py}]" "${conf_file}"

    log "Generated Conf-${inst}.py: tz=${tz_filter}, codes=[${codes_py}]"
}

for region in Americas Europe Asia; do
    prefix="${TZ_PREFIXES[$region]}"
    codes_arr=(${REGION_CODES[$region]})
    count=${#codes_arr[@]}
    [ "$count" -eq 0 ] && continue

    batches=$(( (count + BATCH_SIZE - 1) / BATCH_SIZE ))
    log "${region}: ${count} securities -> ${batches} batches"

    for ((b=0; b<batches; b++)); do
        start=$((b * BATCH_SIZE))
        batch_codes="${codes_arr[@]:$start:$BATCH_SIZE}"

        if [ "$b" -eq 0 ]; then
            inst=${FIRST_INST[$region]}
        else
            inst=$next_inst
            next_inst=$((next_inst + 1))
        fi

        batch_count=$(echo "$batch_codes" | wc -w)
        log "  Batch $((b+1))/${batches}: instance=${inst}, ${batch_count} securities: ${batch_codes}"

        if [ "$DRY_RUN" = false ]; then
            generate_conf "$inst" "$prefix" "$batch_codes"
        fi

        ALL_INSTANCES+=("${inst}:${region}:${prefix}:${batch_codes}")
    done
done

if [ "$DRY_RUN" = true ]; then
    log "=== DRY RUN COMPLETE (no instances deployed/launched) ==="
    exit 0
fi

# ============================================================
# Step 3: Stop any previous calibration instances (not 4 or 5)
# ============================================================
log "Stopping previous calibration instances..."
for entry in "${ALL_INSTANCES[@]}"; do
    inst=$(echo "$entry" | cut -d: -f1)
    bash "${STOP_SCRIPT}" "$inst" 2>/dev/null || true
done
sleep 2

# ============================================================
# Step 4: Deploy and launch with max MAX_PARALLEL at a time
# ============================================================
MIN_PASSES=1          # Wait for at least N complete calibration passes
POLL_INTERVAL=60      # Check logs every N seconds
MAX_WAIT=43200        # Safety timeout: 12 hours max per region

# Get the expected seccodes for an instance from its Conf file
get_instance_codes() {
    local inst=$1
    local conf="${DATA_DIR}/${inst}/Dolph/Configuration/Conf.py"
    python3 -c "
import ast, re
with open('${conf}') as f:
    for line in f:
        m = re.match(r'SECURITY_CODES_FILTER\s*=\s*(.+)', line)
        if m:
            print(' '.join(ast.literal_eval(m.group(1))))
            break
" 2>/dev/null
}

# Check if instance has completed MIN_PASSES calibration passes for all its seccodes
instance_calibration_done() {
    local inst=$1
    local logfile="${DATA_DIR}/${inst}/Dolph/log/Dolph.log"
    [ ! -f "$logfile" ] && return 1

    local codes
    codes=$(get_instance_codes "$inst")
    [ -z "$codes" ] && return 1

    for code in $codes; do
        local count
        count=$(grep -c "seccode=${code} calibration complete" "$logfile" 2>/dev/null || true)
        count=$((count + 0))  # force integer
        if [ "$count" -lt "$MIN_PASSES" ]; then
            return 1
        fi
    done
    return 0
}

# Stop an instance gracefully
stop_instance() {
    local inst=$1
    log "Stopping instance ${inst}..."
    bash "${STOP_SCRIPT}" "$inst" 2>/dev/null || true
    # Double check
    local pid
    pid=$(pgrep -f "python ${DATA_DIR}/${inst}/Dolph/DolphRobot.py" 2>/dev/null) || return 0
    sleep 2
    kill "$pid" 2>/dev/null || true
}

deploy_instance() {
    local inst=$1
    log "Deploying instance ${inst}..."
    bash "${DEPLOY_SCRIPT}" "$inst" --no-launch 2>&1 | sed "s/^/  [deploy-${inst}] /"
    # Copy the generated Conf-N.py AFTER deploy (deploy clones fresh from GitHub)
    local conf_src="${DOLPH_DIR}/Configuration/Conf-${inst}.py"
    local conf_dst="${DATA_DIR}/${inst}/Dolph/Configuration/Conf.py"
    if [ -f "$conf_src" ]; then
        cp "$conf_src" "$conf_dst"
        log "Copied Conf-${inst}.py -> instance ${inst} Conf.py"
    fi
    # Overlay local code changes (not yet on GitHub) over the cloned repo
    cp "${DOLPH_DIR}/DolphRobot.py" "${DATA_DIR}/${inst}/Dolph/DolphRobot.py"
    cp "${DOLPH_DIR}/DataManagement/DataServer.py" "${DATA_DIR}/${inst}/Dolph/DataManagement/DataServer.py"
    cp "${DOLPH_DIR}/Configuration/SecurityDefs.py" "${DATA_DIR}/${inst}/Dolph/Configuration/SecurityDefs.py"
    cp "${DOLPH_DIR}/DataVisualization/TrendViewer.py" "${DATA_DIR}/${inst}/Dolph/DataVisualization/TrendViewer.py"
    cp "${DOLPH_DIR}/TradingPlatforms/TradingPlatform.py" "${DATA_DIR}/${inst}/Dolph/TradingPlatforms/TradingPlatform.py" 2>/dev/null || true
    cp "${DOLPH_DIR}/PredictionModels/MinerviniClaude.py" "${DATA_DIR}/${inst}/Dolph/PredictionModels/MinerviniClaude.py" 2>/dev/null || true
    log "Overlaid local code changes -> instance ${inst}"
}

launch_instance() {
    local inst=$1
    log "Launching instance ${inst}..."
    cd "${DATA_DIR}/${inst}/Dolph"
    nohup python "${DATA_DIR}/${inst}/Dolph/DolphRobot.py" > /dev/null 2>&1 &
    log "Instance ${inst} launched (PID=$!)"
    sleep 3  # stagger launches
}

# ============================================================
# Step 4: Deploy all instances, then run as a pool
# ============================================================

# Collect ALL instance numbers in order
all_inst_nums=()
for entry in "${ALL_INSTANCES[@]}"; do
    all_inst_nums+=("$(echo "$entry" | cut -d: -f1)")
done

log "Deploying ${#all_inst_nums[@]} instances..."
for inst in "${all_inst_nums[@]}"; do
    deploy_instance "$inst"
done

# ============================================================
# Step 5: Pool executor — keep MAX_PARALLEL running at all times
# ============================================================
log "=== Pool executor: ${#all_inst_nums[@]} instances, MAX_PARALLEL=${MAX_PARALLEL} ==="

declare -A RUNNING_INSTS  # inst -> 1 (running)
declare -A DONE_INSTS     # inst -> 1 (completed)
queue_idx=0               # next instance to launch from all_inst_nums
start_time=$(date +%s)

# Fill pool up to MAX_PARALLEL
fill_pool() {
    while [ ${#RUNNING_INSTS[@]} -lt "$MAX_PARALLEL" ] && [ "$queue_idx" -lt "${#all_inst_nums[@]}" ]; do
        local inst="${all_inst_nums[$queue_idx]}"
        queue_idx=$((queue_idx + 1))

        # Skip if already done
        [ "${DONE_INSTS[$inst]:-}" = "1" ] && continue

        launch_instance "$inst"
        RUNNING_INSTS[$inst]=1
    done
}

fill_pool

log "Pool started: running=${!RUNNING_INSTS[*]}, queued=$((${#all_inst_nums[@]} - queue_idx))"

while [ ${#RUNNING_INSTS[@]} -gt 0 ] || [ "$queue_idx" -lt "${#all_inst_nums[@]}" ]; do

    for inst in "${!RUNNING_INSTS[@]}"; do
        # Check if process died
        if ! pgrep -f "python ${DATA_DIR}/${inst}/Dolph/DolphRobot.py" &>/dev/null; then
            log "Instance ${inst}: process exited"
            unset RUNNING_INSTS[$inst]
            DONE_INSTS[$inst]=1
            fill_pool
            continue
        fi

        # Check calibration completion
        if instance_calibration_done "$inst"; then
            log "Instance ${inst}: all seccodes completed >= ${MIN_PASSES} passes"
            stop_instance "$inst"
            unset RUNNING_INSTS[$inst]
            DONE_INSTS[$inst]=1
            fill_pool
        fi
    done

    # Safety timeout
    elapsed=$(( $(date +%s) - start_time ))
    if [ "$elapsed" -ge "$MAX_WAIT" ]; then
        log "WARNING: timeout ${MAX_WAIT}s reached, force-stopping all"
        for inst in "${!RUNNING_INSTS[@]}"; do
            stop_instance "$inst"
        done
        break
    fi

    # Status update every 10 polls (~10 min)
    if (( elapsed % (POLL_INTERVAL * 10) < POLL_INTERVAL )); then
        running_list="${!RUNNING_INSTS[*]}"
        done_count=${#DONE_INSTS[@]}
        total=${#all_inst_nums[@]}
        log "Status: running=[${running_list}], done=${done_count}/${total}, elapsed=$((elapsed/60))min"
    fi

    sleep "$POLL_INTERVAL"
done

log "=== All ${#all_inst_nums[@]} instances completed ==="

# ============================================================
# Step 5: Cleanup dynamic instances (>= 6)
# ============================================================
log "Cleaning up dynamic instances..."
for entry in "${ALL_INSTANCES[@]}"; do
    inst=$(echo "$entry" | cut -d: -f1)
    [ "$inst" -lt 6 ] && continue
    # Stop if still running
    bash "${STOP_SCRIPT}" "$inst" 2>/dev/null || true
    # Remove data directory
    if [ -d "${DATA_DIR}/${inst}" ]; then
        rm -rf "${DATA_DIR}/${inst}"
        log "Removed ${DATA_DIR}/${inst}"
    fi
    # Remove generated Conf file
    rm -f "${DOLPH_DIR}/Configuration/Conf-${inst}.py"
    log "Removed Conf-${inst}.py"
done

log "=== Calibration Orchestrator COMPLETE ==="
