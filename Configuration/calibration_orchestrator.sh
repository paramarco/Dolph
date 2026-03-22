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
MAX_PARALLEL=2
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
wait_for_instance() {
    local inst=$1
    local pid
    pid=$(pgrep -f "python ${DATA_DIR}/${inst}/Dolph/DolphRobot.py" 2>/dev/null) || return 0
    log "Waiting for instance ${inst} (PID=${pid}) to finish..."
    while kill -0 "$pid" 2>/dev/null; do
        sleep 30
    done
    log "Instance ${inst} finished."
}

# Process each timezone group sequentially
for region in Americas Europe Asia; do
    prefix="${TZ_PREFIXES[$region]}"

    # Collect instances for this region
    region_instances=()
    for entry in "${ALL_INSTANCES[@]}"; do
        entry_region=$(echo "$entry" | cut -d: -f2)
        [ "$entry_region" = "$region" ] && region_instances+=("$(echo "$entry" | cut -d: -f1)")
    done

    [ ${#region_instances[@]} -eq 0 ] && continue
    log "=== Processing ${region}: ${#region_instances[@]} instances ==="

    # Deploy all instances for this region (--no-launch: orchestrator controls launch)
    for inst in "${region_instances[@]}"; do
        log "Deploying instance ${inst}..."
        bash "${DEPLOY_SCRIPT}" "$inst" --no-launch 2>&1 | sed "s/^/  [deploy-${inst}] /"
        # Copy the generated Conf-N.py AFTER deploy (deploy clones fresh from GitHub)
        conf_src="${DOLPH_DIR}/Configuration/Conf-${inst}.py"
        conf_dst="${DATA_DIR}/${inst}/Dolph/Configuration/Conf.py"
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
        log "Overlaid local code changes -> instance ${inst}"
    done

    # Launch in pairs of MAX_PARALLEL, wait for each pair to complete
    idx=0
    while [ $idx -lt ${#region_instances[@]} ]; do
        running_pids=()
        running_insts=()

        for ((p=0; p<MAX_PARALLEL && idx<${#region_instances[@]}; p++, idx++)); do
            inst="${region_instances[$idx]}"
            log "Launching instance ${inst}..."
            cd "${DATA_DIR}/${inst}/Dolph"
            nohup python "${DATA_DIR}/${inst}/Dolph/DolphRobot.py" > /dev/null 2>&1 &
            lpid=$!
            running_pids+=("$lpid")
            running_insts+=("$inst")
            log "Instance ${inst} launched (PID=${lpid})"
            sleep 3  # stagger launches
        done

        # Wait for all in this batch to complete
        for i in "${!running_pids[@]}"; do
            inst="${running_insts[$i]}"
            pid="${running_pids[$i]}"
            log "Waiting for instance ${inst} (PID=${pid})..."
            wait "$pid" 2>/dev/null || true
            log "Instance ${inst} completed."
        done
    done

    log "=== ${region} complete ==="
done

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
