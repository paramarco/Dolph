#!/usr/bin/env bash
# =============================================================================
# back-up.sh — Backup and restore /home/dolph_user + PostgreSQL database
#
# Usage:
#   ./back-up.sh              # Create backup
#   ./back-up.sh --restore    # Restore from latest backup
# =============================================================================
set -euo pipefail

BACKUP_DIR="/home/dolph_user/backups"
HOME_DIR="/home/dolph_user"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="backup_${TIMESTAMP}"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"

# DB connection
DB_HOST="127.0.0.1"
DB_PORT="4713"
DB_USER="dolph_user"
DB_NAME="dolph_db"
export PGPASSWORD="dolph_password"

# ─────────────────────────────────────────────────────────────────────────────
# BACKUP
# ─────────────────────────────────────────────────────────────────────────────
do_backup() {
    echo "=== BACKUP started at $(date) ==="
    mkdir -p "${BACKUP_PATH}"

    # 1. Database dump (all tables, schema + data)
    echo "[1/3] Dumping database ${DB_NAME} ..."
    pg_dump -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" \
        --format=custom --compress=6 \
        "${DB_NAME}" > "${BACKUP_PATH}/db.dump"
    echo "      Database dump: $(du -sh "${BACKUP_PATH}/db.dump" | cut -f1)"

    # 2. Home directory (excluding backups dir itself and large non-essential files)
    echo "[2/3] Archiving ${HOME_DIR} ..."
    tar czf "${BACKUP_PATH}/home.tar.gz" \
        --exclude="${BACKUP_DIR}" \
        --exclude="${HOME_DIR}/.cache" \
        --exclude="${HOME_DIR}/.claude" \
        -C / "home/dolph_user"
    echo "      Home archive: $(du -sh "${BACKUP_PATH}/home.tar.gz" | cut -f1)"

    # 3. Write manifest
    echo "[3/3] Writing manifest ..."
    cat > "${BACKUP_PATH}/manifest.txt" <<EOF
Backup: ${BACKUP_NAME}
Date:   $(date -Iseconds)
Host:   $(hostname)
DB:     ${DB_NAME} @ ${DB_HOST}:${DB_PORT}
Home:   ${HOME_DIR}
DB size: $(du -sh "${BACKUP_PATH}/db.dump" | cut -f1)
Home size: $(du -sh "${BACKUP_PATH}/home.tar.gz" | cut -f1)
EOF

    # Symlink latest
    ln -snf "${BACKUP_PATH}" "${BACKUP_DIR}/latest"

    echo "=== BACKUP complete: ${BACKUP_PATH} ==="
    echo "    Total size: $(du -sh "${BACKUP_PATH}" | cut -f1)"
}

# ─────────────────────────────────────────────────────────────────────────────
# RESTORE
# ─────────────────────────────────────────────────────────────────────────────
do_restore() {
    # Find backup to restore
    local restore_from="${1:-${BACKUP_DIR}/latest}"

    if [ ! -d "${restore_from}" ]; then
        echo "ERROR: Backup not found at ${restore_from}"
        echo "Available backups:"
        ls -1d "${BACKUP_DIR}"/backup_* 2>/dev/null || echo "  (none)"
        exit 1
    fi

    # Resolve symlink
    restore_from=$(readlink -f "${restore_from}")

    echo "=== RESTORE from ${restore_from} ==="
    cat "${restore_from}/manifest.txt" 2>/dev/null || true
    echo ""

    # Safety: confirm
    read -rp "This will OVERWRITE the database and home directory. Continue? [y/N] " confirm
    if [[ "${confirm}" != "y" && "${confirm}" != "Y" ]]; then
        echo "Aborted."
        exit 0
    fi

    # 1. Stop all Dolph instances
    echo "[1/4] Stopping Dolph instances ..."
    for inst in 1 2 3 4 5; do
        pid_file="${HOME_DIR}/data/${inst}/Dolph/Dolph.pid"
        if [ -f "${pid_file}" ]; then
            pid=$(cat "${pid_file}")
            if kill -0 "${pid}" 2>/dev/null; then
                echo "      Stopping instance ${inst} (PID ${pid}) ..."
                kill "${pid}" 2>/dev/null || true
            fi
        fi
    done
    sleep 2

    # 2. Restore database (create DB if it doesn't exist)
    echo "[2/4] Restoring database ${DB_NAME} ..."
    if ! psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" \
            -d "${DB_NAME}" -c "SELECT 1;" &>/dev/null; then
        echo "      Database ${DB_NAME} does not exist, creating ..."
        psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" \
            -d postgres -c "CREATE DATABASE ${DB_NAME};" 2>/dev/null || true
    fi
    pg_restore -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" \
        --dbname="${DB_NAME}" --clean --if-exists \
        --no-owner --no-privileges \
        "${restore_from}/db.dump"
    echo "      Database restored."

    # 3. Restore home directory
    echo "[3/4] Restoring ${HOME_DIR} ..."
    tar xzf "${restore_from}/home.tar.gz" -C /
    echo "      Home directory restored."

    # 4. Verify
    echo "[4/4] Verifying ..."
    local table_count
    table_count=$(psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" \
        -t -c "SELECT COUNT(*) FROM security;" 2>/dev/null | tr -d ' ')
    echo "      Securities in DB: ${table_count}"
    echo "      Home dir size: $(du -sh "${HOME_DIR}" --exclude="${BACKUP_DIR}" | cut -f1)"

    echo "=== RESTORE complete ==="
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
case "${1:-}" in
    --restore)
        do_restore "${2:-}"
        ;;
    --list)
        echo "Available backups:"
        for d in "${BACKUP_DIR}"/backup_*; do
            [ -d "$d" ] || continue
            size=$(du -sh "$d" | cut -f1)
            name=$(basename "$d")
            latest=""
            [ "$(readlink -f "${BACKUP_DIR}/latest" 2>/dev/null)" = "$(readlink -f "$d")" ] && latest=" (latest)"
            echo "  ${name}  ${size}${latest}"
        done
        ;;
    --help|-h)
        echo "Usage:"
        echo "  $0              Create a new backup"
        echo "  $0 --restore    Restore from latest backup"
        echo "  $0 --restore <path>  Restore from specific backup"
        echo "  $0 --list       List available backups"
        echo "  $0 --help       Show this help"
        ;;
    "")
        do_backup
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use --help for usage."
        exit 1
        ;;
esac
