#!/usr/bin/env bash
# =============================================================================
# transfer.sh — Transfer backups between laptop and Docker container via VPS
#
# Run from laptop (afrodita):
#   ./transfer.sh              # Download: Container → Laptop
#   ./transfer.sh --upload     # Upload:   Laptop → Container
#
# Path: Container(dolph_user) → VPS(ubuntu) → Laptop(afrodita)
# =============================================================================
set -euo pipefail

VPS_USER=""
VPS_HOST=""
VPS_PORT=""
CONTAINER=""
REMOTE_BACKUPS="/home/dolph_user/backups"
LOCAL_BACKUPS="${HOME}/dolph_backups"
VPS_TMP="/tmp/dolph_backup_staging"
SSH_CMD="ssh -p ${VPS_PORT}"
SSH_TARGET="${VPS_USER}@${VPS_HOST}"

# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD: Container → VPS staging → rsync → Laptop
# ─────────────────────────────────────────────────────────────────────────────
do_download() {
    echo "=== DOWNLOAD: Container → Laptop ==="
    mkdir -p "${LOCAL_BACKUPS}"

    # 1. Stage from container to VPS temp dir
    echo "[1/3] Copying from container to VPS staging ..."
    ${SSH_CMD} ${SSH_TARGET} "\
        sudo rm -rf ${VPS_TMP} && \
        sudo docker cp ${CONTAINER}:${REMOTE_BACKUPS} ${VPS_TMP} && \
        sudo chown -R ${VPS_USER}:${VPS_USER} ${VPS_TMP}"

    # 2. rsync from VPS to laptop (incremental, compressed)
    echo "[2/3] Syncing VPS → Laptop ..."
    rsync -zaPe "${SSH_CMD}" "${SSH_TARGET}:${VPS_TMP}/" "${LOCAL_BACKUPS}/"

    # 3. Cleanup VPS staging
    echo "[3/3] Cleaning up VPS staging ..."
    ${SSH_CMD} ${SSH_TARGET} "sudo rm -rf ${VPS_TMP}"

    echo "=== DOWNLOAD complete: ${LOCAL_BACKUPS} ==="
    du -sh "${LOCAL_BACKUPS}"
}

# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD: Laptop → rsync → VPS staging → Container
# ─────────────────────────────────────────────────────────────────────────────
do_upload() {
    echo "=== UPLOAD: Laptop → Container ==="

    if [ ! -d "${LOCAL_BACKUPS}" ]; then
        echo "ERROR: Local backups not found at ${LOCAL_BACKUPS}"
        exit 1
    fi

    # 1. rsync from laptop to VPS staging (incremental, compressed)
    echo "[1/3] Syncing Laptop → VPS ..."
    ${SSH_CMD} ${SSH_TARGET} "sudo rm -rf ${VPS_TMP} && mkdir -p ${VPS_TMP}"
    rsync -zaPe "${SSH_CMD}" "${LOCAL_BACKUPS}/" "${SSH_TARGET}:${VPS_TMP}/"

    # 2. Copy from VPS staging into container
    echo "[2/3] Copying from VPS to container ..."
    ${SSH_CMD} ${SSH_TARGET} "\
        sudo docker cp ${VPS_TMP}/. ${CONTAINER}:${REMOTE_BACKUPS}/ && \
        sudo docker exec ${CONTAINER} chown -R dolph_user:dolph_user ${REMOTE_BACKUPS}"

    # 3. Cleanup VPS staging
    echo "[3/3] Cleaning up VPS staging ..."
    ${SSH_CMD} ${SSH_TARGET} "sudo rm -rf ${VPS_TMP}"

    echo "=== UPLOAD complete ==="
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
case "${1:-}" in
    --upload)
        do_upload
        ;;
    --help|-h)
        echo "Usage (run from laptop):"
        echo "  $0              Download backups: Container → Laptop"
        echo "  $0 --upload     Upload backups:   Laptop → Container"
        echo ""
        echo "Local dir:  ${LOCAL_BACKUPS}"
        echo "Remote dir: ${REMOTE_BACKUPS} (inside container ${CONTAINER})"
        ;;
    "")
        do_download
        ;;
    *)
        echo "Unknown option: $1. Use --help for usage."
        exit 1
        ;;
esac
