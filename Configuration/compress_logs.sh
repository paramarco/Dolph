#!/bin/bash
# Script to compress Dolph.log for each instance dynamically

# Calculate yesterday's date
YESTERDAY=$(date -d "yesterday" +"%Y-%m-%d")

# Base directory for instances
BASE_DIR="/home/dolph_user/data"

# Iterate through all instance directories in the base directory
for instance_dir in "$BASE_DIR"/*; do
    # Skip non-directories and the "lost+found" directory
    if [ -d "$instance_dir" ] && [ "$(basename "$instance_dir")" != "lost+found" ]; then
        LOG_FILE="${instance_dir}/Dolph/log/Dolph.log"
        ARCHIVE_FILE="${instance_dir}/Dolph/log/Dolph.log_${YESTERDAY}.tar.gz"

        # Check if the log file exists and compress it
        if [ -f "$LOG_FILE" ]; then
            echo "Compressing log for instance: $(basename "$instance_dir")"
            tar -czf "$ARCHIVE_FILE" "$LOG_FILE" && echo "Log compressed to $ARCHIVE_FILE"
            echo "log rotation ..." > "$LOG_FILE" && echo "Log rotation performed"
        else
            echo "Log file $LOG_FILE does not exist for instance: $(basename "$instance_dir")"
        fi
    else
        echo "Skipping directory: $(basename "$instance_dir")"
    fi
done
