#!/bin/bash
# Script to compress Dolph.log with a dynamic filename based on yesterday's date

# Calculate yesterday's date
YESTERDAY=$(date -d "yesterday" +"%Y-%m-%d")

# Define file paths
LOG_FILE="/home/dolph_user/data/Dolph/log/Dolph.log"
ARCHIVE_FILE="/home/dolph_user/data/Dolph/log/Dolph.log_${YESTERDAY}.tar.gz"

# Compress the log file
if [ -f "$LOG_FILE" ]; then
    tar -czf "$ARCHIVE_FILE" "$LOG_FILE" && echo "Log compressed to $ARCHIVE_FILE"
else
    echo "Log file $LOG_FILE does not exist!"
fi

