#!/bin/bash
# -*- coding: utf-8 -*-

# Activate Python virtual environment
echo "Activating Python virtual environment for dolph_user..."
source /opt/venv/bin/activate

# Define the base data directory
BASE_DIR="/home/dolph_user/data"

# Check if an instance number is provided as an argument
if [ -z "$1" ]; then
  echo "No instance number provided. Starting all instances in $BASE_DIR..."
  for instance_dir in "$BASE_DIR"/*; do
    # Skip non-directories and the lost+found directory
    if [ -d "$instance_dir" ] && [ "$(basename "$instance_dir")" != "lost+found" ]; then
      instance=$(basename "$instance_dir")
      echo "Starting instance: $instance"

      cd "$instance_dir/Dolph" || { echo "Directory $instance_dir/Dolph not found. Skipping."; continue; }

      # Launch the application
      echo "Launching DolphRobot.py for instance $instance..."
      nohup python "$instance_dir/Dolph/DolphRobot.py" > /dev/null 2>&1 &
      echo "Instance $instance started."
    fi
  done
else
  instance="$BASE_DIR/$1"
  echo "Starting instance: $1"

  cd "$instance/Dolph" || { echo "Directory $instance/Dolph not found. Exiting."; exit 1; }

  # Launch the application
  echo "Launching DolphRobot.py..."
  nohup python "$instance/Dolph/DolphRobot.py" > /dev/null 2>&1 &
  echo "Instance $1 started."
fi
