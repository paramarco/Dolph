#!/bin/bash
# -*- coding: utf-8 -*-

# Activate Python virtual environment
echo "Activating Python virtual environment for dolph_user..."
source /opt/venv/bin/activate

# Define the base data directory
BASE_DIR="/home/dolph_user/data"

# Check if an instance number is provided as an argument
if [ -z "$1" ]; then
  echo "No instance number provided. Stopping all instances in $BASE_DIR..."
  for instance_dir in "$BASE_DIR"/*; do
    # Skip non-directories and the lost+found directory
    if [ -d "$instance_dir" ] && [ "$(basename "$instance_dir")" != "lost+found" ]; then
      instance=$(basename "$instance_dir")
      echo "Stopping instance: $instance"

      # Find and kill the DolphRobot.py process
      pid=$(pgrep -f "python $instance_dir/Dolph/DolphRobot.py")
      if [ -z "$pid" ]; then
        echo "No running process of DolphRobot.py found for instance $instance."
      else
        echo "Killing the process with PID: $pid"
        kill -2 "$pid"
        echo "Process killed successfully for instance $instance."
      fi
    fi
  done
else
  instance="$BASE_DIR/$1"
  echo "Stopping instance: $1"

  # Find and kill the DolphRobot.py process
  pid=$(pgrep -f "python $instance/Dolph/DolphRobot.py")
  if [ -z "$pid" ]; then
    echo "No running process of DolphRobot.py found for instance $1."
  else
    echo "Killing the process with PID: $pid"
    kill -2 "$pid"
    echo "Process killed successfully for instance $1."
  fi
fi

