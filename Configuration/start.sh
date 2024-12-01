#!/bin/bash
# -*- coding: utf-8 -*-

# Check if an instance number is provided as an argument
if [ -z "$1" ]; then
  echo "Error: No instance number provided."
  echo "Usage: $0 <instance_number>"
  exit 1
fi

# Activate Python virtual environment
echo "Activating Python virtual environment for dolph_user..."
. /opt/venv/bin/activate

instance="/home/dolph_user/data/$1"

# Find and kill the DolphRobot.py process
echo "Searching for the running DolphRobot.py process..."
pid=$(pgrep -f "python ${instance}/Dolph/DolphRobot.py")

if [ -z "$pid" ]; then
  echo "No running process of DolphRobot.py found."
else
  echo "Killing the process with PID: $pid"
  kill -2 "$pid"
  echo "Process killed successfully."
fi

cd ${instance}/Dolph

echo "Just wait 5 seconds ..."
sleep 5 

# Launch the application
echo "Launching DolphRobot.py..."
nohup python ${instance}/Dolph/DolphRobot.py > /dev/null 2>&1 &

