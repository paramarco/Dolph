#!/bin/bash
# -*- coding: utf-8 -*-

# Activate Python virtual environment
echo "Activating Python virtual environment for dolph_user..."
. /opt/venv/bin/activate

data="/home/dolph_user/data"

# Find and kill the DolphRobot.py process
echo "Searching for the running DolphRobot.py process..."
pid=$(pgrep -f "python ${data}/Dolph/DolphRobot.py")

if [ -z "$pid" ]; then
  echo "No running process of DolphRobot.py found."
else
  echo "Killing the process with PID: $pid"
  kill -2 "$pid"
  echo "Process killed successfully."
fi

cd $data/Dolph
mkdir log

# Launch the application
echo "Launching DolphRobot.py..."
nohup python $data/Dolph/DolphRobot.py > /dev/null 2>&1 &
