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

# Delete old log file
echo "Deleting old log file..."
rm -rf $data/Dolph/log/Dolph.log

# Remove the old Dolph directory
echo "Removing the old Dolph directory..."
rm -rf $data/Dolph/

# Clone the repository
echo "Cloning the Dolph repository from GitHub..."
git clone https://github.com/paramarco/Dolph.git $data/Dolph
cd $data/Dolph
mkdir log

# Launch the application
echo "Launching DolphRobot.py..."
nohup python $data/Dolph/DolphRobot.py > /dev/null 2>&1 &
