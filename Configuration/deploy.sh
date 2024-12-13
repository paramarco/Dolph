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
mkdir $instance

# Deployment for the instance number $1
echo "Deployment for the instance in: ${instance}"

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

# Delete old log file
echo "Deleting old log file..."
rm -rf ${instance}/Dolph/log/Dolph.log

# Remove the old Dolph directory
echo "Removing the old Dolph directory..."
rm -rf ${instance}/Dolph/

# Clone the repository
echo "Cloning the Dolph repository from GitHub..."
git clone https://github.com/paramarco/Dolph.git ${instance}/Dolph
cd ${instance}/Dolph
mkdir log

# Replacing template for TradingPlatfomSettings ...
echo "Replacing template for TradingPlatfomSettings ..."
cp /home/dolph_user/TradingPlatfomSettings-$1.py ${instance}/Dolph/Configuration/TradingPlatfomSettings.py

# Replacing template for Conf.py ...
echo "Replacing template for Conf.py  ..."
cp /home/dolph_user/Conf-$1.py ${instance}/Dolph/Configuration/Conf.py


# Launch the application
echo "Launching DolphRobot.py..."
nohup python ${instance}/Dolph/DolphRobot.py > /dev/null 2>&1 &
