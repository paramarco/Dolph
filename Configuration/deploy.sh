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
  sleep 7
fi

# Preserve logs before removing old Dolph directory
if [ -d "${instance}/Dolph/log" ]; then
  echo "Preserving logs..."
  mv ${instance}/Dolph/log ${instance}/Dolph_log_backup
fi

# Remove the old Dolph directory
echo "Removing the old Dolph directory..."
rm -rf ${instance}/Dolph/

# Clone the repository
echo "Cloning the Dolph repository from GitHub..."
git clone https://github.com/paramarco/Dolph.git ${instance}/Dolph
cd ${instance}/Dolph

# Restore preserved logs or create empty log directory
if [ -d "${instance}/Dolph_log_backup" ]; then
  echo "Restoring logs..."
  mv ${instance}/Dolph_log_backup ${instance}/Dolph/log
else
  mkdir log
fi

# Replacing TradingPlatfomSettings-$1.py of the container by the TradingPlatfomSettings.py in the instance ...
echo "Replacing TradingPlatfomSettings-$1.py of the container by the TradingPlatfomSettings.py in the instance ..."
cp /home/dolph_user/TradingPlatfomSettings-$1.py ${instance}/Dolph/Configuration/TradingPlatfomSettings.py

# Replacing Conf-X.py of the instance in the repository by the Conf.py for the instance ...
echo "Replacing Conf-X.py of the instance in the repository by the Conf.py for the instance  ..."
cp ${instance}/Dolph/Configuration/Conf-$1.py ${instance}/Dolph/Configuration/Conf.py


# Launch the application
echo "Launching DolphRobot.py..."
nohup python ${instance}/Dolph/DolphRobot.py > /dev/null 2>&1 &
