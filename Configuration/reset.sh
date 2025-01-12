#!/bin/bash
# -*- coding: utf-8 -*-

# Activate Python virtual environment
echo "Activating Python virtual environment for dolph_user..."

# Define the base data directory
BASE_DIR="/home/dolph_user/data"

# Check if an instance number is provided as an argument
if [ -z "$1" ]; then
  echo "No instance number provided. "
else
  instance="$1"
  echo "resetting instance: $1"
  nohup ./stop.sh $instance && nohup ./deploy.sh $instance  ;

  tail -F data/$instance/Dolph/log/Dolph.log

fi
