#!/bin/bash

# Get the current host IP address
HOST_IP=$(ifconfig | grep "inet " | grep -Fv 127.0.0.1 | awk '{print $2}')
echo "Current host IP: $HOST_IP"

# Create hostfile directory (if it doesn't exist)
mkdir -p ./hostfile

# Write IP address to hostfile
echo "$HOST_IP slots=8" > ./hostfile/hostfile
echo "IP address written to ./hostfile/hostfile"

# Run deepspeed training
deepspeed train_rm.py --deepspeed
