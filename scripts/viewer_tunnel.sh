#!/usr/bin/env bash
# Forward the Jetson viewer to http://localhost:8090 on this Mac. Reconnects when Wi-Fi drops SSH.
# Usage: scripts/viewer_tunnel.sh [ssh host]   (default: jetson-usb, the USB-C cable link)
# Stop with Ctrl-C (twice if it is between attempts).
HOST="${1:-jetson-usb}"
while true; do
  ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3 \
    -o ConnectTimeout=5 -L 8090:localhost:8090 "$HOST"
  echo "$(date +%T) tunnel dropped; reconnecting"
  sleep 2
done
