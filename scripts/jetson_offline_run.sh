#!/usr/bin/env bash
# Run the end-to-end pipeline on the Jetson with Wi-Fi switched off, then switch it back on.
# Needs root for nmcli; the pipeline itself runs as the normal user. Start it detached, so it
# survives the SSH session dropping when Wi-Fi goes off:
#
#   ssh -t jetson 'sudo -v && sudo -b setsid bash ~/kcp/scripts/jetson_offline_run.sh'
#
# Results: ~/kcp/e2e/out_offline/ and ~/kcp/e2e/offline.log
set -u
USER_NAME=ishantikku
KCP=/home/$USER_NAME/kcp
LOG=$KCP/e2e/offline.log
exec > "$LOG" 2>&1

wifi_on() { nmcli radio wifi on; echo "$(date -Is) wifi on"; }
trap wifi_on EXIT
(sleep 300 && nmcli radio wifi on) &  # safety net if anything hangs

sleep 5  # let the SSH command return before the network goes away
echo "$(date -Is) wifi off"
nmcli radio wifi off
sleep 5
echo "--- network check (expect no Wi-Fi address and no route) ---"
ip -br addr show
ping -c 1 -W 3 1.1.1.1 && echo "NETWORK STILL UP" || echo "ping failed: offline"
echo "--- pipeline ---"
cd "$KCP" || exit 1
sudo -u "$USER_NAME" env PYTHONPATH=src:pylib timeout 240 python3 -m kcp.edge.e2e run \
  e2e edge/s3d_selected graphs/tofu_stir_fry.instant_pot.v0.json e2e/out_offline
echo "pipeline exit $?"
