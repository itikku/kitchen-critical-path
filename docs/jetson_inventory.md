# Jetson Inventory

Status on 2026-11-01: **blocked — setup pending.** The Jetson setup waits for the USB flash drive and the DisplayPort-to-HDMI adapter. See `../../jetson-setup/README.md`.

Known hardware: Jetson Orin Nano Super Developer Kit (8 GB), Patriot P320 256 GB NVMe SSD.

After setup, run these read-only commands over SSH and paste the output here:

```sh
cat /etc/nv_tegra_release
nvpmodel -q
free -h
df -h /
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
ls /dev/video*
```
