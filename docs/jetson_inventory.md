# Jetson Inventory

Status: **done** (Phase 0.7). Captured 2026-11-05 with read-only commands; nothing was installed to produce it.

| Item | Value |
|---|---|
| Device | Jetson Orin Nano Super Developer Kit (8 GB) |
| JetPack | 7.2.1 (`nvidia-jetpack 7.2.1-b49`) |
| Jetson Linux | R39.2.1, kernel `6.8.12-1021-tegra` |
| Power mode | MAXN_SUPER (nvpmodel mode 2) |
| RAM | 7.3 GiB, shared by CPU and GPU; ~5.6 GiB available at idle |
| Swap | none |
| Root disk | Patriot P320 256 GB NVMe, 189 GB free |
| CUDA | 13.2.2 |
| cuDNN | 9.20.0 |
| TensorRT | 10.16.2 (`trtexec` at `/usr/src/tensorrt/bin/trtexec`; Python bindings via `python3-libnvinfer`) |
| Python | 3.12.3, NumPy 1.26.4; no `pip3`, no PyTorch |
| Camera | none connected |
| Docker | 29.1.3 with `nvidia-container-toolkit` 1.19.1 |
| Idle power and temperature | ~5.2 W input; ~49 °C |

## What this means for the project

- **Inference runs on TensorRT**, which JetPack includes. PyTorch is not needed on the device: the model ships as ONNX and TensorRT compiles it.
- **No swap:** anything over ~5.6 GB of RAM is killed. The S3D encoder and head are 34 MB on disk.
- **No camera yet:** on-device runs use recorded clips until a camera is chosen (Phase 3).
- **Wi-Fi only:** long jobs run under `nohup`, because one SSH session already dropped during setup.

## Raw checks

```text
$ cat /etc/nv_tegra_release
# R39 (release), REVISION: 2.1, GCID: 46758480, BOARD: generic, EABI: aarch64, DATE: Fri Aug  7 05:54:22 AM UTC 2026

$ nvpmodel -q
NV Power Mode: MAXN_SUPER
2

$ free -h
               total        used        free      shared  buff/cache   available
Mem:           7.3Gi       1.7Gi       4.4Gi       5.7Mi       1.4Gi       5.6Gi
Swap:             0B          0B          0B

$ df -h /
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p1  233G   32G  189G  15% /

$ python3 -c "import torch"
ModuleNotFoundError: No module named 'torch'

$ ls /dev/video*
ls: cannot access '/dev/video*': No such file or directory

$ python3 --version
Python 3.12.3
```
