#!/bin/bash
#SBATCH -J allegro_L_corr
#SBATCH -A mat281
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:55:00
# Note: Frontier's bin policy caps 1-node jobs at 2h walltime (any QoS).
# Config max_time=1d is enforced lightning-side; to use it fully, chain
# multiple submissions via --dependency=afterok and last.ckpt restart.
#SBATCH -o /lustre/orion/mat281/scratch/demirand/projects/sparse_delta/experiments/2-allegro_L_correction/logs/%x_%j.out
#SBATCH -e /lustre/orion/mat281/scratch/demirand/projects/sparse_delta/experiments/2-allegro_L_correction/logs/%x_%j.err

set -euo pipefail

# === Frontier modules: ROCm 6.4.2 to match torch==2.9.1+rocm6.4 ===
# (rocm6.2 wheels are pinned to torch 2.5.x, which is below the >=2.6
#  required by nequip / nequip-allegro for compile_mode: compile (PT2).)
module load amd-mixed/6.4.2

# Common locale
export LANG=en_US.utf8
export LC_ALL=en_US.utf8

# Frontier compute nodes carry 8 GCDs (4× MI250X). Even with
# --gpus-per-node=1 the node is allocated exclusively, so the OS still
# exposes all 8 GCDs to the process and torch.cuda.device_count() returns
# 8. Lightning then auto-selects 8 devices and crashes against
# --ntasks-per-node=1. Pin visibility to GCD 0 (matched by devices=1 in
# the trainer config). For multi-GPU training, set this to 0-7 and bump
# ntasks-per-node + Trainer.devices in lock-step.
export ROCR_VISIBLE_DEVICES=0

# Friendly fail if the venv is missing
PROJECT_ROOT=/lustre/orion/mat281/scratch/demirand/projects/sparse_delta
cd "$PROJECT_ROOT"
if [[ ! -x "$PROJECT_ROOT/.venv/bin/nequip-train" ]]; then
    echo "ERROR: .venv/bin/nequip-train not found." >&2
    echo "       On Frontier, build the venv with: 'uv sync --extra rocm64'." >&2
    exit 1
fi

mkdir -p experiments/2-allegro_L_correction/logs

# Fail-fast venv check: torch must be a ROCm build and at least one GPU
# must be visible. A CUDA-wheel torch slips in if `uv sync` was run with
# the wrong extra; without this, the job would only fail several seconds
# into trainer init, wasting a queue slot.
.venv/bin/python - <<'PY' || { echo "[init] venv check failed — re-run 'uv sync --extra rocm64'" >&2; exit 2; }
import sys, torch
hip = torch.version.hip
dev = torch.cuda.device_count()
print(f"[init] torch={torch.__version__}  hip={hip}  device_count={dev}")
for i in range(dev):
    print(f"  device[{i}] = {torch.cuda.get_device_name(i)}")
if hip is None or dev == 0:
    sys.exit(1)
PY

# Single-run experiment: config.yaml lives in this directory. nequip-train uses
# Hydra; -cd specifies the config dir, -cn the config file (without .yaml).
.venv/bin/nequip-train \
    -cd "$PROJECT_ROOT/experiments/2-allegro_L_correction" \
    -cn config
