#!/bin/bash
#SBATCH -J allegro_L_corr
#SBATCH -A mat281
#SBATCH -p batch
#SBATCH -q debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
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

# Use all 8 GCDs per node via DDP. SLURM allocates the node exclusively
# (4× MI250X = 8 GCDs); --ntasks-per-node=8 spawns one PyTorch process
# per GCD, --cpus-per-task=7 fits 8×7=56 of the 64 cores. Each rank
# gets a distinct device via Lightning's SLURMEnvironment auto-detect
# from SLURM_LOCALID. ROCR_VISIBLE_DEVICES is not set — each task sees
# all 8 GCDs but Lightning binds to rank-specific cuda:LOCAL_RANK.

# Frontier compute nodes have NO outbound internet — wandb's API host
# `api.wandb.ai` is unreachable and `wandb.init` blocks indefinitely in
# its retry loop, freezing the trainer at the LightningLogger init hook.
# Force offline mode; metrics buffer to runs/<run>/wandb/, then sync
# from a login node with:
#   wandb sync runs/2-allegro_L_correction/wandb/run-*  (or just */wandb/)
# The run resumes by id on subsequent chained jobs as long as the hydra
# run dir is reused.
export WANDB_MODE=offline

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
# srun spawns 8 ranks (one per GCD); Lightning's SLURMEnvironment plugin
# auto-detects this and wires up DDP.
srun .venv/bin/nequip-train \
    -cd "$PROJECT_ROOT/experiments/2-allegro_L_correction" \
    -cn config
