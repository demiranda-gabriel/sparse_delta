#!/bin/bash
#SBATCH -J composite_pA
#SBATCH -A mat281
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:55:00
# Frontier 1-node walltime cap is 2h regardless of QoS; max_time=1d in
# config.yaml is enforced lightning-side, full multi-day runs need
# chained submissions via --dependency=afterok and last.ckpt.
#SBATCH -o /lustre/orion/mat281/scratch/demirand/projects/sparse_delta/experiments/3-composite_phase_A/logs/%x_%j.out
#SBATCH -e /lustre/orion/mat281/scratch/demirand/projects/sparse_delta/experiments/3-composite_phase_A/logs/%x_%j.err

set -euo pipefail

module load amd-mixed/6.4.2

export LANG=en_US.utf8
export LC_ALL=en_US.utf8

# Frontier nodes expose 8 GCDs even when --gpus-per-node=1. Pin to GCD 0
# to match Trainer.devices=1; bump in lockstep for multi-GPU.
export ROCR_VISIBLE_DEVICES=0

PROJECT_ROOT=/lustre/orion/mat281/scratch/demirand/projects/sparse_delta
cd "$PROJECT_ROOT"
if [[ ! -x "$PROJECT_ROOT/.venv/bin/nequip-train" ]]; then
    echo "ERROR: .venv/bin/nequip-train not found." >&2
    echo "       On Frontier, build the venv with: 'uv sync --extra rocm64'." >&2
    exit 1
fi

mkdir -p experiments/3-composite_phase_A/logs

# Fail-fast venv check (per sparse_delta_feedback_verify_venv memory):
# torch must be a ROCm build and at least one GPU must be visible.
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

.venv/bin/nequip-train \
    -cd "$PROJECT_ROOT/experiments/3-composite_phase_A" \
    -cn config
