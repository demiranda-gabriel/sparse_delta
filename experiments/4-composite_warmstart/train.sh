#!/bin/bash
#SBATCH -J sd_warmstart
#SBATCH -p kozinsky_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH -o /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta/experiments/4-composite_warmstart/logs/%x_%j.out
#SBATCH -e /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta/experiments/4-composite_warmstart/logs/%x_%j.err

# 4-composite_warmstart: joint training of M0 + gate + M1 with M1
# architecturally warm-started from M0's pre-final TP output.
#
# Partition `kozinsky_gpu` chosen over `gpu` for priority on lab-owned
# nodes (7-day cap vs 3-day, both fine for this 500-epoch run with
# max_time=3d). Falls back to `gpu` (#SBATCH -p gpu, same A100 80GB
# nodes) or `gpu_h200` (141GB, in case M1 backward OOMs) if the lab
# queue is saturated.

set -euo pipefail

export LANG=en_US.utf8
export LC_ALL=en_US.utf8

# Allocator tuning. The composite holds M0 + gate + M1 activations
# through the single backward pass — fragmentation risk is higher than
# for a standalone Allegro.
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

PROJECT_ROOT=/n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta
cd "$PROJECT_ROOT"

if [[ ! -x "$PROJECT_ROOT/.venv/bin/nequip-train" ]]; then
    echo "ERROR: .venv/bin/nequip-train not found." >&2
    echo "       Run: cd software && bash install.sh" >&2
    exit 1
fi

mkdir -p experiments/4-composite_warmstart/logs

# Fail-fast venv check: torch must see at least one GPU.
.venv/bin/python - <<'PY' || { echo "[init] venv check failed — re-run 'bash software/install.sh'" >&2; exit 2; }
import sys, torch
cuda = torch.version.cuda
dev = torch.cuda.device_count()
print(f"[init] torch={torch.__version__}  cuda={cuda}  device_count={dev}")
for i in range(dev):
    print(f"  device[{i}] = {torch.cuda.get_device_name(i)}")
if cuda is None or dev == 0:
    sys.exit(1)
PY

# Auto-resume from last.ckpt if a prior link of a chain wrote one.
# `kozinsky_gpu` allows 7-day jobs so chaining is rarely needed for a
# 3-day cap, but keep the logic so a manual resubmit picks up cleanly.
LAST_CKPT="$PROJECT_ROOT/runs/4-composite_warmstart/last.ckpt"
if [[ -f "$LAST_CKPT" ]]; then
    echo "[chain] resuming from $LAST_CKPT"
    CKPT_OVERRIDE="+ckpt_path=$LAST_CKPT"
else
    echo "[chain] no last.ckpt found — starting fresh"
    CKPT_OVERRIDE=""
fi

srun .venv/bin/nequip-train \
    -cd "$PROJECT_ROOT/experiments/4-composite_warmstart" \
    -cn config \
    $CKPT_OVERRIDE
