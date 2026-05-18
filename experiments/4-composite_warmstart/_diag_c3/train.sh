#!/bin/bash
#SBATCH -J sd_warm_diag_c3
#SBATCH -p gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH -o /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta/experiments/4-composite_warmstart/_diag_c3/logs/%x_%j.out
#SBATCH -e /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta/experiments/4-composite_warmstart/_diag_c3/logs/%x_%j.err

# Diagnostic run for the warm-start composite. Eager + no
# cuEquivariance modifier + Lightning detect_anomaly + NaNBatchLogger.
# Expected to reach ~epoch 15 in ~30-40 min, then stop on the first
# NaN train step with a traceback to the producing op and a saved
# `nan_batch.pt` file. 2h walltime budget is conservative.

set -euo pipefail

export LANG=en_US.utf8
export LC_ALL=en_US.utf8

export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

PROJECT_ROOT=/n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta
cd "$PROJECT_ROOT"

if [[ ! -x "$PROJECT_ROOT/.venv/bin/nequip-train" ]]; then
    echo "ERROR: .venv/bin/nequip-train not found." >&2
    echo "       Run: SPARSE_DELTA_UV_EXTRA=cuda bash software/install.sh" >&2
    exit 1
fi

mkdir -p experiments/4-composite_warmstart/_diag_c3/logs

.venv/bin/python - <<'PY' || { echo "[init] venv check failed" >&2; exit 2; }
import sys, torch
cuda = torch.version.cuda
dev = torch.cuda.device_count()
print(f"[init] torch={torch.__version__}  cuda={cuda}  device_count={dev}")
for i in range(dev):
    print(f"  device[{i}] = {torch.cuda.get_device_name(i)}")
# NaNBatchLogger import check
from sparse_delta_core.train import NaNBatchLogger
print(f"[init] NaNBatchLogger ready: {NaNBatchLogger}")
if cuda is None or dev == 0:
    sys.exit(1)
PY

LAST_CKPT="$PROJECT_ROOT/runs/4-composite_warmstart_diag_c3/last.ckpt"
if [[ -f "$LAST_CKPT" ]]; then
    echo "[chain] resuming from $LAST_CKPT"
    CKPT_OVERRIDE="+ckpt_path=$LAST_CKPT"
else
    echo "[chain] no last.ckpt found — starting fresh"
    CKPT_OVERRIDE=""
fi

srun .venv/bin/nequip-train \
    -cd "$PROJECT_ROOT/experiments/4-composite_warmstart/_diag_c3" \
    -cn config \
    $CKPT_OVERRIDE
