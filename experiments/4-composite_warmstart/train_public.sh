#!/bin/bash
#SBATCH -J sd_warmstart_pub
#SBATCH -p gpu,seas_gpu,gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH -o /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta/experiments/4-composite_warmstart/logs/%x_%j.out
#SBATCH -e /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta/experiments/4-composite_warmstart/logs/%x_%j.err

# Public-pool fallback for the warm-start composite. Runs the same
# config.yaml as train.sh; only the SLURM headers differ:
#
#   - Partitions: `gpu` (A100 80GB) / `seas_gpu` (A100 or H200) /
#     `gpu_h200` (H200 141GB). Whichever opens first.
#   - GRES syntax: `--gpus=1` (the FASRC submit filter on these
#     partitions rejects typed-gres like `gpu:nvidia_a100-sxm4-80gb:1`
#     and demands the canonical `--gpus=N` form).
#
# Use side-by-side with train.sh (lab partition) when the lab queue is
# deep; cancel the loser once one starts. `kozinsky_gpu` typically
# gives our lab the best long-run fairshare; this script is for cases
# like a 2-node partition with one node in maintenance + a queue of
# higher-priority lab jobs ahead.

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

mkdir -p experiments/4-composite_warmstart/logs

.venv/bin/python - <<'PY' || { echo "[init] venv check failed" >&2; exit 2; }
import sys, torch
cuda = torch.version.cuda
dev = torch.cuda.device_count()
print(f"[init] torch={torch.__version__}  cuda={cuda}  device_count={dev}")
for i in range(dev):
    print(f"  device[{i}] = {torch.cuda.get_device_name(i)}")
try:
    import cuequivariance, cuequivariance_torch
    print(f"[init] cuequivariance={cuequivariance.__version__} "
          f"cuequivariance_torch={cuequivariance_torch.__version__}")
except ImportError as e:
    print(f"[init] cuequivariance NOT available: {e}", file=sys.stderr)
    sys.exit(1)
if cuda is None or dev == 0:
    sys.exit(1)
PY

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
