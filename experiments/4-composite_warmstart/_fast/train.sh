#!/bin/bash
#SBATCH -J sd_warmstart_fast
#SBATCH -p gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta/experiments/4-composite_warmstart/_fast/logs/%x_%j.out
#SBATCH -e /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta/experiments/4-composite_warmstart/_fast/logs/%x_%j.err

# Fast-path full training on gpu_test (12h cap, A100 MIG 20GB slice).
# Same model + dataset as ../config.yaml; just a smaller GPU and
# shorter walltime. Submitted because the cluster-wide 8apower2
# maintenance reservation holds kozinsky_gpu at half capacity until
# 2026-05-23 and the public gpu/seas_gpu pools have 1500+ jobs ahead.
#
# gpu_test schedules essentially instantly (16 running / 0 pending at
# the time of submit). Trade-off: we cut at 11h lightning cap; the
# composite at this size should still see >50 epochs in that window
# (smoke timing: epoch ~6-10s steady-state on the same MIG slice).

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

mkdir -p experiments/4-composite_warmstart/_fast/logs

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

LAST_CKPT="$PROJECT_ROOT/runs/4-composite_warmstart_fast_v2/last.ckpt"
if [[ -f "$LAST_CKPT" ]]; then
    echo "[chain] resuming from $LAST_CKPT"
    CKPT_OVERRIDE="+ckpt_path=$LAST_CKPT"
else
    echo "[chain] no last.ckpt found — starting fresh"
    CKPT_OVERRIDE=""
fi

srun .venv/bin/nequip-train \
    -cd "$PROJECT_ROOT/experiments/4-composite_warmstart/_fast" \
    -cn config \
    $CKPT_OVERRIDE
