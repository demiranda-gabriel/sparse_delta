#!/bin/bash
#SBATCH -J sd_warmstart_smoke
#SBATCH -p gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH -o /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta/experiments/4-composite_warmstart/_smoke/logs/%x_%j.out
#SBATCH -e /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta/experiments/4-composite_warmstart/_smoke/logs/%x_%j.err

# Tiny smoke run for the warm-start composite. Validates:
#   - data pipeline (cameron_plus_bulkpt splits)
#   - composite forward + backward end-to-end
#   - torch.compile (compile_mode=compile) doesn't crash
#   - cuEquivariance modifier swaps in cleanly + kernel runs
#
# 30-min wall (15 min lightning cap + compile/init overhead). 20 GB
# MIG slice is plenty for the tiny config.

set -euo pipefail

export LANG=en_US.utf8
export LC_ALL=en_US.utf8

export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

PROJECT_ROOT=/n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta
cd "$PROJECT_ROOT"

if [[ ! -x "$PROJECT_ROOT/.venv/bin/nequip-train" ]]; then
    echo "ERROR: .venv/bin/nequip-train not found." >&2
    echo "       Run: SPARSE_DELTA_UV_EXTRA=cuda cd software && bash install.sh" >&2
    exit 1
fi

mkdir -p experiments/4-composite_warmstart/_smoke/logs

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

srun .venv/bin/nequip-train \
    -cd "$PROJECT_ROOT/experiments/4-composite_warmstart/_smoke" \
    -cn config
