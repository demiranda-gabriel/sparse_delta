#!/bin/bash
#SBATCH -J sd_staged_smoke
#SBATCH -p gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH -o /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta/experiments/5-composite_staged/_smoke/logs/%x_%j.out
#SBATCH -e /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta/experiments/5-composite_staged/_smoke/logs/%x_%j.err

# Smoke for 5-composite_staged. Runs stages 1 → 2 → 3 back-to-back in
# one job (each capped at 3 epochs × 30 train batches). Validates:
#   - M0 standalone builder + ForceStressOutput.
#   - extract_model_weights.py produces a loadable .pt.
#   - LoadWeightsCallback + FreezeByNameCallback wire up correctly.
#   - Warm-start composite with l_max_M0=1, l_max_M1=2 + adapter +
#     M1 edge SH builds and trains end-to-end.
#   - gate_mode=constant and gate_mode=learned both produce sensible
#     λ outputs.

set -euo pipefail

export LANG=en_US.utf8
export LC_ALL=en_US.utf8
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

PROJECT_ROOT=/n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta
cd "$PROJECT_ROOT"

SMOKE=$PROJECT_ROOT/experiments/5-composite_staged/_smoke
RUN_ROOT=$PROJECT_ROOT/runs/5-composite_staged_smoke
mkdir -p "$SMOKE/logs"

.venv/bin/python - <<'PY' || { echo "[init] venv check failed" >&2; exit 2; }
import sys, torch
if torch.version.cuda is None or torch.cuda.device_count() == 0:
    sys.exit(1)
print(f"[init] torch={torch.__version__} cuda={torch.version.cuda}")
PY

echo
echo "============================================================"
echo "Stage 1: M0 standalone"
echo "============================================================"
.venv/bin/nequip-train -cd "$SMOKE/configs" -cn stage1_m0

STAGE1_BEST="$RUN_ROOT/5-composite_staged_smoke-stage1_m0/best.ckpt"
STAGE1_WEIGHTS_MIRROR="$RUN_ROOT/stage1_m0/stage1_weights.pt"
echo "[chain] extracting stage 1 weights → $STAGE1_WEIGHTS_MIRROR"
mkdir -p "$(dirname "$STAGE1_WEIGHTS_MIRROR")"
.venv/bin/python "$PROJECT_ROOT/scripts/extract_model_weights.py" \
    "$STAGE1_BEST" "$STAGE1_WEIGHTS_MIRROR"

echo
echo "============================================================"
echo "Stage 2: warm-start composite (constant gate), M0 frozen"
echo "============================================================"
.venv/bin/nequip-train -cd "$SMOKE/configs" -cn stage2_m1

STAGE2_BEST="$RUN_ROOT/5-composite_staged_smoke-stage2_m1/best.ckpt"
STAGE2_WEIGHTS_MIRROR="$RUN_ROOT/stage2_m1/stage2_weights.pt"
echo "[chain] extracting stage 2 weights → $STAGE2_WEIGHTS_MIRROR"
mkdir -p "$(dirname "$STAGE2_WEIGHTS_MIRROR")"
.venv/bin/python "$PROJECT_ROOT/scripts/extract_model_weights.py" \
    "$STAGE2_BEST" "$STAGE2_WEIGHTS_MIRROR"

echo
echo "============================================================"
echo "Stage 3: warm-start composite (learned gate), M0+M1+adapter frozen"
echo "============================================================"
.venv/bin/nequip-train -cd "$SMOKE/configs" -cn stage3_gate

echo
echo "[smoke] all three stages completed."
