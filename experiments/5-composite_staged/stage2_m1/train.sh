#!/bin/bash
#SBATCH -J sd_staged_s2
#SBATCH -p kozinsky_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH -o /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta/experiments/5-composite_staged/stage2_m1/logs/%x_%j.out
#SBATCH -e /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta/experiments/5-composite_staged/stage2_m1/logs/%x_%j.err

# Stage 2 of the staged composite training: warm-start composite with
# constant gate (λ=1), M0 frozen, adapter + M1 trainable. Loads M0
# weights via LoadWeightsCallback from stage 1's extracted .pt.

set -euo pipefail

export LANG=en_US.utf8
export LC_ALL=en_US.utf8
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

PROJECT_ROOT=/n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta
cd "$PROJECT_ROOT"

EXP=experiments/5-composite_staged
STAGE_DIR=$EXP/stage2_m1
mkdir -p "$STAGE_DIR/logs"

WEIGHTS_IN="$PROJECT_ROOT/runs/5-composite_staged/stage1_m0/stage1_weights.pt"
if [[ ! -f "$WEIGHTS_IN" ]]; then
    echo "ERROR: stage 1 weights not found at $WEIGHTS_IN" >&2
    echo "       Did stage 1 complete successfully?" >&2
    exit 4
fi

.venv/bin/python - <<'PY' || { echo "[init] venv check failed" >&2; exit 2; }
import sys, torch
if torch.version.cuda is None or torch.cuda.device_count() == 0:
    sys.exit(1)
PY

LAST_CKPT="$PROJECT_ROOT/runs/5-composite_staged/5-composite_staged-stage2_m1/last.ckpt"
if [[ -f "$LAST_CKPT" ]]; then
    echo "[chain] resuming from $LAST_CKPT"
    CKPT_OVERRIDE="+ckpt_path=$LAST_CKPT"
else
    echo "[chain] no last.ckpt — starting fresh; loading M0 from stage 1"
    CKPT_OVERRIDE=""
fi

srun .venv/bin/nequip-train \
    -cd "$PROJECT_ROOT/$EXP/configs" \
    -cn stage2_m1 \
    $CKPT_OVERRIDE

# Extract weights for stage 3.
RUN_DIR="$PROJECT_ROOT/runs/5-composite_staged/5-composite_staged-stage2_m1"
BEST_CKPT="$RUN_DIR/best.ckpt"
WEIGHTS_OUT="$RUN_DIR/stage2_weights.pt"

if [[ -f "$BEST_CKPT" ]]; then
    echo "[chain] extracting model weights → $WEIGHTS_OUT"
    .venv/bin/python "$PROJECT_ROOT/scripts/extract_model_weights.py" \
        "$BEST_CKPT" "$WEIGHTS_OUT"
    MIRROR="$PROJECT_ROOT/runs/5-composite_staged/stage2_m1/stage2_weights.pt"
    mkdir -p "$(dirname "$MIRROR")"
    cp -f "$WEIGHTS_OUT" "$MIRROR"
    echo "[chain] mirrored to $MIRROR"
else
    echo "[chain] WARNING: $BEST_CKPT not found; stage 3 will fail." >&2
    exit 3
fi
