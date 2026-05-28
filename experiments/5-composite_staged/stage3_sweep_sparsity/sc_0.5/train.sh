#!/bin/bash
#SBATCH -J sd_sw_sc0.5
#SBATCH -p gpu,seas_gpu,kozinsky_gpu,gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH -o /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta/experiments/5-composite_staged/stage3_sweep_sparsity/sc_0.5/logs/%x_%j.out
#SBATCH -e /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta/experiments/5-composite_staged/stage3_sweep_sparsity/sc_0.5/logs/%x_%j.err

# stage 3 sweep, sparsity_coeff=0.5.
# Loads stage 2 weights via LoadWeightsCallback; freezes M0+M1+adapter;
# trains EdgeFeaturesToNodeLambda gate only.

set -euo pipefail
export LANG=en_US.utf8
export LC_ALL=en_US.utf8
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

PROJECT_ROOT=/n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta
cd "$PROJECT_ROOT"

SWEEP_DIR=$PROJECT_ROOT/experiments/5-composite_staged/stage3_sweep_sparsity
mkdir -p "$SWEEP_DIR/sc_0.5/logs"

WEIGHTS_IN="$PROJECT_ROOT/runs/5-composite_staged/stage2_m1/stage2_weights.pt"
if [[ ! -f "$WEIGHTS_IN" ]]; then
    echo "ERROR: stage 2 weights not found at $WEIGHTS_IN" >&2
    exit 4
fi

.venv/bin/python - <<'PY' || { echo "[init] venv check failed" >&2; exit 2; }
import sys, torch
if torch.version.cuda is None or torch.cuda.device_count() == 0:
    sys.exit(1)
PY

LAST_CKPT="$PROJECT_ROOT/runs/5-composite_staged/5-composite_staged_sweep-sc_0.5/last.ckpt"
if [[ -f "$LAST_CKPT" ]]; then
    echo "[chain] resuming from $LAST_CKPT"
    CKPT_OVERRIDE="+ckpt_path=$LAST_CKPT"
else
    echo "[chain] no last.ckpt -- starting fresh"
    CKPT_OVERRIDE=""
fi

srun .venv/bin/nequip-train \
    -cd "$SWEEP_DIR/configs" \
    -cn sc_0.5 \
    $CKPT_OVERRIDE
