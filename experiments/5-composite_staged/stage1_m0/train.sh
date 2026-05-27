#!/bin/bash
#SBATCH -J sd_staged_s1
#SBATCH -p gpu,seas_gpu,kozinsky_gpu,gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH -o /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta/experiments/5-composite_staged/stage1_m0/logs/%x_%j.out
#SBATCH -e /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta/experiments/5-composite_staged/stage1_m0/logs/%x_%j.err

# Stage 1 of the staged composite training: M0 standalone.
# After training, scripts/extract_model_weights.py is invoked to
# materialise stage1_weights.pt for stage 2's LoadWeightsCallback.

set -euo pipefail

export LANG=en_US.utf8
export LC_ALL=en_US.utf8
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

PROJECT_ROOT=/n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta
cd "$PROJECT_ROOT"

if [[ ! -x "$PROJECT_ROOT/.venv/bin/nequip-train" ]]; then
    echo "ERROR: .venv/bin/nequip-train not found." >&2
    echo "       Run: cd software && bash install.sh" >&2
    exit 1
fi

EXP=experiments/5-composite_staged
STAGE_DIR=$EXP/stage1_m0
mkdir -p "$STAGE_DIR/logs"

# Fail-fast venv check.
.venv/bin/python - <<'PY' || { echo "[init] venv check failed" >&2; exit 2; }
import sys, torch
cuda = torch.version.cuda
dev = torch.cuda.device_count()
print(f"[init] torch={torch.__version__}  cuda={cuda}  device_count={dev}")
if cuda is None or dev == 0:
    sys.exit(1)
PY

# Resume from last.ckpt if a prior submission of THIS stage was
# requeued / killed mid-run.
LAST_CKPT="$PROJECT_ROOT/runs/5-composite_staged/5-composite_staged-stage1_m0/last.ckpt"
if [[ -f "$LAST_CKPT" ]]; then
    echo "[chain] resuming from $LAST_CKPT"
    CKPT_OVERRIDE="+ckpt_path=$LAST_CKPT"
else
    echo "[chain] no last.ckpt — starting fresh"
    CKPT_OVERRIDE=""
fi

srun .venv/bin/nequip-train \
    -cd "$PROJECT_ROOT/$EXP/configs" \
    -cn stage1_m0 \
    $CKPT_OVERRIDE

# === Post-training: extract weights for stage 2 ===
# Only runs if nequip-train exited 0 (set -e above). Writes
# stage1_weights.pt next to the best.ckpt so the path is stable
# regardless of which login node / partition stage 2 lands on.
RUN_DIR="$PROJECT_ROOT/runs/5-composite_staged/5-composite_staged-stage1_m0"
BEST_CKPT="$RUN_DIR/best.ckpt"
WEIGHTS_OUT="$RUN_DIR/stage1_weights.pt"

if [[ -f "$BEST_CKPT" ]]; then
    echo "[chain] extracting model weights → $WEIGHTS_OUT"
    .venv/bin/python "$PROJECT_ROOT/scripts/extract_model_weights.py" \
        "$BEST_CKPT" "$WEIGHTS_OUT"

    # Mirror into the canonical stage_root location stage 2 expects.
    MIRROR="$PROJECT_ROOT/runs/5-composite_staged/stage1_m0/stage1_weights.pt"
    mkdir -p "$(dirname "$MIRROR")"
    cp -f "$WEIGHTS_OUT" "$MIRROR"
    echo "[chain] mirrored to $MIRROR"
else
    echo "[chain] WARNING: $BEST_CKPT not found; stage 2 will fail." >&2
    exit 3
fi
