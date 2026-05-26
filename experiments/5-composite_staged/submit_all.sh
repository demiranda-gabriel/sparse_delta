#!/bin/bash
# submit_all.sh — chain the three staged training jobs via SLURM
# --dependency=afterok. Each stage's train.sh runs the corresponding
# nequip-train + extracts weights for the next stage on the same node
# (so paths land consistently in the run dir).

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[submit_all] stage 1: M0 standalone"
JOB1_OUT=$(sbatch "$EXP_DIR/stage1_m0/train.sh")
JOB1_ID=$(echo "$JOB1_OUT" | awk '{print $NF}')
echo "  → job $JOB1_ID"

echo "[submit_all] stage 2: warm-start composite, M0 frozen (afterok:$JOB1_ID)"
JOB2_OUT=$(sbatch --dependency=afterok:"$JOB1_ID" "$EXP_DIR/stage2_m1/train.sh")
JOB2_ID=$(echo "$JOB2_OUT" | awk '{print $NF}')
echo "  → job $JOB2_ID"

echo "[submit_all] stage 3: learned gate, M0+M1+adapter frozen (afterok:$JOB2_ID)"
JOB3_OUT=$(sbatch --dependency=afterok:"$JOB2_ID" "$EXP_DIR/stage3_gate/train.sh")
JOB3_ID=$(echo "$JOB3_OUT" | awk '{print $NF}')
echo "  → job $JOB3_ID"

echo
echo "[submit_all] chain submitted:"
echo "  stage 1 → $JOB1_ID"
echo "  stage 2 → $JOB2_ID  (afterok stage 1)"
echo "  stage 3 → $JOB3_ID  (afterok stage 2)"
