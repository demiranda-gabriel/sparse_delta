#!/bin/bash
#SBATCH -J allegro_L_corr
#SBATCH -A mat281
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:55:00
# Note: Frontier's bin policy caps 1-node jobs at 2h walltime (any QoS).
# Config max_time=1d is enforced lightning-side; to use it fully, chain
# multiple submissions via --dependency=afterok and last.ckpt restart.
#SBATCH -o /lustre/orion/mat281/scratch/demirand/projects/sparse_delta/experiments/2-allegro_L_correction/logs/%x_%j.out
#SBATCH -e /lustre/orion/mat281/scratch/demirand/projects/sparse_delta/experiments/2-allegro_L_correction/logs/%x_%j.err

set -euo pipefail

# === Frontier modules: ROCm 6.2.4 to match torch==2.5.1+rocm6.2 ===
module load amd-mixed/6.2.4

# Common locale
export LANG=en_US.utf8
export LC_ALL=en_US.utf8

# Friendly fail if the venv is missing
PROJECT_ROOT=/lustre/orion/mat281/scratch/demirand/projects/sparse_delta
cd "$PROJECT_ROOT"
if [[ ! -x "$PROJECT_ROOT/.venv/bin/nequip-train" ]]; then
    echo "ERROR: .venv/bin/nequip-train not found." >&2
    echo "       On Frontier, build the venv with: 'uv sync --extra rocm62'." >&2
    exit 1
fi

mkdir -p experiments/2-allegro_L_correction/logs

# Resolve to a friendly device count log line
.venv/bin/python - <<'PY'
import torch
print(f"[init] torch={torch.__version__}  hip={torch.version.hip}  device_count={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  device[{i}] = {torch.cuda.get_device_name(i)}")
PY

# Single-run experiment: config.yaml lives in this directory. nequip-train uses
# Hydra; -cd specifies the config dir, -cn the config file (without .yaml).
.venv/bin/nequip-train \
    -cd "$PROJECT_ROOT/experiments/2-allegro_L_correction" \
    -cn config
