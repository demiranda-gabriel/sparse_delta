#!/bin/bash
#SBATCH -J sd_analyze_gate
#SBATCH -p gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_a100_3g.20gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH -o /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta/experiments/5-composite_staged/results/analyze_gate_%j.out
#SBATCH -e /n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta/experiments/5-composite_staged/results/analyze_gate_%j.err

# Run analyze_gate.py on a GPU node — the stage 3 checkpoint was
# trained with cuEquivariance kernels that require libcuda.so.1, which
# the login node lacks.

set -euo pipefail
export PYTORCH_ALLOC_CONF=expandable_segments:True

PROJECT_ROOT=/n/netscratch/kozinsky_lab/Lab/demiranda/projects/sparse_delta
cd "$PROJECT_ROOT"

srun .venv/bin/python experiments/5-composite_staged/results/analyze_gate.py
