#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=run_simple
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/run_simple_%A.out

set -euo pipefail

module purge
module load 2025

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate videoflextok

cd "$SLURM_SUBMIT_DIR"

python simple.py
