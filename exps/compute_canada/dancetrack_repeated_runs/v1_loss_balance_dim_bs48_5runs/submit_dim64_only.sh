#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."

echo "Submitting only dim64: 5 repeats"
sbatch --array=0-4 exps/compute_canada/dancetrack_repeated_runs/v1_loss_balance_dim_bs48_5runs/train_v1_loss_balance_dim_bs48_5runs_array.slurm