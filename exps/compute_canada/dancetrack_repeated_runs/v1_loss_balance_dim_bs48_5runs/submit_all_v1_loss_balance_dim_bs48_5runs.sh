#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."

echo "Submitting JDE V1 loss-balancing dim ablation: 4 dims × 5 repeats = 20 jobs"
echo "Batch size: 48"
echo "Clean aggregate folders will be under ~/scratch/YOLOX-jde/YOLOX_outputs/"

sbatch exps/compute_canada/dancetrack_repeated_runs/v1_loss_balance_dim_bs48_5runs/train_v1_loss_balance_dim_bs48_5runs_array.slurm