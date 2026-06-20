#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

SLURM_FILE="exps/compute_canada/dancetrack_ablations/v1_epoch_ablation_dim128_lb_compact/train_v1_epoch_dim128_lb_generic.slurm"

echo "Submitting JDE V1 epoch sweep with per-epoch time limits..."
echo "Based on your observation: 8 epochs needs ~36 min actual."
echo "Requested times include buffer but are still efficient."

echo "Submitting max_epoch=12 with --time=01:30:00"
sbatch --time=01:30:00 --job-name=dt_v1_ep12 --export=ALL,MAX_EPOCH=12,EPOCH_NAME=012 "$SLURM_FILE"

echo "Submitting max_epoch=16 with --time=02:00:00"
sbatch --time=02:00:00 --job-name=dt_v1_ep16 --export=ALL,MAX_EPOCH=16,EPOCH_NAME=016 "$SLURM_FILE"

echo "Submitting max_epoch=20 with --time=02:30:00"
sbatch --time=02:30:00 --job-name=dt_v1_ep20 --export=ALL,MAX_EPOCH=20,EPOCH_NAME=020 "$SLURM_FILE"

echo "Submitting max_epoch=25 with --time=03:00:00"
sbatch --time=03:00:00 --job-name=dt_v1_ep25 --export=ALL,MAX_EPOCH=25,EPOCH_NAME=025 "$SLURM_FILE"

echo "Submitting max_epoch=30 with --time=03:30:00"
sbatch --time=03:30:00 --job-name=dt_v1_ep30 --export=ALL,MAX_EPOCH=30,EPOCH_NAME=030 "$SLURM_FILE"
