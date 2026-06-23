#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
CONCURRENCY="${CONCURRENCY:-1}"
echo "Submitting JDE-V2 LB dim=128 reid_match_weight sweep: 6 values x 5 runs = 30 tasks"
echo "Values: 0.05 0.10 0.20 0.30 0.40 1.00"
echo "Concurrency: $CONCURRENCY"
sbatch --array=0-29%${CONCURRENCY} exps/compute_canada/dancetrack_repeated_runs/v2_loss_balance_dim128_rmw_bs48_5runs/train_v2_lb_dim128_rmw_bs48_5runs_array.slurm