#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
CONCURRENCY="${CONCURRENCY:-1}"
echo "Submitting reid_match_weight=1.50, tasks 35-39, 5 runs, concurrency=$CONCURRENCY"
sbatch --array=35-39%${CONCURRENCY} exps/compute_canada/dancetrack_repeated_runs/v2_loss_balance_dim128_rmw_bs48_5runs/train_v2_lb_dim128_rmw_bs48_5runs_array.slurm
