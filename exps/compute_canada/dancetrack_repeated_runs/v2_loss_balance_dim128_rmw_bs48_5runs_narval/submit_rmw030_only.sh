#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
CONCURRENCY="${CONCURRENCY:-1}"
echo "Submitting reid_match_weight=0.30, tasks 15-19, 5 runs, concurrency=$CONCURRENCY"
sbatch --array=15-19%${CONCURRENCY} exps/compute_canada/dancetrack_repeated_runs/v2_loss_balance_dim128_rmw_bs48_5runs_narval_narval/train_v2_lb_dim128_rmw_bs48_5runs_array.slurm