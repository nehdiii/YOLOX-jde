#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
CONCURRENCY="${CONCURRENCY:-1}"
echo "Submitting reid_match_weight=0.10, tasks 5-9, 5 runs, concurrency=$CONCURRENCY"
sbatch --array=5-9%${CONCURRENCY} exps/compute_canada/dancetrack_repeated_runs/v2_loss_balance_dim128_rmw_bs48_5runs_narval_narval/train_v2_lb_dim128_rmw_bs48_5runs_array.slurm