#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
CONCURRENCY="${CONCURRENCY:-1}"
echo "Submitting reid_match_max_cost=1.0, tasks 5-9, 5 runs, concurrency=$CONCURRENCY"
sbatch --array=5-9%${CONCURRENCY} exps/compute_canada/dancetrack_repeated_runs/v2_loss_balance_dim128_rmc_nody_bs48_5runs/train_v2_lb_dim128_rmc_nody_bs48_5runs_array.slurm
