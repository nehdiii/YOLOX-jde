#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
CONCURRENCY="${CONCURRENCY:-1}"
echo "Submitting Narval reid_match_max_cost=3.5, tasks 25-29, 5 runs, concurrency=$CONCURRENCY"
sbatch --array=25-29%${CONCURRENCY} exps/compute_canada/dancetrack_repeated_runs/v2_loss_balance_dim128_rmc_nody_bs48_5runs_narval/train_v2_lb_dim128_rmc_nody_bs48_5runs_narval_array.slurm
