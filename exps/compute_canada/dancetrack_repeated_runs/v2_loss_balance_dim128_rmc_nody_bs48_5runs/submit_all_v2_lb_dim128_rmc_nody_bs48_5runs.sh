#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
CONCURRENCY="${CONCURRENCY:-1}"
echo "Submitting JDE-V2 LB dim=128 reid_match_max_cost sweep: 6 values x 5 runs = 30 tasks"
echo "Fixed: reid_match_weight=0.1, use_reid_in_dynamic_k=False"
echo "Values: 0.5 1.0 1.5 2.5 3.0 3.5"
echo "Concurrency: $CONCURRENCY"
sbatch --array=0-29%${CONCURRENCY} exps/compute_canada/dancetrack_repeated_runs/v2_loss_balance_dim128_rmc_nody_bs48_5runs/train_v2_lb_dim128_rmc_nody_bs48_5runs_array.slurm
