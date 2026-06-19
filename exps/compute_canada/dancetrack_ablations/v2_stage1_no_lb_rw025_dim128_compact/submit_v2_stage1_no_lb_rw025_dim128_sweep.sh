#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."

echo "Submitting compact JDE V2 Stage 1 no-LB rw=0.25 dim=128 sweep as one Slurm array..."
sbatch exps/compute_canada/dancetrack_ablations/v2_stage1_no_lb_rw025_dim128_compact/train_v2_stage1_no_lb_rw025_dim128_array.slurm