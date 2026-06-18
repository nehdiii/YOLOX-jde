#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."

echo "Submitting v1 loss-balance dim ablations..."
sbatch exps/compute_canada/dancetrack_ablations/loss_balance_dim/v1_lb_rw100_dim064/train_v1_lb_rw100_dim064.slurm
sbatch exps/compute_canada/dancetrack_ablations/loss_balance_dim/v1_lb_rw100_dim128/train_v1_lb_rw100_dim128.slurm
sbatch exps/compute_canada/dancetrack_ablations/loss_balance_dim/v1_lb_rw100_dim256/train_v1_lb_rw100_dim256.slurm
sbatch exps/compute_canada/dancetrack_ablations/loss_balance_dim/v1_lb_rw100_dim512/train_v1_lb_rw100_dim512.slurm