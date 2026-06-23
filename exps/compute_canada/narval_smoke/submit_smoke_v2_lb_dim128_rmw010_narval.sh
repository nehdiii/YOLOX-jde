#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

echo "Submitting one Narval smoke job..."
sbatch exps/compute_canada/narval_smoke/smoke_v2_lb_dim128_rmw010_narval.slurm