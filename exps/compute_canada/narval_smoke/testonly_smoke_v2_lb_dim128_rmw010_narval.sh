#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

echo "Testing Slurm acceptance only..."
sbatch --test-only exps/compute_canada/narval_smoke/smoke_v2_lb_dim128_rmw010_narval.slurm