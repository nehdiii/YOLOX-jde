#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"

cd "$REPO_DIR"
mkdir -p logs_compute_canada
export REPO_DIR

echo "Submitting one Narval smoke job..."
echo "Repository: $REPO_DIR"
sbatch exps/compute_canada/narval_smoke/smoke_v2_lb_dim128_rmw010_narval.slurm
