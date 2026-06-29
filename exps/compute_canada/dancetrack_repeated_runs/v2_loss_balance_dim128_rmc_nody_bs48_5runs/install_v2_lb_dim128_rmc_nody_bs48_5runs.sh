#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."

BUNDLE_DIR="exps/compute_canada/dancetrack_repeated_runs/v2_loss_balance_dim128_rmc_nody_bs48_5runs"
SLURM_FILE="$BUNDLE_DIR/train_v2_lb_dim128_rmc_nody_bs48_5runs_array.slurm"
EXP_FILE="$BUNDLE_DIR/exp_v2_lb_dim128_rmc_nody_param.py"
PREFLIGHT="$BUNDLE_DIR/preflight_v2_lb_dim128_rmc_nody_repeats.py"

test -f "$SLURM_FILE"
test -f "$EXP_FILE"
test -f "$PREFLIGHT"

chmod +x "$BUNDLE_DIR"/*.sh "$PREFLIGHT"
mkdir -p logs_compute_canada

echo "Installed JDE-V2 reid_match_max_cost repeated-run bundle."
echo "Bundle: $BUNDLE_DIR"
echo
echo "Preflight:"
echo "  python $PREFLIGHT"
echo
echo "Submit all values:"
echo "  bash $BUNDLE_DIR/submit_all_v2_lb_dim128_rmc_nody_bs48_5runs.sh"
echo
echo "Submit one value:"
echo "  bash $BUNDLE_DIR/submit_rmc050_only.sh"
echo "  bash $BUNDLE_DIR/submit_rmc100_only.sh"
echo "  bash $BUNDLE_DIR/submit_rmc150_only.sh"
echo "  bash $BUNDLE_DIR/submit_rmc200_only.sh"
