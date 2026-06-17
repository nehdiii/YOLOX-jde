#!/bin/bash
set -euo pipefail

# Run exactly one HPO config by index from a grid file.
#
# Example:
#   CONFIG_INDEX=0 PARALLEL_WORKERS=10 PARALLEL_GPUS=0,1 bash scripts/tracking_hpo/run_one_jde_v1_hpo_config.sh

cd "$(dirname "$0")/../.."

CKPT="${CKPT:-YOLOX_outputs/yolox_x_dancetrack_jde_v1_cc_2nodes_8gpu/best_ckpt.pth.tar}"
GRID="${GRID:-configs/tracking_hpo/jde_v1_focused_grid.json}"
CONFIG_INDEX="${CONFIG_INDEX:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
python tools/tracking_hpo/run_jde_v1_tracking_hpo_sequential.py \
  --grid "$GRID" \
  --exp-file exps/example/dancetrack/yolox_x_dancetrack_jde_v1.py \
  --ckpt "$CKPT" \
  --base-expn hpo_jde_v1_single_seq \
  --parallel-workers "${PARALLEL_WORKERS:-10}" \
  --parallel-gpus "${PARALLEL_GPUS:-0,1}" \
  --start-index "$CONFIG_INDEX" \
  --max-runs 1 \
  --sleep-between 0 \
  --execute