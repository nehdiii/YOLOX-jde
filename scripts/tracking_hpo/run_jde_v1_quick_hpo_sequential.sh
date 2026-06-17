#!/bin/bash
set -euo pipefail

# Execute the quick grid with strict sequential HPO scheduling.
# HPO configs run one by one.
# Each HPO config still uses --parallel-workers for sequence-level parallelism.

cd "$(dirname "$0")/../.."

CKPT="${CKPT:-YOLOX_outputs/yolox_x_dancetrack_jde_v1_cc_2nodes_8gpu/best_ckpt.pth.tar}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
python tools/tracking_hpo/run_jde_v1_tracking_hpo_sequential.py \
  --grid configs/tracking_hpo/jde_v1_quick_grid.json \
  --exp-file exps/example/dancetrack/yolox_x_dancetrack_jde_v1.py \
  --ckpt "$CKPT" \
  --base-expn hpo_jde_v1_quick_seq \
  --parallel-workers "${PARALLEL_WORKERS:-10}" \
  --parallel-gpus "${PARALLEL_GPUS:-0,1}" \
  --sleep-between "${SLEEP_BETWEEN:-5}" \
  --execute