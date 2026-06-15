#!/bin/bash
#SBATCH --account=def-mpederso_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=dt_det_x_1gpu
#SBATCH --output=logs_compute_canada/%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

export REPO_DIR
export DATA_SOURCE="${DATA_SOURCE:-$REPO_DIR/datasets/data.zip}"
export EXP_FILE="${EXP_FILE:-$REPO_DIR/exps/example/mot/yolox_dancetrack_val.py}"
export EXP_NAME="${EXP_NAME:-yolox_x_dancetrack_detector_cc_1gpu}"
export CKPT="${CKPT:-$REPO_DIR/pretrained/yolox_x.pth}"
export BATCH_SIZE="${BATCH_SIZE:-48}"
export NUM_DEVICES="${NUM_DEVICES:-1}"
export MAKE_JDE=0

bash "$REPO_DIR/exps/compute_canada/run_dancetrack_training_rorqual.sh"