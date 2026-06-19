#!/bin/bash
#SBATCH --account=def-mpederso_gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --job-name=dt_v1_lb_rw075
#SBATCH --output=logs_compute_canada/%x-%j.out

set -euo pipefail

cd "$HOME/links/projects/YOLOX-jde"

export REPO_DIR="$HOME/links/projects/YOLOX-jde"
export DATA_SOURCE="$REPO_DIR/datasets/data.zip"
export SCRATCH_ROOT="$HOME/scratch/YOLOX-jde"
export OUTPUT_ROOT="$SCRATCH_ROOT/YOLOX_outputs"

# JDE V1 loss-balancing reid_weight ablation.
# Fixed:
#   reid_dim = 128
#   use_uncertainty = True
# Swept:
#   reid_weight = 0.75
export EXP_FILE="$REPO_DIR/exps/compute_canada/dancetrack_ablation/loss_balance_rw_v1/v1_lb_rw075_dim128/exp_v1_lb_rw075_dim128.py"
export EXP_NAME="yolox_x_dancetrack_jde_v1_lb_rw075_dim128_cc_2nodes_8gpu"
export CKPT="$REPO_DIR/pretrained/yolox_x.pth"

# 2 nodes × 4 H100 per node = 8 GPUs total.
# YOLOX -b is global batch. Global batch 48 => 6 images/GPU.
export NUM_DEVICES=4
export NUM_MACHINES=2
export BATCH_SIZE=48

# JDE needs train_jde.json with global identity labels.
export MAKE_JDE=1

# Use your existing Rorqual activation helper.
export ACTIVATE_SCRIPT="$REPO_DIR/exps/compute_canada/activate_track_h100.sh"

srun --ntasks=2 --ntasks-per-node=1 bash -lc '
    cd "$REPO_DIR"
    export MACHINE_RANK="$SLURM_NODEID"
    bash "$REPO_DIR/exps/compute_canada/run_dancetrack_training_rorqual.sh"
'