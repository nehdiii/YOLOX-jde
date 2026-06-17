#!/bin/bash
#SBATCH --account=def-mpederso_gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --job-name=dt_v1_rw050_dim128
#SBATCH --output=logs_compute_canada/%x-%j.out

set -euo pipefail

cd "$HOME/links/projects/YOLOX-jde"

export REPO_DIR="$HOME/links/projects/YOLOX-jde"
export DATA_SOURCE="$REPO_DIR/datasets/data.zip"
export SCRATCH_ROOT="$HOME/scratch/YOLOX-jde"
export OUTPUT_ROOT="$SCRATCH_ROOT/YOLOX_outputs"

# DanceTrack JDE V1 ablation: v1_rw050_dim128
# reid_weight=0.5 | reid_dim=128 | max_epoch=8
export EXP_FILE="$REPO_DIR/exps/compute_canada/dancetrack_ablations/v1_rw050_dim128/exp_v1_rw050_dim128.py"
export EXP_NAME="yolox_x_dancetrack_jde_v1_rw050_dim128_cc_2nodes_8gpu"
export CKPT="$REPO_DIR/pretrained/yolox_x.pth"

# 2 nodes × 4 H100 per node = 8 GPUs total.
# YOLOX -b is global batch, so global batch 48 = 6 images/GPU.
export NUM_DEVICES=4
export NUM_MACHINES=2
export BATCH_SIZE=48

# JDE V1 needs train_jde.json with global identity labels.
export MAKE_JDE=1

# Use your working activation helper and training launcher.
export ACTIVATE_SCRIPT="$REPO_DIR/exps/compute_canada/activate_track_h100.sh"

# Important: run one launcher per node; each launcher uses 4 local GPUs.
srun --ntasks=2 --ntasks-per-node=1 bash -lc '
    cd "$REPO_DIR"
    export MACHINE_RANK="$SLURM_NODEID"
    bash "$REPO_DIR/exps/compute_canada/run_dancetrack_training_rorqual.sh"
'