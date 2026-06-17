#!/bin/bash
#SBATCH --account=def-mpederso_gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=01:15:00
#SBATCH --job-name=dt_v1_rw025_dim128
#SBATCH --output=logs_compute_canada/%x-%j.out

set -euo pipefail

cd "$HOME/links/projects/YOLOX-jde"

export REPO_DIR="$HOME/links/projects/YOLOX-jde"
export DATA_SOURCE="$REPO_DIR/datasets/data.zip"
export SCRATCH_ROOT="$HOME/scratch/YOLOX-jde"
export OUTPUT_ROOT="$SCRATCH_ROOT/YOLOX_outputs"

# JDE V1 ablation: v1_rw025_dim128
# Exp file overrides reid_weight/reid_dim/max_epoch.
export EXP_FILE="$REPO_DIR/exps/compute_canada/dancetrack_ablations/reid_weight/exp_v1_rw025_dim128.py"
export EXP_NAME="yolox_x_dancetrack_jde_v1_rw025_dim128_cc_2nodes_8gpu"
export CKPT="$REPO_DIR/pretrained/yolox_x.pth"

# 2 nodes × 4 H100 per node = 8 GPUs total.
# YOLOX -b is global batch, so global batch 48 means 6 images/GPU.
export NUM_DEVICES=4
export NUM_MACHINES=2
export BATCH_SIZE=48

# JDE V1 needs train_jde.json with global identity labels.
# prepare_dancetrack_on_node.sh will create it if missing.
export MAKE_JDE=1

# Use the same activation helper created by create_env_rorqual.sh.
export ACTIVATE_SCRIPT="$REPO_DIR/exps/compute_canada/activate_track_h100.sh"

srun --ntasks=2 --ntasks-per-node=1 bash -lc '
    cd "$REPO_DIR"
    export MACHINE_RANK="$SLURM_NODEID"
    bash "$REPO_DIR/exps/compute_canada/run_dancetrack_training_rorqual.sh"
'