#!/bin/bash
#SBATCH --account=def-mpederso_gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=03:30:00
#SBATCH --job-name=dt_v1_epoch
#SBATCH --output=logs_compute_canada/%x-%j.out

set -euo pipefail

cd "$HOME/links/projects/YOLOX-jde"

export REPO_DIR="$HOME/links/projects/YOLOX-jde"
export DATA_SOURCE="$REPO_DIR/datasets/data.zip"
export SCRATCH_ROOT="$HOME/scratch/YOLOX-jde"
export OUTPUT_ROOT="$SCRATCH_ROOT/YOLOX_outputs"

# Compact JDE V1 epoch ablation.
# Fixed: JDE V1, reid_dim=128, use_uncertainty=True, no_aug_epochs=1.
# Swept through sbatch --export: MAX_EPOCH=12,16,20,25,30.
export MAX_EPOCH="${MAX_EPOCH:?MAX_EPOCH must be provided, e.g. 12}"
export EPOCH_NAME="${EPOCH_NAME:-${MAX_EPOCH}}"

export EXP_FILE="$REPO_DIR/exps/compute_canada/dancetrack_ablations/v1_epoch_ablation_dim128_lb_compact/exp_v1_epoch_dim128_lb_param.py"
export EXP_NAME="yolox_x_dancetrack_jde_v1_epoch${EPOCH_NAME}_dim128_lb_cc_2nodes_8gpu"
export CKPT="$REPO_DIR/pretrained/yolox_x.pth"

export NUM_DEVICES=4
export NUM_MACHINES=2
export BATCH_SIZE=48
export MAKE_JDE=1
export ACTIVATE_SCRIPT="$REPO_DIR/exps/compute_canada/activate_track_h100.sh"

echo "=============================================================="
echo "JDE V1 epoch ablation"
echo "MAX_EPOCH=$MAX_EPOCH"
echo "EPOCH_NAME=$EPOCH_NAME"
echo "EXP_FILE=$EXP_FILE"
echo "EXP_NAME=$EXP_NAME"
echo "reid_dim=128"
echo "use_uncertainty=True"
echo "no_aug_epochs=1"
echo "=============================================================="

srun --ntasks=2 --ntasks-per-node=1 bash -lc '
    cd "$REPO_DIR"
    export MACHINE_RANK="$SLURM_NODEID"
    bash "$REPO_DIR/exps/compute_canada/run_dancetrack_training_rorqual.sh"
'