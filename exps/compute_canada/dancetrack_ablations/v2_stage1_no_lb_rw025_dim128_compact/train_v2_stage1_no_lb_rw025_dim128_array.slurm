#!/bin/bash
#SBATCH --account=def-mpederso_gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --job-name=dt_v2_nolb_rmw
#SBATCH --array=0-4
#SBATCH --output=logs_compute_canada/%x-%A_%a.out

set -euo pipefail

cd "$HOME/links/projects/YOLOX-jde"

export REPO_DIR="$HOME/links/projects/YOLOX-jde"
export DATA_SOURCE="$REPO_DIR/datasets/data.zip"
export SCRATCH_ROOT="$HOME/scratch/YOLOX-jde"
export OUTPUT_ROOT="$SCRATCH_ROOT/YOLOX_outputs"

# Compact JDE V2 Stage 1, no loss balancing.
# Fixed: reid_dim=128, reid_weight=0.25, use_uncertainty=False.
# Swept by Slurm array: reid_match_weight.
RMW_VALUES=(0.00 0.05 0.10 0.20 0.30)
RMW_NAMES=(000 005 010 020 030)

export REID_MATCH_WEIGHT="${RMW_VALUES[$SLURM_ARRAY_TASK_ID]}"
export RMW_NAME="${RMW_NAMES[$SLURM_ARRAY_TASK_ID]}"

export EXP_FILE="$REPO_DIR/exps/compute_canada/dancetrack_ablations/v2_stage1_no_lb_rw025_dim128_compact/exp_v2_stage1_no_lb_rw025_dim128_param.py"
export EXP_NAME="yolox_x_dancetrack_jde_v2_stage1_no_lb_rw025_dim128_rmw${RMW_NAME}_cc_2nodes_8gpu"
export CKPT="$REPO_DIR/pretrained/yolox_x.pth"

export NUM_DEVICES=4
export NUM_MACHINES=2
export BATCH_SIZE=48
export MAKE_JDE=1
export ACTIVATE_SCRIPT="$REPO_DIR/exps/compute_canada/activate_track_h100.sh"

echo "=============================================================="
echo "JDE V2 Stage 1 compact no-LB"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "REID_MATCH_WEIGHT=$REID_MATCH_WEIGHT"
echo "EXP_NAME=$EXP_NAME"
echo "=============================================================="

srun --ntasks=2 --ntasks-per-node=1 bash -lc '
    cd "$REPO_DIR"
    export MACHINE_RANK="$SLURM_NODEID"
    bash "$REPO_DIR/exps/compute_canada/run_dancetrack_training_rorqual.sh"
'