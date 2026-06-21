#!/bin/bash
#SBATCH --account=def-mpederso_gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --job-name=dt_v1_lb_dim5
#SBATCH --array=0-19
#SBATCH --output=logs_compute_canada/%x-%A_%a.out

set -euo pipefail

cd "$HOME/links/projects/YOLOX-jde"

export REPO_DIR="$HOME/links/projects/YOLOX-jde"
export DATA_SOURCE="$REPO_DIR/datasets/data.zip"
export SCRATCH_ROOT="$HOME/scratch/YOLOX-jde"
export OUTPUT_ROOT="$SCRATCH_ROOT/YOLOX_outputs"

# JDE V1 loss-balancing dim ablation repeated 5 times.
# DIMS = 64, 128, 256, 512
# REPEATS = 1, 2, 3, 4, 5
# Batch size = 128
DIMS=(64 128 256 512)
NUM_REPEATS=5

DIM_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_REPEATS))
REPEAT_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_REPEATS + 1))

export REID_DIM="${DIMS[$DIM_INDEX]}"
export TRIAL_ID="$REPEAT_INDEX"

# Aggregate folder: one folder per dim containing only best_ckpt_i + train_log_i.
export AGG_EXP_NAME="yolox_x_dancetrack_jde_v1_lb_dim${REID_DIM}_bs48_5runs"

# Temporary unique training folder, deleted after best/log collection.
export TMP_EXP_NAME="${AGG_EXP_NAME}_tmp_trial${TRIAL_ID}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

export EXP_FILE="$REPO_DIR/exps/compute_canada/dancetrack_repeated_runs/v1_loss_balance_dim_bs48_5runs/exp_v1_loss_balance_dim_param.py"
export EXP_NAME="$TMP_EXP_NAME"
export CKPT="$REPO_DIR/pretrained/yolox_x.pth"

# 2 nodes × 4 H100 = 8 GPUs total.
# YOLOX -b is global batch. Batch 128 => 6 images/GPU.
export NUM_DEVICES=4
export NUM_MACHINES=2
export BATCH_SIZE=48
export MAKE_JDE=1
export ACTIVATE_SCRIPT="$REPO_DIR/exps/compute_canada/activate_track_h100.sh"

# Do not force a fixed seed. These are independent random repeats for mean/std.
unset SEED || true
unset YOLOX_SEED || true

echo "=============================================================="
echo "JDE V1 loss-balancing dim repeated-run ablation"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "REID_DIM=$REID_DIM"
echo "TRIAL_ID=$TRIAL_ID"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "TMP_EXP_NAME=$TMP_EXP_NAME"
echo "AGG_EXP_NAME=$AGG_EXP_NAME"
echo "Output aggregate: $OUTPUT_ROOT/$AGG_EXP_NAME"
echo "=============================================================="

srun --ntasks=2 --ntasks-per-node=1 bash -lc '
    cd "$REPO_DIR"
    export MACHINE_RANK="$SLURM_NODEID"
    bash "$REPO_DIR/exps/compute_canada/run_dancetrack_training_rorqual.sh"
'

# After all distributed training ranks finish, collect only best checkpoint + train log.
bash "$REPO_DIR/exps/compute_canada/repeated_runs/collect_best_and_clean.sh" "$OUTPUT_ROOT" "$TMP_EXP_NAME" "$AGG_EXP_NAME" "$TRIAL_ID"

# Print clean folder contents for verification.
echo "=============================================================="
echo "Final clean aggregate folder: $OUTPUT_ROOT/$AGG_EXP_NAME"
find "$OUTPUT_ROOT/$AGG_EXP_NAME" -maxdepth 1 -type f -printf '  %f
' | sort
echo "=============================================================="