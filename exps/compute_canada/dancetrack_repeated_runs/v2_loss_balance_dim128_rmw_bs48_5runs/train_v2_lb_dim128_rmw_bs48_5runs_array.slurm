#!/bin/bash
#SBATCH --account=def-mpederso_gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --job-name=dt_v2_lb_rmw5
#SBATCH --array=0-44
#SBATCH --output=logs_compute_canada/%x-%A_%a.out

set -euo pipefail
cd "$HOME/links/projects/YOLOX-jde"

export REPO_DIR="$HOME/links/projects/YOLOX-jde"
export DATA_SOURCE="$REPO_DIR/datasets/data.zip"
export SCRATCH_ROOT="$HOME/scratch/YOLOX-jde"
export OUTPUT_ROOT="$SCRATCH_ROOT/YOLOX_outputs"

RMW_VALUES=(0.05 0.10 0.20 0.30 0.40 1.00 1.20 1.50 2.00)
RMW_NAMES=(005 010 020 030 040 100 120 150 200)
RUNS_PER_RMW=5
RMW_INDEX=$((SLURM_ARRAY_TASK_ID / RUNS_PER_RMW))
RUN_ID=$((SLURM_ARRAY_TASK_ID % RUNS_PER_RMW + 1))

export REID_MATCH_WEIGHT="${RMW_VALUES[$RMW_INDEX]}"
export RMW_NAME="${RMW_NAMES[$RMW_INDEX]}"
export RUN_ID
export SEED=$((12345 + SLURM_ARRAY_TASK_ID))

FINAL_EXP_NAME="yolox_x_dancetrack_jde_v2_lb_dim128_rmw${RMW_NAME}_bs48_5runs"
TEMP_EXP_NAME="${FINAL_EXP_NAME}_run${RUN_ID}_tmp"
export EXP_FILE="$REPO_DIR/exps/compute_canada/dancetrack_repeated_runs/v2_loss_balance_dim128_rmw_bs48_5runs/exp_v2_lb_dim128_rmw_param.py"
export EXP_NAME="$TEMP_EXP_NAME"
export CKPT="$REPO_DIR/pretrained/yolox_x.pth"
export NUM_DEVICES=4
export NUM_MACHINES=2
export BATCH_SIZE=48
export MAKE_JDE=1
export ACTIVATE_SCRIPT="$REPO_DIR/exps/compute_canada/activate_track_h100.sh"
FINAL_DIR="$OUTPUT_ROOT/$FINAL_EXP_NAME"
TEMP_DIR="$OUTPUT_ROOT/$TEMP_EXP_NAME"

echo "=============================================================="
echo "JDE-V2 LB dim=128 rmw repeated run"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "REID_MATCH_WEIGHT=$REID_MATCH_WEIGHT"
echo "RUN_ID=$RUN_ID"
echo "SEED=$SEED"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "TEMP_EXP_NAME=$TEMP_EXP_NAME"
echo "FINAL_EXP_NAME=$FINAL_EXP_NAME"
echo "=============================================================="

rm -rf "$TEMP_DIR"
srun --ntasks=2 --ntasks-per-node=1 bash -lc '
    cd "$REPO_DIR"
    export MACHINE_RANK="$SLURM_NODEID"
    bash "$REPO_DIR/exps/compute_canada/run_dancetrack_training_rorqual.sh"
'
META_TEXT="method=JDE-V2; loss_balance=True; reid_dim=128; reid_weight=1.0; reid_match_weight=$REID_MATCH_WEIGHT; reid_match_max_cost=2.0; use_reid_in_dynamic_k=True; batch_size=48; run_id=$RUN_ID; seed=$SEED; slurm_task=$SLURM_ARRAY_TASK_ID"
bash "$REPO_DIR/exps/compute_canada/repeated_runs/collect_best_and_clean.sh" "$TEMP_DIR" "$FINAL_DIR" "$RUN_ID" "$META_TEXT"
echo "Done repeated run."
