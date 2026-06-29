#!/bin/bash
#SBATCH --account=def-mpederso_gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --job-name=dt_v2_lb_rmc_nd
#SBATCH --array=0-19
#SBATCH --output=logs_compute_canada/%x-%A_%a.out

set -euo pipefail
cd "$HOME/links/projects/YOLOX-jde"

export REPO_DIR="$HOME/links/projects/YOLOX-jde"
export DATA_SOURCE="$REPO_DIR/datasets/data.zip"
export SCRATCH_ROOT="$HOME/scratch/YOLOX-jde"
export OUTPUT_ROOT="$SCRATCH_ROOT/YOLOX_outputs"

RMC_VALUES=(0.5 1.0 1.5 2.0)
RMC_NAMES=(050 100 150 200)
RUNS_PER_RMC=5
RMC_INDEX=$((SLURM_ARRAY_TASK_ID / RUNS_PER_RMC))
RUN_ID=$((SLURM_ARRAY_TASK_ID % RUNS_PER_RMC + 1))

export REID_MATCH_MAX_COST="${RMC_VALUES[$RMC_INDEX]}"
export RMC_NAME="${RMC_NAMES[$RMC_INDEX]}"
export RUN_ID
export SEED=$((22345 + SLURM_ARRAY_TASK_ID))

FINAL_EXP_NAME="yolox_x_dancetrack_jde_v2_lb_dim128_rmw010_rmc${RMC_NAME}_nody_bs48_5runs"
TEMP_EXP_NAME="${FINAL_EXP_NAME}_run${RUN_ID}_tmp"
export EXP_FILE="$REPO_DIR/exps/compute_canada/dancetrack_repeated_runs/v2_loss_balance_dim128_rmc_nody_bs48_5runs/exp_v2_lb_dim128_rmc_nody_param.py"
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
echo "JDE-V2 LB dim=128 rmw=0.10 rmc no-dynamic-k repeated run"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "REID_MATCH_MAX_COST=$REID_MATCH_MAX_COST"
echo "REID_MATCH_WEIGHT=0.1"
echo "USE_REID_IN_DYNAMIC_K=False"
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
META_TEXT="method=JDE-V2; loss_balance=True; reid_dim=128; reid_weight=1.0; reid_match_weight=0.1; reid_match_max_cost=$REID_MATCH_MAX_COST; use_reid_in_dynamic_k=False; batch_size=48; run_id=$RUN_ID; seed=$SEED; slurm_task=$SLURM_ARRAY_TASK_ID"
bash "$REPO_DIR/exps/compute_canada/dancetrack_repeated_runs/repeated_runs/collect_best_and_clean.sh" "$TEMP_DIR" "$FINAL_DIR" "$RUN_ID" "$META_TEXT"
echo "Done repeated run."
