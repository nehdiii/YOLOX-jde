#!/bin/bash
#SBATCH --account=def-mpederso_gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --job-name=dt_v2_costdyn
#SBATCH --output=logs_compute_canada/%x-%j.out

set -euo pipefail

cd "$HOME/links/projects/YOLOX-jde"

export REPO_DIR="$HOME/links/projects/YOLOX-jde"
export DATA_SOURCE="$REPO_DIR/datasets/data.zip"
export SCRATCH_ROOT="$HOME/scratch/YOLOX-jde"
export OUTPUT_ROOT="$SCRATCH_ROOT/YOLOX_outputs"

# JDE V2 dynamic-k ablation: COST + DYNAMIC-K.
# Fixed:
#   reid_dim = 128
#   reid_weight = 0.25
#   use_uncertainty = False
#   reid_match_weight = 0.10
#   reid_match_max_cost = 2.0
# Ablation:
#   use_reid_in_dynamic_k = True
export EXP_FILE="$REPO_DIR/exps/compute_canada/dancetrack_ablations/v2_dynamic_k_ablation/cost_plus_dynamic_k/exp_v2_cost_plus_dynamic_k.py"
export EXP_NAME="yolox_x_dancetrack_jde_v2_cost_plus_dynamic_k_cc_2nodes_8gpu"
export CKPT="$REPO_DIR/pretrained/yolox_x.pth"

export NUM_DEVICES=4
export NUM_MACHINES=2
export BATCH_SIZE=48
export MAKE_JDE=1
export ACTIVATE_SCRIPT="$REPO_DIR/exps/compute_canada/activate_track_h100.sh"

echo "=============================================================="
echo "JDE V2 COST + DYNAMIC-K"
echo "EXP_FILE=$EXP_FILE"
echo "EXP_NAME=$EXP_NAME"
echo "reid_dim=128"
echo "reid_weight=0.25"
echo "use_uncertainty=False"
echo "reid_match_weight=0.10"
echo "reid_match_max_cost=2.0"
echo "use_reid_in_dynamic_k=True"
echo "=============================================================="

srun --ntasks=2 --ntasks-per-node=1 bash -lc '
    cd "$REPO_DIR"
    export MACHINE_RANK="$SLURM_NODEID"
    bash "$REPO_DIR/exps/compute_canada/run_dancetrack_training_rorqual.sh"
'