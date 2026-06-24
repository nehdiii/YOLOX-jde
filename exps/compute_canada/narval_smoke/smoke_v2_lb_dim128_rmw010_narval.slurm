#!/bin/bash
#SBATCH --account=def-mpederso_gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --job-name=nv_v2_smoke
#SBATCH --output=logs_compute_canada/%x-%j.out

set -euo pipefail

# The submit wrapper exports the repository's real path.  Falling back to
# SLURM_SUBMIT_DIR also keeps direct `sbatch ...` submissions working, provided
# they are made from the repository root.
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [ -z "$REPO_DIR" ] || [ ! -f "$REPO_DIR/tools/train.py" ]; then
    echo "[ERROR] Could not find the HybridSORT repository."
    echo "REPO_DIR=${REPO_DIR:-not_set}"
    echo "Submit with the provided submit_smoke_v2_lb_dim128_rmw010_narval.sh wrapper,"
    echo "or export REPO_DIR=/absolute/path/to/your/repository before sbatch."
    exit 1
fi

export REPO_DIR
cd "$REPO_DIR"
export DATA_SOURCE="$REPO_DIR/datasets/data.zip"
export SCRATCH_ROOT="$HOME/scratch/YOLOX-jde"
export OUTPUT_ROOT="$SCRATCH_ROOT/YOLOX_outputs"

export EXP_FILE="$REPO_DIR/exps/compute_canada/narval_smoke/exp_v2_lb_dim128_rmw010_smoke.py"
export EXP_NAME="yolox_x_dancetrack_jde_v2_lb_dim128_rmw010_narval_smoke"
export CKPT="$REPO_DIR/pretrained/yolox_x.pth"

export NUM_DEVICES=4
export NUM_MACHINES=2
export BATCH_SIZE=48
export MAKE_JDE=1
export ACTIVATE_SCRIPT="$REPO_DIR/exps/compute_canada/activate_track_narval.sh"

echo "=============================================================="
echo "Narval smoke training: JDE-V2 LB dim128 rmw010, 1 epoch"
echo "EXP_FILE=$EXP_FILE"
echo "EXP_NAME=$EXP_NAME"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "=============================================================="

srun --ntasks=2 --ntasks-per-node=1 bash -lc '
    cd "$REPO_DIR"
    export MACHINE_RANK="$SLURM_NODEID"
    bash "$REPO_DIR/exps/compute_canada/run_dancetrack_training_narval.sh"
'
