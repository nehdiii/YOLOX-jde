#!/bin/bash
# Shared launcher used by the Slurm scripts.
# Keep this file in the repo and call it from a Slurm allocation/job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

ACTIVATE_SCRIPT="${ACTIVATE_SCRIPT:-$HOME/pyenv/Track/bin/activate_track}"
DATA_SOURCE="${DATA_SOURCE:-$REPO_DIR/datasets/data.zip}"
SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch/YOLOX-jde}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH_ROOT/YOLOX_outputs}"
EXP_FILE="${EXP_FILE:?EXP_FILE must be set by the Slurm script}"
EXP_NAME="${EXP_NAME:?EXP_NAME must be set by the Slurm script}"
CKPT="${CKPT:-$REPO_DIR/pretrained/yolox_x.pth}"
BATCH_SIZE="${BATCH_SIZE:-48}"
MAKE_JDE="${MAKE_JDE:-0}"

if [ -z "${NUM_DEVICES:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        NUM_DEVICES=$(python - <<'PY'
import os
v = os.environ.get('CUDA_VISIBLE_DEVICES', '')
items = [x for x in v.split(',') if x.strip()]
print(len(items) if items else 1)
PY
)
    else
        NUM_DEVICES=1
    fi
fi

mkdir -p "$SCRATCH_ROOT" "$OUTPUT_ROOT" "$REPO_DIR/logs_compute_canada"

if [ ! -f "$ACTIVATE_SCRIPT" ]; then
    echo "[ERROR] Activate script not found: $ACTIVATE_SCRIPT"
    echo "Create it first with: bash exps/compute_canada/create_env_rorqual.sh $REPO_DIR"
    exit 1
fi

if [ ! -d "$REPO_DIR" ]; then
    echo "[ERROR] Repo not found: $REPO_DIR"
    exit 1
fi

if [ ! -f "$EXP_FILE" ]; then
    echo "[ERROR] Exp file not found: $EXP_FILE"
    exit 1
fi

if [ ! -f "$CKPT" ]; then
    echo "[ERROR] Checkpoint not found: $CKPT"
    echo "Expected YOLOX pretrained weights at: $CKPT"
    exit 1
fi

echo "================================================================================"
echo "Rorqual YOLOX-JDE training launcher"
echo "Host          : $(hostname)"
echo "Job ID        : ${SLURM_JOB_ID:-none}"
echo "Repo          : $REPO_DIR"
echo "Data source   : $DATA_SOURCE"
echo "Scratch root  : $SCRATCH_ROOT"
echo "Output root   : $OUTPUT_ROOT"
echo "Exp file      : $EXP_FILE"
echo "Exp name      : $EXP_NAME"
echo "Checkpoint    : $CKPT"
echo "Batch size    : $BATCH_SIZE"
echo "Num devices   : $NUM_DEVICES"
echo "CUDA_VISIBLE  : ${CUDA_VISIBLE_DEVICES:-not_set}"
echo "Make JDE ann  : $MAKE_JDE"
echo "================================================================================"

source "$ACTIVATE_SCRIPT"
cd "$REPO_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export YOLOX_DATADIR="${YOLOX_DATADIR:-}"

DATA_ROOT="${SLURM_TMPDIR:-$SCRATCH_ROOT/tmp_${SLURM_JOB_ID:-manual}}/datasets"
mkdir -p "$DATA_ROOT"

if [ "$MAKE_JDE" = "1" ]; then
    PREP_EXTRA="--make-jde"
else
    PREP_EXTRA=""
fi

echo "==> Preparing DanceTrack on node-local storage"
bash "$REPO_DIR/exps/compute_canada/prepare_dancetrack_on_node.sh" \
    "$REPO_DIR" \
    "$DATA_SOURCE" \
    "$DATA_ROOT" \
    $PREP_EXTRA

export YOLOX_DATADIR="$DATA_ROOT"
echo "==> YOLOX_DATADIR=$YOLOX_DATADIR"

echo "==> Dataset annotations available:"
find "$YOLOX_DATADIR/dancetrack/annotations" -maxdepth 1 -type f -name '*.json' -printf '  %f\n' | sort

echo "==> Starting training"
python -u tools/train.py \
    -f "$EXP_FILE" \
    -expn "$EXP_NAME" \
    -d "$NUM_DEVICES" \
    -b "$BATCH_SIZE" \
    -c "$CKPT" \
    --fp16 \
    -o \
    output_dir "$OUTPUT_ROOT"

echo "==> Training finished"
echo "Output folder: $OUTPUT_ROOT/$EXP_NAME"
echo "Train log    : $OUTPUT_ROOT/$EXP_NAME/train_log.txt"