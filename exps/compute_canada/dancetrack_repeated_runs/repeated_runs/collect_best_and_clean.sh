#!/bin/bash
# Collect one YOLOX training trial into a clean aggregate folder and remove temporary weights.
# Usage:
#   collect_best_and_clean.sh OUTPUT_ROOT TMP_EXP_NAME AGG_EXP_NAME TRIAL_ID
#
# Result inside $OUTPUT_ROOT/$AGG_EXP_NAME:
#   best_ckpt_<TRIAL_ID>.pth.tar
#   train_log_<TRIAL_ID>.txt
#
# The temporary folder $OUTPUT_ROOT/$TMP_EXP_NAME is deleted after successful collection.

set -euo pipefail

OUTPUT_ROOT="${1:?OUTPUT_ROOT is required}"
TMP_EXP_NAME="${2:?TMP_EXP_NAME is required}"
AGG_EXP_NAME="${3:?AGG_EXP_NAME is required}"
TRIAL_ID="${4:?TRIAL_ID is required}"

TMP_DIR="$OUTPUT_ROOT/$TMP_EXP_NAME"
AGG_DIR="$OUTPUT_ROOT/$AGG_EXP_NAME"

BEST_SRC="$TMP_DIR/best_ckpt.pth.tar"
LOG_SRC="$TMP_DIR/train_log.txt"

mkdir -p "$AGG_DIR"

if [ ! -f "$BEST_SRC" ]; then
    echo "[ERROR] Missing best checkpoint: $BEST_SRC"
    echo "[INFO] Keeping temporary folder for debugging: $TMP_DIR"
    exit 1
fi

if [ ! -f "$LOG_SRC" ]; then
    echo "[WARN] Missing train_log.txt: $LOG_SRC"
    echo "[WARN] Creating small placeholder log in aggregate folder."
    echo "train_log.txt was missing in $TMP_DIR" > "$AGG_DIR/train_log_${TRIAL_ID}.txt"
else
    cp -f "$LOG_SRC" "$AGG_DIR/train_log_${TRIAL_ID}.txt"
fi

cp -f "$BEST_SRC" "$AGG_DIR/best_ckpt_${TRIAL_ID}.pth.tar"

# Keep the aggregate folder clean: only best checkpoints and train logs.
# The Slurm stdout log remains in repo/logs_compute_canada, not here.
rm -rf "$TMP_DIR"

echo "[OK] Collected trial $TRIAL_ID"
echo "     checkpoint: $AGG_DIR/best_ckpt_${TRIAL_ID}.pth.tar"
echo "     train log : $AGG_DIR/train_log_${TRIAL_ID}.txt"
echo "     removed tmp: $TMP_DIR"