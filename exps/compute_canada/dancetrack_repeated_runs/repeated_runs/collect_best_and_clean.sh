#!/bin/bash
set -euo pipefail
if [ "$#" -lt 4 ]; then echo "Usage: collect_best_and_clean.sh <temp_exp_dir> <final_exp_dir> <run_id> <metadata_text>"; exit 2; fi
TEMP_DIR="$1"; FINAL_DIR="$2"; RUN_ID="$3"; META_TEXT="$4"
mkdir -p "$FINAL_DIR"
BEST_SRC="$TEMP_DIR/best_ckpt.pth.tar"
LOG_SRC="$TEMP_DIR/train_log.txt"
if [ ! -f "$BEST_SRC" ]; then echo "ERROR: missing best checkpoint: $BEST_SRC"; echo "Temporary folder kept: $TEMP_DIR"; exit 1; fi
cp -f "$BEST_SRC" "$FINAL_DIR/best_ckpt_${RUN_ID}.pth.tar"
if [ -f "$LOG_SRC" ]; then cp -f "$LOG_SRC" "$FINAL_DIR/train_log_${RUN_ID}.txt"; else echo "WARNING: missing train_log.txt" > "$FINAL_DIR/train_log_${RUN_ID}.txt"; echo "$META_TEXT" >> "$FINAL_DIR/train_log_${RUN_ID}.txt"; fi
{ echo "$META_TEXT"; echo "TEMP_DIR=$TEMP_DIR"; echo "FINAL_DIR=$FINAL_DIR"; echo "RUN_ID=$RUN_ID"; echo "COLLECTED_AT=$(date -Is)"; } > "$FINAL_DIR/metadata_run_${RUN_ID}.txt"
rm -rf "$TEMP_DIR"
echo "Collected $FINAL_DIR/best_ckpt_${RUN_ID}.pth.tar and train_log_${RUN_ID}.txt"