#!/bin/bash
# Activation helper for Fir.
# Source this file from Slurm scripts before launching YOLOX-JDE training.

set -euo pipefail

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.11

# Fir module names may be versioned. Run `module spider opencv` /
# `module spider faiss` on Fir if either of these names changes.
module load opencv
module load faiss/1.7.4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
TRACK_ENV="${TRACK_ENV:-${ENV_DIR:-$HOME/pyenv/Track}}"

if [ ! -f "$TRACK_ENV/bin/activate" ]; then
    echo "[ERROR] TRACK_ENV not found on Fir: $TRACK_ENV"
    echo "Create it first with:"
    echo "  bash $REPO_DIR/exps/compute_canada/create_env_fir.sh"
    return 1 2>/dev/null || exit 1
fi

source "$TRACK_ENV/bin/activate"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "==> Fir environment activated"
echo "Host: $(hostname)"
echo "TRACK_ENV=$TRACK_ENV"
python - <<'PY'
import torch, sys
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
PY
