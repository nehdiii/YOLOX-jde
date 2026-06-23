#!/bin/bash
# Activation helper for Narval.
# Source this file from Slurm scripts before launching YOLOX-JDE training.

set -euo pipefail

module --force purge

# Narval is an Alliance cluster. The exact module list may change.
# These defaults match the Compute Canada PyTorch stack style used in this project.
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.11

# Optional modules. Load them if available; do not fail if the site module name differs.
module load opencv/4.13.0 2>/dev/null || true
module load faiss/1.7.4 2>/dev/null || true

# Prefer the same env name/path, but allow override:
#   export TRACK_ENV=/path/to/env
TRACK_ENV="${TRACK_ENV:-$HOME/pyenv/Track}"

if [ ! -f "$TRACK_ENV/bin/activate" ]; then
    echo "[ERROR] TRACK_ENV not found on Narval: $TRACK_ENV"
    echo "Create/copy your environment on Narval first, or submit with:"
    echo "  sbatch --export=ALL,TRACK_ENV=/path/to/env <slurm_file>"
    return 1 2>/dev/null || exit 1
fi

source "$TRACK_ENV/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "==> Narval environment activated"
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