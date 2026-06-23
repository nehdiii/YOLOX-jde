#!/bin/bash
set -euo pipefail

# Create a Narval Python environment for YOLOX-JDE.
# Run from repo root:
#   cd ~/links/projects/YOLOX-jde
#   bash exps/compute_canada/narval_setup/create_track_env_narval.sh

REPO_DIR="${REPO_DIR:-$PWD}"
TRACK_ENV="${TRACK_ENV:-$HOME/pyenv/Track}"

echo "=============================================================="
echo "Creating Narval env for YOLOX-JDE"
echo "REPO_DIR=$REPO_DIR"
echo "TRACK_ENV=$TRACK_ENV"
echo "=============================================================="

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.11

# Optional but useful if available on Narval.
module load opencv/4.13.0 2>/dev/null || true
module load faiss/1.7.4 2>/dev/null || true

mkdir -p "$(dirname "$TRACK_ENV")"

if [ ! -d "$TRACK_ENV" ]; then
    python -m venv --system-site-packages "$TRACK_ENV"
fi

source "$TRACK_ENV/bin/activate"

python -m pip install --no-index --upgrade pip setuptools wheel || true

echo "==> Installing PyTorch stack from Alliance wheelhouse/site packages"
python -m pip install --no-index torch torchvision torchaudio || true

echo "==> Installing training dependencies"
# Install one by one so a missing optional wheel does not stop everything.
REQ_PKGS=(
  numpy
  loguru
  scikit-image
  tqdm
  Pillow
  thop
  ninja
  tabulate
  tensorboard
  filterpy
  h5py
  pycocotools
  Cython
)

for pkg in "${REQ_PKGS[@]}"; do
    echo "---- pip install --no-index $pkg"
    python -m pip install --no-index "$pkg" || echo "[WARN] Could not install $pkg from wheelhouse. Preflight will tell us if it is required."
done

# Optional dependencies used by some tracking/eval paths.
OPT_PKGS=(
  lap
  cython-bbox
)

for pkg in "${OPT_PKGS[@]}"; do
    echo "---- optional pip install --no-index $pkg"
    python -m pip install --no-index "$pkg" || echo "[WARN] Optional package missing: $pkg"
done

echo "==> Installing YOLOX-JDE repo in editable mode"
cd "$REPO_DIR"
python -m pip install -e . --no-build-isolation --no-deps

echo "=============================================================="
echo "Environment creation finished"
echo "Run:"
echo "  source exps/compute_canada/activate_track_narval.sh"
echo "  bash exps/compute_canada/narval_setup/preflight_track_env_narval.sh"
echo "=============================================================="