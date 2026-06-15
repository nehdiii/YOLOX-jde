#!/bin/bash
set -euo pipefail

# ============================================================
# Rorqual / Compute Canada environment for YOLOX-JDE
#
# Key fixes:
#   1. Load OpenCV module BEFORE activating the virtual env.
#   2. Do NOT pip install opencv-python or opencv-python-headless.
#   3. Use setuptools==69.5.1 because newer setuptools may remove pkg_resources.
#   4. Install YOLOX with --no-build-isolation because setup.py imports torch.
# ============================================================

echo "============================================================"
echo "[INFO] Creating YOLOX-JDE environment on Compute Canada/Rorqual"
echo "============================================================"

# -----------------------------
# Paths
# -----------------------------
ENV_ROOT="${HOME}/pyenv"
ENV_NAME="Track"
ENV_DIR="${ENV_ROOT}/${ENV_NAME}"
REPO_DIR="${HOME}/links/projects/YOLOX-jde"

mkdir -p "${ENV_ROOT}"

# -----------------------------
# Load modules
# -----------------------------
echo "[INFO] Loading modules..."

module --force purge

module load StdEnv/2023
module load gcc/12.3
module load python/3.10
module load cuda/12.2
module load cudnn/8.9.5.29

# IMPORTANT:
# OpenCV must be loaded before activating the virtual environment.
# If this version does not exist, run:
#   module spider opencv
# then replace opencv/4.10.0 with the available version.
module load opencv/4.10.0

echo "[INFO] Loaded modules:"
module list

# -----------------------------
# Create / reuse env
# -----------------------------
if [ -d "${ENV_DIR}" ]; then
    echo "[INFO] Reusing existing environment: ${ENV_DIR}"
else
    echo "[INFO] Creating new environment: ${ENV_DIR}"
    python -m venv "${ENV_DIR}"
fi

source "${ENV_DIR}/bin/activate"

echo "[INFO] Python info:"
which python
python --version

echo "[INFO] Pip info:"
which pip
pip --version

# -----------------------------
# Pip / setuptools
# -----------------------------
echo "[INFO] Installing stable pip tools..."

python -m pip install --upgrade pip

# IMPORTANT:
# setuptools >= 70 can break old torch cpp_extension import:
#   ModuleNotFoundError: No module named 'pkg_resources'
python -m pip install --no-cache-dir --force-reinstall \
    "setuptools==69.5.1" \
    wheel \
    packaging

echo "[INFO] Checking pkg_resources..."
python - <<'PY'
import pkg_resources
print("[OK] pkg_resources available")
PY

# -----------------------------
# PyTorch
# -----------------------------
echo "[INFO] Installing PyTorch..."

pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

echo "[INFO] Checking torch..."
python - <<'PY'
import torch
print("[OK] torch:", torch.__version__)
print("[OK] cuda available:", torch.cuda.is_available())
print("[OK] cuda device count:", torch.cuda.device_count())
PY

# -----------------------------
# Python dependencies
# -----------------------------
echo "[INFO] Installing Python dependencies..."

pip install --no-cache-dir \
    numpy \
    cython \
    loguru \
    tqdm \
    thop \
    ninja \
    tabulate \
    tensorboard \
    scipy \
    scikit-learn \
    pandas \
    matplotlib \
    seaborn \
    pyyaml \
    yacs \
    termcolor \
    gdown \
    lap \
    filterpy \
    h5py \
    easydict \
    motmetrics \
    pycocotools

# -----------------------------
# OpenCV check
# -----------------------------
echo "[INFO] Checking OpenCV from Compute Canada module..."

python - <<'PY'
import cv2
print("[OK] cv2 imported from:", cv2.__file__)
print("[OK] cv2 version:", cv2.__version__)
PY

# -----------------------------
# Install repo
# -----------------------------
echo "[INFO] Installing YOLOX-JDE repo..."

if [ ! -d "${REPO_DIR}" ]; then
    echo "[ERROR] Repo directory not found: ${REPO_DIR}"
    echo "Expected repo path:"
    echo "  ${REPO_DIR}"
    echo ""
    echo "Clone your repo first, for example:"
    echo "  mkdir -p ~/links/projects"
    echo "  cd ~/links/projects"
    echo "  git clone <YOUR_GITHUB_REPO_URL> YOLOX-jde"
    exit 1
fi

cd "${REPO_DIR}"

# IMPORTANT:
# setup.py imports torch, so pip build isolation fails because torch is invisible
# inside the temporary build environment.
pip install --no-cache-dir --no-build-isolation -e .

# -----------------------------
# Final checks
# -----------------------------
echo "============================================================"
echo "[INFO] Final environment check"
echo "============================================================"

python - <<'PY'
import torch
import cv2
import numpy as np
import pycocotools
import yolox

print("[OK] torch:", torch.__version__)
print("[OK] cuda available:", torch.cuda.is_available())
print("[OK] cuda device count:", torch.cuda.device_count())
print("[OK] cv2:", cv2.__version__)
print("[OK] numpy:", np.__version__)
print("[OK] pycocotools imported")
print("[OK] yolox imported")
PY

echo "============================================================"
echo "[DONE] Environment ready"
echo ""
echo "Activate later with:"
echo "  module --force purge"
echo "  module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2 cudnn/8.9.5.29 opencv/4.10.0"
echo "  source ${ENV_DIR}/bin/activate"
echo "============================================================"
