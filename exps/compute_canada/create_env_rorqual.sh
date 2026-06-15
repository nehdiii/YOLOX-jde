#!/bin/bash
set -euo pipefail

# ============================================================
# Rorqual / Compute Canada environment for YOLOX-JDE
# Important:
#   OpenCV must be loaded as a module BEFORE activating venv.
#   Do NOT rely only on pip opencv-python / opencv-python-headless.
# ============================================================

echo "============================================================"
echo "[INFO] Creating YOLOX-JDE environment on Compute Canada/Rorqual"
echo "============================================================"

# -----------------------------
# 0. Paths
# -----------------------------
ENV_ROOT="${HOME}/pyenv"
ENV_NAME="Track"
ENV_DIR="${ENV_ROOT}/${ENV_NAME}"

REPO_DIR="${HOME}/links/projects/YOLOX-jde"

mkdir -p "${ENV_ROOT}"

# -----------------------------
# 1. Load modules
# -----------------------------
echo "[INFO] Loading Compute Canada modules..."

module --force purge

# Base toolchain
module load StdEnv/2023
module load gcc/12.3
module load python/3.10
module load cuda/12.2
module load cudnn/8.9.5.29

# Critical: OpenCV module must be loaded BEFORE venv activation.
# If this exact version does not exist on Rorqual, run:
#   module spider opencv
# and replace opencv/4.10.0 with the available one.
module load opencv/4.10.0

echo "[INFO] Loaded modules:"
module list

# -----------------------------
# 2. Create virtual environment
# -----------------------------
if [ -d "${ENV_DIR}" ]; then
    echo "[INFO] Existing environment found at ${ENV_DIR}"
    echo "[INFO] Reusing it. To recreate from zero, remove it manually:"
    echo "       rm -rf ${ENV_DIR}"
else
    echo "[INFO] Creating virtual environment at ${ENV_DIR}"
    python -m venv "${ENV_DIR}"
fi

# Activate after module load
source "${ENV_DIR}/bin/activate"

echo "[INFO] Python:"
which python
python --version

echo "[INFO] Pip:"
which pip
pip --version

# -----------------------------
# 3. Upgrade pip tools
# -----------------------------
echo "[INFO] Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

# -----------------------------
# 4. Install PyTorch
# -----------------------------
echo "[INFO] Installing PyTorch..."

# For CUDA 12.2, cu121 wheels usually work.
# If Compute Canada module CUDA changes, this can still work because wheels bundle CUDA runtime.
pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# -----------------------------
# 5. Install core Python packages
# -----------------------------
echo "[INFO] Installing core packages..."

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
# 6. Important OpenCV handling
# -----------------------------
echo "[INFO] Checking OpenCV from module..."

python - <<'PY'
import cv2
print("[OK] cv2 imported from:", cv2.__file__)
print("[OK] cv2 version:", cv2.__version__)
PY

# Do NOT pip install opencv-python-headless on Compute Canada.
# Compute Canada provides OpenCV through module load opencv/x.y.z.
# Installing opencv-python-headless may trigger:
# opencv_noinstall-9999+dummy.computecanada.tar.gz error.

# -----------------------------
# 7. Install YOLOX repo in editable mode
# -----------------------------
if [ -d "${REPO_DIR}" ]; then
    echo "[INFO] Installing repo in editable mode: ${REPO_DIR}"
    cd "${REPO_DIR}"
    pip install --no-cache-dir -e .
else
    echo "[WARNING] Repo directory does not exist yet: ${REPO_DIR}"
    echo "[WARNING] Clone your repo there, then run:"
    echo "          cd ${REPO_DIR}"
    echo "          source ${ENV_DIR}/bin/activate"
    echo "          pip install -e ."
fi

# -----------------------------
# 8. Final checks
# -----------------------------
echo "============================================================"
echo "[INFO] Final environment check"
echo "============================================================"

python - <<'PY'
import torch
import cv2
import pycocotools
import numpy as np

print("[OK] torch:", torch.__version__)
print("[OK] cuda available:", torch.cuda.is_available())
print("[OK] cuda device count:", torch.cuda.device_count())
print("[OK] cv2:", cv2.__version__)
print("[OK] numpy:", np.__version__)

try:
    import yolox
    print("[OK] yolox imported")
except Exception as e:
    print("[WARNING] yolox import failed:", repr(e))
PY

echo "============================================================"
echo "[DONE] Environment ready"
echo "Activate with:"
echo "source ${ENV_DIR}/bin/activate"
echo "============================================================"