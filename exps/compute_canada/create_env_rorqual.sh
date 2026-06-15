#!/bin/bash
set -euo pipefail

echo "============================================================"
echo "[INFO] Creating YOLOX-JDE environment on Compute Canada/Rorqual"
echo "============================================================"

ENV_ROOT="${HOME}/pyenv"
ENV_NAME="Track"
ENV_DIR="${ENV_ROOT}/${ENV_NAME}"
REPO_DIR="${HOME}/links/projects/YOLOX-jde"

mkdir -p "${ENV_ROOT}"

echo "[INFO] Loading modules..."

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load python/3.10
module load cuda/12.2
module load cudnn/8.9.5.29
module load opencv/4.10.0

module list

if [ -d "${ENV_DIR}" ]; then
    echo "[INFO] Reusing existing env: ${ENV_DIR}"
else
    echo "[INFO] Creating env: ${ENV_DIR}"
    python -m venv "${ENV_DIR}"
fi

source "${ENV_DIR}/bin/activate"

echo "[INFO] Python:"
which python
python --version

echo "[INFO] Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

echo "[INFO] Installing PyTorch..."
pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

echo "[INFO] Checking torch..."
python - <<'PY'
import torch
print("[OK] torch:", torch.__version__)
print("[OK] cuda available:", torch.cuda.is_available())
PY

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

echo "[INFO] Checking OpenCV from module..."

python - <<'PY'
import cv2
print("[OK] cv2 imported from:", cv2.__file__)
print("[OK] cv2 version:", cv2.__version__)
PY

echo "[INFO] Installing YOLOX-JDE repo..."

if [ -d "${REPO_DIR}" ]; then
    cd "${REPO_DIR}"

    # Important:
    # YOLOX setup.py imports torch.
    # Normal pip editable install uses build isolation, where torch is invisible.
    # Therefore we must disable build isolation.
    pip install --no-cache-dir --no-build-isolation -e .
else
    echo "[ERROR] Repo directory not found: ${REPO_DIR}"
    echo "Clone your repo first:"
    echo "  mkdir -p ~/links/projects"
    echo "  cd ~/links/projects"
    echo "  git clone <your_repo_url> YOLOX-jde"
    exit 1
fi

echo "============================================================"
echo "[INFO] Final check"
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
print("[OK] yolox imported")
PY

echo "============================================================"
echo "[DONE] Environment ready"
echo "Activate later with:"
echo "source ~/pyenv/Track/bin/activate"
echo "============================================================"