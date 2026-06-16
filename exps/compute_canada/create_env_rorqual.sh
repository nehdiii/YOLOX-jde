#!/bin/bash
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/pyenv/Track}"

echo "============================================================"
echo "Creating YOLOX-JDE environment"
echo "ENV_DIR: $ENV_DIR"
echo "Python : 3.9"
echo "Torch  : 1.11.0+cu113"
echo "FAISS  : disabled for detector/JDE training env"
echo "============================================================"

module --force purge
module load StdEnv/2020 || true
module load gcc/9.3.0 || true
module load python/3.9 || module load python/3.9.6 || module load python/3.9.13

rm -rf "$ENV_DIR"

python -m virtualenv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade "pip<25" "setuptools<70" "wheel<0.45"

# Keep NumPy old enough for YOLOX / PyTorch 1.x / cython_bbox compatibility
pip install "numpy==1.23.5" cython

# PyTorch 1.11.0 + CUDA 11.3
# We do not need a cuda/11.3 module here; the PyTorch wheel includes cu113 runtime.
pip install \
  torch==1.11.0+cu113 \
  torchvision==0.12.0+cu113 \
  torchaudio==0.11.0 \
  --extra-index-url https://download.pytorch.org/whl/cu113

# YOLOX / MOT / training dependencies
# IMPORTANT: do NOT install faiss-cpu here.
pip install \
  loguru \
  tqdm \
  tabulate \
  matplotlib \
  seaborn \
  opencv-python-headless \
  scikit-image \
  scipy \
  pandas \
  pillow \
  thop \
  ninja \
  yacs \
  termcolor \
  tensorboard \
  pycocotools \
  cython_bbox \
  "protobuf<4"

echo "============================================================"
echo "Environment created"
echo "Activate with:"
echo "module --force purge"
echo "module load StdEnv/2020"
echo "module load gcc/9.3.0"
echo "module load python/3.9"
echo "source $ENV_DIR/bin/activate"
echo "============================================================"

python - <<'PY'
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

try:
    import torchvision
    print("torchvision:", torchvision.__version__)
except Exception as e:
    print("torchvision import failed:", e)

try:
    import torchaudio
    print("torchaudio:", torchaudio.__version__)
except Exception as e:
    print("torchaudio import failed:", e)

try:
    from cython_bbox import bbox_overlaps
    print("cython_bbox OK")
except Exception as e:
    print("cython_bbox failed:", e)

try:
    import cv2
    print("opencv:", cv2.__version__)
except Exception as e:
    print("opencv import failed:", e)

try:
    import pycocotools
    print("pycocotools OK")
except Exception as e:
    print("pycocotools failed:", e)
PY