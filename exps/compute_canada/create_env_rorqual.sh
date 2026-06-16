#!/bin/bash
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/pyenv/Track}"

echo "============================================================"
echo "Creating YOLOX-JDE environment"
echo "ENV_DIR: $ENV_DIR"
echo "Python : 3.8"
echo "Torch  : 1.10.0+cu111"
echo "============================================================"


module --force purge
module load StdEnv/2020 || true
module load gcc/9.3.0 || true
module load cuda/11.1 || module load cuda/11.1.1 || true
module load python/3.8 || module load python/3.8.10

rm -rf "$ENV_DIR"

python -m virtualenv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade "pip==24.0" "setuptools<70" wheel
pip install "numpy<1.24" cython
# PyTorch 1.10.0 + CUDA 11.1
pip install \
  torch==1.10.0+cu111 \
  torchvision==0.11.1+cu111 \
  torchaudio==0.10.0 \
  -f https://download.pytorch.org/whl/torch_stable.html

# YOLOX / MOT / tracking dependencies
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
  faiss-cpu

# Optional but often needed by older fastreid/detectron-style code
pip install "protobuf<4" "setuptools<60"

echo "============================================================"
echo "Environment created"
echo "Activate with:"
echo "source $ENV_DIR/bin/activate"
echo "============================================================"

python - <<'PY'
import sys
import torch
print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
try:
    import faiss
    print("faiss:", faiss.__version__)
except Exception as e:
    print("faiss import failed:", e)
try:
    from cython_bbox import bbox_overlaps
    print("cython_bbox OK")
except Exception as e:
    print("cython_bbox failed:", e)
PY