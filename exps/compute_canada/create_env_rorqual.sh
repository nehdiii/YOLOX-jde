#!/bin/bash
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/pyenv/Track}"

echo "============================================================"
echo "Creating YOLOX-JDE H100 environment"
echo "ENV_DIR: $ENV_DIR"
echo "Python : 3.10"
echo "Torch  : 2.5.1+cu124"
echo "GPU    : H100 compatible"
echo "============================================================"

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.10
module load opencv

rm -rf "$ENV_DIR"

python -m virtualenv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

pip install "numpy<2" cython

pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

pip install \
  loguru \
  tqdm \
  tabulate \
  matplotlib \
  seaborn \
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
  filterpy \
  "typing_extensions<5" \
  "protobuf<4"

echo "============================================================"
echo "Environment created"
echo "Activate with:"
echo "module --force purge"
echo "module load StdEnv/2023"
echo "module load gcc/12.3"
echo "module load cuda/12.2"
echo "module load python/3.10"
echo "module load opencv"
echo "source $ENV_DIR/bin/activate"
echo "============================================================"

python - <<'PY'
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

try:
    import cv2
    print("opencv:", cv2.__version__)
except Exception as e:
    print("opencv failed:", e)

try:
    from cython_bbox import bbox_overlaps
    print("cython_bbox OK")
except Exception as e:
    print("cython_bbox failed:", e)

try:
    from filterpy.kalman import KalmanFilter
    print("filterpy OK")
except Exception as e:
    print("filterpy failed:", e)
PY