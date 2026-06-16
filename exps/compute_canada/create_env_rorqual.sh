#!/bin/bash
# Robust Rorqual / Compute Canada environment for YOLOX-JDE on H100.
#
# Final clean stack:
#   - StdEnv/2023
#   - gcc/12.3
#   - cuda/12.2
#   - python/3.11
#   - PyTorch 2.5.1 + CUDA 12.4 wheel
#   - OpenCV from Compute Canada module
#   - FAISS from Compute Canada module
#
# Why Python 3.11:
#   On Rorqual, loading opencv/faiss under StdEnv/2023 reloads python/3.10 to python/3.11.
#   So we use Python 3.11 consistently instead of fighting the module stack.
#
# What this script fixes:
#   - H100 requires modern PyTorch. Old torch 1.10/1.11 CUDA 11.x fails with:
#       "no kernel image is available for execution on the device"
#   - Compute Canada blocks pip OpenCV; use the opencv module.
#   - FAISS should come from the Compute Canada faiss module, not pip faiss-cpu.
#   - YOLOX setup.py imports torch; editable install must use --no-build-isolation.
#   - Old FastReID code uses collections.Mapping; patch to collections.abc.Mapping.

set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
    echo "[ERROR] Please run this script with bash, not sh."
    echo "Use:"
    echo "  bash exps/compute_canada/create_env_rorqual.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
ENV_DIR="${ENV_DIR:-$HOME/pyenv/Track}"

echo "============================================================"
echo "Creating YOLOX-JDE H100 environment"
echo "REPO_DIR: $REPO_DIR"
echo "ENV_DIR : $ENV_DIR"
echo "Stack   : StdEnv/2023 + gcc/12.3 + cuda/12.2 + python/3.11"
echo "Torch   : 2.5.1 + CUDA 12.4 wheel"
echo "OpenCV  : Compute Canada module"
echo "FAISS   : Compute Canada faiss/1.7.4 module"
echo "============================================================"

if [ ! -d "$REPO_DIR" ]; then
    echo "[ERROR] REPO_DIR not found: $REPO_DIR"
    exit 1
fi

if [ ! -f "$REPO_DIR/tools/train.py" ]; then
    echo "[ERROR] This does not look like the YOLOX-JDE repo: $REPO_DIR"
    echo "Expected: $REPO_DIR/tools/train.py"
    exit 1
fi

echo "==> Loading Compute Canada modules"
module --force purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.11

# Compute Canada OpenCV must be loaded as a module, not installed with pip.
if ! module load opencv; then
    echo "[ERROR] Could not load OpenCV module."
    echo "Run:"
    echo "  module spider opencv"
    exit 1
fi

# Your module spider output says faiss/1.7.4 requires:
#   StdEnv/2023 gcc/12.3 cuda/12.2
# and compatible python/3.10 or python/3.11.
if ! module load faiss/1.7.4; then
    echo "[ERROR] Could not load faiss/1.7.4."
    echo "Run:"
    echo "  module spider faiss/1.7.4"
    exit 1
fi

echo "==> Module stack"
module list || true

echo "==> Removing old environment: $ENV_DIR"
rm -rf "$ENV_DIR"

echo "==> Creating virtualenv"
# --system-site-packages lets the venv see module-provided Python extensions
# such as cv2 and faiss through the module environment.
python -m virtualenv --system-site-packages "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python - <<'PY'
import sys
print("Python executable:", sys.executable)
print("Python version   :", sys.version)
assert sys.version_info[:2] == (3, 11), "This env must use python/3.11"
PY

echo "==> Installing build tools"
python -m pip install --upgrade "pip<25" "setuptools<70" "wheel<0.45"

echo "==> Installing core numeric/build packages"
python -m pip install --no-cache-dir \
    "numpy==1.26.4" \
    "cython" \
    "ninja" \
    "packaging" \
    "typing_extensions<5"

echo "==> Installing H100-compatible PyTorch trio"
# Official matching PyTorch 2.5.1 wheel set:
#   torch==2.5.1
#   torchvision==0.20.1
#   torchaudio==2.5.1
#   CUDA 12.4 wheels
python -m pip uninstall -y torch torchvision torchaudio || true
python -m pip install --no-cache-dir --force-reinstall \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

echo "==> Verifying torch/torchvision before installing the rest"
python - <<'PY'
import torch
import torchvision
from torchvision.ops import nms

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("torchvision nms OK")
PY

echo "==> Installing YOLOX / tracking / FastReID Python dependencies"
# Important:
#   - do NOT install opencv-python
#   - do NOT install opencv-python-headless
#   - do NOT install faiss-cpu
#   - do NOT install faiss-gpu
#
# OpenCV comes from module load opencv.
# FAISS comes from module load faiss/1.7.4.
python -m pip install --no-cache-dir \
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
    yacs \
    termcolor \
    tensorboard \
    pycocotools \
    cython_bbox \
    filterpy \
    lap \
    h5py \
    scikit-learn \
    pyyaml \
    prettytable \
    easydict \
    gdown \
    xmltodict \
    "protobuf<4"

echo "==> Patching old FastReID imports for Python 3.10+"
python - <<'PY'
from pathlib import Path
import os

repo = Path(os.environ["REPO_DIR"])

for p in repo.rglob("*.py"):
    # Only patch FastReID files to avoid touching unrelated code too much.
    if "fast_reid" not in str(p):
        continue

    s = p.read_text(errors="ignore")
    old = s

    s = s.replace(
        "from collections import Mapping, OrderedDict",
        "from collections import OrderedDict\nfrom collections.abc import Mapping",
    )
    s = s.replace(
        "from collections import OrderedDict, Mapping",
        "from collections import OrderedDict\nfrom collections.abc import Mapping",
    )
    s = s.replace(
        "from collections import Mapping",
        "from collections.abc import Mapping",
    )
    s = s.replace(
        "from collections import MutableMapping",
        "from collections.abc import MutableMapping",
    )
    s = s.replace(
        "from collections import Sequence",
        "from collections.abc import Sequence",
    )
    s = s.replace(
        "from collections import Iterable",
        "from collections.abc import Iterable",
    )

    if s != old:
        p.write_text(s)
        print("[PATCHED]", p)
PY

echo "==> Installing YOLOX editable without build isolation"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export MAX_JOBS="${MAX_JOBS:-4}"

# Important:
# setup.py imports torch.
# Without --no-build-isolation, pip creates a temporary build env with no torch and fails.
python -m pip install -e . --no-build-isolation --no-deps

echo "==> Writing activation helper"
cat > "$REPO_DIR/exps/compute_canada/activate_track_h100.sh" <<'SH'
#!/bin/bash
module --force purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.11
module load opencv
module load faiss/1.7.4
source "$HOME/pyenv/Track/bin/activate"
export PYTHONPATH="$HOME/links/projects/YOLOX-jde:${PYTHONPATH:-}"
SH

chmod +x "$REPO_DIR/exps/compute_canada/activate_track_h100.sh"

echo "==> Final verification"
python - <<'PY'
import sys
print("python:", sys.version)

import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

import torchvision
print("torchvision:", torchvision.__version__)

from torchvision.ops import nms
print("torchvision nms OK")

import cv2
print("opencv:", cv2.__version__)

import faiss
print("faiss:", faiss.__version__)
try:
    print("faiss gpus:", faiss.get_num_gpus())
except Exception as e:
    print("faiss gpu check failed:", e)

from cython_bbox import bbox_overlaps
print("cython_bbox OK")

from filterpy.kalman import KalmanFilter
print("filterpy OK")

import sklearn
print("sklearn:", sklearn.__version__)

import yolox
print("yolox:", yolox.__file__)

# This verifies the YOLOX C++ extension used by COCOeval_opt.
from yolox.layers import COCOeval_opt
print("YOLOX COCOeval_opt OK")

# This verifies FastReID import path and faiss dependency.
from fast_reid.fast_reid_interfece import FastReIDInterface
print("FastReIDInterface OK")
PY

echo "============================================================"
echo "Environment created successfully."
echo
echo "To activate manually:"
echo "  source $REPO_DIR/exps/compute_canada/activate_track_h100.sh"
echo
echo "To submit detector training:"
echo "  sbatch $REPO_DIR/exps/compute_canada/train_dancetrack_detector_x_1gpu_rorqual.slurm"
echo "============================================================"