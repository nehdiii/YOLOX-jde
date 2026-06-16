#!/bin/bash
# Robust Rorqual / Compute Canada environment for YOLOX-JDE on H100.
#
# What this script fixes:
#   - H100 requires a modern PyTorch build. Old torch 1.10/1.11 CUDA 11.x fails with:
#       "no kernel image is available for execution on the device"
#   - Compute Canada blocks pip OpenCV; use the opencv module.
#   - FAISS should come from the Compute Canada faiss module, not pip faiss-cpu.
#   - YOLOX setup.py imports torch; editable install must use --no-build-isolation.
#   - Old FastReID code uses collections.Mapping; patch to collections.abc.Mapping.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
ENV_DIR="${ENV_DIR:-$HOME/pyenv/Track}"

echo "============================================================"
echo "Creating YOLOX-JDE H100 environment"
echo "REPO_DIR: $REPO_DIR"
echo "ENV_DIR : $ENV_DIR"
echo "Stack   : StdEnv/2023 + gcc/12.3 + cuda/12.2 + python/3.10"
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
module load python/3.10

# Compute Canada OpenCV must be loaded as a module, not installed with pip.
if ! module load opencv; then
    echo "[ERROR] Could not load OpenCV module."
    echo "Run: module spider opencv"
    echo "Then replace 'module load opencv' with the exact version if needed."
    exit 1
fi

# Your module spider output says faiss/1.7.4 requires:
#   StdEnv/2023 gcc/12.3 cuda/12.2
# and compatible python/3.10 or python/3.11.
if ! module load faiss/1.7.4; then
    echo "[ERROR] Could not load faiss/1.7.4."
    echo "Run: module spider faiss/1.7.4"
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
assert sys.version_info[:2] == (3, 10), "This env must use python/3.10"
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
# torch==2.5.1, torchvision==0.20.1, torchaudio==2.5.1, CUDA 12.4 wheels.
python -m pip uninstall -y torch torchvision torchaudio || true
python -m pip install --no-cache-dir --force-reinstall \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

echo "==> Installing YOLOX / tracking / FastReID Python dependencies"
# Do NOT install opencv-python or opencv-python-headless on Compute Canada.
# Do NOT install faiss-cpu/faiss-gpu with pip here; FAISS is loaded as a module.
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

repo = Path(__import__("os").environ["REPO_DIR"])
targets = [
    repo / "fast_reid/fastreid/evaluation/testing.py",
    repo / "fast_reid/fastreid/data/build.py",
]

for p in targets:
    if not p.exists():
        print("[WARN] missing:", p)
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
    if s != old:
        p.write_text(s)
        print("[PATCHED]", p)
    else:
        print("[OK]", p)
PY

echo "==> Installing YOLOX editable without build isolation"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export MAX_JOBS="${MAX_JOBS:-4}"

# Important: setup.py imports torch. Without --no-build-isolation, pip creates a temp
# build env with no torch and fails with "ModuleNotFoundError: torch".
python -m pip install -e . --no-build-isolation --no-deps

echo "==> Writing activation helper"
cat > "$REPO_DIR/exps/compute_canada/activate_track_h100.sh" <<'SH'
#!/bin/bash
module --force purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.10
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