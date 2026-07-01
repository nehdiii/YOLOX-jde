#!/bin/bash
# Robust Fir / Compute Canada environment for YOLOX-JDE on H100.
#
# Final Compute Canada stack:
#   - StdEnv/2023
#   - gcc/12.3
#   - cuda/12.2
#   - python/3.11
#   - torch / torchvision / torchaudio from Compute Canada wheelhouse
#   - OpenCV from Compute Canada module
#   - FAISS from Compute Canada module
#
# Important:
#   On login node, torch.cuda.is_available() can be False.
#   That is normal. It should be True inside a Slurm GPU job.
#
# Do NOT:
#   - use torch 1.10/1.11 on H100
#   - pip install opencv-python or opencv-python-headless
#   - pip install faiss-cpu/faiss-gpu
#   - mix torch from Compute Canada with torchvision from PyTorch cu124
#
# This script also safely patches old FastReID code for:
#   - Python 3.11 collections.abc changes
#   - PyTorch 2.x removal of torch._six

set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
    echo "[ERROR] Please run this script with bash, not sh."
    echo "Use:"
    echo "  bash exps/compute_canada/create_env_fir.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
ENV_DIR="${ENV_DIR:-$HOME/pyenv/Track}"

export REPO_DIR
export ENV_DIR

echo "============================================================"
echo "Creating YOLOX-JDE H100 environment"
echo "REPO_DIR: $REPO_DIR"
echo "ENV_DIR : $ENV_DIR"
echo "Stack   : StdEnv/2023 + gcc/12.3 + cuda/12.2 + python/3.11"
echo "Torch   : Compute Canada torch/torchvision/torchaudio"
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

if ! module load opencv; then
    echo "[ERROR] Could not load OpenCV module."
    echo "Run: module spider opencv"
    exit 1
fi

if ! module load faiss/1.7.4; then
    echo "[ERROR] Could not load faiss/1.7.4."
    echo "Run: module spider faiss/1.7.4"
    exit 1
fi

echo "==> Module stack"
module list || true

echo "==> Restoring FastReID source before patching"
cd "$REPO_DIR"

if [ -d "$REPO_DIR/.git" ] && [ -d "$REPO_DIR/fast_reid" ]; then
    BACKUP_DIR="$REPO_DIR/fast_reid_backup_before_env_patch_$(date +%Y%m%d_%H%M%S)"
    cp -a "$REPO_DIR/fast_reid" "$BACKUP_DIR"
    echo "Backed up current fast_reid to:"
    echo "  $BACKUP_DIR"

    # This removes the corrupted patch from the previous attempt.
    git checkout -- fast_reid || {
        echo "[WARN] git checkout -- fast_reid failed. Continuing with repair patch."
    }
else
    echo "[WARN] Git repo or fast_reid folder not found. Continuing without git restore."
fi

echo "==> Removing old environment: $ENV_DIR"
rm -rf "$ENV_DIR"

echo "==> Creating virtualenv"
python -m virtualenv --system-site-packages "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python - <<'PY'
import sys
print("Python executable:", sys.executable)
print("Python version   :", sys.version)
assert sys.version_info[:2] == (3, 11), "This env must use python/3.11"
PY

echo "==> Upgrading build tools"
python -m pip install --upgrade "pip<25" "setuptools<70" "wheel<0.45"

echo "==> Cleaning previous broken torch stack"
python -m pip uninstall -y torch torchvision torchaudio || true
python -m pip uninstall -y torch torchvision torchaudio || true

SITE_PACKAGES="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

rm -rf "$SITE_PACKAGES/torch" \
       "$SITE_PACKAGES/torch-"*.dist-info \
       "$SITE_PACKAGES/torchvision" \
       "$SITE_PACKAGES/torchvision-"*.dist-info \
       "$SITE_PACKAGES/torchaudio" \
       "$SITE_PACKAGES/torchaudio-"*.dist-info || true

echo "==> Installing stable base packages"
python -m pip install --no-cache-dir --force-reinstall \
    "numpy==1.26.4" \
    "typing_extensions<5" \
    "packaging" \
    "cython" \
    "ninja"

echo "==> Installing Compute Canada torch stack"
python -m pip install --no-cache-dir --force-reinstall \
    "torch==2.5.1+computecanada" \
    "torchvision==0.20.1+computecanada" \
    "torchaudio==2.5.1+computecanada"

echo "==> Verifying torch stack"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

import torchvision
print("torchvision:", torchvision.__version__)

from torchvision.ops import nms
print("torchvision nms OK")

assert torch.__version__.startswith("2.5.1"), f"Unexpected torch version: {torch.__version__}"
assert torchvision.__version__.startswith("0.20.1"), f"Unexpected torchvision version: {torchvision.__version__}"
assert torch.version.cuda is not None, "Torch has no CUDA support"
print("torch stack verification OK")
PY

echo "==> Re-pinning NumPy after torch install"
python -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

echo "==> Installing YOLOX / tracking / FastReID Python dependencies"
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

echo "==> Installing cython_bbox without build isolation"
python -m pip install --no-cache-dir --no-build-isolation cython_bbox

echo "==> Final NumPy pin"
python -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

echo "==> Safely patching old FastReID imports for Python 3.11 + PyTorch 2.x"
python - <<'PY'
from pathlib import Path
import os

repo = Path(os.environ["REPO_DIR"])

def patch_file(path: Path):
    text = path.read_text(errors="ignore")
    original = text
    new_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        indent = line[:len(line) - len(line.lstrip())]

        # Repair corruption produced by the previous bad patch.
        if stripped in {".abc import Mapping", ".abc import Iterable", ".abc import Sequence", ".abc import MutableMapping"}:
            name = stripped.replace(".abc import ", "")
            new_lines.append(f"{indent}from collections.abc import {name}")
            continue

        # Python 3.10+ collections fixes.
        if stripped == "from collections import Mapping, OrderedDict":
            new_lines.append(f"{indent}from collections import OrderedDict")
            new_lines.append(f"{indent}from collections.abc import Mapping")
            continue

        if stripped == "from collections import OrderedDict, Mapping":
            new_lines.append(f"{indent}from collections import OrderedDict")
            new_lines.append(f"{indent}from collections.abc import Mapping")
            continue

        if stripped == "from collections import Mapping":
            new_lines.append(f"{indent}from collections.abc import Mapping")
            continue

        if stripped == "from collections import MutableMapping":
            new_lines.append(f"{indent}from collections.abc import MutableMapping")
            continue

        if stripped == "from collections import Sequence":
            new_lines.append(f"{indent}from collections.abc import Sequence")
            continue

        if stripped == "from collections import Iterable":
            new_lines.append(f"{indent}from collections.abc import Iterable")
            continue

        # PyTorch 2.x removed torch._six.
        if stripped.startswith("from torch._six import "):
            names = [x.strip() for x in stripped.replace("from torch._six import ", "").split(",")]
            for name in names:
                if name == "container_abcs":
                    new_lines.append(f"{indent}import collections.abc as container_abcs")
                elif name == "string_classes":
                    new_lines.append(f"{indent}string_classes = (str,)")
                elif name == "int_classes":
                    new_lines.append(f"{indent}int_classes = (int,)")
                else:
                    new_lines.append(f"{indent}# Removed unsupported torch._six import: {name}")
            continue

        new_lines.append(line)

    patched = "\n".join(new_lines) + "\n"

    if patched != original:
        path.write_text(patched)
        print("[PATCHED]", path)

for p in repo.rglob("*.py"):
    if "fast_reid" not in str(p):
        continue
    patch_file(p)
PY

echo "==> Checking FastReID source syntax"
python - <<'PY'
import compileall
import os
from pathlib import Path

repo = Path(os.environ["REPO_DIR"])
fast_reid_dir = repo / "fast_reid"

ok = compileall.compile_dir(str(fast_reid_dir), quiet=1)
if not ok:
    raise SystemExit("[ERROR] FastReID syntax check failed.")
print("FastReID syntax check OK.")
PY

echo "==> Checking that torch._six imports are gone"
python - <<'PY'
from pathlib import Path
import os

repo = Path(os.environ["REPO_DIR"])
bad = []

for p in repo.rglob("*.py"):
    if "fast_reid" not in str(p):
        continue
    s = p.read_text(errors="ignore")
    if "torch._six" in s:
        bad.append(str(p))

if bad:
    print("[ERROR] Still found torch._six in:")
    for x in bad:
        print("  ", x)
    raise SystemExit(1)

print("No torch._six imports left in fast_reid.")
PY

echo "==> Installing YOLOX editable without build isolation"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export MAX_JOBS="${MAX_JOBS:-4}"

python -m pip install -e . --no-build-isolation --no-deps

echo "==> Writing activation helper"
cat > "$REPO_DIR/exps/compute_canada/activate_track_fir.sh" <<SH
#!/bin/bash
module --force purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.11
module load opencv
module load faiss/1.7.4
source "$ENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR:\${PYTHONPATH:-}"
SH

chmod +x "$REPO_DIR/exps/compute_canada/activate_track_fir.sh"

echo "==> Final verification"
python - <<'PY'
import sys
print("python:", sys.version)

import numpy as np
print("numpy:", np.__version__)

import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
else:
    print("No GPU visible here. This is normal on login node.")

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

from yolox.layers import COCOeval_opt
print("YOLOX COCOeval_opt OK")

from fast_reid.fast_reid_interfece import FastReIDInterface
print("FastReIDInterface OK")
PY

echo "============================================================"
echo "Environment created successfully."
echo
echo "To activate manually:"
echo "  source $REPO_DIR/exps/compute_canada/activate_track_fir.sh"
echo
echo "Fir repeated-run Slurm scripts use:"
echo "  ACTIVATE_SCRIPT=$REPO_DIR/exps/compute_canada/activate_track_fir.sh"
echo "============================================================"
