#!/bin/bash
set -euo pipefail

REPO_DIR="${1:-$HOME/links/projects/YOLOX-jde}"
HOME_DIR="${HOME}"
ENV_ROOT="${HOME_DIR}/pyenv"
ENV_NAME="Track"
ENV_DIR="${ENV_ROOT}/${ENV_NAME}"

if [ ! -d "$REPO_DIR" ]; then
    echo "[ERROR] Repo directory not found: $REPO_DIR"
    echo "Pass the repo path explicitly, e.g."
    echo "  bash $0 $HOME/projects/def-mpederso/tnehdi/JDE_YOLOX"
    exit 1
fi

# ---- modules ----
module --force purge
module load StdEnv/2020

# Prefer Python 3.9 / 3.8 because Torch 1.10 is much safer there.
if module load python/3.9.6 2>/dev/null; then
    PY_MOD="python/3.9.6"
elif module load python/3.8.10 2>/dev/null; then
    PY_MOD="python/3.8.10"
else
    echo "[ERROR] Could not load python/3.9.6 or python/3.8.10."
    echo "Run: module spider python"
    exit 1
fi

module load cuda/11.4
module load gcc/9.3.0
module load opencv/4.6.0

if module load faiss/1.7.1 2>/dev/null; then
    FAISS_MOD="faiss/1.7.1"
else
    echo "[ERROR] Could not load faiss/1.7.1."
    echo "Run: module spider faiss/1.7.1"
    exit 1
fi

mkdir -p "$ENV_ROOT"

if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtualenv at $ENV_DIR"
    python -m venv "$ENV_DIR"
else
    echo "Reusing existing virtualenv at $ENV_DIR"
fi

source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# ---- core compatibility pins ----
# NumPy<1.24 avoids many np.float / np.int breakages in older code.
# Cython<3 is safer for older pycocotools / cython_bbox builds.
# Pillow<10 is a conservative choice for old CV code.
pip install --no-cache-dir --force-reinstall \
    "numpy<1.24" \
    "scipy==1.10.1" \
    "Cython<3" \
    "Pillow<10" \
    "setuptools<70"

# ---- PyTorch ----
# Keep this conservative for old JDE_YOLOX / FastReID code.
pip install --no-cache-dir --force-reinstall \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    torch==1.10.0+cu113 \
    torchvision==0.11.1+cu113 \
    torchaudio==0.10.0+cu113

# ---- main JDE_YOLOX / YOLOX deps ----
# opencv-python is intentionally NOT installed by pip because we already load
# the Alliance OpenCV module above.
pip install --no-cache-dir \
    loguru \
    scikit-image \
    tqdm \
    thop \
    ninja \
    tabulate \
    tensorboard \
    lap \
    filterpy \
    h5py \
    imageio \
    timm==0.5.4 \
    loralib \
    scikit-learn \
    pandas \
    xmltodict \
    matplotlib \
    pycocotools \
    cython_bbox \
    prettytable \
    easydict \
    pyyaml \
    yacs \
    termcolor \
    gdown

# ---- optional ONNX export stack ----
# Uncomment only if you need export / deployment tools.
# pip install --no-cache-dir onnx==1.8.1 onnxruntime==1.8.0 onnx-simplifier==0.3.5

# ---- install repo in develop mode ----
cd "$REPO_DIR"
python setup.py develop

# Make the bundled fast_reid package easy to import later.
ACTIVATE_EXTRA="$ENV_DIR/bin/activate_track"
cat > "$ACTIVATE_EXTRA" <<ACTEOF
#!/bin/bash
module --force purge
module load StdEnv/2020
module load ${PY_MOD}
module load cuda/11.4
module load gcc/9.3.0
module load opencv/4.6.0
module load ${FAISS_MOD}
source "$ENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/fast_reid:\${PYTHONPATH:-}"
cd "$REPO_DIR"
ACTEOF
chmod +x "$ACTIVATE_EXTRA"

# Also export PYTHONPATH in the current shell for the sanity checks below.
export PYTHONPATH="$REPO_DIR:$REPO_DIR/fast_reid:${PYTHONPATH:-}"

# ---- sanity checks ----
python - <<'PY'
import sys
import torch
import torchvision
import numpy as np
import cv2
import scipy
import faiss

print("Python      :", sys.version.split()[0])
print("Torch       :", torch.__version__)
print("Torchvision :", torchvision.__version__)
print("NumPy       :", np.__version__)
print("SciPy       :", scipy.__version__)
print("OpenCV      :", cv2.__version__)
print("FAISS       : OK")
PY

python - <<'PY'
import yolox
from fast_reid.fast_reid_interfece import FastReIDInterface  # noqa: F401
print("YOLOX import            : OK")
print("FastReID interface import: OK")
PY

echo
echo "✅ Environment is ready."
echo "Repo      : $REPO_DIR"
echo "Env path  : $ENV_DIR"
echo
echo "Use it later with:"
echo "  source $ACTIVATE_EXTRA"
echo
echo "Notes:"
echo "  1) This setup is intentionally conservative for old JDE_YOLOX / FastReID code."
echo "  2) Run this script with bash, not sh."
echo "  3) If you reuse an old env, stale packages may remain; a fresh env is safer."