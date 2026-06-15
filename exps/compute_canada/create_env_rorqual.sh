#!/bin/bash
# Compute Canada / Alliance environment setup for the YOLOX-JDE DanceTrack repo.
# This follows the structure of JDE_YOLOX-main(27)/exps/compute_canada/create_env.sh,
# but makes paths configurable for the Rorqual layout.

set -euo pipefail

REPO_DIR="${1:-${REPO_DIR:-$HOME/links/projects/YOLOX-jde}}"
ENV_ROOT="${ENV_ROOT:-$HOME/pyenv}"
ENV_NAME="${ENV_NAME:-Track}"
ENV_DIR="$ENV_ROOT/$ENV_NAME"

if [ ! -d "$REPO_DIR" ]; then
    echo "[ERROR] Repo directory not found: $REPO_DIR"
    echo "Pass it explicitly, e.g."
    echo "  bash $0 $HOME/links/projects/YOLOX-jde"
    exit 1
fi

# -------- modules: same philosophy as the old repo, but tolerant to Rorqual availability --------
module --force purge
if module load StdEnv/2023 2>/dev/null; then
    STDENV_MOD="StdEnv/2023"
elif module load StdEnv/2020 2>/dev/null; then
    STDENV_MOD="StdEnv/2020"
else
    echo "[WARN] Could not load StdEnv/2023 or StdEnv/2020; continuing with current module environment."
    STDENV_MOD=""
fi

load_first_module() {
    local outvar="$1"
    shift
    local mod
    for mod in "$@"; do
        if module load "$mod" 2>/dev/null; then
            printf -v "$outvar" '%s' "$mod"
            return 0
        fi
    done
    return 1
}

if ! load_first_module PY_MOD python/3.10 python/3.9.6 python/3.9 python/3.8.10 python/3.8; then
    echo "[ERROR] Could not load a Python module. Run: module spider python"
    exit 1
fi

# CUDA is mainly needed for compiling/runtime compatibility. PyTorch wheels include CUDA runtime libs.
if load_first_module CUDA_MOD cuda/11.8 cuda/11.7 cuda/11.4 cuda/12.2 cuda/12.1; then
    echo "Loaded CUDA module: $CUDA_MOD"
else
    echo "[WARN] Could not load a CUDA module. Run: module spider cuda"
    CUDA_MOD=""
fi

if load_first_module GCC_MOD gcc/12.3 gcc/11.3 gcc/9.3.0 gcc/9.3; then
    echo "Loaded GCC module: $GCC_MOD"
else
    echo "[WARN] Could not load a GCC module. Continuing."
    GCC_MOD=""
fi

# Prefer system OpenCV when present, like old repo. If unavailable, install opencv-python-headless below.
if load_first_module OPENCV_MOD opencv/4.8.0 opencv/4.6.0 opencv/4.5.5; then
    echo "Loaded OpenCV module: $OPENCV_MOD"
    INSTALL_PIP_OPENCV=0
else
    echo "[WARN] OpenCV module not found; pip will install opencv-python-headless."
    OPENCV_MOD=""
    INSTALL_PIP_OPENCV=1
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

# Conservative pins: keep NumPy old enough for old YOLOX/FastReID-style code.
pip install --no-cache-dir --force-reinstall \
    "numpy<1.24" \
    "scipy==1.10.1" \
    "Cython<3" \
    "Pillow<10" \
    "setuptools<70"

# Default is stable for Python 3.9/3.10 and old YOLOX code.
# You can override before running this script, e.g.
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 \
#   TORCH_PACKAGES='torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118' \
#   bash exps/compute_canada/create_env_rorqual.sh
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu117}"
TORCH_PACKAGES="${TORCH_PACKAGES:-torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1}"

pip install --no-cache-dir --force-reinstall \
    --extra-index-url "$TORCH_INDEX_URL" \
    $TORCH_PACKAGES

COMMON_PKGS=(
    loguru
    scikit-image
    tqdm
    thop
    ninja
    tabulate
    tensorboard
    lap
    filterpy
    h5py
    imageio
    timm==0.5.4
    loralib
    scikit-learn
    pandas
    xmltodict
    matplotlib
    pycocotools
    cython_bbox
    prettytable
    easydict
    pyyaml
    yacs
    termcolor
    gdown
    motmetrics
)

if [ "$INSTALL_PIP_OPENCV" = "1" ]; then
    COMMON_PKGS+=(opencv-python-headless)
fi

pip install --no-cache-dir "${COMMON_PKGS[@]}"

cd "$REPO_DIR"
python setup.py develop

ACTIVATE_EXTRA="$ENV_DIR/bin/activate_track"
cat > "$ACTIVATE_EXTRA" <<ACTEOF
#!/bin/bash
module --force purge
$( [ -n "$STDENV_MOD" ] && echo "module load $STDENV_MOD" )
module load $PY_MOD
$( [ -n "$CUDA_MOD" ] && echo "module load $CUDA_MOD" )
$( [ -n "$GCC_MOD" ] && echo "module load $GCC_MOD" )
$( [ -n "$OPENCV_MOD" ] && echo "module load $OPENCV_MOD" )
source "$ENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/fast_reid:\${PYTHONPATH:-}"
cd "$REPO_DIR"
ACTEOF
chmod +x "$ACTIVATE_EXTRA"

export PYTHONPATH="$REPO_DIR:$REPO_DIR/fast_reid:${PYTHONPATH:-}"

python - <<'PY'
import sys
import torch
import torchvision
import numpy as np
import cv2
import scipy
print('Python      :', sys.version.split()[0])
print('Torch       :', torch.__version__)
print('Torchvision :', torchvision.__version__)
print('CUDA avail  :', torch.cuda.is_available())
print('NumPy       :', np.__version__)
print('SciPy       :', scipy.__version__)
print('OpenCV      :', cv2.__version__)
PY

python - <<'PY'
import yolox
print('YOLOX import: OK')
try:
    from fast_reid.fast_reid_interfece import FastReIDInterface  # noqa: F401
    print('FastReID interface import: OK')
except Exception as e:
    print('FastReID interface import: WARN:', repr(e))
PY

echo
echo "✅ Environment ready."
echo "Repo     : $REPO_DIR"
echo "Env path : $ENV_DIR"
echo "Activate : source $ACTIVATE_EXTRA"