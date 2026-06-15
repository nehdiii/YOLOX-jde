#!/bin/bash
# Create the YOLOX-JDE Python environment on Compute Canada / Alliance.
# Run this on the login node, not inside an sbatch job.
set -euo pipefail

REPO_DIR="${1:-${REPO_DIR:-$PWD}}"
ENV_DIR="${ENV_DIR:-$HOME/venvs/yolox-jde-rorqual}"

if [ ! -d "$REPO_DIR" ]; then
  echo "[ERROR] REPO_DIR does not exist: $REPO_DIR"
  echo "Usage: bash compute_canada_rorqual/create_env_rorqual.sh /path/to/YOLOX-jde"
  exit 1
fi

# ---- modules: try common Alliance stacks conservatively ----
module --force purge
if module load StdEnv/2023 2>/dev/null; then
  STDENV_LOADED="StdEnv/2023"
elif module load StdEnv/2020 2>/dev/null; then
  STDENV_LOADED="StdEnv/2020"
else
  echo "[WARN] Could not load StdEnv/2023 or StdEnv/2020; continuing with default modules."
  STDENV_LOADED=""
fi

PY_MOD=""
for m in python/3.10.13 python/3.10 python/3.9.6 python/3.9 python/3.8.10 python/3.8; do
  if module load "$m" 2>/dev/null; then
    PY_MOD="$m"
    break
  fi
done
if [ -z "$PY_MOD" ]; then
  echo "[ERROR] Could not load a Python module. Try: module spider python"
  exit 1
fi

CUDA_MOD=""
for m in cuda/11.8 cuda/11.7 cuda/11.4 cuda/12.2 cuda/12.1 cuda; do
  if module load "$m" 2>/dev/null; then
    CUDA_MOD="$m"
    break
  fi
done
if [ -z "$CUDA_MOD" ]; then
  echo "[WARN] Could not load a CUDA module. PyTorch wheel may still run if driver is compatible."
fi

GCC_MOD=""
for m in gcc/12.3 gcc/11.3 gcc/9.3.0 gcc; do
  if module load "$m" 2>/dev/null; then
    GCC_MOD="$m"
    break
  fi
done

mkdir -p "$(dirname "$ENV_DIR")"
if [ ! -d "$ENV_DIR" ]; then
  echo "==> Creating venv: $ENV_DIR"
  python -m venv "$ENV_DIR"
else
  echo "==> Reusing venv: $ENV_DIR"
fi
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Conservative pins for this YOLOX/FastReID-era code.
python -m pip install --no-cache-dir --force-reinstall \
  "numpy<1.24" "Cython<3" "setuptools<70" "Pillow<10"

# PyTorch 1.13.1+cu117 is close to your local track1 setup and works for old YOLOX code.
python -m pip install --no-cache-dir --force-reinstall \
  --extra-index-url https://download.pytorch.org/whl/cu117 \
  torch==1.13.1+cu117 torchvision==0.14.1+cu117

python -m pip install --no-cache-dir \
  loguru scikit-image tqdm thop ninja tabulate tensorboard \
  lap filterpy h5py imageio timm==0.5.4 loralib scikit-learn \
  pandas xmltodict matplotlib pycocotools cython_bbox prettytable \
  easydict pyyaml yacs termcolor opencv-python-headless

cd "$REPO_DIR"
export MAX_JOBS="${MAX_JOBS:-4}"
python setup.py develop

# Activation helper used by all Slurm scripts.
ACTIVATE_SCRIPT="$ENV_DIR/bin/activate_yolox_jde_rorqual"
cat > "$ACTIVATE_SCRIPT" <<ACTEOF
#!/bin/bash
module --force purge
[ -n "$STDENV_LOADED" ] && module load "$STDENV_LOADED"
module load "$PY_MOD"
[ -n "$CUDA_MOD" ] && module load "$CUDA_MOD"
[ -n "$GCC_MOD" ] && module load "$GCC_MOD"
source "$ENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/fast_reid:\${PYTHONPATH:-}"
cd "$REPO_DIR"
ACTEOF
chmod +x "$ACTIVATE_SCRIPT"

export PYTHONPATH="$REPO_DIR:$REPO_DIR/fast_reid:${PYTHONPATH:-}"
python - <<'PY'
import sys, torch, torchvision, numpy as np, cv2
print("Python      :", sys.version.split()[0])
print("Torch       :", torch.__version__)
print("Torchvision :", torchvision.__version__)
print("CUDA avail  :", torch.cuda.is_available())
print("NumPy       :", np.__version__)
print("OpenCV      :", cv2.__version__)
import yolox
print("YOLOX import: OK")
PY

echo
echo "Environment ready. Use:"
echo "  source $ACTIVATE_SCRIPT"
echo
echo "Repo: $REPO_DIR"
echo "Env : $ENV_DIR"