#!/bin/bash
set -euo pipefail

cd "${REPO_DIR:-$PWD}"

source exps/compute_canada/activate_track_narval.sh

echo "=============================================================="
echo "Narval environment preflight"
echo "=============================================================="

python - <<'PY'
import sys
print("Python:", sys.version)
mods = [
    "torch",
    "torchvision",
    "numpy",
    "cv2",
    "loguru",
    "skimage",
    "tqdm",
    "PIL",
    "thop",
    "tabulate",
    "tensorboard",
    "filterpy",
    "h5py",
    "pycocotools",
    "yolox",
]
for m in mods:
    try:
        mod = __import__(m)
        ver = getattr(mod, "__version__", "ok")
        print(f"[OK] {m}: {ver}")
    except Exception as e:
        print(f"[FAIL] {m}: {type(e).__name__}: {e}")
        raise

import torch
print("Torch version:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count visible now:", torch.cuda.device_count())

try:
    from torchvision.ops import nms
    import torch
    boxes = torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]])
    scores = torch.tensor([0.9,0.8])
    print("[OK] torchvision.ops.nms:", nms(boxes, scores, 0.5).tolist())
except Exception as e:
    print("[FAIL] torchvision.ops.nms:", repr(e))
    raise
PY

python exps/compute_canada/narval_smoke/preflight_smoke_v2_exp.py

echo "=============================================================="
echo "NARVAL ENV PREFLIGHT OK"
echo "=============================================================="