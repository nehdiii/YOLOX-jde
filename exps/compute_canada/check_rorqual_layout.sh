#!/bin/bash
# Quick pre-flight check before sbatch.
# This version is for the safe Rorqual workflow where DanceTrack stays zipped
# on shared storage as datasets/data.zip and is extracted only inside $SLURM_TMPDIR.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_SOURCE="${DATA_SOURCE:-$REPO_DIR/datasets/data.zip}"
CKPT="${CKPT:-$REPO_DIR/pretrained/yolox_x.pth}"
ACTIVATE_SCRIPT="${ACTIVATE_SCRIPT:-$HOME/pyenv/Track/bin/activate}"

echo "Repo          : $REPO_DIR"
echo "Data zip/source: $DATA_SOURCE"
echo "Checkpoint    : $CKPT"
echo "Activate      : $ACTIVATE_SCRIPT"
echo

ok=1
check_path() {
    local p="$1"; local label="$2"
    if [ -e "$p" ]; then
        echo "[OK]   $label: $p"
    else
        echo "[MISS] $label: $p"
        ok=0
    fi
}

check_file() {
    local p="$1"; local label="$2"
    if [ -f "$p" ]; then
        echo "[OK]   $label: $p"
    else
        echo "[MISS] $label: $p"
        ok=0
    fi
}

check_path "$REPO_DIR/tools/train.py" "train.py"
check_path "$REPO_DIR/exps/example/mot/yolox_dancetrack_val.py" "detector exp"
check_path "$REPO_DIR/exps/example/dancetrack/yolox_x_dancetrack_jde_v1.py" "JDE V1 exp"
check_path "$REPO_DIR/tools/convert_dance_to_coco_jde.py" "JDE converter"
check_file "$DATA_SOURCE" "DanceTrack archive/data.zip"
check_file "$CKPT" "YOLOX-X pretrained checkpoint"
check_file "$ACTIVATE_SCRIPT" "environment activate script"

# For the login node we only inspect the archive metadata. We do NOT unzip here.
if [ -f "$DATA_SOURCE" ]; then
    case "$DATA_SOURCE" in
        *.zip)
            echo
            echo "Inspecting zip layout without extracting..."
            set +e
            python - "$DATA_SOURCE" <<'PY'
import sys, zipfile
path = sys.argv[1]
ok = True
with zipfile.ZipFile(path) as z:
    names = z.namelist()

def has_prefix(prefix):
    return any(n.startswith(prefix.rstrip('/') + '/') or n == prefix.rstrip('/') for n in names)

checks = [
    ("dancetrack/", "top-level dancetrack folder"),
    ("dancetrack/train/", "dancetrack/train inside zip"),
    ("dancetrack/val/", "dancetrack/val inside zip"),
]
for prefix, label in checks:
    if has_prefix(prefix):
        print(f"[OK]   {label}: {prefix}")
    else:
        print(f"[MISS] {label}: {prefix}")
        ok = False

if has_prefix("dancetrack/test/"):
    print("[OK]   dancetrack/test inside zip: dancetrack/test/")
else:
    print("[WARN] dancetrack/test missing inside zip; training can run, test tracking cannot.")

if has_prefix("dancetrack/annotations/"):
    print("[OK]   dancetrack/annotations inside zip: dancetrack/annotations/")
else:
    print("[WARN] dancetrack/annotations missing inside zip; Slurm job will create annotations in $SLURM_TMPDIR if GT files exist.")

sys.exit(0 if ok else 2)
PY
            py_status=$?
            set -e
            if [ "$py_status" != "0" ]; then
                ok=0
            fi
            ;;
        *.tar|*.tar.gz|*.tgz)
            echo "[OK]   archive format supported: $DATA_SOURCE"
            echo "[INFO] not inspecting tar contents on login node. It will be checked after extraction in the Slurm job."
            ;;
        *)
            echo "[MISS] unsupported DATA_SOURCE format. Expected .zip, .tar, .tar.gz, or .tgz"
            ok=0
            ;;
    esac
fi

echo
if [ "$ok" = "1" ]; then
    echo "Layout check passed. The data will be extracted on the compute node into \$SLURM_TMPDIR/datasets/dancetrack."
else
    echo "Layout check has missing required items. Fix them before sbatch."
    exit 1
fi