#!/bin/bash
# Stage DanceTrack to node-local storage for YOLOX/JDE training.
# Rorqual-safe workflow:
#   login/shared filesystem stores only datasets/data.zip;
#   Slurm job extracts it into $SLURM_TMPDIR/datasets;
#   YOLOX_DATADIR points to $SLURM_TMPDIR/datasets.

set -euo pipefail

usage() {
    echo "Usage: bash $0 REPO_DIR DATA_SOURCE DATA_ROOT [--make-jde]"
    echo
    echo "REPO_DIR    : path to YOLOX-jde repo"
    echo "DATA_SOURCE : archive path, default is repo/datasets/data.zip"
    echo "DATA_ROOT   : node-local parent folder, usually \$SLURM_TMPDIR/datasets"
    echo "--make-jde  : create train_jde.json if missing"
}

if [ "$#" -lt 3 ]; then
    usage
    exit 1
fi

REPO_DIR="$1"
DATA_SOURCE="$2"
DATA_ROOT="$3"
MAKE_JDE=0
if [ "${4:-}" = "--make-jde" ]; then
    MAKE_JDE=1
fi

TARGET="$DATA_ROOT/dancetrack"

printf 'REPO_DIR    = %s\n' "$REPO_DIR"
printf 'DATA_SOURCE = %s\n' "$DATA_SOURCE"
printf 'DATA_ROOT   = %s\n' "$DATA_ROOT"
printf 'TARGET      = %s\n' "$TARGET"
printf 'MAKE_JDE    = %s\n' "$MAKE_JDE"

if [ ! -d "$REPO_DIR" ]; then
    echo "[ERROR] Repo directory not found: $REPO_DIR"
    exit 1
fi

if [ ! -e "$DATA_SOURCE" ]; then
    echo "[ERROR] DATA_SOURCE not found: $DATA_SOURCE"
    echo "Expected by default: $REPO_DIR/datasets/data.zip"
    exit 1
fi

mkdir -p "$DATA_ROOT"
rm -rf "$TARGET"

copy_dir() {
    local src="$1"
    local dst="$2"
    mkdir -p "$(dirname "$dst")"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --info=progress2 "$src"/ "$dst"/
    else
        mkdir -p "$dst"
        cp -a "$src"/. "$dst"/
    fi
}

if [ -d "$DATA_SOURCE" ]; then
    # Kept for manual/debug runs. Normal Rorqual workflow should use data.zip.
    if [ -d "$DATA_SOURCE/train" ] && [ -d "$DATA_SOURCE/val" ]; then
        echo "==> Copying DanceTrack folder to node-local storage"
        copy_dir "$DATA_SOURCE" "$TARGET"
    elif [ -d "$DATA_SOURCE/dancetrack" ]; then
        echo "==> Copying DATA_SOURCE/dancetrack to node-local storage"
        copy_dir "$DATA_SOURCE/dancetrack" "$TARGET"
    else
        echo "[ERROR] Folder does not look like DanceTrack root or parent containing dancetrack: $DATA_SOURCE"
        exit 1
    fi
else
    echo "==> Extracting archive into node-local storage: $DATA_ROOT"
    case "$DATA_SOURCE" in
        *.zip)
            unzip -q "$DATA_SOURCE" -d "$DATA_ROOT"
            ;;
        *.tar.gz|*.tgz)
            tar -xzf "$DATA_SOURCE" -C "$DATA_ROOT"
            ;;
        *.tar)
            tar -xf "$DATA_SOURCE" -C "$DATA_ROOT"
            ;;
        *)
            echo "[ERROR] Unknown archive format: $DATA_SOURCE"
            exit 1
            ;;
    esac

    # Expected case: data.zip contains dancetrack/ directly.
    if [ ! -d "$TARGET" ]; then
        echo "==> Normalizing archive layout"
        cand=$(find "$DATA_ROOT" -maxdepth 4 -type d -name dancetrack | head -n 1 || true)
        if [ -n "$cand" ]; then
            mkdir -p "$(dirname "$TARGET")"
            mv "$cand" "$TARGET"
        else
            cand=$(find "$DATA_ROOT" -maxdepth 4 -type d \( -name train -o -name val \) | head -n 1 || true)
            if [ -n "$cand" ]; then
                parent=$(dirname "$cand")
                if [ -d "$parent/train" ] && [ -d "$parent/val" ]; then
                    mv "$parent" "$TARGET"
                fi
            fi
        fi
    fi
fi

if [ ! -d "$TARGET" ]; then
    echo "[ERROR] Could not create node-local dancetrack folder: $TARGET"
    echo "[DEBUG] Contents of DATA_ROOT:"
    find "$DATA_ROOT" -maxdepth 3 -type d | sort | head -100
    exit 1
fi

# Verify expected folders.
for split in train val; do
    if [ ! -d "$TARGET/$split" ]; then
        echo "[ERROR] Missing $TARGET/$split"
        echo "[DEBUG] Contents of $TARGET:"
        ls -lah "$TARGET" || true
        exit 1
    fi
done

if [ ! -d "$TARGET/test" ]; then
    echo "[WARN] Missing $TARGET/test; training can continue, test tracking cannot."
fi

mkdir -p "$TARGET/annotations"

# If normal COCO annotations are missing, build them in the staged folder.
# This avoids unzipping/converting on the login node.
if [ ! -f "$TARGET/annotations/train.json" ] || [ ! -f "$TARGET/annotations/val.json" ]; then
    echo "==> Creating standard train.json/val.json/test.json in node-local folder"
    python "$REPO_DIR/tools/convert_dance_to_coco_jde.py" \
        --data-root "$TARGET" \
        --splits train val test \
        --out-suffix "" \
        --write-standard-names \
        --verify
fi

if [ "$MAKE_JDE" = "1" ] && [ ! -f "$TARGET/annotations/train_jde.json" ]; then
    echo "==> Creating zero-based global-ID JDE annotations in node-local folder"
    python "$REPO_DIR/tools/convert_dance_to_coco_jde.py" \
        --data-root "$TARGET" \
        --splits train val test \
        --out-suffix _jde \
        --verify
fi

# Final checks.
for ann in train.json val.json; do
    if [ ! -f "$TARGET/annotations/$ann" ]; then
        echo "[ERROR] Missing $TARGET/annotations/$ann"
        exit 1
    fi
done

if [ "$MAKE_JDE" = "1" ] && [ ! -f "$TARGET/annotations/train_jde.json" ]; then
    echo "[ERROR] Missing $TARGET/annotations/train_jde.json after JDE conversion"
    exit 1
fi

echo "==> DanceTrack staged successfully"
echo "Node-local dataset root: $TARGET"
echo "YOLOX_DATADIR should be: $DATA_ROOT"
find "$TARGET/annotations" -maxdepth 1 -type f -name '*.json' -printf '  %f\n' | sort