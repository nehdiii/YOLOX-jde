#!/usr/bin/env python3
# encoding: utf-8
"""
Convert DanceTrack MOT-format annotations to COCO-style JSON for YOLOX-JDE.

Main JDE requirement:
    raw DanceTrack track IDs are only unique inside one sequence.
    For identity-classification/ReID training, IDs must be globally unique.

This converter maps:
    (split, sequence_name, raw_track_id) -> global zero-based track_id

Why zero-based?
    YOLOX-JDE heads usually use nn.CrossEntropyLoss / Linear(num_ids), so valid
    identity labels must be exactly in [0, num_ids - 1].

Default outputs:
    datasets/dancetrack/annotations/train_jde.json
    datasets/dancetrack/annotations/val_jde.json
    datasets/dancetrack/annotations/test_jde.json

Expected DanceTrack layout:
    datasets/dancetrack/
      train/<seq>/img1/00000001.jpg
      train/<seq>/gt/gt.txt
      val/<seq>/img1/00000001.jpg
      val/<seq>/gt/gt.txt
      test/<seq>/img1/00000001.jpg

Example:
    python tools/convert_dance_to_coco_jde.py \
        --data-root datasets/dancetrack \
        --splits train val test \
        --out-suffix _jde \
        --verify
"""

import argparse
import configparser
import csv
import json
import os
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DanceTrack to COCO JSON with globally unique zero-based JDE IDs."
    )
    parser.add_argument(
        "--data-root",
        default="datasets/dancetrack",
        help="DanceTrack root folder containing train/val/test folders.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to convert. Default: train val test.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output annotation directory. Default: <data-root>/annotations.",
    )
    parser.add_argument(
        "--out-suffix",
        default="_jde",
        help="Suffix for output JSON names. Default gives train_jde.json.",
    )
    parser.add_argument(
        "--category-name",
        default="dancer",
        help="COCO category name. Default: dancer.",
    )
    parser.add_argument(
        "--id-start",
        type=int,
        default=0,
        help="First global identity ID. Keep 0 for normal JDE training.",
    )
    parser.add_argument(
        "--keep-ignored",
        action="store_true",
        help=(
            "Keep rows with MOT mark/conf <= 0. Default is to drop them when the "
            "column exists, which is safer for training."
        ),
    )
    parser.add_argument(
        "--write-standard-names",
        action="store_true",
        help=(
            "Also write train.json/val.json/test.json. Not recommended unless you "
            "want to overwrite the normal detector annotations."
        ),
    )
    parser.add_argument(
        "--no-identity-map",
        action="store_true",
        help="Do not store identity_map metadata in JSON. Usually leave disabled.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run strict validation after writing each JSON.",
    )
    return parser.parse_args()


def natural_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), path.name)
    return (1, stem, path.name)


def list_sequences(split_dir: Path) -> List[Path]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")
    seqs = [p for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return sorted(seqs, key=lambda p: p.name)


def list_images(img_dir: Path) -> List[Path]:
    if not img_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {img_dir}")
    images = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    images = sorted(images, key=natural_key)
    return images


def frame_id_from_name(img_path: Path, fallback: int) -> int:
    if img_path.stem.isdigit():
        return int(img_path.stem)
    return fallback


def read_seqinfo(seq_dir: Path) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Return height, width, seq_length from seqinfo.ini when available."""
    ini_path = seq_dir / "seqinfo.ini"
    if not ini_path.exists():
        return None, None, None

    cfg = configparser.ConfigParser()
    cfg.read(str(ini_path))
    if "Sequence" not in cfg:
        return None, None, None

    sec = cfg["Sequence"]
    try:
        width = int(sec.get("imWidth")) if sec.get("imWidth") else None
        height = int(sec.get("imHeight")) if sec.get("imHeight") else None
        seq_length = int(sec.get("seqLength")) if sec.get("seqLength") else None
    except ValueError:
        return None, None, None
    return height, width, seq_length


def read_image_shape(first_img: Path) -> Tuple[int, int]:
    img = cv2.imread(str(first_img))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {first_img}")
    h, w = img.shape[:2]
    return int(h), int(w)


def load_mot_txt(gt_path: Path) -> np.ndarray:
    if not gt_path.exists() or gt_path.stat().st_size == 0:
        return np.zeros((0, 9), dtype=np.float32)

    rows = []
    with gt_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for line_no, row in enumerate(reader, start=1):
            if not row or all(str(x).strip() == "" for x in row):
                continue
            try:
                vals = [float(x) for x in row]
            except ValueError as exc:
                raise ValueError(f"Invalid numeric row in {gt_path}:{line_no}: {row}") from exc
            rows.append(vals)

    if not rows:
        return np.zeros((0, 9), dtype=np.float32)

    max_len = max(len(r) for r in rows)
    arr = np.zeros((len(rows), max_len), dtype=np.float32)
    for i, r in enumerate(rows):
        arr[i, : len(r)] = np.asarray(r, dtype=np.float32)
    return arr


def is_valid_gt_row(row: np.ndarray, keep_ignored: bool) -> bool:
    if len(row) < 6:
        return False

    raw_id = int(row[1])
    x, y, w, h = row[2:6]

    if raw_id <= 0:
        return False
    if w <= 0 or h <= 0:
        return False

    # MOT-style gt usually uses column 7 / index 6 as mark/conf.
    # For training, rows with mark=0 are ignored by default.
    if not keep_ignored and len(row) > 6 and float(row[6]) <= 0:
        return False

    return True


def make_output_name(split: str, suffix: str) -> str:
    if suffix:
        return f"{split}{suffix}.json"
    return f"{split}.json"


def convert_split(
    data_root: Path,
    split: str,
    out_dir: Path,
    out_suffix: str,
    category_name: str,
    id_start: int,
    keep_ignored: bool,
    include_identity_map: bool,
    verify: bool,
    write_standard_names: bool,
) -> Path:
    split_dir = data_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / make_output_name(split, out_suffix)

    out = {
        "images": [],
        "annotations": [],
        "videos": [],
        "categories": [{"id": 1, "name": category_name}],
        "jde_zero_based_track_ids": True,
        "id_start": int(id_start),
    }

    seqs = list_sequences(split_dir)
    image_id = 0
    ann_id = 0
    video_id = 0
    global_track_map: "OrderedDict[Tuple[str, int], int]" = OrderedDict()

    split_stats = []

    for seq_dir in seqs:
        seq = seq_dir.name
        img_dir = seq_dir / "img1"
        gt_path = seq_dir / "gt" / "gt.txt"

        images = list_images(img_dir)
        if not images:
            print(f"[WARN] {split}/{seq}: no images found in {img_dir}; skipping")
            continue

        height, width, seq_len = read_seqinfo(seq_dir)
        if height is None or width is None:
            height, width = read_image_shape(images[0])

        if seq_len is not None and seq_len != len(images):
            print(
                f"[WARN] {split}/{seq}: seqinfo seqLength={seq_len}, "
                f"but found {len(images)} image files"
            )

        video_id += 1
        out["videos"].append({"id": video_id, "file_name": seq})

        frame_to_image_id: Dict[int, int] = {}
        prev_global_image_id = -1

        for local_idx, img_path in enumerate(images, start=1):
            frame_id = frame_id_from_name(img_path, fallback=local_idx)
            image_id += 1
            curr_image_id = image_id
            frame_to_image_id[frame_id] = curr_image_id

            # next_image_id is filled from sorted image list; it stays inside the sequence.
            next_image_id = curr_image_id + 1 if local_idx < len(images) else -1

            out["images"].append(
                {
                    "file_name": f"{seq}/img1/{img_path.name}",
                    "id": curr_image_id,
                    "frame_id": int(frame_id),
                    "video_id": video_id,
                    "height": int(height),
                    "width": int(width),
                    "prev_image_id": int(prev_global_image_id),
                    "next_image_id": int(next_image_id),
                }
            )
            prev_global_image_id = curr_image_id

        num_anns_before = len(out["annotations"])
        num_ids_before = len(global_track_map)

        if split.lower() != "test":
            anns = load_mot_txt(gt_path)
            missing_frame_rows = 0
            ignored_rows = 0

            for row in anns:
                frame_id = int(row[0]) if len(row) > 0 else -1
                if frame_id not in frame_to_image_id:
                    missing_frame_rows += 1
                    continue
                if not is_valid_gt_row(row, keep_ignored=keep_ignored):
                    ignored_rows += 1
                    continue

                raw_track_id = int(row[1])
                key = (seq, raw_track_id)
                if key not in global_track_map:
                    # Critical: zero-based by default, valid for CE targets.
                    global_track_map[key] = id_start + len(global_track_map)

                track_id = int(global_track_map[key])
                x, y, w, h = [float(v) for v in row[2:6]]
                ann_id += 1

                ann = {
                    "id": ann_id,
                    "image_id": int(frame_to_image_id[frame_id]),
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                    # JDE identity target. Zero-based and globally unique inside this split.
                    "track_id": track_id,
                    # Debug / traceability metadata.
                    "raw_track_id": raw_track_id,
                    "global_track_id_1based": track_id + 1,
                    "seq_name": seq,
                    "conf": float(row[6]) if len(row) > 6 else 1.0,
                }
                if len(row) > 7:
                    ann["raw_category_id"] = int(row[7])
                if len(row) > 8:
                    ann["visibility"] = float(row[8])

                out["annotations"].append(ann)

            if missing_frame_rows > 0:
                print(
                    f"[WARN] {split}/{seq}: skipped {missing_frame_rows} GT rows "
                    "because their frame_id has no image file"
                )
            if ignored_rows > 0:
                print(
                    f"[INFO] {split}/{seq}: ignored {ignored_rows} invalid/marked GT rows"
                )

        seq_ann_count = len(out["annotations"]) - num_anns_before
        seq_id_count = len(global_track_map) - num_ids_before
        split_stats.append((seq, len(images), seq_ann_count, seq_id_count))
        print(
            f"{split}/{seq}: images={len(images)} anns={seq_ann_count} "
            f"new_ids={seq_id_count}"
        )

    # For normal id_start=0, num_ids equals number of identities.
    # If a nonzero id_start is used, classifier size must cover max_id+1.
    if global_track_map:
        max_track_id = max(global_track_map.values())
        out["num_ids"] = int(max_track_id + 1)
        out["num_actual_ids"] = int(len(global_track_map))
    else:
        out["num_ids"] = 0
        out["num_actual_ids"] = 0

    out["num_videos"] = len(out["videos"])
    out["num_images"] = len(out["images"])
    out["num_annotations"] = len(out["annotations"])

    if include_identity_map:
        out["identity_map"] = [
            {"seq_name": seq, "raw_track_id": raw_id, "track_id": gid}
            for (seq, raw_id), gid in global_track_map.items()
        ]

    with out_path.open("w") as f:
        json.dump(out, f)

    if write_standard_names:
        std_path = out_dir / f"{split}.json"
        with std_path.open("w") as f:
            json.dump(out, f)
        print(f"[WRITE] {std_path}")

    print(
        f"[WRITE] {out_path} | videos={out['num_videos']} images={out['num_images']} "
        f"anns={out['num_annotations']} num_ids={out['num_ids']}"
    )

    if verify:
        verify_json(out_path, data_root=data_root, split=split)

    return out_path


def verify_json(json_path: Path, data_root: Path, split: str) -> None:
    with json_path.open("r") as f:
        data = json.load(f)

    image_ids = [im["id"] for im in data.get("images", [])]
    if len(image_ids) != len(set(image_ids)):
        raise AssertionError(f"Duplicate image IDs in {json_path}")
    image_id_set = set(image_ids)

    ann_ids = [ann["id"] for ann in data.get("annotations", [])]
    if len(ann_ids) != len(set(ann_ids)):
        raise AssertionError(f"Duplicate annotation IDs in {json_path}")

    # Verify image paths exist.
    missing_images = []
    for im in data.get("images", []):
        p = data_root / split / im["file_name"]
        if not p.exists():
            missing_images.append(str(p))
            if len(missing_images) >= 5:
                break
    if missing_images:
        raise FileNotFoundError(
            "Some image paths from the JSON do not exist. Examples:\n" + "\n".join(missing_images)
        )

    num_ids = int(data.get("num_ids", 0))
    id_to_keys = defaultdict(set)

    for ann in data.get("annotations", []):
        if ann["image_id"] not in image_id_set:
            raise AssertionError(f"Annotation {ann['id']} references missing image_id={ann['image_id']}")
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            raise AssertionError(f"Annotation {ann['id']} has invalid bbox={ann['bbox']}")
        if "track_id" in ann:
            tid = int(ann["track_id"])
            if tid < 0 or tid >= num_ids:
                raise AssertionError(
                    f"Invalid track_id={tid} in ann {ann['id']} for num_ids={num_ids}. "
                    "JDE identity labels must satisfy 0 <= track_id < num_ids."
                )
            id_to_keys[tid].add((ann.get("seq_name", ""), int(ann.get("raw_track_id", -1))))

    collisions = {tid: keys for tid, keys in id_to_keys.items() if len(keys) > 1}
    if collisions:
        first_tid = next(iter(collisions))
        raise AssertionError(
            f"Global ID collision in {json_path}: track_id={first_tid} maps to {collisions[first_tid]}"
        )

    ids = sorted(id_to_keys.keys())
    if ids:
        expected = list(range(min(ids), max(ids) + 1))
        if ids != expected:
            missing = sorted(set(expected) - set(ids))[:20]
            print(f"[WARN] non-contiguous used IDs in {json_path}; first missing={missing}")
        if min(ids) != 0:
            print(
                f"[WARN] min track_id is {min(ids)} not 0. This is okay only if you intentionally used --id-start."
            )

    print(
        f"[VERIFY OK] {json_path} | images={len(image_ids)} anns={len(ann_ids)} num_ids={num_ids}"
    )


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir) if args.out_dir else data_root / "annotations"

    print("=" * 90)
    print("DanceTrack -> COCO-JDE conversion")
    print(f"data_root: {data_root}")
    print(f"out_dir:   {out_dir}")
    print(f"splits:    {args.splits}")
    print("ID rule:   (sequence_name, raw_track_id) -> zero-based global track_id")
    print("=" * 90)

    written = []
    for split in args.splits:
        written.append(
            convert_split(
                data_root=data_root,
                split=split,
                out_dir=out_dir,
                out_suffix=args.out_suffix,
                category_name=args.category_name,
                id_start=args.id_start,
                keep_ignored=args.keep_ignored,
                include_identity_map=not args.no_identity_map,
                verify=args.verify,
                write_standard_names=args.write_standard_names,
            )
        )

    print("=" * 90)
    print("Done. Generated files:")
    for p in written:
        print(f"  {p}")
    print("For JDE training, use train_ann='train_jde.json' in your exp file.")
    print("=" * 90)


if __name__ == "__main__":
    main()