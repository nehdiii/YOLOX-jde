#!/usr/bin/env python3
# encoding: utf-8
"""Summarize JDE V1 tracking HPO results CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PREFERRED_COLUMNS = [
    "index", "name", "expn", "returncode",
    "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "Frag", "MT", "ML",
    "track_thresh", "low_thresh", "iou_thresh", "inertia",
    "EG_weight_high_score", "EG_weight_low_score",
    "alpha", "high_score_matching_thresh", "low_score_matching_thresh",
    "TCM_first_step_weight", "TCM_byte_step_weight",
    "summary_file",
]


def to_float(x, default=-1e9):
    try:
        return float(x)
    except Exception:
        return default


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--sort", default="HOTA", help="Metric to sort by, e.g. HOTA, IDF1, AssA, MOTA")
    p.add_argument("--topk", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.csv)
    rows = list(csv.DictReader(path.open()))

    rows.sort(key=lambda r: (to_float(r.get(args.sort)), to_float(r.get("IDF1")), to_float(r.get("AssA"))), reverse=True)

    if not rows:
        print("No rows.")
        return 0

    cols = [c for c in PREFERRED_COLUMNS if c in rows[0]]
    # Add anything else not in preferred.
    for c in rows[0]:
        if c not in cols:
            cols.append(c)

    print(f"Loaded {len(rows)} rows from {path}")
    print(f"Sorted by {args.sort} desc\n")

    top = rows[: args.topk]

    # simple width formatting
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in top)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in top:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))

    best = rows[0]
    print("\nBEST CONFIG:")
    for c in cols:
        if c in best:
            print(f"{c}: {best[c]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())