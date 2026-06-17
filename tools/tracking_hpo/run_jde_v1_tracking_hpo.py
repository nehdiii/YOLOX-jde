#!/usr/bin/env python3
# encoding: utf-8
"""Run a sequential local grid search for JDE V1 DanceTrack tracking HPs.

This script wraps tools/run_hybrid_sort_dance_jde_parallel.py and passes tracking
hyperparameters through YOLOX exp opts.

Example:
  CUDA_VISIBLE_DEVICES=0,1 python tools/tracking_hpo/run_jde_v1_tracking_hpo.py \
    --grid configs/tracking_hpo/jde_v1_quick_grid.json \
    --exp-file exps/example/dancetrack/yolox_x_dancetrack_jde_v1.py \
    --ckpt YOLOX_outputs/yolox_x_dancetrack_jde_v1_cc_2nodes_8gpu/best_ckpt.pth.tar \
    --base-expn hpo_jde_v1 \
    --parallel-workers 10 \
    --parallel-gpus 0,1 \
    --execute
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args():
    p = argparse.ArgumentParser("JDE V1 tracking HPO runner")
    p.add_argument("--grid", default="configs/tracking_hpo/jde_v1_quick_grid.json")
    p.add_argument("--exp-file", default="exps/example/dancetrack/yolox_x_dancetrack_jde_v1.py")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--base-expn", default="hpo_jde_v1")
    p.add_argument("--parallel-workers", type=int, default=10)
    p.add_argument("--parallel-gpus", default="0,1")
    p.add_argument("--cuda-visible-devices", default=None)
    p.add_argument("--output-csv", default=None)
    p.add_argument("--execute", action="store_true", help="Actually run commands. Default is dry-run.")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true", help="Skip if a TrackEval summary exists.")
    p.add_argument("--no-fuse", action="store_true", help="Disable --fuse.")
    p.add_argument("--no-fp16", action="store_true", help="Disable --fp16.")
    p.add_argument("--conf", type=float, default=None, help="Optional detector confidence override.")
    p.add_argument("--nms", type=float, default=None, help="Optional NMS override.")
    return p.parse_args()


def load_grid(path: str) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data["configs"]


def clean_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def opts_to_tokens(opts: dict[str, Any]) -> list[str]:
    tokens = []
    for k, v in opts.items():
        # Keep booleans out of opts because YOLOX BaseExp.merge casts bool("False") -> True.
        if isinstance(v, bool):
            continue
        tokens += [str(k), str(v)]
    return tokens


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def find_summary_for_expn(expn: str) -> Path | None:
    root = Path("YOLOX_outputs") / expn
    if not root.exists():
        return None
    candidates = list(root.rglob("*_summary.txt"))
    if not candidates:
        return None
    # Prefer pedestrian_summary.txt if present.
    ped = [p for p in candidates if p.name == "pedestrian_summary.txt"]
    candidates = ped or candidates
    # Choose newest.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_summary_file(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}

    lines = [ln.strip() for ln in path.read_text(errors="ignore").splitlines() if ln.strip()]
    if len(lines) < 2:
        return {"summary_file": str(path)}

    header = re.split(r"\s+", lines[0])
    values = re.split(r"\s+", lines[1])
    out: dict[str, Any] = {"summary_file": str(path)}

    # TrackEval summary is normally header line then one values line.
    for k, v in zip(header, values):
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v

    return out


def row_metric_value(row: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in row:
            return row[k]
    return ""


def main() -> int:
    args = parse_args()
    grid = load_grid(args.grid)
    selected = grid[args.start_index:]
    if args.max_runs is not None:
        selected = selected[: args.max_runs]

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path("tracking_hpo_runs") / f"{args.base_expn}_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.output_csv) if args.output_csv else run_dir / "results.csv"
    commands_path = run_dir / "commands.sh"

    cuda_visible = args.cuda_visible_devices or args.parallel_gpus
    env_base = os.environ.copy()
    if cuda_visible:
        env_base["CUDA_VISIBLE_DEVICES"] = cuda_visible

    rows = []
    commands = ["#!/bin/bash", "set -euo pipefail", "cd \"$(dirname \"$0\")/../..\" 2>/dev/null || true", ""]

    print("=" * 100)
    print("JDE V1 Tracking HPO")
    print("grid:", args.grid)
    print("exp:", args.exp_file)
    print("ckpt:", args.ckpt)
    print("runs:", len(selected))
    print("execute:", args.execute)
    print("parallel_workers:", args.parallel_workers)
    print("parallel_gpus:", args.parallel_gpus)
    print("CUDA_VISIBLE_DEVICES:", cuda_visible)
    print("run_dir:", run_dir)
    print("=" * 100)

    for idx, cfg in enumerate(selected, start=args.start_index):
        name = clean_name(cfg["name"])
        expn = f"{args.base_expn}_{idx:03d}_{name}"
        opts = cfg.get("opts", {})
        summary_path = find_summary_for_expn(expn)

        if args.skip_existing and summary_path is not None:
            print(f"[SKIP existing] {idx:03d} {name} -> {summary_path}")
            metrics = parse_summary_file(summary_path)
            row = {"index": idx, "name": name, "expn": expn, "returncode": 0, "duration_sec": 0, **opts, **metrics}
            rows.append(row)
            continue

        cmd = [
            sys.executable,
            "tools/run_hybrid_sort_dance_jde_parallel.py",
            "-f", args.exp_file,
            "-c", args.ckpt,
            "-b", "1",
            "-d", "1",
        ]

        if not args.no_fp16:
            cmd.append("--fp16")
        if not args.no_fuse:
            cmd.append("--fuse")
        if args.conf is not None:
            cmd += ["--conf", str(args.conf)]
        if args.nms is not None:
            cmd += ["--nms", str(args.nms)]

        cmd += [
            "--expn", expn,
            "--parallel-workers", str(args.parallel_workers),
            "--parallel-gpus", args.parallel_gpus,
            "--parallel-clean-results",
            "--parallel-progress-mode", "frames",
        ]

        cmd += opts_to_tokens(opts)

        cmd_line = f"CUDA_VISIBLE_DEVICES={shlex.quote(cuda_visible)} " + shell_join(cmd)
        commands.append(cmd_line)
        commands.append("")

        print("\n" + "-" * 100)
        print(f"[{idx:03d}] {name}")
        print("expn:", expn)
        print("opts:", json.dumps(opts, sort_keys=True))
        print(cmd_line)

        start = time.time()
        returncode = None
        if args.execute:
            proc = subprocess.run(cmd, env=env_base)
            returncode = int(proc.returncode)
        else:
            returncode = -999  # dry-run marker

        duration = time.time() - start
        metrics = parse_summary_file(find_summary_for_expn(expn))

        row = {
            "index": idx,
            "name": name,
            "expn": expn,
            "returncode": returncode,
            "duration_sec": round(duration, 2),
            **opts,
            **metrics,
        }
        rows.append(row)

        # Save after every run.
        all_keys = []
        for r in rows:
            for k in r.keys():
                if k not in all_keys:
                    all_keys.append(k)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(rows)

        if returncode not in (0, -999):
            print(f"[ERROR] run failed with returncode={returncode}. Stopping.")
            break

    commands_path.write_text("\n".join(commands) + "\n")
    commands_path.chmod(0o755)

    print("\n" + "=" * 100)
    print("Done.")
    print("CSV:", csv_path)
    print("Commands:", commands_path)
    print("To summarize:")
    print(f"  python tools/tracking_hpo/summarize_jde_v1_tracking_hpo.py --csv {csv_path}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())