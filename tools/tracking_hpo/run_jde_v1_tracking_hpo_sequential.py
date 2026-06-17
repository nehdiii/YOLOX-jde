#!/usr/bin/env python3
# encoding: utf-8
"""Strict sequential tracking HPO runner for JDE V1 DanceTrack.

This fixes the important scheduling behavior:

  - HPO configs are executed ONE BY ONE.
  - The next HPO config does not start until the previous tracking evaluation process exits.
  - Sequence-level parallelism is kept unchanged through --parallel-workers and --parallel-gpus.
  - A lock file prevents accidentally launching two HPO grids at the same time.
  - Each HPO config gets its own stdout/stderr log.
  - results.csv is updated after every config.

This does NOT reduce parallel sequence workers. If you set --parallel-workers 10, every
single HPO evaluation still runs with 10 sequence workers.
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
    p = argparse.ArgumentParser("Strict sequential JDE V1 tracking HPO runner")
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
    p.add_argument("--sleep-between", type=float, default=5.0, help="Seconds to wait between HPO configs.")
    p.add_argument("--continue-on-error", action="store_true", help="Continue grid even if one config fails.")
    p.add_argument("--lock-file", default="tracking_hpo_runs/.jde_v1_hpo.lock")
    p.add_argument("--no-lock", action="store_true")
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
    tokens: list[str] = []
    for k, v in opts.items():
        # Keep booleans out of opts because YOLOX BaseExp.merge casts bool("False") -> True.
        if isinstance(v, bool):
            continue
        tokens += [str(k), str(v)]
    return tokens


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def find_summary_for_expn(expn: str) -> Path | None:
    root = Path("YOLOX_outputs") / expn
    if not root.exists():
        return None
    candidates = list(root.rglob("*_summary.txt"))
    if not candidates:
        return None
    ped = [p for p in candidates if p.name == "pedestrian_summary.txt"]
    candidates = ped or candidates
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

    for k, v in zip(header, values):
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v

    return out


class RunLock:
    def __init__(self, path: Path, enabled: bool = True):
        self.path = path
        self.enabled = enabled

    def __enter__(self):
        if not self.enabled:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            old = self.path.read_text(errors="ignore")
            raise SystemExit(
                f"[ERROR] HPO lock exists: {self.path}\n"
                f"Another HPO run may already be active.\n\n"
                f"Lock content:\n{old}\n\n"
                f"If you are sure no HPO is running, remove it:\n"
                f"  rm -f {self.path}\n"
            )
        self.path.write_text(
            f"pid={os.getpid()}\n"
            f"time={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"cwd={Path.cwd()}\n"
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self.path.exists():
            self.path.unlink()


def write_csv(csv_path: Path, rows: list[dict[str, Any]]):
    all_keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    grid = load_grid(args.grid)
    selected = grid[args.start_index:]
    if args.max_runs is not None:
        selected = selected[: args.max_runs]

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path("tracking_hpo_runs") / f"{args.base_expn}_{run_stamp}"
    log_dir = run_dir / "per_config_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.output_csv) if args.output_csv else run_dir / "results.csv"
    commands_path = run_dir / "commands.sh"

    cuda_visible = args.cuda_visible_devices or args.parallel_gpus
    env_base = os.environ.copy()
    if cuda_visible:
        env_base["CUDA_VISIBLE_DEVICES"] = cuda_visible

    rows: list[dict[str, Any]] = []
    commands = [
        "#!/bin/bash",
        "set -euo pipefail",
        "# Commands generated by strict sequential HPO runner.",
        "# Run ONE at a time. Do not background these commands.",
        "",
    ]

    print("=" * 100)
    print("STRICT SEQUENTIAL JDE V1 Tracking HPO")
    print("grid:", args.grid)
    print("exp:", args.exp_file)
    print("ckpt:", args.ckpt)
    print("runs:", len(selected))
    print("execute:", args.execute)
    print("parallel_workers PER EVALUATION:", args.parallel_workers)
    print("parallel_gpus PER EVALUATION:", args.parallel_gpus)
    print("CUDA_VISIBLE_DEVICES:", cuda_visible)
    print("run_dir:", run_dir)
    print("lock_file:", "disabled" if args.no_lock else args.lock_file)
    print("=" * 100)

    with RunLock(Path(args.lock_file), enabled=(args.execute and not args.no_lock)):
        for seq_i, cfg in enumerate(selected):
            idx = args.start_index + seq_i
            name = clean_name(cfg["name"])
            expn = f"{args.base_expn}_{idx:03d}_{name}"
            opts = cfg.get("opts", {})
            summary_path = find_summary_for_expn(expn)

            if args.skip_existing and summary_path is not None:
                print(f"[SKIP existing] {idx:03d} {name} -> {summary_path}")
                metrics = parse_summary_file(summary_path)
                row = {
                    "index": idx,
                    "name": name,
                    "expn": expn,
                    "status": "skipped_existing",
                    "returncode": 0,
                    "duration_sec": 0,
                    **opts,
                    **metrics,
                }
                rows.append(row)
                write_csv(csv_path, rows)
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

            stdout_log = log_dir / f"{idx:03d}_{name}.stdout.log"
            stderr_log = log_dir / f"{idx:03d}_{name}.stderr.log"

            print("\n" + "-" * 100)
            print(f"[{idx:03d}] START {name}")
            print("This is HPO config", seq_i + 1, "of", len(selected))
            print("The next HPO config will NOT start until this process exits.")
            print("Sequence parallelism inside this config:")
            print("  parallel_workers:", args.parallel_workers)
            print("  parallel_gpus   :", args.parallel_gpus)
            print("expn:", expn)
            print("opts:", json.dumps(opts, sort_keys=True))
            print("stdout log:", stdout_log)
            print("stderr log:", stderr_log)
            print(cmd_line)

            start = time.time()
            if args.execute:
                with open(stdout_log, "w") as fout, open(stderr_log, "w") as ferr:
                    proc = subprocess.Popen(cmd, stdout=fout, stderr=ferr, env=env_base)
                    print(f"launched pid={proc.pid}; waiting for completion...")
                    returncode = proc.wait()
                print(f"[{idx:03d}] FINISHED returncode={returncode}")
            else:
                returncode = -999
                print("[DRY RUN] command not executed.")

            duration = time.time() - start
            metrics = parse_summary_file(find_summary_for_expn(expn))

            row = {
                "index": idx,
                "name": name,
                "expn": expn,
                "status": "ok" if returncode == 0 else ("dry_run" if returncode == -999 else "failed"),
                "returncode": int(returncode),
                "duration_sec": round(duration, 2),
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                **opts,
                **metrics,
            }
            rows.append(row)
            write_csv(csv_path, rows)

            if returncode not in (0, -999):
                print(f"[ERROR] HPO config failed with returncode={returncode}.")
                print("Check:")
                print(f"  tail -n 120 {stdout_log}")
                print(f"  tail -n 120 {stderr_log}")
                print(f"  tail -n 120 YOLOX_outputs/{expn}/parallel_worker_logs/worker_00.log")
                if not args.continue_on_error:
                    print("Stopping because --continue-on-error was not set.")
                    break

            if args.execute and args.sleep_between > 0 and seq_i != len(selected) - 1:
                print(f"Sleeping {args.sleep_between:.1f}s before next HPO config...")
                time.sleep(args.sleep_between)

    commands_path.write_text("\n".join(commands) + "\n")
    commands_path.chmod(0o755)

    print("\n" + "=" * 100)
    print("Sequential HPO runner finished.")
    print("CSV:", csv_path)
    print("Commands:", commands_path)
    print("Per-config logs:", log_dir)
    print("Summarize:")
    print(f"  python tools/tracking_hpo/summarize_jde_v1_tracking_hpo.py --csv {csv_path} --sort HOTA --topk 20")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())