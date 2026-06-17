#!/usr/bin/env python3
# encoding: utf-8
"""
Sequence-parallel DanceTrack/YOLOX-JDE + motion-only HybridSORT evaluation runner.

Why this exists:
- Trackers must process frames sequentially inside each video.
- Standard DDP/DistributedSampler splits frames, which breaks tracker state.
- This script splits the validation/test set by complete sequence names and launches
  one normal tracking process per subset. Each worker writes different <seq>.txt files
  into the same result folder. The parent runs TrackEval once after all workers finish.

Example:
  CUDA_VISIBLE_DEVICES=0,1 python tools/run_hybrid_sort_dance_jde_parallel.py \
    -f exps/example/dancetrack/yolox_x_dancetrack_jde_v1.py \
    -c YOLOX_outputs/yolox_x_dancetrack_jde_v1/best_ckpt.pth.tar \
    -b 1 -d 1 --fp16 --fuse --expn my_jde_exp --parallel-workers 2
"""

from loguru import logger
from tqdm import tqdm

import argparse
import math
import os
import random
import subprocess
import sys
import warnings
import time
import json
import contextlib
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, setup_logger
try:
    from yolox.evaluators.mot_evaluator_dance_jde_motion import MOTEvaluatorJDEMotion as MOTEvaluator
except Exception as _jde_import_error:
    MOTEvaluator = None
    _JDE_IMPORT_ERROR = _jde_import_error

from utils.args import make_parser, args_merge_params_form_exp


def remove_argparse_action(parser, dest):
    """Remove an argparse action by dest so we can insert new args before opts."""
    for action in list(parser._actions):
        if action.dest == dest:
            parser._remove_action(action)
            for group in parser._action_groups:
                if action in group._group_actions:
                    group._group_actions.remove(action)
            return action
    return None


def build_parser():
    parser = make_parser()

    # utils.args.make_parser() adds positional opts with nargs=REMAINDER at the end.
    # Remove it temporarily, add our parallel args, then add opts back.
    remove_argparse_action(parser, "opts")

    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of sequence-parallel tracking workers. Use 1 for original behavior.",
    )
    parser.add_argument(
        "--parallel-gpus",
        type=str,
        default=None,
        help="Comma-separated GPU ids for workers, e.g. '0,1,2,3'. Defaults to CUDA_VISIBLE_DEVICES if set.",
    )
    parser.add_argument(
        "--parallel-worker-rank",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--parallel-worker-world",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--parallel-seqs",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--parallel-worker-progress-file",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-trackeval",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-coco-eval",
        action="store_true",
        help="Skip per-worker COCO detector eval. Recommended for parallel tracking.",
    )
    parser.add_argument(
        "--dry-run-seqs",
        action="store_true",
        help="Only print the sequence split and exit.",
    )

    parser.add_argument(
        "--parallel-progress-interval",
        type=float,
        default=2.0,
        help="Seconds between parent progress-bar refreshes while workers run.",
    )
    parser.add_argument(
        "--parallel-progress-mode",
        type=str,
        default="frames",
        choices=["frames", "seqs"],
        help="Parent progress bar mode. 'frames' shows global processed frames like the original tqdm; 'seqs' shows completed sequences.",
    )
    parser.add_argument(
        "--parallel-worker-log-dir",
        type=str,
        default=None,
        help="Optional folder for per-worker stdout/stderr logs. Defaults to YOLOX_outputs/<expn>/parallel_worker_logs.",
    )
    parser.add_argument(
        "--parallel-clean-results",
        action="store_true",
        help="Delete old result .txt files for the target sequences before launching workers. Useful when reusing the same --expn.",
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def result_dir_name(args):
    split_name = "test" if args.test else "val"
    return (
        f"{args.expn}_{split_name}"
        + "_EGWeightHigh" + str(args.EG_weight_high_score)
        + "_EGWeightLow" + str(args.EG_weight_low_score)
        + "_WithLongTermReIDCorrection" + str(args.with_longterm_reid_correction)
        + "_LongTermReIDCorrectionThresh" + str(args.longterm_reid_correction_thresh)
        + "_LongTermReIDCorrectionThreshLow" + str(args.longterm_reid_correction_thresh_low)
        + "_IoUThresh" + str(args.iou_thresh)
        + "_ScoreDifInterval" + str(args.TCM_first_step_weight)
        + "_SecScoreDifInterval" + str(args.TCM_byte_step_weight)
    )


def get_results_folder(exp, args):
    file_name = os.path.join(exp.output_dir, args.expn)
    return os.path.join(file_name, result_dir_name(args))


def sequence_names_from_loader(loader):
    seqs = []
    seen = set()
    for ann in loader.dataset.annotations:
        # annotation tuple is (res, img_info, file_name)
        file_name = ann[2]
        seq = file_name.split("/")[0]
        if seq not in seen:
            seen.add(seq)
            seqs.append(seq)
    return seqs


def filter_loader_to_sequences(loader, keep_sequences):
    keep_sequences = set(keep_sequences)
    dataset = loader.dataset

    keep_indices = []
    for idx, ann in enumerate(dataset.annotations):
        file_name = ann[2]
        seq = file_name.split("/")[0]
        if seq in keep_sequences:
            keep_indices.append(idx)

    dataset.ids = [dataset.ids[i] for i in keep_indices]
    dataset.annotations = [dataset.annotations[i] for i in keep_indices]
    return len(keep_indices)


def split_contiguous(items, n_chunks):
    """
    Contiguous chunks keep video_id order, matching the original evaluator assumption
    that the previous video is video_id - 1 when it flushes results.
    """
    n_chunks = max(1, min(n_chunks, len(items)))
    base = len(items) // n_chunks
    extra = len(items) % n_chunks
    chunks = []
    start = 0
    for i in range(n_chunks):
        end = start + base + (1 if i < extra else 0)
        chunks.append(items[start:end])
        start = end
    return chunks


def parse_gpu_list(args):
    if args.parallel_gpus:
        return [x.strip() for x in args.parallel_gpus.split(",") if x.strip() != ""]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [x.strip() for x in visible.split(",") if x.strip() != ""]

    count = torch.cuda.device_count()
    return [str(i) for i in range(count)] if count > 0 else [""]


def build_eval_loader_quiet(exp, args):
    """Build the eval loader while hiding noisy dataset/coco indexing logs.

    This only affects the parent process. Workers still write full details to
    their worker_XX.log files.
    """
    disabled_names = [
        "yolox.data.datasets.mot",
        "pycocotools.coco",
    ]
    for name in disabled_names:
        try:
            logger.disable(name)
        except Exception:
            pass
    try:
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                return exp.get_eval_loader(args.batch_size, False, args.test, run_tracking=True)
    finally:
        for name in disabled_names:
            try:
                logger.enable(name)
            except Exception:
                pass




def atomic_write_json(path, payload):
    """Write a small JSON file atomically so the parent never reads half-written data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


class WorkerFrameProgress:
    """Small tqdm replacement used inside each worker.

    The evaluator still calls progress_bar(self.dataloader) exactly as before.
    Instead of printing a tqdm bar from every worker, this wrapper updates one
    JSON file. The parent sums all JSON files and prints one clean global
    frame-level progress bar.
    """

    def __init__(self, iterable, progress_file, rank=0, total=None, update_every=1):
        self.iterable = iterable
        self.progress_file = progress_file
        self.rank = rank
        self.total = int(total if total is not None else len(iterable))
        self.update_every = max(1, int(update_every))
        self.current = 0
        self._write(status="running")

    def __iter__(self):
        try:
            for item in self.iterable:
                yield item
                self.current += 1
                if self.current % self.update_every == 0 or self.current >= self.total:
                    self._write(status="running")
        finally:
            # If the worker exits normally, the evaluator will usually reach total.
            # If it crashes, this file still shows the last completed frame.
            self._write(status="finished_iter", current=self.current)

    def __len__(self):
        return self.total

    def _write(self, status="running", current=None):
        atomic_write_json(
            self.progress_file,
            {
                "rank": self.rank,
                "current": int(self.current if current is None else current),
                "total": int(self.total),
                "status": status,
                "updated_at": time.time(),
            },
        )


def patch_worker_tqdm_for_frame_progress(args, total_frames, progress_file):
    """Patch the JDE evaluator module-level tqdm symbol for this worker only."""
    try:
        import yolox.evaluators.mot_evaluator_dance_jde_motion as mot_eval_jde
    except Exception as exc:
        logger.warning(f"Could not patch JDE evaluator tqdm for frame progress: {exc}")
        return

    def quiet_progress(iterable, *a, **kw):
        return WorkerFrameProgress(
            iterable,
            progress_file=progress_file,
            rank=int(args.parallel_worker_rank or 0),
            total=total_frames,
            update_every=1,
        )

    mot_eval_jde.tqdm = quiet_progress


def read_worker_progress(progress_files):
    """Return summed current/total across worker JSON progress files."""
    current = 0
    total = 0
    seen = 0
    stale = 0
    now = time.time()
    for path in progress_files:
        path = Path(path)
        if not path.exists():
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
            cur = int(data.get("current", 0))
            tot = int(data.get("total", 0))
            current += max(0, min(cur, tot))
            total += max(0, tot)
            seen += 1
            if now - float(data.get("updated_at", 0)) > 60:
                stale += 1
        except Exception:
            continue
    return current, total, seen, stale

def run_trackeval(exp, args, results_folder):
    if args.test:
        logger.info("Test mode: skipping TrackEval because ground truth is not available.")
        return 0

    if args.dataset == "dancetrack":
        cmd = (
            "python3 TrackEval/scripts/run_mot_challenge.py "
            "--SPLIT_TO_EVAL val "
            "--METRICS HOTA CLEAR Identity "
            "--GT_FOLDER datasets/dancetrack/val "
            "--SEQMAP_FILE datasets/dancetrack/val/val_seqmap.txt "
            "--SKIP_SPLIT_FOL True "
            "--TRACKERS_TO_EVAL '' "
            "--TRACKER_SUB_FOLDER '' "
            "--USE_PARALLEL True "
            "--NUM_PARALLEL_CORES 8 "
            "--PLOT_CURVES False "
            "--TRACKERS_FOLDER " + results_folder
        )
    elif args.dataset == "mot17":
        cmd = (
            "python TrackEval/scripts/run_mot_challenge.py "
            "--BENCHMARK MOT17 "
            "--SPLIT_TO_EVAL train "
            "--TRACKERS_TO_EVAL '' "
            "--METRICS HOTA CLEAR Identity VACE "
            "--TIME_PROGRESS False "
            "--USE_PARALLEL False "
            "--NUM_PARALLEL_CORES 1 "
            "--GT_FOLDER datasets/mot/ "
            "--TRACKERS_FOLDER " + results_folder + " "
            "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_val_half.txt"
        )
    elif args.dataset == "mot20":
        cmd = (
            "python TrackEval/scripts/run_mot_challenge.py "
            "--BENCHMARK MOT20 "
            "--SPLIT_TO_EVAL train "
            "--TRACKERS_TO_EVAL '' "
            "--METRICS HOTA CLEAR Identity VACE "
            "--TIME_PROGRESS False "
            "--USE_PARALLEL False "
            "--NUM_PARALLEL_CORES 1 "
            "--GT_FOLDER datasets/MOT20/ "
            "--TRACKERS_FOLDER " + results_folder + " "
            "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_val_half.txt"
        )
    else:
        raise ValueError(f"Unsupported dataset for TrackEval: {args.dataset}")

    logger.info("Running TrackEval after all parallel workers finish:")
    logger.info(cmd)
    return os.system(cmd)


def force_jde_runtime_args(args):
    """Force motion-only tracking using detections from a YOLOX-JDE checkpoint.

    This runner still loads the JDE model and obtains detections through
    postprocess_jde(), but it disables all ReID association. The tracker is
    Hybrid_Sort, not Hybrid_Sort_ReID.
    """
    args.hybrid_sort_with_reid = False
    args.with_jde_reid = False
    args.with_fastreid = False
    return args


def ensure_jde_dependencies():
    if MOTEvaluator is None:
        raise ImportError(
            "Could not import yolox.evaluators.mot_evaluator_dance_jde_motion.MOTEvaluatorJDEMotion. "
            "Install/apply the YOLOX-JDE V1 bundle first, then rerun this script. "
            f"Original import error: {_JDE_IMPORT_ERROR}"
        )


def run_worker(exp, args):
    ensure_jde_dependencies()
    force_jde_runtime_args(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("You have chosen to seed testing. This turns on CUDNN deterministic mode.")

    cudnn.benchmark = True
    rank = 0
    torch.cuda.set_device(rank)

    file_name = os.path.join(exp.output_dir, args.expn)
    results_folder = get_results_folder(exp, args)
    os.makedirs(results_folder, exist_ok=True)
    setup_logger(
        file_name,
        distributed_rank=0,
        filename=f"val_log_worker_{args.parallel_worker_rank}.txt",
        mode="a",
    )

    keep_sequences = [x for x in (args.parallel_seqs or "").split(",") if x]
    logger.info(f"Worker {args.parallel_worker_rank}/{args.parallel_worker_world} sequences: {keep_sequences}")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    val_loader = exp.get_eval_loader(args.batch_size, False, args.test, run_tracking=True)
    kept_frames = filter_loader_to_sequences(val_loader, keep_sequences)
    logger.info(f"Worker {args.parallel_worker_rank}: kept {kept_frames} frames.")

    # Parent-visible frame progress. This suppresses per-worker tqdm and lets the
    # parent display one global bar: processed_frames / total_frames.
    progress_file = getattr(args, "parallel_worker_progress_file", None)
    if progress_file:
        patch_worker_tqdm_for_frame_progress(args, total_frames=len(val_loader), progress_file=progress_file)

    evaluator = MOTEvaluator(
        args=args,
        dataloader=val_loader,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        num_classes=exp.num_classes,
    )

    if args.skip_coco_eval:
        evaluator.evaluate_prediction = lambda data_dict, statistics: (0, 0, "Skipped COCO eval in parallel worker.")

    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        ckpt_file = args.ckpt if args.ckpt is not None else os.path.join(file_name, "best_ckpt.pth.tar")
        logger.info(f"loading checkpoint: {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location=f"cuda:{rank}")
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse and args.batch_size == 1, "TensorRT does not support fuse or batch_size > 1 here."
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(trt_file), "TensorRT model not found. Run tools/trt.py first."
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    # JDE motion-only path: use YOLOX-JDE detector boxes, but no embeddings.
    *_, summary = evaluator.evaluate_hybrid_sort_motion(
        args, model, False, args.fp16, trt_file, decoder, exp.test_size, results_folder
    )

    logger.info(f"Worker {args.parallel_worker_rank} completed. Results folder: {results_folder}")
    return 0


def run_parent(exp, args, original_argv):
    ensure_jde_dependencies()
    force_jde_runtime_args(args)

    """Launch sequence-level workers with clean terminal output.

    Workers still run the same evaluator/tracker code, but their noisy stdout/stderr
    including tqdm bars and repeated args/model logs are redirected into separate
    log files. The parent terminal shows one compact sequence progress bar.
    """
    file_name = os.path.join(exp.output_dir, args.expn)
    os.makedirs(file_name, exist_ok=True)
    setup_logger(file_name, distributed_rank=0, filename="val_log_parallel_parent.txt", mode="a")

    val_loader = build_eval_loader_quiet(exp, args)
    seqs = sequence_names_from_loader(val_loader)
    chunks = split_contiguous(seqs, args.parallel_workers)

    results_folder = get_results_folder(exp, args)
    os.makedirs(results_folder, exist_ok=True)

    log_dir = args.parallel_worker_log_dir
    if log_dir is None:
        log_dir = os.path.join(file_name, "parallel_worker_logs")
    os.makedirs(log_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Sequence-parallel YOLOX-JDE + HybridSORT Motion-Only evaluation")
    logger.info(f"Experiment: {args.expn}")
    logger.info(f"Sequences: {len(seqs)} | Workers: {len(chunks)}")
    logger.info(f"Results folder: {results_folder}")
    logger.info(f"Worker logs: {log_dir}")
    logger.info("=" * 80)

    for i, chunk in enumerate(chunks):
        first = chunk[0] if chunk else "-"
        last = chunk[-1] if chunk else "-"
        logger.info(f"Worker {i:02d}: {len(chunk):3d} seqs | {first} -> {last}")

    if args.dry_run_seqs:
        logger.info("Dry run only. No worker launched.")
        return 0

    expected_txt = {f"{seq}.txt" for seq in seqs}

    if args.parallel_clean_results:
        removed = 0
        for txt_name in expected_txt:
            p = Path(results_folder) / txt_name
            if p.exists():
                try:
                    p.unlink()
                    removed += 1
                except OSError as exc:
                    logger.warning(f"Could not remove old result file {p}: {exc}")
        logger.info(f"Removed {removed} old result files before launch.")

    # Progress must ignore old .txt files from previous runs.
    # We only count result files modified after this timestamp.
    progress_start_time = time.time()

    gpus = parse_gpu_list(args)
    logger.info(f"GPU assignment list: {gpus}")

    procs = []
    log_files = []
    finished = set()
    script = str(Path(__file__).resolve())

    for rank, chunk in enumerate(chunks):
        gpu = gpus[rank % len(gpus)]
        env = os.environ.copy()
        if gpu != "":
            env["CUDA_VISIBLE_DEVICES"] = gpu

        worker_log = os.path.join(log_dir, f"worker_{rank:02d}.log")
        worker_progress = os.path.join(log_dir, f"worker_{rank:02d}_progress.json")
        # Remove stale progress from older runs so frame progress starts at 0.
        try:
            Path(worker_progress).unlink()
        except FileNotFoundError:
            pass
        log_fh = open(worker_log, "w", buffering=1)
        log_files.append(log_fh)

        cmd = [
            sys.executable,
            script,
            *original_argv,
            "--parallel-worker-rank", str(rank),
            "--parallel-worker-world", str(len(chunks)),
            "--parallel-seqs", ",".join(chunk),
            "--parallel-worker-progress-file", worker_progress,
            "--skip-trackeval",
            "--skip-coco-eval",
        ]

        logger.info(
            f"Launching worker {rank:02d} on CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')} "
            f"| log: {worker_log}"
        )
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        procs.append((rank, proc, worker_log))

    done_txt = set()
    progress_files = []
    for rank, _, _ in procs:
        progress_files.append(os.path.join(log_dir, f"worker_{rank:02d}_progress.json"))

    def scan_completed_results():
        current = set()
        for p in Path(results_folder).glob("*.txt"):
            if p.name not in expected_txt:
                continue
            try:
                if p.stat().st_mtime >= progress_start_time:
                    current.add(p.name)
            except OSError:
                continue
        return current

    failed = []
    try:
        if args.parallel_progress_mode == "frames":
            total_frames = len(val_loader)
            pbar_total = total_frames
            pbar_unit = "frame"
            pbar_desc = f"Tracking frames ({len(chunks)} workers)"
        else:
            pbar_total = len(seqs)
            pbar_unit = "seq"
            pbar_desc = f"Tracking seqs ({len(chunks)} workers)"

        last_frame_current = 0
        with tqdm(
            total=pbar_total,
            desc=pbar_desc,
            unit=pbar_unit,
            dynamic_ncols=True,
        ) as pbar:
            while True:
                alive = 0
                for rank, proc, worker_log in procs:
                    ret = proc.poll()
                    if ret is None:
                        alive += 1
                    elif rank not in finished:
                        finished.add(rank)
                        if ret == 0:
                            logger.info(f"Worker {rank:02d} finished successfully. log: {worker_log}")
                        else:
                            failed.append((rank, ret, worker_log))
                            logger.error(f"Worker {rank:02d} failed with code {ret}. log: {worker_log}")

                # Sequence completion is still tracked to validate output files and to
                # optionally display seq-level mode, but frame mode is driven by JSON
                # worker counters, not old result files.
                current_done = scan_completed_results()
                new_done = current_done - done_txt
                if new_done:
                    done_txt.update(new_done)

                if args.parallel_progress_mode == "frames":
                    frame_current, frame_total_seen, progress_seen, stale = read_worker_progress(progress_files)
                    # If workers are still initializing, keep the total as the known
                    # dataset total from the parent loader.
                    frame_current = max(last_frame_current, min(frame_current, pbar_total))
                    delta = frame_current - last_frame_current
                    if delta > 0:
                        pbar.update(delta)
                        last_frame_current = frame_current
                    postfix = {
                        "frames": f"{last_frame_current}/{pbar_total}",
                        "seqs": f"{len(done_txt)}/{len(seqs)}",
                        "alive": alive,
                    }
                    if progress_seen < len(chunks):
                        postfix["ready"] = f"{progress_seen}/{len(chunks)}"
                    if stale:
                        postfix["stale"] = stale
                    pbar.set_postfix(postfix)
                else:
                    # Old/fallback seq-level display.
                    if new_done:
                        pbar.update(len(new_done))
                    pbar.set_postfix(done=f"{len(done_txt)}/{len(seqs)}", alive=alive)

                if len(finished) == len(procs):
                    # Final refresh after all workers flush progress/result files.
                    current_done = scan_completed_results()
                    done_txt.update(current_done)
                    if args.parallel_progress_mode == "frames":
                        frame_current, _, _, _ = read_worker_progress(progress_files)
                        frame_current = min(max(frame_current, last_frame_current), pbar_total)
                        if frame_current > last_frame_current:
                            pbar.update(frame_current - last_frame_current)
                            last_frame_current = frame_current
                        # If all workers succeeded, the evaluator may finish before the
                        # last JSON write on network filesystems. Complete the bar only
                        # after processes ended, not before.
                        if not failed and last_frame_current < pbar_total:
                            pbar.update(pbar_total - last_frame_current)
                            last_frame_current = pbar_total
                        pbar.set_postfix(frames=f"{last_frame_current}/{pbar_total}", seqs=f"{len(done_txt)}/{len(seqs)}", alive=0)
                    else:
                        if len(done_txt) > pbar.n:
                            pbar.update(len(done_txt) - pbar.n)
                    break

                time.sleep(max(0.2, float(args.parallel_progress_interval)))
    finally:
        for fh in log_files:
            try:
                fh.close()
            except Exception:
                pass

    if failed:
        logger.error("Some parallel workers failed:")
        for rank, ret, worker_log in failed:
            logger.error(f"  worker {rank:02d}: return code {ret}, log: {worker_log}")
        logger.error("Open the worker log above to see the original error traceback.")
        return 1

    missing = sorted(expected_txt - done_txt)
    if missing:
        logger.error(f"Workers finished but {len(missing)} fresh result files were not detected.")
        logger.error(f"First missing files: {missing[:10]}")
        logger.error("Stopping before TrackEval to avoid evaluating stale result files from an older run.")
        logger.error("Tip: rerun with --parallel-clean-results if you are reusing the same --expn.")
        return 1

    logger.info("All expected fresh sequence result files were produced.")

    if not args.skip_trackeval:
        return run_trackeval(exp, args, results_folder)
    return 0

def main():
    parser = build_parser()
    args = parser.parse_args()
    original_argv = sys.argv[1:]

    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    args_merge_params_form_exp(args, exp)
    force_jde_runtime_args(args)

    if not args.expn:
        args.expn = exp.exp_name

    # Worker mode: process only the sequences assigned by the parent.
    if args.parallel_worker_rank is not None:
        return run_worker(exp, args)

    # Original behavior if --parallel-workers 1.
    if args.parallel_workers <= 1:
        args.parallel_worker_rank = 0
        args.parallel_worker_world = 1
        # use all sequences
        tmp_loader = exp.get_eval_loader(args.batch_size, False, args.test, run_tracking=True)
        args.parallel_seqs = ",".join(sequence_names_from_loader(tmp_loader))
        args.skip_coco_eval = False
        return run_worker(exp, args)

    # Parent mode: split sequences, launch workers, then run TrackEval once.
    return run_parent(exp, args, original_argv)


if __name__ == "__main__":
    raise SystemExit(main())