#!/usr/bin/env python3
# encoding: utf-8
"""
Sequence-parallel DanceTrack/HybridSORT evaluation runner.

Why this exists:
- Trackers must process frames sequentially inside each video.
- Standard DDP/DistributedSampler splits frames, which breaks tracker state.
- This script splits the validation/test set by complete sequence names and launches
  one normal tracking process per subset. Each worker writes different <seq>.txt files
  into the same result folder. The parent runs TrackEval once after all workers finish.

Example:
  CUDA_VISIBLE_DEVICES=0,1 python tools/run_hybrid_sort_dance_parallel.py \
    -f exps/example/mot/yolox_dancetrack_val_hybrid_sort.py \
    -b 1 -d 1 --fp16 --fuse --expn my_exp --parallel-workers 2
"""

from loguru import logger

import argparse
import math
import os
import random
import subprocess
import sys
import warnings
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, setup_logger
from yolox.evaluators import MOTEvaluatorDance as MOTEvaluator

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


def run_worker(exp, args):
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

    if not args.hybrid_sort_with_reid:
        *_, summary = evaluator.evaluate_hybrid_sort(
            args, model, False, args.fp16, trt_file, decoder, exp.test_size, results_folder
        )
    else:
        *_, summary = evaluator.evaluate_hybrid_sort_reid(
            args, model, False, args.fp16, trt_file, decoder, exp.test_size, results_folder
        )

    logger.info(f"Worker {args.parallel_worker_rank} completed. Results folder: {results_folder}")
    return 0


def run_parent(exp, args, original_argv):
    file_name = os.path.join(exp.output_dir, args.expn)
    os.makedirs(file_name, exist_ok=True)
    setup_logger(file_name, distributed_rank=0, filename="val_log_parallel_parent.txt", mode="a")

    val_loader = exp.get_eval_loader(args.batch_size, False, args.test, run_tracking=True)
    seqs = sequence_names_from_loader(val_loader)
    chunks = split_contiguous(seqs, args.parallel_workers)

    logger.info(f"Found {len(seqs)} sequences. Starting {len(chunks)} parallel workers.")
    for i, chunk in enumerate(chunks):
        logger.info(f"Worker {i}: {chunk}")

    if args.dry_run_seqs:
        return 0

    results_folder = get_results_folder(exp, args)
    os.makedirs(results_folder, exist_ok=True)

    gpus = parse_gpu_list(args)
    logger.info(f"GPU assignment list: {gpus}")

    procs = []
    script = str(Path(__file__).resolve())
    for rank, chunk in enumerate(chunks):
        gpu = gpus[rank % len(gpus)]
        env = os.environ.copy()
        if gpu != "":
            env["CUDA_VISIBLE_DEVICES"] = gpu

        cmd = [
            sys.executable,
            script,
            *original_argv,
            "--parallel-worker-rank", str(rank),
            "--parallel-worker-world", str(len(chunks)),
            "--parallel-seqs", ",".join(chunk),
            "--skip-trackeval",
            "--skip-coco-eval",
        ]

        logger.info(f"Launching worker {rank} on CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}")
        logger.info(" ".join(cmd))
        procs.append(subprocess.Popen(cmd, env=env))

    failed = []
    for rank, proc in enumerate(procs):
        ret = proc.wait()
        if ret != 0:
            failed.append((rank, ret))

    if failed:
        logger.error(f"Some parallel workers failed: {failed}")
        return 1

    logger.info("All parallel workers completed.")
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