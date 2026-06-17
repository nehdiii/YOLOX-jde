#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""DanceTrack evaluator for YOLOX-JDE detector + motion-only HybridSORT.

Purpose
-------
This evaluator tests the detector/bbox quality of a trained JDE checkpoint while
removing the ReID embedding from tracking. It still runs the YOLOX-JDE model and
uses postprocess_jde(), but it slices detections back to the normal 7 columns:

    [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]

Then it feeds those detections into the motion-only HybridSORT tracker:

    trackers.hybrid_sort_tracker.hybrid_sort.Hybrid_Sort

Therefore the boxes come from the JDE checkpoint, but association uses motion/IoU
only and does not use detector embeddings or external FastReID features.
"""

from collections import defaultdict
import itertools
import os
import time

import torch
from tqdm import tqdm

from yolox.evaluators.mot_evaluator_dance import MOTEvaluator as BaseMOTEvaluator
from yolox.utils import gather, is_main_process, postprocess, synchronize, time_synchronized
from yolox.utils.jde_postprocess import postprocess_jde
from trackers.hybrid_sort_tracker.hybrid_sort import Hybrid_Sort
from utils.utils import write_results_no_score


class MOTEvaluatorJDEMotion(BaseMOTEvaluator):
    def evaluate_hybrid_sort_motion(
        self,
        args,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
    ):
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()

        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        ori_thresh = self.args.track_thresh
        tracker = None

        for cur_iter, (imgs, _, info_imgs, ids, raw_image) in enumerate(progress_bar(self.dataloader)):
            with torch.no_grad():
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split("/")[0]

                # Keep the same threshold adaptation logic as the existing evaluators.
                if video_name == "MOT17-05-FRCNN" or video_name == "MOT17-06-FRCNN":
                    self.args.track_buffer = 14
                elif video_name == "MOT17-13-FRCNN" or video_name == "MOT17-14-FRCNN":
                    self.args.track_buffer = 25
                else:
                    self.args.track_buffer = 30

                if video_name == "MOT17-01-FRCNN":
                    self.args.track_thresh = 0.65
                elif video_name == "MOT17-06-FRCNN":
                    self.args.track_thresh = 0.65
                elif video_name == "MOT17-12-FRCNN":
                    self.args.track_thresh = 0.7
                elif video_name == "MOT17-14-FRCNN":
                    self.args.track_thresh = 0.67
                else:
                    self.args.track_thresh = ori_thresh

                if video_name == "MOT20-06" or video_name == "MOT20-08":
                    self.args.track_thresh = 0.3
                else:
                    self.args.track_thresh = ori_thresh

                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time_synchronized()

                if video_name not in video_names:
                    video_names[video_id] = video_name

                if frame_id == 1:
                    tracker = Hybrid_Sort(
                        args,
                        det_thresh=self.args.track_thresh,
                        iou_threshold=self.args.iou_thresh,
                        asso_func=self.args.asso,
                        delta_t=self.args.deltat,
                        inertia=self.args.inertia,
                        use_byte=self.args.use_byte,
                    )
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, f"{video_names[video_id - 1]}.txt")
                        write_results_no_score(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)
                raw_outputs = model(imgs)
                if decoder is not None:
                    raw_outputs = decoder(raw_outputs, dtype=raw_outputs.type())

                # Important: use JDE postprocess because raw JDE outputs contain embedding columns.
                # Then remove embeddings before motion-only tracking.
                outputs = postprocess_jde(raw_outputs, self.num_classes, self.confthre, self.nmsthre)
                if outputs[0] is not None and outputs[0].shape[1] > 7:
                    outputs[0] = outputs[0][:, :7].contiguous()
                elif outputs[0] is not None and outputs[0].shape[1] != 7:
                    # Fallback for non-JDE detector heads if this runner is accidentally used there.
                    outputs = postprocess(raw_outputs, self.num_classes, self.confthre, self.nmsthre)

            if is_time_record:
                infer_end = time_synchronized()
                inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6 if self.args.dataset in ["mot17", "mot20"] else False
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end

            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, f"{video_names[video_id]}.txt")
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results