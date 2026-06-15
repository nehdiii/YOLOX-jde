# encoding: utf-8
"""YOLOX-X DanceTrack JDE V1 training/tracking config.

JDE V1 means:
- detector assignment remains YOLOX/SimOTA only;
- ReID loss is computed after assignment on positive anchors;
- no ReID term is used inside SimOTA matching.
"""

import json
import os

import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir


def infer_num_ids_from_coco(ann_path, fallback=1):
    """Infer identity classifier size from zero-based global-ID annotations.

    The converter `convert_dance_to_coco_jde.py` writes zero-based track IDs and
    top-level `num_ids`. Therefore we use `num_ids` directly. Do not add +1.
    """
    if not os.path.exists(ann_path):
        print(f"[DanceTrack-JDE-V1] WARNING: annotation file not found: {ann_path}")
        print(f"[DanceTrack-JDE-V1] Using fallback num_ids={fallback}")
        return int(fallback)

    with open(ann_path, "r") as f:
        data = json.load(f)

    if "num_ids" in data and int(data["num_ids"]) > 0:
        num_ids = int(data["num_ids"])
        print(f"[DanceTrack-JDE-V1] num_ids from top-level metadata: {num_ids}")
        return num_ids

    tids = []
    for ann in data.get("annotations", []):
        tid = ann.get("track_id", -1)
        try:
            tid = int(tid)
        except Exception:
            continue
        if tid >= 0:
            tids.append(tid)

    if not tids:
        print("[DanceTrack-JDE-V1] WARNING: no valid track_id found; using fallback")
        return int(fallback)

    num_ids = max(tids) + 1
    print(
        f"[DanceTrack-JDE-V1] inferred num_ids={num_ids} "
        f"from {len(set(tids))} unique IDs, max_track_id={max(tids)}"
    )
    return int(num_ids)


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Requires the JDE converter output. Keep val/test normal for detector AP
        # and TrackEval compatibility; only train needs global identity labels.
        self.train_ann = "train_jde.json"
        self.val_ann = "val.json"
        self.test_ann = "test.json"

        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.random_size = (18, 32)
        self.max_epoch = 8
        self.print_interval = 20
        self.eval_interval = 5
        self.test_conf = 0.1
        self.nmsthre = 0.7
        self.no_aug_epochs = 1
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1

        dancetrack_root = os.path.join(get_yolox_datadir(), "dancetrack")
        self.num_ids = infer_num_ids_from_coco(
            os.path.join(dancetrack_root, "annotations", self.train_ann),
            fallback=1,
        )

        # JDE branch settings.
        self.reid_dim = 128
        self.reid_weight = 1.0
        self.use_uncertainty = False
        self.label_id_index = 5
        
    def get_model(self, sublinear=False):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from yolox.models.yolox import YOLOX
            from yolox.models.yolo_pafpn import YOLOPAFPN
            from yolox.models.yolo_head_dense_reid_v1 import YOLOXHead

            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, depthwise=False)
            head = YOLOXHead(
                num_classes=self.num_classes,
                width=self.width,
                in_channels=in_channels,
                depthwise=False,
                reid_dim=self.reid_dim,
                num_ids=self.num_ids,
                reid_weight=self.reid_weight,
                use_uncertainty=self.use_uncertainty,
                label_id_index=self.label_id_index,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        dataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "dancetrack"),
            json_file=self.train_ann,
            name="train",
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )
        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        return DataLoader(self.dataset, **dataloader_kwargs)

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, run_tracking=False):
        from yolox.data import MOTDataset, ValTransform

        if testdev:
            valdataset = MOTDataset(
                data_dir=os.path.join(get_yolox_datadir(), "dancetrack"),
                json_file=self.test_ann,
                img_size=self.test_size,
                name="test",
                preproc=ValTransform(
                    rgb_means=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                run_tracking=run_tracking,
            )
        else:
            valdataset = MOTDataset(
                data_dir=os.path.join(get_yolox_datadir(), "dancetrack"),
                json_file=self.val_ann,
                img_size=self.test_size,
                name="val",
                preproc=ValTransform(
                    rgb_means=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                run_tracking=run_tracking,
            )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "batch_size": batch_size,
        }
        return torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev, run_tracking=False)
        return COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )