#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """YOLOX model wrapper.

    This version stays backward-compatible with the normal detector head and also
    accepts the JDE V1 head output:

    normal detector head:
        (loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg)

    JDE head:
        (loss, iou_loss, conf_loss, cls_loss, l1_loss,
         id_loss, id_acc, valid_id_count, num_fg)
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            head_out = self.head(fpn_outs, targets, x)

            if len(head_out) == 6:
                loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = head_out
                id_loss = x.new_tensor(0.0)
                id_acc = x.new_tensor(0.0)
                valid_id_count = x.new_tensor(0.0)
            elif len(head_out) == 9:
                (
                    loss,
                    iou_loss,
                    conf_loss,
                    cls_loss,
                    l1_loss,
                    id_loss,
                    id_acc,
                    valid_id_count,
                    num_fg,
                ) = head_out
            else:
                raise ValueError(
                    f"Unexpected YOLOX head output length {len(head_out)}. "
                    "Expected 6 for detector or 9 for JDE."
                )

            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "id_loss": id_loss,
                "id_acc": id_acc,
                "valid_id_count": valid_id_count,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs