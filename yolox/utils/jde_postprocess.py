#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Post-processing helpers for YOLOX-JDE outputs.

Normal YOLOX postprocess drops channels after the selected class. A dense JDE
head appends embeddings after the class logits, so this variant keeps those
embedding channels after NMS:

input prediction layout:
    [cx, cy, w, h, obj, cls_0..cls_C-1, emb_0..emb_D-1]

output detection layout:
    [x1, y1, x2, y2, obj_conf, class_conf, class_pred, emb_0..emb_D-1]
"""

import torch
import torchvision


def postprocess_jde(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        if not image_pred.size(0):
            continue

        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()

        # Keep embeddings after normal detector columns.
        extra = image_pred[:, 5 + num_classes :]
        detections = torch.cat(
            (image_pred[:, :5], class_conf, class_pred.float(), extra), dim=1
        )
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        output[i] = detections if output[i] is None else torch.cat((output[i], detections))

    return output