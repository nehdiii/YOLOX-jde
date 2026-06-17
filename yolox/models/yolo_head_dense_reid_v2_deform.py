#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# YOLOX-JDE V2 + Deformable ReID dense head.
# Based on the local YOLOX-JDE V2 SimOTA-ReID head.
# V2 changes SimOTA assignment: identity-aware ReID matching cost is added
# to the detection assignment cost, and can also affect dynamic-k quality.
# Deform adds a box-aware deformable sampler on the dense ReID map before
# identity loss, SimOTA-ReID matching cost, and tracking feature export.

from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou

import math

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv
from .deformable_reid_sampler import BoxAwareDeformableReIDSampler


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        reid_dim=0,
        num_ids=0,
        reid_weight=1.0,
        use_uncertainty=False,
        label_id_index=5,
        reid_match_weight=0.2,
        reid_match_max_cost=2.0,
        use_reid_in_dynamic_k=True,
        # Deformable body-aware ReID sampling. Detection branch stays unchanged.
        deform_reid=False,
        deform_reid_num_points=5,
        deform_reid_offset_scale=0.25,
        deform_reid_mix_init=0.25,
        deform_reid_use_box_scale=True,
        deform_reid_detach_box_scale=True,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): wheather apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        # JDE V2: ReID is trained after assignment like V1, but predicted
        # embeddings are also used inside SimOTA matching.
        self.reid_dim = int(reid_dim)
        self.num_ids = int(num_ids)
        self.use_reid = self.reid_dim > 0 and self.num_ids > 0
        self.reid_weight = float(reid_weight)
        self.use_uncertainty = bool(use_uncertainty)
        self.label_id_index = int(label_id_index)

        # JDE V2 / contribution-2 controls:
        # Unlike V1, these values affect SimOTA matching itself.
        # The ReID cost is computed between each GT identity classifier prototype
        # and each candidate anchor embedding.
        self.reid_match_weight = float(reid_match_weight)
        self.reid_match_max_cost = float(reid_match_max_cost)
        self.use_reid_in_dynamic_k = bool(use_reid_in_dynamic_k)

        # V2+Deform / contribution-2 variant:
        # deform_reid only changes the dense identity descriptor map. The detector
        # logits/regression and SimOTA candidate geometry stay unchanged. Because
        # V2 uses reid_outputs inside SimOTA, assignment also receives the
        # body-aware descriptor when deform_reid=True.
        self.deform_reid = bool(deform_reid)
        self.deform_reid_num_points = int(deform_reid_num_points)
        self.deform_reid_offset_scale = float(deform_reid_offset_scale)
        self.deform_reid_mix_init = float(deform_reid_mix_init)
        self.deform_reid_use_box_scale = bool(deform_reid_use_box_scale)
        self.deform_reid_detach_box_scale = bool(deform_reid_detach_box_scale)

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        self.reid_convs = nn.ModuleList()
        self.reid_preds = nn.ModuleList()
        self.deform_reid_samplers = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

            if self.use_reid:
                self.reid_convs.append(
                    nn.Sequential(
                        *[
                            Conv(
                                in_channels=int(256 * width),
                                out_channels=int(256 * width),
                                ksize=3,
                                stride=1,
                                act=act,
                            ),
                            Conv(
                                in_channels=int(256 * width),
                                out_channels=int(256 * width),
                                ksize=3,
                                stride=1,
                                act=act,
                            ),
                        ]
                    )
                )
                self.reid_preds.append(
                    nn.Conv2d(
                        in_channels=int(256 * width),
                        out_channels=self.reid_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
                self.deform_reid_samplers.append(
                    BoxAwareDeformableReIDSampler(
                        hidden_dim=int(256 * width),
                        reid_dim=self.reid_dim,
                        num_points=self.deform_reid_num_points,
                        offset_scale=self.deform_reid_offset_scale,
                        mix_init=self.deform_reid_mix_init,
                        use_box_scale=self.deform_reid_use_box_scale,
                        detach_box_scale=self.deform_reid_detach_box_scale,
                    )
                )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)
        self.expanded_strides = [None] * len(in_channels)

        self.reid_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if self.use_reid:
            self.reid_classifier = nn.Linear(self.reid_dim, self.num_ids)
            self.emb_scale = math.sqrt(2) * math.log(self.num_ids - 1) if self.num_ids > 1 else 1.0
            if self.use_uncertainty:
                self.s_det = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
                self.s_id = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        reid_outputs = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.use_reid:
                reid_feat = self.reid_convs[k](x)
                reid_output = self.reid_preds[k](reid_feat)
                if self.deform_reid:
                    reid_output = self.deform_reid_samplers[k](
                        reid_map=reid_output,
                        reg_raw=reg_output,
                        guide_feat=reid_feat,
                    )

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

                if self.use_reid:
                    reid_outputs.append(
                        reid_output.flatten(start_dim=2).permute(0, 2, 1).contiguous()
                    )

            else:
                if self.use_reid:
                    # Append normalized dense JDE embeddings after detector columns.
                    # Final raw output layout before flattening:
                    # [bbox4, obj1, clsC, embD].
                    reid_output = F.normalize(reid_output, dim=1)
                    output = torch.cat(
                        [reg_output, obj_output.sigmoid(), cls_output.sigmoid(), reid_output], 1
                    )
                else:
                    output = torch.cat(
                        [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                    )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                reid_outputs=torch.cat(reid_outputs, 1) if self.use_reid else None,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        reid_outputs=None,
        dtype=None,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        id_targets = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                id_target = outputs.new_zeros((0,), dtype=torch.long)
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                if labels.shape[2] > self.label_id_index:
                    gt_track_ids_per_image = labels[batch_idx, :num_gt, self.label_id_index].long()
                else:
                    gt_track_ids_per_image = labels.new_full((num_gt,), -1).long()
                bboxes_preds_per_image = bbox_preds[batch_idx]
                
                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        gt_track_ids_per_image,
                        reid_outputs,
                    )
                except RuntimeError:
                    logger.info(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    print("OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size.")
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        gt_track_ids_per_image,
                        reid_outputs,
                        "cpu",
                    )
                
                
                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                id_target = gt_track_ids_per_image[matched_gt_inds]

                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            id_targets.append(id_target)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        id_targets = torch.cat(id_targets, 0) if len(id_targets) else outputs.new_zeros((0,), dtype=torch.long)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        det_loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        id_loss = outputs.new_tensor(0.0)
        id_acc = outputs.new_tensor(0.0)
        valid_id_count = outputs.new_tensor(0.0)

        if self.use_reid:
            # Keep ReID branch/classifier in the autograd graph even when a rare
            # batch has no valid identity. This avoids DDP unused-parameter issues
            # without changing the detector loss.
            zero_id_graph = reid_outputs.sum() * 0.0
            for p in self.reid_classifier.parameters():
                zero_id_graph = zero_id_graph + p.sum() * 0.0

            if reid_outputs is not None and id_targets.numel() > 0 and fg_masks.sum() > 0:
                reid_preds = reid_outputs.reshape(-1, self.reid_dim)[fg_masks]
                id_targets = id_targets.to(reid_preds.device).long()
                valid_mask = (id_targets >= 0) & (id_targets < self.num_ids)
                valid_id_count = valid_mask.sum().float()

                if valid_mask.any():
                    embeddings = F.normalize(reid_preds[valid_mask], dim=1) * self.emb_scale
                    id_logits = self.reid_classifier(embeddings.float())
                    id_loss = self.reid_loss(id_logits, id_targets[valid_mask])
                    with torch.no_grad():
                        id_acc = (id_logits.argmax(dim=1) == id_targets[valid_mask]).float().mean()
                else:
                    id_loss = zero_id_graph
            else:
                id_loss = zero_id_graph

        if self.use_reid and self.use_uncertainty:
            loss = torch.exp(-self.s_det) * det_loss + self.s_det + torch.exp(-self.s_id) * id_loss + self.s_id
        else:
            loss = det_loss + (self.reid_weight * id_loss if self.use_reid else 0.0)

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            id_loss,
            id_acc,
            valid_id_count,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        gt_ids_per_image=None,
        reid_outputs=None,
        mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            if gt_ids_per_image is not None:
                gt_ids_per_image = gt_ids_per_image.cpu()
            if reid_outputs is not None:
                reid_outputs = reid_outputs.cpu()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        img_size = imgs.shape[2:]
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
            img_size
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        if self.use_reid and reid_outputs is not None:
            reid_preds_ = reid_outputs[batch_idx][fg_mask]
        else:
            reid_preds_ = None
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        det_cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss

        pair_wise_reid_cost = self._pairwise_reid_matching_cost(
            gt_ids_per_image,
            reid_preds_,
            num_gt,
        )

        # V2: identity-aware SimOTA assignment.
        # A smaller identity distance lowers the total assignment cost for
        # anchors whose predicted embedding matches the GT identity prototype.
        cost = (
            det_cost
            + self.reid_match_weight * pair_wise_reid_cost
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(
            cost,
            pair_wise_ious,
            gt_classes,
            num_gt,
            fg_mask,
            pair_wise_reid_cost=pair_wise_reid_cost,
        )
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss, pair_wise_reid_cost

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )


    def _pairwise_reid_matching_cost(self, gt_ids_per_image, reid_preds_, num_gt):
        """Return [num_gt, num_candidates] identity matching cost for SimOTA.

        The cost uses the ReID classifier weights as identity prototypes. Invalid
        IDs get zero cost so no-ID boxes behave like detector-only SimOTA.
        In V2+Deform, reid_preds_ comes from the body-aware deformable ReID map.
        """
        if (
            not self.use_reid
            or self.reid_match_weight <= 0
            or self.reid_classifier is None
            or gt_ids_per_image is None
            or reid_preds_ is None
            or reid_preds_.numel() == 0
        ):
            n = 0 if reid_preds_ is None else reid_preds_.shape[0]
            if reid_preds_ is not None:
                return reid_preds_.new_zeros((num_gt, n))
            device = gt_ids_per_image.device if gt_ids_per_image is not None else None
            return torch.zeros((num_gt, n), device=device)

        gt_ids = gt_ids_per_image.long()
        valid = (gt_ids >= 0) & (gt_ids < self.num_ids)
        cost = reid_preds_.new_zeros((num_gt, reid_preds_.shape[0]))
        if not valid.any():
            return cost

        emb = F.normalize(reid_preds_.float(), p=2, dim=1)
        weight = F.normalize(self.reid_classifier.weight.float(), p=2, dim=1)
        gt_proto = weight[gt_ids[valid]]

        valid_cost = 1.0 - torch.mm(gt_proto, emb.t())
        valid_cost = valid_cost.clamp(min=0.0, max=self.reid_match_max_cost)
        valid_cost = valid_cost / valid_cost.mean(dim=1, keepdim=True).clamp(min=1e-6)
        cost[valid] = valid_cost.to(cost.dtype)
        return cost

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
        img_size
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5
        # clip center inside image
        gt_bboxes_per_image_clip = gt_bboxes_per_image[:, 0:2].clone()
        gt_bboxes_per_image_clip[:, 0] = torch.clamp(gt_bboxes_per_image_clip[:, 0], min=0, max=img_size[1])
        gt_bboxes_per_image_clip[:, 1] = torch.clamp(gt_bboxes_per_image_clip[:, 1], min=0, max=img_size[0])

        gt_bboxes_per_image_l = (gt_bboxes_per_image_clip[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image_clip[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image_clip[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image_clip[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        del gt_bboxes_per_image_clip
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask, pair_wise_reid_cost=None):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        dynamic_quality = ious_in_boxes_matrix
        if self.use_reid_in_dynamic_k and pair_wise_reid_cost is not None:
            dynamic_quality = ious_in_boxes_matrix * torch.exp(
                -self.reid_match_weight * pair_wise_reid_cost
            )
        n_candidate_k = min(10, dynamic_quality.size(1))
        topk_ious, _ = torch.topk(dynamic_quality, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            cost_min, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds