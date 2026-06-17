#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Box-aware deformable ReID sampler for dense TAL-JDE.

This module replaces a single point-level ReID descriptor E[p] with a small
learned aggregation of body-part samples around p:

    z_body(p) = (1 - m) E[p] + m * sum_k a_k(p) E[p + Delta p_k]

where Delta p_k is initialized from simple body-part anchors and then refined
by learned offsets. The offsets are scaled by the predicted box width/height
in feature-cell units, so the sampled positions adapt to person size.

The detector branch is not changed. This module is used only on the ReID map.
"""

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoxAwareDeformableReIDSampler(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        reid_dim: int,
        num_points: int = 5,
        offset_scale: float = 0.25,
        mix_init: float = 0.25,
        use_box_scale: bool = True,
        detach_box_scale: bool = True,
        max_box_scale: float = 32.0,
        padding_mode: str = "border",
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.reid_dim = int(reid_dim)
        self.num_points = int(num_points)
        self.offset_scale = float(offset_scale)
        self.use_box_scale = bool(use_box_scale)
        self.detach_box_scale = bool(detach_box_scale)
        self.max_box_scale = float(max_box_scale)
        self.padding_mode = str(padding_mode)

        self.offset_pred = nn.Conv2d(
            self.hidden_dim,
            2 * self.num_points,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.attn_pred = nn.Conv2d(
            self.hidden_dim,
            self.num_points,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Start from deterministic body-like anchors and uniform attention.
        # Learned residual offsets are zero-initialized, so the initial module is stable.
        nn.init.constant_(self.offset_pred.weight, 0.0)
        nn.init.constant_(self.offset_pred.bias, 0.0)
        nn.init.constant_(self.attn_pred.weight, 0.0)
        nn.init.constant_(self.attn_pred.bias, 0.0)

        base_offsets = self._make_base_offsets(self.num_points)  # [K, 2] as (x, y)
        self.register_buffer("base_offsets", base_offsets, persistent=True)

        # Mix between original point descriptor and deformable body descriptor.
        # mix = sigmoid(logit). mix_init=0.25 means mostly baseline at start.
        mix_init = min(max(float(mix_init), 1e-4), 1.0 - 1e-4)
        mix_logit = math.log(mix_init / (1.0 - mix_init))
        self.mix_logit = nn.Parameter(torch.tensor(mix_logit, dtype=torch.float32))

    @staticmethod
    def _make_base_offsets(num_points: int) -> torch.Tensor:
        """Body-part anchors in normalized box coordinates: (x, y)."""
        if num_points <= 1:
            offsets = [(0.0, 0.0)]
        elif num_points == 3:
            offsets = [
                (0.0, 0.0),
                (0.0, -0.25),
                (0.0, 0.25),
            ]
        elif num_points == 5:
            offsets = [
                (0.0, 0.0),     # center/torso
                (0.0, -0.25),    # upper body
                (0.0, 0.25),     # lower body
                (-0.20, 0.0),    # left body side
                (0.20, 0.0),     # right body side
            ]
        else:
            # Center + points on an ellipse inside the box.
            offsets = [(0.0, 0.0)]
            remain = num_points - 1
            for i in range(remain):
                angle = 2.0 * math.pi * i / max(remain, 1)
                offsets.append((0.25 * math.cos(angle), 0.35 * math.sin(angle)))
        return torch.tensor(offsets[:num_points], dtype=torch.float32)

    @staticmethod
    def _meshgrid_xy(height: int, width: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        y_range = torch.arange(height, device=device, dtype=torch.float32)
        x_range = torch.arange(width, device=device, dtype=torch.float32)
        try:
            yy, xx = torch.meshgrid(y_range, x_range, indexing="ij")
        except TypeError:
            yy, xx = torch.meshgrid(y_range, x_range)
        return xx, yy

    def _build_sampling_grid(
        self,
        reg_raw: torch.Tensor,
        offsets_norm: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Return grid [B, K, H, W, 2] in normalized grid_sample coords."""
        bsz = reg_raw.shape[0]
        device = reg_raw.device

        if self.use_box_scale:
            wh_cells = torch.exp(reg_raw[:, 2:4].float()).clamp(
                min=1.0,
                max=self.max_box_scale,
            )  # [B, 2, H, W], width/height in feature-cell units
            if self.detach_box_scale:
                wh_cells = wh_cells.detach()
        else:
            wh_cells = torch.ones(
                (bsz, 2, height, width),
                dtype=torch.float32,
                device=device,
            )

        # offsets_norm: [B, K, 2, H, W] in box-normalized coordinates.
        # scale by box w/h in feature-cell units.
        offsets_cells = offsets_norm.float() * wh_cells.unsqueeze(1)  # [B,K,2,H,W]

        xx, yy = self._meshgrid_xy(height, width, device)
        xx = xx.view(1, 1, height, width)
        yy = yy.view(1, 1, height, width)

        sample_x = xx + offsets_cells[:, :, 0]
        sample_y = yy + offsets_cells[:, :, 1]

        if width > 1:
            sample_x = 2.0 * sample_x / float(width - 1) - 1.0
        else:
            sample_x = torch.zeros_like(sample_x)
        if height > 1:
            sample_y = 2.0 * sample_y / float(height - 1) - 1.0
        else:
            sample_y = torch.zeros_like(sample_y)

        grid = torch.stack([sample_x, sample_y], dim=-1)  # [B,K,H,W,2]
        return grid

    def forward(
        self,
        reid_map: torch.Tensor,
        reg_raw: torch.Tensor,
        guide_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            reid_map:  [B, D, H, W] raw dense ReID map.
            reg_raw:   [B, 4, H, W] raw YOLOX regression logits for this level.
                       exp(reg_raw[:,2:4]) approximates box w/h in feature cells.
            guide_feat:[B, C, H, W] feature used to predict offsets/attention.
        Returns:
            body_reid: [B, D, H, W] body-aware ReID map.
        """
        bsz, dim, height, width = reid_map.shape
        dtype = reid_map.dtype

        # Predict residual offsets around fixed body-part anchors.
        offsets = self.offset_pred(guide_feat)
        offsets = offsets.view(bsz, self.num_points, 2, height, width)
        offsets = self.offset_scale * torch.tanh(offsets.float())

        base = self.base_offsets.to(device=reid_map.device, dtype=torch.float32)
        base = base.view(1, self.num_points, 2, 1, 1)
        offsets_norm = base + offsets

        # Predict aggregation weights.
        attn = self.attn_pred(guide_feat).float()  # [B,K,H,W]
        attn = F.softmax(attn, dim=1)

        grid = self._build_sampling_grid(reg_raw, offsets_norm, height, width)

        # grid_sample for each point. Use fp32 for robust interpolation under --fp16.
        sampled = []
        reid_src = reid_map.float()
        for k in range(self.num_points):
            sampled_k = F.grid_sample(
                reid_src,
                grid[:, k].float(),
                mode="bilinear",
                padding_mode=self.padding_mode,
                align_corners=True,
            )
            sampled.append(sampled_k)
        sampled = torch.stack(sampled, dim=1)  # [B,K,D,H,W]

        body = (sampled * attn.unsqueeze(2)).sum(dim=1)  # [B,D,H,W]
        body = body.to(dtype=dtype)

        mix = torch.sigmoid(self.mix_logit).to(dtype=dtype)
        out = (1.0 - mix) * reid_map + mix * body
        return out

    def debug_values(self):
        return {
            "num_points": self.num_points,
            "offset_scale": self.offset_scale,
            "mix": float(torch.sigmoid(self.mix_logit).detach().cpu()),
            "use_box_scale": self.use_box_scale,
            "detach_box_scale": self.detach_box_scale,
        }