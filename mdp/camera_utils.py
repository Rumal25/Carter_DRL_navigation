# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared camera utility functions used by both observations.py and rewards.py.

Segmentation pipeline (no neural network needed in simulation):
    RGB image → color thresholding → binary mask → compact features

Color conventions (match the USD environment materials):
    Yellow floor:  R > 150, G > 120, B < 100
    Blue wall:     R < 100, G < 100, B > 150
    Other:         classified as floor (safe)

Future upgrade path:
    Replace `segment_image()` with a neural network segmentation model.
    The `image_features_from_seg()` function and all downstream code remain unchanged.
    This is the integration point for the semantic segmentation model from the paper.
"""

import torch


def segment_image(rgb_image: torch.Tensor) -> torch.Tensor:
    """Convert an RGB camera image to a binary wall/floor segmentation mask.

    Uses simple color thresholding — works perfectly in simulation where
    colors are exact and reproducible. Replace this function with a neural
    network segmentation model when deploying on a physical robot.

    Args:
        rgb_image: Camera output tensor of shape [N, H, W, 3] or [N, H, W, 4].
                   Values should be in range [0, 255] (uint8 or float).

    Returns:
        seg_mask: Binary float tensor of shape [N, H, W].
                  1.0 = wall (blue), 0.0 = floor/other (safe).
    """
    # Handle RGBA input (some IsaacLab camera configs return 4 channels)
    rgb = rgb_image[..., :3].float()

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    # Blue wall: low red, low green, high blue
    is_wall = (r < 100.0) & (g < 100.0) & (b > 150.0)

    seg_mask = torch.zeros(rgb_image.shape[:3], dtype=torch.float32, device=rgb_image.device)
    seg_mask[is_wall] = 1.0

    return seg_mask


def image_features_from_seg(seg_mask: torch.Tensor) -> torch.Tensor:
    """Extract 4 compact navigation features from a segmentation mask.

    These 4 numbers give the policy enough information to:
      - Know how dangerous the current view is (wall_ratio)
      - Know which side has more wall → which way to steer (left/right split)
      - Know where the safe floor is horizontally (floor_center_x)

    Args:
        seg_mask: Binary float tensor of shape [N, H, W]. 1=wall, 0=floor.

    Returns:
        features: Float tensor of shape [N, 4]:
            [0] wall_ratio        — fraction of ALL pixels that are wall [0, 1]
            [1] left_wall_ratio   — wall fraction in LEFT half of image [0, 1]
            [2] right_wall_ratio  — wall fraction in RIGHT half of image [0, 1]
            [3] floor_center_x    — horizontal centroid of floor pixels [0, 1]
                                    0.0 = floor is on the left
                                    0.5 = floor is centered (safe corridor)
                                    1.0 = floor is on the right

    Steering intuition:
        left_wall_ratio >> right_wall_ratio  → wall on left  → steer right
        right_wall_ratio >> left_wall_ratio  → wall on right → steer left
        floor_center_x < 0.5               → floor shifted left → steer left
        floor_center_x > 0.5               → floor shifted right → steer right
    """
    N, H, W = seg_mask.shape

    # Overall wall fraction
    wall_ratio = seg_mask.mean(dim=[1, 2])  # [N]

    # Left/right wall fractions (split image vertically at center)
    left_wall_ratio = seg_mask[:, :, : W // 2].mean(dim=[1, 2])   # [N]
    right_wall_ratio = seg_mask[:, :, W // 2 :].mean(dim=[1, 2])  # [N]

    # Horizontal centroid of safe floor pixels (1 - wall = floor)
    floor = 1.0 - seg_mask  # [N, H, W], 1=floor
    x_coords = torch.linspace(0.0, 1.0, W, device=seg_mask.device)  # [W]
    # Weighted sum: where is the floor concentrated horizontally?
    floor_center_x = (floor * x_coords).sum(dim=[1, 2]) / (floor.sum(dim=[1, 2]) + 1e-6)  # [N]

    return torch.stack([wall_ratio, left_wall_ratio, right_wall_ratio, floor_center_x], dim=1)
