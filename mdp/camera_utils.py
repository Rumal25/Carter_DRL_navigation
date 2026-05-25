# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared camera utility functions used by observations.py and rewards.py.

Color conventions (match USD environment materials):
    Yellow floor:  R=255, G=220, B=0     normalised (1.000, 0.863, 0.000)
    Blue wall:     R=30,  G=80,  B=200   normalised (0.118, 0.314, 0.784)

FIX: Isaac Lab CameraCfg returns float values in [0.0, 1.0], NOT [0, 255].
All thresholds use normalised [0, 1] range.
"""

import torch


def segment_image(rgb_image: torch.Tensor) -> torch.Tensor:
    rgb = rgb_image[..., :3].float()

    # Always normalise — camera always returns uint8 [0,255]
    # No conditional needed — just always divide
    rgb = rgb / 255.0

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    is_wall = (r < 0.40) & (g < 0.50) & (b > 0.55)

    seg_mask = torch.zeros(rgb_image.shape[:3], dtype=torch.float32,
                           device=rgb_image.device)
    seg_mask[is_wall] = 1.0
    return seg_mask


def image_features_from_seg(seg_mask: torch.Tensor) -> torch.Tensor:
    """Extract 4 compact navigation features from segmentation mask.

    Args:
        seg_mask: [N, H, W] float32.  1=wall, 0=floor

    Returns:
        features: [N, 4]
            [0] wall_ratio        fraction of ALL pixels = wall
            [1] left_wall_ratio   wall fraction in LEFT half
            [2] right_wall_ratio  wall fraction in RIGHT half
            [3] floor_center_x    horizontal centroid of floor [0=left, 0.5=center, 1=right]
    """
    N, H, W = seg_mask.shape

    wall_ratio       = seg_mask.mean(dim=[1, 2])
    left_wall_ratio  = seg_mask[:, :, : W // 2].mean(dim=[1, 2])
    right_wall_ratio = seg_mask[:, :, W // 2 :].mean(dim=[1, 2])

    floor        = 1.0 - seg_mask
    x_coords     = torch.linspace(0.0, 1.0, W, device=seg_mask.device)
    floor_center_x = (floor * x_coords).sum(dim=[1, 2]) / (
        floor.sum(dim=[1, 2]) + 1e-6
    )

    return torch.stack(
        [wall_ratio, left_wall_ratio, right_wall_ratio, floor_center_x], dim=1
    )


# def debug_camera_output(rgb_image: torch.Tensor, step: int, interval: int = 500) -> None:
#     """Print camera stats every interval steps. Add to observations.py temporarily.

#     Usage:
#         from .camera_utils import segment_image, image_features_from_seg, debug_camera_output
#         debug_camera_output(rgb, env.common_step_counter)
#     """
#     if step % interval != 0:
#         return
#     rgb = rgb_image[..., :3].float()
#     print(f"\n[CameraDebug] step={step}  shape={rgb_image.shape}  "
#           f"dtype={rgb_image.dtype}")
#     print(f"  pixel range : {rgb.min():.4f} to {rgb.max():.4f}")
#     print(f"  mean RGB    : R={rgb[...,0].mean():.4f}  "
#           f"G={rgb[...,1].mean():.4f}  B={rgb[...,2].mean():.4f}")
#     seg = segment_image(rgb_image)
#     wr  = seg.mean(dim=[1, 2])
#     print(f"  wall_ratio  : min={wr.min():.4f}  max={wr.max():.4f}  "
#           f"mean={wr.mean():.4f}")
#     print(f"  wall detected in {(wr > 0.01).sum().item()} / "
#           f"{rgb_image.shape[0]} envs\n")
