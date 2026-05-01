# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for Phase 2 camera-based navigation.

Observation vector (10 values):
    indices 0-1  body-frame linear velocity     [v_fwd, v_lat]
    index   2    yaw angular velocity            [yaw_rate]
    indices 3-5  goal in body frame             [x_b, y_b, heading_b]
    indices 6-9  camera image features          [wall_ratio, left_wall, right_wall, floor_cx]

The first 6 values are identical to Phase 1 (carter_manager).
The last 4 values are new — derived from the RGB camera without a neural network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .camera_utils import image_features_from_seg, segment_image

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Phase 1 observation functions (unchanged from carter_manager)
# ---------------------------------------------------------------------------

def carter_base_lin_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Forward and lateral velocity in the robot's body frame.

    Returns shape: [num_envs, 2] → [v_forward, v_lateral]
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, :2]


def carter_base_ang_vel_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw angular velocity (world frame Z axis).

    Returns shape: [num_envs, 1]
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w[:, 2].unsqueeze(1)


def carter_pose_command(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Goal position and heading error in the robot's body frame.

    The command tensor has 4 columns: [x_b, y_b, z_b, heading_b].
    We return [x_b, y_b, heading_b] — 3 values — dropping z_b (always ~0 on flat ground).

    x_b, y_b decrease toward (0, 0) as the robot approaches the goal.
    heading_b is the angle the robot must rotate to face the goal (radians, [-π, π]).

    Returns shape: [num_envs, 3]
    """
    command = env.command_manager.get_command(command_name)
    return torch.cat([command[:, :2], command[:, 3:4]], dim=1)


# ---------------------------------------------------------------------------
# Phase 2 observation function — camera image features
# ---------------------------------------------------------------------------

def camera_image_features(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
) -> torch.Tensor:
    """Extract 4 compact navigation features from the stereo camera RGB image.

    Pipeline:
        camera.data.output["rgb"]   →   segment_image()   →   image_features_from_seg()

    Feature vector (4 values):
        [0] wall_ratio        — fraction of blue-wall pixels in the full image
        [1] left_wall_ratio   — wall fraction in the LEFT half  (steer right if high)
        [2] right_wall_ratio  — wall fraction in the RIGHT half (steer left if high)
        [3] floor_center_x    — horizontal centroid of yellow floor [0=left, 0.5=center, 1=right]

    Returns zeros on the very first step if the camera has not yet produced a frame,
    or if the camera prim path was not found in the scene.

    Returns shape: [num_envs, 4]
    """
    camera = env.scene[camera_cfg.name]
    rgb = camera.data.output.get("rgb")  # None if camera not ready yet

    # Guard: return neutral zeros when camera data is unavailable.
    # floor_center_x = 0.5 means "floor is centered" — safe default behaviour.
    if rgb is None or rgb.numel() == 0:
        zeros = torch.zeros(env.num_envs, 4, device=env.device)
        zeros[:, 3] = 0.5  # floor_center_x neutral = corridor centre
        return zeros

    seg_mask = segment_image(rgb)                  # [N, H, W]
    features = image_features_from_seg(seg_mask)  # [N, 4]
    return features
