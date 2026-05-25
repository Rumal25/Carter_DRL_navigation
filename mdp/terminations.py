# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_reached(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    distance = torch.norm(command[:, :2], dim=1)
    return distance < threshold


def robot_fell_over(
    env: ManagerBasedRLEnv,
    max_tilt_deg: float = 45.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when robot tilts more than max_tilt_deg from upright."""
    asset = env.scene[asset_cfg.name]
    quat_w = asset.data.root_quat_w
    x = quat_w[:, 1]
    y = quat_w[:, 2]
    z_up = 1.0 - 2.0 * (x * x + y * y)
    cos_threshold = math.cos(max_tilt_deg * math.pi / 180.0)
    return z_up < cos_threshold
