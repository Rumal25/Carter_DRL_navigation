# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for Phase 2 camera-based navigation.

Phase 1 (unchanged):
    goal_reached              — success: robot within 0.5m of goal

Phase 2 (new):
    wall_collision_termination — failure: chassis contact force exceeds threshold
"""

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
    """Terminate (success) when the robot is within threshold metres of the goal.

    Upon success, Isaac Lab resets the episode and the command manager
    resamples a new goal automatically.

    Returns shape: [num_envs]  (bool tensor)
    """
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)
    return distance < threshold


def wall_collision_termination(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 5.0,
) -> torch.Tensor:
    """Terminate (failure) when the chassis detects a horizontal wall-contact force.

    IMPORTANT — only the HORIZONTAL (X, Y) force components are used.
    This prevents false terminations from:
      - Robot landing on the floor after episode reset (large vertical / Z force)
      - Normal driving vibration on the floor surface

    Wall collisions produce force in X and Y (horizontal push against a vertical wall).
    Floor contact from landing produces force in Z only — correctly ignored here.

    Use the SAME threshold as wall_collision_penalty in RewardsCfg.

    Returns shape: [num_envs]  (bool tensor)
    """
    contact_sensor = env.scene[sensor_cfg.name]
    # net_forces_w_history: [N, history_length, num_bodies, 3]
    net_forces = contact_sensor.data.net_forces_w_history
    N = net_forces.shape[0]
    forces_flat = net_forces.view(N, -1, 3)              # [N, history*bodies, 3]
    # Only X and Y: wall contact is horizontal; landing/floor is vertical (Z)
    h_magnitude = torch.norm(forces_flat[..., :2], dim=-1)  # [N, history*bodies]
    max_h_force = h_magnitude.max(dim=-1).values             # [N]
    return max_h_force > threshold


def robot_fell_over(
    env: ManagerBasedRLEnv,
    max_tilt_deg: float = 45.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot tilts more than max_tilt_deg from upright.

    Carter is a differential-drive rover — it has no active balance control.
    Once tilted past ~45° it cannot recover and the episode is wasted.

    Method: projects the robot's local +Z axis onto world +Z.
        z_up = 1.0  → perfectly upright
        z_up = 0.71 → tilted 45°  (default threshold)
        z_up = 0.0  → flat on its side

    Derivation from unit quaternion (w, x, y, z):
        robot_Z_in_world_Z = 1 - 2*(x² + y²)

    Returns shape: [num_envs]  (bool tensor)
    """
    asset = env.scene[asset_cfg.name]
    quat_w = asset.data.root_quat_w   # [N, 4]  (w, x, y, z)
    x = quat_w[:, 1]
    y = quat_w[:, 2]
    z_up = 1.0 - 2.0 * (x * x + y * y)                 # [N]
    cos_threshold = math.cos(max_tilt_deg * math.pi / 180.0)
    return z_up < cos_threshold
