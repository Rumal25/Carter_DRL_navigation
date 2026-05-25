# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .camera_utils import segment_image

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_reached_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    bonus: float,
    command_name: str,
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    distance = torch.norm(command[:, :2], dim=1)
    return (distance < threshold).float() * bonus


def exponential_distance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    scale: float = 2.0,
) -> torch.Tensor:
    """Dense reward: exp(-distance/scale). Strong pull near goal, nonzero gradient far away."""
    command = env.command_manager.get_command(command_name)
    distance = torch.norm(command[:, :2], dim=1)
    return torch.exp(-distance / scale)


def step_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def heading_alignment_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """cos(heading_error): +1 facing goal, 0 sideways, -1 reversed."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return torch.cos(heading_b)


def wall_collision_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 0.2,
) -> torch.Tensor:
    """Binary penalty when filtered wall contact force exceeds threshold.

    Uses force_matrix_w_history which only counts contacts with the filtered
    wall prims (floor contacts excluded).
    """
    sensor = env.scene[sensor_cfg.name]
    N = env.num_envs
    net = sensor.data.net_forces_w_history          # [N, H, B, 3]
    magnitudes = torch.norm(
        net.view(N, -1, 3), dim=-1
    ).max(dim=-1).values                            # [N]
    return (magnitudes > threshold).float()


# def wall_approach_penalty(
#     env: ManagerBasedRLEnv,
#     camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     wall_threshold: float = 0.35,
#     vel_threshold: float = 0.1,
# ) -> torch.Tensor:
#     """Pre-collision penalty: fires when robot actively moves toward a visible wall.

#     Combines camera wall detection with robot velocity direction. Only fires when
#     BOTH a wall is visible on a given side AND the robot is moving toward it.
#     This gives a training signal BEFORE physical contact so the policy learns
#     to use camera observations to anticipate and avoid obstacles.

#     Lateral walls use wall_threshold; front wall uses 1.5x threshold to avoid
#     penalizing normal forward navigation through the corridor.
#     """
#     camera = env.scene[camera_cfg.name]
#     rgb = camera.data.output.get("rgb")
#     if rgb is None or rgb.numel() == 0:
#         return torch.zeros(env.num_envs, device=env.device)

#     seg_mask = segment_image(rgb)   # [N, H, W]
#     N, H, W = seg_mask.shape
#     W3 = W // 3

#     left_wall   = seg_mask[:, :, :W3].mean(dim=[1, 2])
#     center_wall = seg_mask[:, :, W3:2*W3].mean(dim=[1, 2])
#     right_wall  = seg_mask[:, :, 2*W3:].mean(dim=[1, 2])

#     asset = env.scene[asset_cfg.name]
#     v_fwd = asset.data.root_lin_vel_b[:, 0]   # positive = forward
#     v_lat = asset.data.root_lin_vel_b[:, 1]   # positive = left

#     approaching_front = (center_wall > wall_threshold * 1.5) & (v_fwd > vel_threshold)
#     approaching_left  = (left_wall   > wall_threshold)       & (v_lat > vel_threshold)
#     approaching_right = (right_wall  > wall_threshold)       & (v_lat < -vel_threshold)

#     return (approaching_front | approaching_left | approaching_right).float()


def non_success_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for failure terminations only — does NOT fire on goal-reached or timeout."""
    terminated = env.termination_manager.terminated
    timed_out  = env.termination_manager.time_outs
    # command = env.command_manager.get_command("pose_command")
    # distance = torch.norm(command[:, :2], dim=1)
    # at_goal  = distance < 0.6
    failure  = terminated & ~timed_out 
    return failure.float()


def backward_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_backward_vel: float = 0.05,
) -> torch.Tensor:
    """Penalise backward motion beyond dead-band. Camera must face direction of travel."""
    asset = env.scene[asset_cfg.name]
    v_fwd = asset.data.root_lin_vel_b[:, 0]
    return torch.clamp(-v_fwd - max_backward_vel, min=0.0)


def upright_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty proportional to robot tilt from upright. Returns [0, 2]."""
    asset = env.scene[asset_cfg.name]
    quat_w = asset.data.root_quat_w
    x = quat_w[:, 1]
    y = quat_w[:, 2]
    z_up = 1.0 - 2.0 * (x * x + y * y)
    return (1.0 - z_up).clamp(0.0, 2.0)
