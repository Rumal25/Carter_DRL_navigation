# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for Phase 2 camera-based navigation.

Reward design — Morioka Lab paper style (updated 2026-04-29)
-------------------------------------------------------------
The original tanh position-tracking rewards caused the robot to rush directly
toward the goal and crash through walls (strong continuous gradient overcame
the weak wall penalty).  The paper system replaces them with:

  goal_reached_bonus        +100   sparse   robot within 0.5 m of goal
  distance_closing_reward   +0.01  dense    moving toward goal this step
  step_penalty              -0.01  dense    every step (time-efficiency pressure)
  heading_penalty_90        -0.1   dense    facing >90° from goal
  heading_penalty_150       -5.0   dense    facing >150° from goal (almost reversed)
  wall_collision_penalty    -50.0  dense    chassis contacts wall (H-force > 5 N)
  on_floor_reward           +0.05  dense    camera sees >60% yellow floor
  upright_penalty           -1.0   dense    robot tilting (physics stability)
  termination_penalty       -50.0  sparse   non-timeout episode end

Key property: wall collision (-50/step) is 5000× stronger than forward progress
(+0.01/step), so wall shortcuts are never worth it for the policy.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .camera_utils import segment_image

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Navigation rewards — paper system (Morioka Lab)
# ---------------------------------------------------------------------------

def goal_reached_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    bonus: float,
    command_name: str,
) -> torch.Tensor:
    """Sparse bonus when robot is within threshold metres of the goal.

    Paper value: bonus=100.0.  Returns 0.0 or bonus each step.

    Returns shape: [num_envs]
    """
    command = env.command_manager.get_command(command_name)
    distance = torch.norm(command[:, :2], dim=1)
    return (distance < threshold).float() * bonus


def distance_closing_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving toward the goal this step.

    Returns 1.0 if the robot's velocity has a positive component along the
    goal direction (body frame), else 0.0.  Use with weight=+0.01.

    This approximates 'closes the distance to goal' from the paper without
    needing a previous-distance buffer.  The projection onto the goal unit
    vector ensures only genuine approach motion is rewarded (not sideways drift
    that happens to reduce distance slightly).

    Returns shape: [num_envs]
    """
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]                                         # [N, 2]
    dist = torch.norm(des_pos_b, dim=1, keepdim=True).clamp(min=1e-6)
    goal_dir_b = des_pos_b / dist                                      # [N, 2] unit vector to goal

    asset = env.scene[asset_cfg.name]
    vel_b = asset.data.root_lin_vel_b[:, :2]                           # [N, 2] body-frame velocity

    vel_toward_goal = (vel_b * goal_dir_b).sum(dim=1)                  # [N]
    return (vel_toward_goal > 0.0).float()


def step_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant 1.0 every step.  Use with weight=-0.01.

    Provides a small time-efficiency pressure: the robot loses 0.01 per step
    it does not reach the goal.  Over a 30-second episode at 15 Hz this costs
    at most 450 × 0.01 = 4.5 — tiny compared to goal bonus (+100) or wall
    penalty (-50), so it only breaks ties between equally-rewarded paths.

    Returns shape: [num_envs]
    """
    return torch.ones(env.num_envs, device=env.device)


def heading_penalty_90(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Penalty when robot faces more than 90° away from the goal.

    Returns 1.0 if |heading_b| > π/2, else 0.0.  Use with weight=-0.1.

    heading_b = 0   → facing goal directly → no penalty
    heading_b = π/2 → sideways to goal    → penalty starts
    heading_b = π   → facing away         → penalty + heading_penalty_150 fires

    Combined effect with heading_penalty_150:
        0° – 90°:   no heading cost
        90° – 150°: -0.1 per step
        150° – 180°: -0.1 - 5.0 = -5.1 per step

    Returns shape: [num_envs]
    """
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return (heading_b.abs() > math.pi / 2).float()


def heading_penalty_150(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Strong penalty when robot faces more than 150° away from the goal.

    Returns 1.0 if |heading_b| > 5π/6 (150°), else 0.0.  Use with weight=-5.0.

    Fires only in the narrow 30° window where the robot is nearly reversed.
    This strongly discourages the policy from learning to go backward.

    Returns shape: [num_envs]
    """
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return (heading_b.abs() > 5.0 * math.pi / 6.0).float()


# ---------------------------------------------------------------------------
# Wall avoidance + visual safety
# ---------------------------------------------------------------------------

def wall_collision_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 5.0,
) -> torch.Tensor:
    """Per-step penalty when the robot's chassis contacts a wall.

    Only HORIZONTAL (X, Y) force components are checked — prevents false
    penalties from floor-landing impacts (which are purely vertical / Z).

    Paper value: -50.0 per collision.  Since wall_collision_termination ends
    the episode immediately, this fires at most once per episode, giving a
    total wall cost of weight × 1.0 = -50.0, exactly matching the paper.

    Returns shape: [num_envs]  (0.0 = no collision, 1.0 = collision)
    """
    contact_sensor = env.scene[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history   # [N, H, B, 3]
    N = net_forces.shape[0]
    forces_flat = net_forces.view(N, -1, 3)                 # [N, H*B, 3]
    h_magnitude = torch.norm(forces_flat[..., :2], dim=-1)  # [N, H*B]
    max_h_force = h_magnitude.max(dim=-1).values            # [N]
    return (max_h_force > threshold).float()


def on_floor_reward(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    min_floor_ratio: float = 0.60,
) -> torch.Tensor:
    """Small bonus when the camera view shows enough yellow floor.

    Encourages the robot to stay in the corridor center rather than hugging
    walls.  Weight reduced to 0.05 (from 0.1) so it does not compete with
    the heading or distance rewards.

    Returns shape: [num_envs]  (0.0 or 1.0)
    """
    camera = env.scene[camera_cfg.name]
    rgb = camera.data.output["rgb"]
    seg_mask = segment_image(rgb)
    floor_ratio = 1.0 - seg_mask.mean(dim=[1, 2])
    return (floor_ratio >= min_floor_ratio).float()


def upright_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty proportional to how far the robot has tilted from upright.

    Returns 0.0 when perfectly upright, up to 2.0 when completely sideways.
    Provides a dense gradient before robot_fell_over termination fires.
    Use with weight=-1.0.

    Returns shape: [num_envs]
    """
    asset = env.scene[asset_cfg.name]
    quat_w = asset.data.root_quat_w   # [N, 4]  (w, x, y, z)
    x = quat_w[:, 1]
    y = quat_w[:, 2]
    z_up = 1.0 - 2.0 * (x * x + y * y)
    return (1.0 - z_up).clamp(0.0, 2.0)


# ---------------------------------------------------------------------------
# Legacy functions kept for reference — NOT used in current RewardsCfg
# ---------------------------------------------------------------------------

def position_command_error_tanh(env, std, command_name):
    """DEPRECATED — caused wall-rushing. Replaced by distance_closing_reward."""
    command = env.command_manager.get_command(command_name)
    distance = torch.norm(command[:, :2], dim=1)
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env, command_name):
    """DEPRECATED — replaced by heading_penalty_90 + heading_penalty_150."""
    command = env.command_manager.get_command(command_name)
    return command[:, 3].abs()


def forward_velocity_reward(env, asset_cfg=SceneEntityCfg("robot")):
    """DEPRECATED — replaced by distance_closing_reward + step_penalty."""
    asset = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.root_lin_vel_b[:, 0], min=0.0)


def angular_velocity_penalty(env, asset_cfg=SceneEntityCfg("robot")):
    """DEPRECATED — heading penalties now handle turn behavior."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w[:, 2].abs()
