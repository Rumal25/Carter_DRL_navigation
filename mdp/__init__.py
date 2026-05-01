# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP components for Phase 2 camera-based navigation environment.

Merges built-in Isaac Lab MDP functions with Phase 2 custom functions
into a single `mdp` namespace. The env config only needs `from . import mdp`.
"""

# -- Built-in Isaac Lab MDP functions (time_out, is_terminated, reset_root_state_uniform, etc.)
from isaaclab.envs.mdp import *  # noqa: F401, F403

# -- Phase 1 + Phase 2 observation functions
from .observations import (
    camera_image_features,
    carter_base_ang_vel_z,
    carter_base_lin_vel,
    carter_pose_command,
)

# -- Reward functions (paper system + stability)
from .rewards import (
    # Paper system — active
    goal_reached_bonus,
    distance_closing_reward,
    step_penalty,
    heading_penalty_90,
    heading_penalty_150,
    wall_collision_penalty,
    on_floor_reward,
    upright_penalty,
    # Legacy — kept for reference, not in RewardsCfg
    position_command_error_tanh,
    heading_command_error_abs,
    forward_velocity_reward,
    angular_velocity_penalty,
)

# -- Phase 1 + Phase 2 termination functions
from .terminations import goal_reached, robot_fell_over, wall_collision_termination

# -- Camera utilities (available for inspection / debugging)
from .camera_utils import image_features_from_seg, segment_image
