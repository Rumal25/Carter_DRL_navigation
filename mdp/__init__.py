# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .observations import (
    camera_image_features,
    carter_base_ang_vel_z,
    carter_base_lin_vel,
    carter_pose_command,
)

from .rewards import (
    goal_reached_bonus,
    exponential_distance_reward,
    step_penalty,
    heading_alignment_reward,
    wall_collision_penalty,
    # wall_approach_penalty,
    non_success_termination,
    backward_velocity_penalty,
    upright_penalty,
)

from .terminations import (
    goal_reached,
    robot_fell_over,
)

from .camera_utils import image_features_from_seg, segment_image
