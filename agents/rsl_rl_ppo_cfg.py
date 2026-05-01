# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class CarterCameraNavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO config for Phase 2 camera-based navigation.

    Observation space : 10  (6 state + 4 camera features)
    Action space      : 2   (left_wheel_vel, right_wheel_vel)
    Network           : [256, 128] ELU
                        Larger than Phase 1 [128, 128] because the 10-dim obs
                        includes visual features that benefit from more capacity.

    Training takes longer than Phase 1 because:
      - 64 envs (not 4096) → fewer parallel samples per update
      - Longer episodes (30s vs 10s)
      - More complex task (obstacle avoidance + navigation)
    Expect convergence in 2000–4000 iterations.
    """

    num_steps_per_env = 32         # slightly longer rollout per env for corridor tasks
    max_iterations = 3000          # more iterations needed (fewer parallel envs)
    save_interval = 200
    experiment_name = "carter_camera_nav"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128],   # larger than Phase 1 for visual features
        critic_hidden_dims=[256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,    # lower LR than Phase 1 (1e-3) for stability with visual obs
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
