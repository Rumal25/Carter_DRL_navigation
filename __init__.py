# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Carter V1 Phase 2 — camera-based navigation in colored corridor environments.

Pipeline
--------
RGB camera  →  color segmentation  →  image features  →  PPO policy  →  wheel commands

Registered task IDs
-------------------
Isaac-Carter-Camera-Nav-v0         →  training (64 envs — camera rendering cost)
Isaac-Carter-Camera-Nav-Play-v0    →  evaluation (4 envs, no noise)

Usage
-----
Train with SKRL (recommended for Phase 2):
    python scripts/skrl/train.py \\
        --task Isaac-Carter-Camera-Nav-v0

Train with RSL-RL:
    python scripts/rsl_rl/train.py \\
        --task Isaac-Carter-Camera-Nav-v0 \\
        --num_envs 64

Play / evaluate:
    python scripts/rsl_rl/play.py \\
        --task Isaac-Carter-Camera-Nav-Play-v0 \\
        --num_envs 4 \\
        --load_run <run_name>

Train with SB3:
    python scripts/sb3/train.py \\
        --task Isaac-Carter-Camera-Nav-v0
"""

import gymnasium as gym

from . import agents
from .carter_camera_env_cfg import CarterCameraNavEnvCfg, CarterCameraNavEnvCfg_PLAY

gym.register(
    id="Isaac-Carter-Camera-Nav-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CarterCameraNavEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CarterCameraNavPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Carter-Camera-Nav-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CarterCameraNavEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CarterCameraNavPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
