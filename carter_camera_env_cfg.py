# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Carter V1 camera navigation environment — T-shape colored corridor.

Observation space: 10
    [v_fwd, v_lat, yaw_rate]       robot velocity (3)
    [x_b, y_b, heading_b]          goal in body frame (3)
    [wall_ratio, left_wall,
     right_wall, floor_center_x]   camera image features (4)

Action space: 2  (left_wheel_vel, right_wheel_vel), scale x5
"""

from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


# ---------------------------------------------------------------------------
# USD path — Linux training machine
# ---------------------------------------------------------------------------

_T_SHAPE_USD = "/home/rl-ws/myws/Catrter_usd20260424/Carter_T_shape.usd"

if not os.path.exists(_T_SHAPE_USD):
    print(f"\n[carter_camera_nav] USD NOT FOUND: {_T_SHAPE_USD}")
    print("  Copy Carter_T_shape.usd to that path before training.")
    print("  Training will continue on flat ground only.\n")


# ---------------------------------------------------------------------------
# Corridor coordinates
# Open the USD in Isaac Sim and read XY from the Stage Transform panel.
# ---------------------------------------------------------------------------

ENTRANCE_X   = 0.0
ENTRANCE_Y   = -2.5
ENTRANCE_YAW = math.pi   # facing -Y, into the corridor

ENTRANCE_Z   = 0.85      # drops robot onto corridor floor

GOAL_X_MIN = -2.5
GOAL_X_MAX =  2.5
GOAL_Y     =  2.0

ENTRANCE_RAND_XY  = 0.25
ENTRANCE_RAND_YAW = math.pi / 4   # ±45° — forces policy to handle full heading range
GOAL_RAND_XY      = 0.05


# ---------------------------------------------------------------------------
# Robot configuration
# ---------------------------------------------------------------------------

CARTER_V1_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=["left_wheel", "right_wheel"],
    effort_limit_sim=100.0,
    stiffness=0.0,
    damping=50.0,
)

CARTER_V1_CAMERA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Carter/carter_v1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            linear_damping=0.5,
            angular_damping=1.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, ENTRANCE_Z),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={"left_wheel": 0.0, "right_wheel": 0.0, "rear_pivot": 0.0, "rear_axle": 0.0},
        joint_vel={"left_wheel": 0.0, "right_wheel": 0.0, "rear_pivot": 0.0, "rear_axle": 0.0},
    ),
    actuators={"wheels": CARTER_V1_ACTUATOR_CFG},
)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class CarterCameraSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.0)),
    )

    environment = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/environment",
        spawn=sim_utils.UsdFileCfg(usd_path=_T_SHAPE_USD),
    ) if os.path.exists(_T_SHAPE_USD) else None

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )

    robot: ArticulationCfg = CARTER_V1_CAMERA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/chassis_link/camera_mount/CameraRGB",
    #     update_period=0.0,
    #     height=84,
    #     width=84,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955),
    # )
    
    # In CarterCameraSceneCfg — replace your entire camera definition with this

    camera = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/chassis_link/camera_mount/RLCamera",
    update_period=0.0,
    height=84,
    width=84,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        horizontal_aperture=20.955,
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.103, -0.06, 0.3095),
        rot=(0.5, 0.5, -0.5, 0.5),
        convention="ros",
    ),
)

    contact_forces_chassis = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis_link",
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/environment/TShapeCorridor/Walls.*",
            # "{ENV_REGEX_NS}/environment/TShapeCorridor/Walls/WestStem",
            # "{ENV_REGEX_NS}/environment/TShapeCorridor/Walls/EastStem",
            # "{ENV_REGEX_NS}/environment/TShapeCorridor/Walls/North",
            # "{ENV_REGEX_NS}/environment/TShapeCorridor/Walls/WestBar",
            # "{ENV_REGEX_NS}/environment/TShapeCorridor/Walls/EastBar",
            # "{ENV_REGEX_NS}/environment/TShapeCorridor/Walls/JunctionSW",
            # "{ENV_REGEX_NS}/environment/TShapeCorridor/Walls/JunctionSE",
        ],
        update_period=0.0,
        history_length=3,
        track_air_time=False,
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@configclass
class ActionsCfg:
    wheel_velocities = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_wheel", "right_wheel"],
        scale=5.0,
        use_default_offset=True,
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

@configclass
class ObservationsCfg:
    """10-D observation: [v_fwd, v_lat, yaw_rate, x_b, y_b, heading_b, wall_ratio, left_wall, right_wall, floor_cx]"""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel  = ObsTerm(func=mdp.carter_base_lin_vel)
        base_ang_vel  = ObsTerm(func=mdp.carter_base_ang_vel_z)
        pose_command  = ObsTerm(func=mdp.carter_pose_command, params={"command_name": "pose_command"})
        image_features = ObsTerm(func=mdp.camera_image_features, params={"camera_cfg": SceneEntityCfg("camera")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@configclass
class CommandsCfg:
    """Goal sampled uniformly in X [-2.5, +2.5 m], Y fixed at 2.0 m (far end of T corridor)."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(20.0, 20.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(GOAL_X_MIN, GOAL_X_MAX),
            pos_y=(GOAL_Y - GOAL_RAND_XY, GOAL_Y + GOAL_RAND_XY),
            heading=(-math.pi, math.pi),
        ),
    )


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@configclass
class EventCfg:
    """Reset Carter near the corridor entrance each episode."""

    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (ENTRANCE_X - ENTRANCE_RAND_XY, ENTRANCE_X + ENTRANCE_RAND_XY),
                "y": (ENTRANCE_Y - ENTRANCE_RAND_XY, ENTRANCE_Y + ENTRANCE_RAND_XY),
                "yaw": (ENTRANCE_YAW - ENTRANCE_RAND_YAW, ENTRANCE_YAW + ENTRANCE_RAND_YAW),
            },
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
    )


# ---------------------------------------------------------------------------
# Rewards (v3 — 2026-05-21)
# ---------------------------------------------------------------------------

@configclass
class RewardsCfg:
    """
    Term                   Weight   Type    Range / step
    ----                   ------   ----    ------------
    goal_reached           +1.0     sparse  0 or +100
    exponential_distance   +0.5     dense   +0.02 to +0.50
    step_penalty           -0.01    dense   always -0.01
    heading_alignment      +0.1     dense   -0.10 to +0.10
    wall_collision_chassis -50.0    dense   0 or -50
    wall_approach_penalty  -5.0     dense   0 or -5  (pre-collision)
    backward_penalty       -2.0     dense   0 to -2
    upright_penalty        -1.0     dense   0 to -2
    termination_penalty    -50.0    sparse  0 or -50  (fell_over only)
    """

    goal_reached = RewTerm(
        func=mdp.goal_reached_bonus,
        weight=1.0,
        params={"threshold": 0.5, "bonus": 100.0, "command_name": "pose_command"},
    )

    exponential_distance = RewTerm(
        func=mdp.exponential_distance_reward,
        weight=0.5,
        params={"command_name": "pose_command", "scale": 2.0},
    )

    step_penalty = RewTerm(func=mdp.step_penalty, weight=-0.01)

    heading_alignment = RewTerm(
        func=mdp.heading_alignment_reward,
        weight=0.05,
        params={"command_name": "pose_command"},
    )

    wall_collision_chassis = RewTerm(
        func=mdp.wall_collision_penalty,
        weight=-50.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces_chassis"),
            "threshold": 0.2,
        },
    )

    # wall_approach = RewTerm(
    #     func=mdp.wall_approach_penalty,
    #     weight=0.0,
    #     params={
    #         "camera_cfg": SceneEntityCfg("camera"),
    #         "wall_threshold": 0.35,
    #         "vel_threshold": 0.1,
    #     },
    # )

    backward_penalty = RewTerm(
        func=mdp.backward_velocity_penalty,
        weight=-2.0,
        params={"max_backward_vel": 0.05},
    )

    upright_penalty = RewTerm(func=mdp.upright_penalty, weight=-1.0)

    termination_penalty = RewTerm(func=mdp.non_success_termination, weight=-20.0)


# ---------------------------------------------------------------------------
# Terminations (v3 — 2026-05-21)
# ---------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    """Three valid episode endings: timeout, goal reached, fell over.

    Wall collision is handled by reward penalty only (episode continues).
    This allows the robot to learn to recover after light wall contact.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={"threshold": 0.5, "command_name": "pose_command"},
    )

    robot_fell_over = DoneTerm(
        func=mdp.robot_fell_over,
        params={"max_tilt_deg": 45.0},
    )


# ---------------------------------------------------------------------------
# Main environment config
# ---------------------------------------------------------------------------

@configclass
class CarterCameraNavEnvCfg(ManagerBasedRLEnvCfg):

    scene: CarterCameraSceneCfg = CarterCameraSceneCfg(num_envs=64, env_spacing=15.0)

    actions: ActionsCfg       = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg     = CommandsCfg()
    events: EventCfg          = EventCfg()
    rewards: RewardsCfg       = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 1.0 / 60.0
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0


@configclass
class CarterCameraNavEnvCfg_PLAY(CarterCameraNavEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 15.0
        self.observations.policy.enable_corruption = False
