# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 2 manager-based environment — camera-based obstacle avoidance in colored corridors.

Architecture
------------
This environment builds on carter_manager (Phase 1) and adds:
  1. Stereo camera sensor (84×84 RGB) mounted on Carter's chassis
  2. Contact sensor on Carter's chassis for wall collision detection
  3. Custom USD corridor environment (T / L / U shape — yellow floor, blue walls)
  4. Color-segmentation image features in the observation (no deep-learning needed in sim)
  5. Wall-collision penalty and termination rewards
  6. Fixed goal and spawn positions (corridor entrance → dead-end)

Observation space: 10
    [v_fwd, v_lat, yaw_rate]          ← robot velocity (3)
    [x_b, y_b, heading_b]             ← goal in body frame (3)
    [wall_ratio, left_wall,
     right_wall, floor_center_x]      ← camera image features (4)

Action space: 2  (left_wheel_vel, right_wheel_vel), scale ×5

IMPORTANT — before running, set the USD paths and corridor coordinates below.
Open each USD in Isaac Sim's Stage panel to read the correct entry/goal XY positions.
"""

from __future__ import annotations

import math
import os
import platform

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
# USER CONFIGURATION — adjust these before running
# ---------------------------------------------------------------------------

# Automatically find USD files depending on which OS is running.
#
# WINDOWS (your development PC):
#   USD files are read directly from the vault's LLM-wiki/ folder.
#
# LINUX (your IsaacLab training machine):
#   USD files are expected at: /home/rl-ws/myws/Catrter_usd20260424/
#   If you move them, update _USD_DIR in the else-branch below.

if platform.system() == "Windows":
    _USD_DIR = os.path.join(
        r"C:\Users\edu2025\Desktop\Rumal Personal\Second Brain\DRL-Research-Vault",
        "LLM-wiki",
    )
    _L_SHAPE = "Carter _L_shape.usd"   # original filename on Windows (has space)
else:
    # Linux training machine — USD files live here directly
    _USD_DIR = "/home/rl-ws/myws/Catrter_usd20260424"
    _L_SHAPE = "Carter_L_shape.usd"    # renamed on Linux (no space)

# USD file paths for each corridor shape
USD_PATHS = {
    "T": os.path.join(_USD_DIR, "Carter_T_shape.usd"),
    "L": os.path.join(_USD_DIR, _L_SHAPE),
    "U": os.path.join(_USD_DIR, "Carter_u_shape.usd"),
}

# Which shape to train on. Change to "L" or "U" to switch corridors.
ACTIVE_SHAPE = "T"

# Validate at import time so the user gets a clear message instead of a silent empty scene.
if not os.path.exists(USD_PATHS[ACTIVE_SHAPE]):
    print("\n" + "=" * 70)
    print(f"[carter_camera_nav] USD FILE NOT FOUND: {USD_PATHS[ACTIVE_SHAPE]}")
    print(f"  On Linux — copy the USD files to: {_USD_DIR}/")
    print("  Commands:")
    print(f"    mkdir -p {_USD_DIR}")
    print(f"    cp /path/to/LLM-wiki/Carter_T_shape.usd   {_USD_DIR}/")
    print(f"    cp /path/to/LLM-wiki/'Carter _L_shape.usd' {_USD_DIR}/")
    print(f"    cp /path/to/LLM-wiki/Carter_u_shape.usd   {_USD_DIR}/")
    print("  Training will continue with NO corridor (flat ground only).")
    print("=" * 70 + "\n")

# Robot spawn position (corridor entrance) relative to the USD environment origin.
# VERIFY these by loading the USD in Isaac Sim and checking XY coordinates of the entrance.
ENTRANCE_X = 0.0        # metres — X position at corridor entrance (verified from USD)
ENTRANCE_Y = 0.0        # metres — Y position at corridor entrance (verified from USD)
ENTRANCE_YAW = math.pi  # radians — π = facing -Y (into the corridor toward goal at lower Y)

# Z height to spawn the robot above the corridor floor so physics drops it correctly.
# The T-shape floor sits above world Z=0 — spawning at Z=0.05 puts the robot UNDER the floor.
# Set this to slightly above the corridor floor height you see in Isaac Sim.
# HOW TO FIND IT: open the USD, click the floor at the entrance, read the Z from Transform.
# If unsure, keep 1.0 — the robot will free-fall onto whatever floor is below it.
ENTRANCE_Z = 0.85       # metres — Z spawn height (drops onto corridor floor)

# ── FIX 1 ──────────────────────────────────────────────────────────────────
# ORIGINAL (invalid): GOAL_X = -2.5 to 2.5
#   "to" is not a Python keyword; a scalar variable cannot hold a range.
#
# FIXED: split into GOAL_X_MIN / GOAL_X_MAX so CommandsCfg can use them as
#   a proper (min, max) tuple in UniformPose2dCommandCfg.Ranges.pos_x.
GOAL_X_MIN = -2.5       # metres — minimum X position of goal (left edge of corridor)
GOAL_X_MAX =  2.5       # metres — maximum X position of goal (right edge of corridor)
# ───────────────────────────────────────────────────────────────────────────

GOAL_Y = 2.0            # metres — Y position of goal dead-end (fixed, verified from USD)

# Small randomization around entrance/goal for robustness (set 0.0 for fully fixed)
ENTRANCE_RAND_XY  = 0.25           # ± metres of position noise at reset
ENTRANCE_RAND_YAW = math.pi / 8    # ± radians of heading noise at reset
GOAL_RAND_XY      = 0.05           # ± metres of goal Y noise (X is already a full range)


# ---------------------------------------------------------------------------
# Carter V1 Robot Configuration (Phase 2 — contact sensors enabled)
# ---------------------------------------------------------------------------

CARTER_V1_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=["left_wheel", "right_wheel"],
    effort_limit_sim=100.0,
    stiffness=0.0,
    damping=50.0,
)

# Same as Phase 1 but with activate_contact_sensors=True for wall detection
CARTER_V1_CAMERA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Carter/carter_v1.usd",
        activate_contact_sensors=True,   # required for ContactSensorCfg
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            linear_damping=0.5,           # increased from 0.1 — damps linear oscillation on landing
            angular_damping=1.0,          # increased from 0.1 — critical: damps roll/pitch from floor impact
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,  # lowered from 1000 — prevents large impulses when meshes overlap
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
        pos=(0.0, 0.0, ENTRANCE_Z),  # Z uses ENTRANCE_Z so robot drops onto corridor floor
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "left_wheel": 0.0,
            "right_wheel": 0.0,
            "rear_pivot": 0.0,
            "rear_axle": 0.0,
        },
        joint_vel={
            "left_wheel": 0.0,
            "right_wheel": 0.0,
            "rear_pivot": 0.0,
            "rear_axle": 0.0,
        },
    ),
    actuators={"wheels": CARTER_V1_ACTUATOR_CFG},
)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class CarterCameraSceneCfg(InteractiveSceneCfg):
    """Scene: colored corridor USD + Carter robot + camera sensor + contact sensor."""

    # Safety ground plane — placed 1 m BELOW world Z=0 so it never interferes with
    # the corridor floor (which sits near Z=0).  It only catches the robot if it
    # somehow falls through all corridor geometry (acts as a last-resort safety net).
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.0)),
    )

    # Colored corridor environment (T / L / U shape, yellow floor + blue walls).
    # Only spawned when the USD file exists; skipped silently otherwise.
    environment = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/environment",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATHS[ACTIVE_SHAPE],
        ),
    ) if os.path.exists(USD_PATHS[ACTIVE_SHAPE]) else None

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )

    robot: ArticulationCfg = CARTER_V1_CAMERA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # RGB camera at Carter's stereo camera mount.
    # Prim path confirmed: /World/carter_v1/chassis_link/camera_mount
    # A new camera sensor is spawned at this location (84×84 resolution).
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis_link/camera_mount/CameraRGB",
        update_period=0.0,   # update every simulation step
        height=84,
        width=84,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
        ),
    )

    # Contact sensor on Carter's chassis to detect wall collisions.
    # Returns net contact force (N) at each step.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis_link",
        update_period=0.0,
        history_length=3,    # keep 3 steps of history to catch brief contacts
        track_air_time=False,
    )


# ---------------------------------------------------------------------------
# MDP — Actions
# ---------------------------------------------------------------------------

@configclass
class ActionsCfg:
    """Same as Phase 1: joint velocity control for left/right drive wheels."""

    # ── FIX 3 ──────────────────────────────────────────────────────────────
    # ORIGINAL comment said "# restored to 10.0" but scale value was 5.0.
    # Comment was stale/misleading. Value 5.0 is kept; comment corrected.
    wheel_velocities = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_wheel", "right_wheel"],
        scale=5.0,         # action scale ×5 — change to 10.0 for faster robot motion
        use_default_offset=True,
    )
    # ───────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# MDP — Observations
# ---------------------------------------------------------------------------

@configclass
class ObservationsCfg:
    """10-dimensional observation: velocity + goal (body frame) + camera features.

    Layout:
        [v_fwd, v_lat]            indices 0-1  body-frame velocity
        [yaw_rate]                index  2     angular velocity
        [x_b, y_b, heading_b]    indices 3-5  goal in body frame
        [wall_ratio,              index  6     fraction of blue wall pixels
         left_wall_ratio,         index  7     wall ratio in left half of image
         right_wall_ratio,        index  8     wall ratio in right half
         floor_center_x]          index  9     horizontal centroid of yellow floor
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # Velocity observations (body frame) — same as Phase 1
        base_lin_vel = ObsTerm(func=mdp.carter_base_lin_vel)           # [N, 2]
        base_ang_vel = ObsTerm(func=mdp.carter_base_ang_vel_z)         # [N, 1]

        # Goal command (body frame) — same as Phase 1
        pose_command = ObsTerm(
            func=mdp.carter_pose_command,
            params={"command_name": "pose_command"},
        )                                                               # [N, 3]

        # Camera-derived image features — NEW in Phase 2
        image_features = ObsTerm(
            func=mdp.camera_image_features,
            params={"camera_cfg": SceneEntityCfg("camera")},
        )                                                               # [N, 4]

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# MDP — Commands (fixed goal inside corridor)
# ---------------------------------------------------------------------------

@configclass
class CommandsCfg:
    """Goal is sampled uniformly along X (-2.5 → +2.5 m), Y is fixed at 2.0 m.

    GOAL_X_MIN / GOAL_X_MAX define the full X range sampled each episode reset.
    GOAL_Y ± GOAL_RAND_XY gives a near-fixed Y with tiny noise for robustness.

    The command is expressed in the robot body frame (updated every step),
    so the policy naturally generalises to any starting position and heading.
    """

    # ── FIX 1 (continued) ──────────────────────────────────────────────────
    # ORIGINAL: pos_x=(GOAL_X - GOAL_RAND_XY, GOAL_X + GOAL_RAND_XY)
    #   GOAL_X was undefined (syntax error above), so this would crash at import.
    #
    # FIXED: pos_x uses (GOAL_X_MIN, GOAL_X_MAX) = (-2.5, 2.5) directly.
    #   Goal X is now sampled uniformly across the full corridor width each episode.
    #   Goal Y stays fixed at GOAL_Y ± GOAL_RAND_XY (small noise only).
    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(20.0, 20.0),   # matches episode length
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(GOAL_X_MIN, GOAL_X_MAX),                         # random X: -2.5 → +2.5 m
            pos_y=(GOAL_Y - GOAL_RAND_XY, GOAL_Y + GOAL_RAND_XY),  # fixed Y:  1.95 → 2.05 m
            heading=(-math.pi, math.pi),
        ),
    )
    # ───────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# MDP — Events (reset robot at corridor entrance)
# ---------------------------------------------------------------------------

@configclass
class EventCfg:
    """Reset Carter near the corridor entrance each episode.

    The pose_range offsets are added to init_state.pos=(0,0,ENTRANCE_Z), which is
    then shifted by env_origin. So the robot spawns at approximately:
      (ENTRANCE_X ± ENTRANCE_RAND_XY, ENTRANCE_Y ± ENTRANCE_RAND_XY, ENTRANCE_Z) + env_origin

    Small YAW randomization (±ENTRANCE_RAND_YAW rad) prevents the policy from
    memorizing a single starting direction and improves generalization to corridors
    with unknown robot initial orientation (important for physical deployment).

    NOTE: Goal position is NOT set here — it is controlled entirely by CommandsCfg
    (UniformPose2dCommandCfg), which resamples goal XY at every episode reset.
    """

    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (ENTRANCE_X - ENTRANCE_RAND_XY, ENTRANCE_X + ENTRANCE_RAND_XY),
                "y": (ENTRANCE_Y - ENTRANCE_RAND_XY, ENTRANCE_Y + ENTRANCE_RAND_XY),
                "yaw": (ENTRANCE_YAW - ENTRANCE_RAND_YAW, ENTRANCE_YAW + ENTRANCE_RAND_YAW),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # ── FIX 2 ──────────────────────────────────────────────────────────────
    # ORIGINAL (invalid): goal=pose_command=EventTerm(x=-2.5 to 2.5, y=2.0)
    #   Three problems:
    #     (a) "goal=pose_command=" — chained assignment on a class attribute is
    #         invalid in a @configclass body.
    #     (b) EventTerm does not accept x= / y= keyword arguments.
    #     (c) "to" is not valid Python syntax.
    #
    # FIXED: Line removed entirely. Goal randomization is handled by CommandsCfg
    #   (UniformPose2dCommandCfg resamples goal XY at every episode reset automatically).
    #   No EventTerm is needed or correct here.
    # ───────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# MDP — Rewards
# ---------------------------------------------------------------------------

@configclass
class RewardsCfg:
    """Reward system based on Morioka Lab paper (updated 2026-04-29).

    WHY this was changed
    --------------------
    The original tanh position-tracking rewards created a strong continuous
    magnetic pull toward the goal at every step.  The robot learned to rush
    in a straight line and crash through walls because the wall penalty (-5/step)
    was too weak to overcome the tanh gradient (up to +2/step combined).
    After 145k steps the robot wandered slowly without converging.

    The paper system flips this: the only positive signal is weak (+0.01 for
    moving toward goal), and wall contact is catastrophically negative (-50).
    The robot must learn to navigate AROUND walls to collect the +100 goal bonus.

    Term                    Weight  Type    Description
    -------                 ------  ------  -----------
    goal_reached_bonus      +1.0    sparse  +100 within 0.5 m of goal
    distance_closing        +0.01   dense   moving toward goal this step
    step_penalty            -0.01   dense   every step (time efficiency)
    heading_penalty_90      -0.1    dense   |heading| > 90° from goal
    heading_penalty_150     -5.0    dense   |heading| > 150° from goal
    wall_collision_penalty  -50.0   dense   chassis H-force > 5 N
    on_floor_reward         +0.05   dense   camera sees >60% floor
    upright_penalty         -1.0    dense   tilt from upright
    termination_penalty     -50.0   sparse  non-timeout episode end
    """

    # ── Goal ────────────────────────────────────────────────────────────────
    goal_reached = RewTerm(
        func=mdp.goal_reached_bonus,
        weight=1.0,
        params={"threshold": 0.5, "bonus": 100.0, "command_name": "pose_command"},
    )

    # ── Progress ─────────────────────────────────────────────────────────────
    distance_closing = RewTerm(
        func=mdp.distance_closing_reward,
        weight=0.01,
        params={"command_name": "pose_command"},
    )

    step_penalty = RewTerm(func=mdp.step_penalty, weight=-0.01)

    # ── Heading penalties ────────────────────────────────────────────────────
    heading_penalty_90 = RewTerm(
        func=mdp.heading_penalty_90,
        weight=-0.1,
        params={"command_name": "pose_command"},
    )
    heading_penalty_150 = RewTerm(
        func=mdp.heading_penalty_150,
        weight=-5.0,
        params={"command_name": "pose_command"},
    )

    # ── Wall collision ───────────────────────────────────────────────────────
    wall_collision_penalty = RewTerm(
        func=mdp.wall_collision_penalty,
        weight=-50.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "threshold": 5.0,
        },
    )

    # ── Visual guidance ──────────────────────────────────────────────────────
    on_floor_reward = RewTerm(
        func=mdp.on_floor_reward,
        weight=0.05,
        params={
            "camera_cfg": SceneEntityCfg("camera"),
            "min_floor_ratio": 0.60,
        },
    )

    # ── Stability ─────────────────────────────────────────────────────────────
    upright_penalty = RewTerm(func=mdp.upright_penalty, weight=-1.0)

    # ── Termination ──────────────────────────────────────────────────────────
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)


# ---------------------------------------------------------------------------
# MDP — Terminations
# ---------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    """Episode ends on timeout, goal reached, OR wall collision."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={"threshold": 0.5, "command_name": "pose_command"},
    )

    wall_collision = DoneTerm(
        func=mdp.wall_collision_termination,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "threshold": 5.0,
        },
    )

    robot_fell_over = DoneTerm(
        func=mdp.robot_fell_over,
        params={"max_tilt_deg": 45.0},
    )


# ---------------------------------------------------------------------------
# Main Environment Config
# ---------------------------------------------------------------------------

@configclass
class CarterCameraNavEnvCfg(ManagerBasedRLEnvCfg):
    """Full Phase 2 environment config — Carter V1 with camera in colored corridors."""

    scene: CarterCameraSceneCfg = CarterCameraSceneCfg(
        num_envs=64,
        env_spacing=15.0,
    )

    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 1.0 / 60.0
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0


@configclass
class CarterCameraNavEnvCfg_PLAY(CarterCameraNavEnvCfg):
    """Reduced-size evaluation variant for visualizing a trained policy."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 15.0
        self.observations.policy.enable_corruption = False