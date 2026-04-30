# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from rocket.actuators import StepperActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, TiledCameraCfg
from isaaclab.sim import SimulationCfg, UrdfConverterCfg
from isaaclab.utils import configclass

from .reward_cfg import RewardCfg

##
# Rocket Robot Configuration
##

ROCKET_PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent.parent
URDF_PATH = os.path.join(ROCKET_PACKAGE_DIR, "data", "Rocket.urdf")

ROCKET_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
        fix_base=False,
        activate_contact_sensors=True,
        merge_fixed_joints=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.16),  # Initial position (spawn at 0.3m height)
        joint_pos={"Revolute.*": 0.0},  # All joints start at 0
    ),
    actuators={
        "servos": ImplicitActuatorCfg(
            joint_names_expr=["Revolute1", "Revolute2"],
            effort_limit=0.25,
            velocity_limit=5.2,
            stiffness=120.0,   # position-controlled: needs spring stiffness
            damping=10.0,
        ),
        "hip_steppers": StepperActuatorCfg(
            joint_names_expr=["Revolute3", "Revolute4"],
            effort_limit=3.0,
            velocity_limit=6.2,
            stiffness=200.0,
            damping=10.0,
            velocity_gain=10.0,
            position_deadband=0.005,
        ),
        "knee_steppers": StepperActuatorCfg(
            joint_names_expr=["Revolute5", "Revolute6"],
            effort_limit=5.0,
            velocity_limit=6.2,
            stiffness=200.0,
            damping=10.0,
            velocity_gain=10.0,
            position_deadband=0.005,
        ),
    },
)

@configclass
class RocketSceneCfg(InteractiveSceneCfg):
    imu: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Torso_1",
        update_period=0.01,  # 100 Hz
        history_length=1,
        debug_vis=True,
        # Torso mesh bottom is at Z=0 in link frame (center XY=0,0).
        # Place IMU 3 inches (0.0762 m) above bottom, centered in XY.
        offset=ImuCfg.OffsetCfg(pos=(0.0, 0.0, 0.05)),
    )

    contact_sensor_calves: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Calf_.*_1",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
    )

    contact_sensor_toes: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Toe_.*_1",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
    )

    # Camera for video recording - 45° angle view
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(2.0, 0.0, 1.5),  # 2m back, 1.5m up
            rot=(0.9239, 0.0, 0.3827, 0.0),  # ~45° pitch down
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,      # shorter = wider FOV
            focus_distance=1.0,     # focus on rocket ~1.0m away
            horizontal_aperture=20.0,
            clipping_range=(0.1, 20.0),
        ),
        width=300,
        height=300,
    )

@configclass
class ObsNoiseCfg:
    """Per-channel observation noise standard deviations (Gaussian, zero-mean).

    Set any value to 0.0 to disable noise for that channel.
    Realistic IMU noise: lin_acc >> ang_vel > quat (filtered). Encoders are relatively clean.
    """
    joint_pos_std: float = 0.02   # rad  — encoder quantization + flex
    joint_vel_std: float = 0.05   # rad/s — numerical differentiation amplifies noise
    ang_vel_std:   float = 0.05   # rad/s — MEMS gyro
    lin_acc_std:   float = 0.10   # m/s²  — MEMS accelerometer (noisiest sensor)
    quat_std:      float = 0.005  # unitless — orientation filter smooths this out


@configclass
class EventCfg:
    """Domain randomization events, ordered from most to least impactful for sim-to-real transfer."""

    # -------------------------------------------------------------------------
    # HIGH IMPACT — startup (applied once at env creation; CPU-only PhysX writes)
    # -------------------------------------------------------------------------

    # Mass variation accounts for battery charge state, payload, and manufacturing tolerances.
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (1.0, 1.05),  # skewed high: parts likely heavier than URDF model
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # Stepper motor gains vary significantly unit-to-unit and with temperature.
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.90, 1.10),  # ±10% (was ±25% — too aggressive for early training)
            "damping_distribution_params": (0.90, 1.10),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -------------------------------------------------------------------------
    # HIGH IMPACT — reset (applied every episode; encourages diverse initial states)
    # -------------------------------------------------------------------------

    # Resets root pose to default (applies env grid origins) and zeroes velocity.
    # Required — without this the robot body stays wherever it fell after termination.
    reset_root_state = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "roll":  (-0.0873, 0.0873),  # ±5°
                "pitch": (-0.0873, 0.0873),  # ±5°
            },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "ang_z": (-0.0, 0.0),
            },
        },
    )

    # Joint position noise prevents the policy from overfitting to a single start pose.
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),  # ±0.05 rad (~3°) — slight nudge
            "velocity_range": (0.0, 0.0),
        },
    )

    # -------------------------------------------------------------------------
    # MEDIUM IMPACT — interval (applied periodically during an episode)
    # -------------------------------------------------------------------------

    # ±0.05 m/s derived from H1 (±0.5 m/s, 1.8m tall) scaled by height ratio to Rocket (0.16m):
    # 0.5 × (0.16/1.8) ≈ 0.044 m/s → rounded to 0.05. Same angular disturbance, right robot scale.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={"velocity_range": {"y": (-0.01, 0.01)}},
    )

    # -------------------------------------------------------------------------
    # MEDIUM IMPACT — startup (manufacturing shifts center of mass)
    # -------------------------------------------------------------------------

    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "com_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.005, 0.005)},  # ±5 mm
        },
    )

    # -------------------------------------------------------------------------
    # LOW IMPACT — startup (floor surface friction varies by environment)
    # -------------------------------------------------------------------------

    # Toe friction affects slip behavior; real floors range from carpet to tile.
    randomize_foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Toe_.*"),
            "static_friction_range": (1.0, 1.0),   # moderate (0.5) → carpet (1.0); 0.3 too slippery for standing
            "dynamic_friction_range": (1.0, 1.0),  # always lower than static
            "restitution_range": (0.0, 0.0),        # real floors don't bounce
            "num_buckets": 64,
        },
    )


@configclass
class RocketEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 20
    episode_length_s = 10.0

    # set to False to disable all startup domain randomization (mass, gains, COM, friction).
    # reset_root_state and reset_joints are always active regardless of this flag.
    # recommended workflow: train without DR first, then enable once policy is stable.
    enable_domain_randomization: bool = True

    # spaces definition - UPDATED for 6 DOF robot
    action_space = 6  # 6 joints to control
    observation_space = 18  # 4 stepper joint pos + 4 stepper joint vel + 3 ang_vel + 3 lin_acc + 4 quat
                            # hip servo pos (R1/R2) excluded — no encoders on those joints
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # robot - UPDATED to use ROCKET_CFG
    robot_cfg: ArticulationCfg = ROCKET_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: RocketSceneCfg = RocketSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # domain randomization
    events: EventCfg = EventCfg()

    # per-channel observation noise (applied in _get_observations)
    obs_noise: ObsNoiseCfg = ObsNoiseCfg()

    # robot joint names (from URDF)
    servo_joint_names = ["Revolute1", "Revolute2"]        # hip yaw (position-controlled servos)
    hip_joint_names = ["Revolute3", "Revolute4"]          # hip roll (velocity-controlled steppers)
    knee_joint_names = ["Revolute5", "Revolute6"]         # knee (velocity-controlled steppers)
    stepper_joint_names = hip_joint_names + knee_joint_names  # hip roll + knee (steppers)

    # target standing pose (for reward calculation)
    # initialize this at runtime so it can be set to the joint pos limits taken from robot.data
    target_standing_pose = ()

    # policy type: "standing" or "walking"
    # determines which reward function and scales are used
    policy_type: str = "standing"

    # active reward config — set by policy_type at env init, or override directly
    rewards: RewardCfg = RewardCfg()  # overwritten at env init from POLICIES[policy_type]

    # termination conditions
    max_tilt_distance = 0.5   # max tilt before termination
    max_airborne_steps = 2    # consecutive steps with both feet off ground before termination
