# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, TiledCameraCfg
from isaaclab.sim import SimulationCfg, UrdfConverterCfg
from isaaclab.utils import configclass

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
        )
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
            stiffness=200.0,   # position-controlled: needs spring stiffness
            damping=10.0,
        ),
        "hip_steppers": ImplicitActuatorCfg(
            joint_names_expr=["Revolute3", "Revolute4"],
            effort_limit=3,
            velocity_limit=6.2,
            stiffness=200.0,
            damping=10.0,
        ),
        "knee_steppers": ImplicitActuatorCfg(
            joint_names_expr=["Revolute5", "Revolute6"],
            effort_limit=6,
            velocity_limit=6.2,
            stiffness=200.0,
            damping=10.0,
        ),
    },
)

@configclass
class RocketSceneCfg(InteractiveSceneCfg):
    imu: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Torso_1",
        update_period=0.0,  # every step
        history_length=1,
        debug_vis=True,
        # Torso mesh bottom is at Z=0 in link frame (center XY=0,0).
        # Place IMU 1 inch (0.0254 m) above bottom, centered in XY.
        offset=ImuCfg.OffsetCfg(pos=(0.0, 0.0, 0.0254)),
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
        history_length=1,
        track_air_time=False,
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
    joint_pos_std: float = 0.01   # rad  — encoder quantization + flex
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
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.85, 1.15),  # ±15% of URDF mass per body
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
            "stiffness_distribution_params": (0.75, 1.25),  # ±25%
            "damping_distribution_params": (0.75, 1.25),
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
            "pose_range": {},     # no randomization, pure reset to URDF init_state + env origin
            "velocity_range": {},
        },
    )

    # Joint position noise prevents the policy from overfitting to a single start pose.
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),  # ±0.1 rad (~6°) around default
            "velocity_range": (0.0, 0.0),
        },
    )

    # -------------------------------------------------------------------------
    # MEDIUM IMPACT — interval (applied periodically during an episode)
    # -------------------------------------------------------------------------

    # Disabled: all Isaac Lab biped configs (Cassie, H1, G1) disable push randomization —
    # it destabilizes training before a stable gait is learned. Re-enable at ±0.1 m/s
    # once the policy can stand/walk reliably.
    push_robot = None
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(3.0, 6.0),
    #     params={"velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}},  # scaled down for small biped
    # )

    # -------------------------------------------------------------------------
    # MEDIUM IMPACT — startup (manufacturing shifts center of mass)
    # -------------------------------------------------------------------------

    # COM offsets model asymmetric mass distribution from wiring, fasteners, and tolerances.
    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},  # ±1 cm
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
            "static_friction_range": (0.3, 1.0),   # tile (0.3) → carpet (1.0), ref: Spot config
            "dynamic_friction_range": (0.2, 0.8),  # always lower than static
            "restitution_range": (0.0, 0.0),        # real floors don't bounce
            "num_buckets": 64,
        },
    )


@configclass
class RocketEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0

    # spaces definition - UPDATED for 6 DOF robot
    action_space = 6  # 6 joints to control
    observation_space = 20  # 6 joint pos + 4 stepper joint vel + 3 ang_vel + 3 lin_acc + 4 quat
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
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

    # reward scales - defaults match standing policy
    rew_scale_alive: float = 5.0
    rew_scale_terminated: float = -5.0
    rew_scale_upright: float = 2.0
    rew_scale_joint_vel: float = -0.05
    rew_scale_torque: float = -0.1
    rew_scale_lin_vel: float = -0.05
    rew_scale_lat_vel: float = -0.05
    rew_scale_vertical_vel: float = -0.5  # penalize vertical bouncing (standing only)
    rew_scale_target_standing_pose: float = 1.0
    rew_scale_height: float = 1.0
    rew_scale_toe_walking: float = 1.0  # reward toe ground contact, penalize calf ground contact
    rew_scale_action_rate: float = -0.1   # penalize rapid action changes to reduce jitter
    rew_scale_jerk: float = -0.05         # penalize second-order action changes (discrete jerk)
    rew_scale_alternating_contact: float = 0.0  # reward alternating toe contact (walking only)

    # reward scale presets (applied at env init based on policy_type)
    standing_reward_scales = {
        "rew_scale_alive":                5.0,
        "rew_scale_terminated":           -5.0,
        "rew_scale_upright":              3.0,
        "rew_scale_joint_vel":           -1.5,
        "rew_scale_torque":              -0.0,
        "rew_scale_lin_vel":             -1.0,
        "rew_scale_lat_vel":             -0.0,
        "rew_scale_target_standing_pose": 2.0,
        "rew_scale_height":               0.0,
        "rew_scale_toe_walking":          3.0,
        "rew_scale_action_rate":         -0.1,
        "rew_scale_vertical_vel":        -1.0,
        "rew_scale_jerk":                -0.1,
        "rew_scale_alternating_contact":  0.0,
    }

    walking_reward_scales = {
        "rew_scale_alive":                5.0,
        "rew_scale_terminated":          -5.0,
        "rew_scale_upright":              2.0,
        "rew_scale_joint_vel":            0.0,
        "rew_scale_torque":               0.0,
        "rew_scale_lin_vel":              4.0,   # positive = reward forward x-velocity
        "rew_scale_lat_vel":             -0.05,  # penalize lateral drift
        "rew_scale_target_standing_pose": 0.0,   # light posture encouragement
        "rew_scale_height":               2.0,   # stay off the ground
        "rew_scale_toe_walking":          2.0,
        "rew_scale_action_rate":         -0.1,
        "rew_scale_vertical_vel":         0.0,  # not penalized during walking
        "rew_scale_jerk":                -0.05,
        "rew_scale_alternating_contact":  1.0,  # reward alternating gait
    }

    # additional conditions
    target_height = 0.14 # at about 0.12 m, the robot is sitting

    # domain randomization ranges (applied per-env on reset)
    dr_stiffness_range: tuple = (0.8, 1.2)    # ±20% stiffness multiplier
    dr_damping_range: tuple = (0.8, 1.2)      # ±20% damping multiplier
    dr_joint_friction_range: tuple = (0.0, 0.05)  # joint friction coefficient
    dr_imu_noise_std: float = 0.01            # gaussian noise on IMU ang_vel and lin_acc

    # termination conditions
    max_tilt_distance = 0.50  # max tilt before termination
    min_height: float = 0.10  # min height before termination [m] (currently unused)
