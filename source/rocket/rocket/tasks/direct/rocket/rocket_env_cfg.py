# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, UrdfConverterCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ImuCfg, TiledCameraCfg, ContactSensorCfg

import torch
import math

##
# Rocket Robot Configuration
##

import os
from pathlib import Path

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
        # device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # robot - UPDATED to use ROCKET_CFG
    robot_cfg: ArticulationCfg = ROCKET_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: RocketSceneCfg = RocketSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

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
    initial_joint_range = [-0.1, 0.1]  # joint angle randomization on reset [rad]
    target_height = 0.14 # at about 0.12 m, the robot is sitting

    # domain randomization ranges (applied per-env on reset)
    dr_stiffness_range: tuple = (0.8, 1.2)    # ±20% stiffness multiplier
    dr_damping_range: tuple = (0.8, 1.2)      # ±20% damping multiplier
    dr_joint_friction_range: tuple = (0.0, 0.05)  # joint friction coefficient
    dr_imu_noise_std: float = 0.01            # gaussian noise on IMU ang_vel and lin_acc

    # termination conditions
    max_tilt_distance = 0.50  # max tilt before termination
    min_height: float = 0.10  # min height before termination [m] (currently unused)
