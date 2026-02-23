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
from isaaclab.sensors import ImuCfg, TiledCameraCfg

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
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0
            )
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),  # Initial position (spawn at 0.3m height)
        joint_pos={"Revolute.*": 0.0},  # All joints start at 0
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["Revolute.*"],  # Matches all 6 joints from URDF
            effort_limit=10.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=0.5,
        ),
    },
)

@configclass
class RocketSceneCfg(InteractiveSceneCfg):
    imu: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Torso_Assy_1",  # Updated to match URDF base link
        update_period=0.0,  # every step
        history_length=1,
        debug_vis=True,
    )

    # Camera for video recording - 45° angle view
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,  # Increase update period for video recording to reduce overhead and speed up training
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.5, 0.5, 0.5),  # 45° angle position
            rot=(0.7071, 0.0, 0.0, 0.7071),  # Look down at robot (quaternion for -45° pitch)
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=48.0,
            focus_distance=400.0,
            horizontal_aperture=21.0,
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
    stepper_joint_names = ["Revolute3", "Revolute4", "Revolute5", "Revolute6"]  # hip roll + knee (steppers)

    # action scale
    action_scale = 10.0  # [Nm] torque multiplier

    # reward scales
    rew_scale_alive:         float =  2.0
    rew_scale_terminated:    float = -5.0
    rew_scale_upright:       float =  2.0 
    rew_scale_joint_vel:     float = -0.1
    rew_scale_energy:        float = -0.1
    rew_scale_lin_vel:       float = -0.5

    # reset/termination conditions
    initial_joint_range = [-0.1, 0.1]  # joint angle randomization on reset [rad]
    max_tilt_distance = 0.75  # max tilt before termination
    min_height: float = 0.1  # min height before termination [m]
