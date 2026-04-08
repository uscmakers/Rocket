# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, quat_apply, quat_inv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg, ContactSensor

from .rocket_env_cfg import RocketEnvCfg
from .rewards import compute_standing_rewards, compute_walking_rewards, rew_toe_walking_debug


class RocketEnv(DirectRLEnv):
    cfg: RocketEnvCfg

    def __init__(self, cfg: RocketEnvCfg, render_mode: str | None = None, **kwargs):
        # Apply policy-type reward scales before the env is fully initialised
        self.policy_type = cfg.policy_type
        if self.policy_type == "walking":
            for k, v in cfg.walking_reward_scales.items():
                setattr(cfg, k, v)
        else:
            for k, v in cfg.standing_reward_scales.items():
                setattr(cfg, k, v)
        print(f"[RocketEnv] Policy type: {self.policy_type}")

        super().__init__(cfg, render_mode, **kwargs)

        # Find joint indices by group
        self._servo_joint_ids, _ = self.robot.find_joints(self.cfg.servo_joint_names)
        self._stepper_joint_ids, _ = self.robot.find_joints(self.cfg.stepper_joint_names)
        self._joint_ids = self._servo_joint_ids + self._stepper_joint_ids
        self.imu = self.scene["imu"]
        self.contact_sensor_calves: ContactSensor = self.scene["contact_sensor_calves"]
        self.contact_sensor_toes: ContactSensor = self.scene["contact_sensor_toes"]

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # Find joint limits for action normalization (actions * self.joint_pos_range + self.joint_pos_mid)
        joint_limits = self.robot.data.soft_joint_pos_limits[:, self._joint_ids]  # (num_envs, num_joints, 2)
        self.joint_pos_mid = (joint_limits[..., 0] + joint_limits[..., 1]) * 0.5
        self.joint_pos_range = (joint_limits[..., 1] - joint_limits[..., 0]) * 0.5  # half-range as scale

        # Calculate the target standing pose (servos should be at 0 degrees, steppers should be at their negative limit to rep 45 deg limit)
        target_pos = torch.zeros(1, len(self._joint_ids), device=self.device)
        # stepper_indices = [self._joint_ids.index(j) for j in self._stepper_joint_ids]
        # target_pos[:, stepper_indices] = self.robot.data.soft_joint_pos_limits[:1, self._stepper_joint_ids, 0]
        self.target_standing_pose = target_pos  # (1, num_joints) broadcasts over envs

        # Log dict for reward components and diagnostics
        self.extras["log"] = {}

        # Print out root z-height for testing purposes
        print("Resting height:", self.robot.data.root_pos_w[:, 2].mean().item())
        print(f"IMU position in world frame: {self.imu.data.pos_w.mean(dim=0)}")
        print(f"IMU orientation in world frame (quat): {self.imu.data.quat_w.mean(dim=0)}")

        # Print out target standing pose for testing purposes
        print("Target standing pose (joint positions):", self.target_standing_pose)

        # --- Contact sensor diagnostics ---
        # net_forces_w shape: (num_envs, num_bodies, 3)
        # num_bodies must be 2 (one per leg) for the toe-walking reward to work correctly.
        calf_shape = self.contact_sensor_calves.data.net_forces_w.shape
        toe_shape  = self.contact_sensor_toes.data.net_forces_w.shape
        print(f"[ContactSensor] calves net_forces_w shape: {calf_shape}  (expect: num_envs x 2 x 3)")
        print(f"[ContactSensor] toes  net_forces_w shape: {toe_shape}   (expect: num_envs x 2 x 3)")
        if hasattr(self.contact_sensor_calves, "body_names"):
            print(f"[ContactSensor] calf bodies tracked: {self.contact_sensor_calves.body_names}")
            print(f"[ContactSensor] toe  bodies tracked: {self.contact_sensor_toes.body_names}")

        # Print out camera config
        if hasattr(cfg.scene, "tiled_camera"):
            print("Scene setup complete with the following camera config: ", cfg.scene.tiled_camera)


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

        self.scene.update(dt=self.physics_dt)
        self.imu.update(dt=self.physics_dt)

    def _apply_action(self) -> None:
        # actions are in [-1, 1] after tanh squashing in the policy head, so we scale and shift them according to joint ranges
        target_joint_pos = self.joint_pos_mid + self.actions * self.joint_pos_range

        # additional clamping to ensure we never exceed joint limits (optional)
        target_joint_pos = target_joint_pos.clamp(
            self.robot.data.soft_joint_pos_limits[:, self._joint_ids, 0],
            self.robot.data.soft_joint_pos_limits[:, self._joint_ids, 1]
        )

        # we control all joints in position control mode in isaac sim (delta pos in real life)
        self.robot.set_joint_position_target(target_joint_pos, joint_ids=self._joint_ids)

    def _get_observations(self) -> dict:
        imu = self.imu.data
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        obs = torch.cat(
            (
                self.joint_pos[:, self._joint_ids],          # (6,) all joint positions
                self.joint_vel[:, self._stepper_joint_ids],  # (4,) stepper joint velocities only
                imu.ang_vel_b,                               # (3,) angular velocity
                imu.lin_acc_b,                               # (3,) linear acceleration
                imu.quat_w,                                  # (4,) orientation quaternion
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        reward_kwargs = dict(
            rew_scale_alive=self.cfg.rew_scale_alive,
            rew_scale_terminated=self.cfg.rew_scale_terminated,
            rew_scale_upright=self.cfg.rew_scale_upright,
            rew_scale_target_standing_pose=self.cfg.rew_scale_target_standing_pose,
            rew_scale_joint_vel=self.cfg.rew_scale_joint_vel,
            rew_scale_torque=self.cfg.rew_scale_torque,
            rew_scale_lin_vel=self.cfg.rew_scale_lin_vel,
            rew_scale_lat_vel=self.cfg.rew_scale_lat_vel,
            rew_scale_height=self.cfg.rew_scale_height,
            rew_scale_toe_walking=self.cfg.rew_scale_toe_walking,
            quat_w=self.imu.data.quat_w,
            root_lin_vel_w=self.robot.data.root_lin_vel_w[:, :3],
            joint_pos=self.joint_pos[:, self._joint_ids],
            joint_vel=self.joint_vel[:, self._joint_ids],
            actions=self.actions,
            torques=self.robot.data.applied_torque[:, self._joint_ids],
            reset_terminated=self.reset_terminated,
            target_standing_pose=self.target_standing_pose,
            z_height=self.robot.data.root_pos_w[:, 2],
            calf_forces=self.contact_sensor_calves.data.net_forces_w,
            toe_forces=self.contact_sensor_toes.data.net_forces_w,
        )
        rew_toe_walking_debug(reward_kwargs["calf_forces"], reward_kwargs["toe_forces"])

        if self.policy_type == "walking":
            total_reward, components = compute_walking_rewards(**reward_kwargs)
        else:
            total_reward, components = compute_standing_rewards(**reward_kwargs)

        self.extras["log"].update({
            f"rewards/{k}": v.mean().item() for k, v in components.items()
        })

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        imu = self.imu.data
        quat_w = imu.quat_w  # (num_envs, 4)
        num_envs = quat_w.shape[0]

        # Rocket's local up axis
        body_z = torch.tensor(
            [0.0, 0.0, 1.0],
            device=quat_w.device,
            dtype=quat_w.dtype
        ).expand(num_envs, 3)

        # World up axis
        world_up = torch.tensor(
            [0.0, 0.0, 1.0],
            device=quat_w.device,
            dtype=quat_w.dtype
        ).expand(num_envs, 3)

        # Rotate rocket up into world frame
        body_z_world = quat_apply(quat_w, body_z)

        # Euclidean distance from upright
        distance = torch.norm(body_z_world - world_up, dim=-1)

        # Termination conditions
        tilted = distance > self.cfg.max_tilt_distance

        return tilted, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        # Randomize all joint positions slightly
        joint_pos[:, self._joint_ids] += sample_uniform(
            self.cfg.initial_joint_range[0],
            self.cfg.initial_joint_range[1],
            joint_pos[:, self._joint_ids].shape,
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Randomize spawn height slightly
        default_root_state[:, 2] += sample_uniform(
            -0.05, 0.05,
            (len(env_ids),),
            default_root_state.device
        )

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)