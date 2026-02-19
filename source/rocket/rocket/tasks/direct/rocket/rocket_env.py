# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, quat_rotate
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg

from .rocket_env_cfg import RocketEnvCfg


class RocketEnv(DirectRLEnv):
    cfg: RocketEnvCfg

    def __init__(self, cfg: RocketEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)
        self.imu = self.scene["imu"]

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

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
        self.imu.update(dt=self.physics_dt, force_compute=True)

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(
            self.actions * self.cfg.action_scale, joint_ids=self._cart_dof_idx
        )

    def _get_observations(self) -> dict:
        imu = self.imu.data
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                
                imu.ang_vel_b,  
                imu.lin_acc_b,
                imu.quat_w
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.cfg.rew_scale_upright,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.imu.data.quat_w,
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(
            torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos,
            dim=1,
        )
        out_of_bounds = out_of_bounds | torch.any(
            torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1
        )

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
        body_z_world = quat_rotate(quat_w, body_z)

        # Euclidean distance from upright
        distance = torch.norm(body_z_world - world_up, dim=-1)

        # Threshold selection:
        # If rocket tilts 45°, distance ≈ 0.765
        tilted = distance > 0.75

        out_of_bounds = out_of_bounds | tilted

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    rew_scale_upright: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    quat_w: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(
        torch.square(pole_pos).unsqueeze(dim=1), dim=-1
    )
    rew_cart_vel = rew_scale_cart_vel * torch.sum(
        torch.abs(cart_vel).unsqueeze(dim=1), dim=-1
    )
    rew_pole_vel = rew_scale_pole_vel * torch.sum(
        torch.abs(pole_vel).unsqueeze(dim=1), dim=-1
    )

    # Distance from upright: rotate body z-axis [0,0,1] into world frame
    # using the IMU quaternion, then measure Euclidean distance to world up [0,0,1].
    # quat_w is (num_envs, 4) in (w, x, y, z) format.
    #
    # For a quaternion q = (w, x, y, z), rotating v = [0,0,1]:
    #   rotated_x = 2*(x*z + w*y)
    #   rotated_y = 2*(y*z - w*x)
    #   rotated_z = 1 - 2*(x^2 + y^2)
    qw = quat_w[:, 0]
    qx = quat_w[:, 1]
    qy = quat_w[:, 2]
    qz = quat_w[:, 3]
    body_z_world_x = 2.0 * (qx * qz + qw * qy)
    body_z_world_y = 2.0 * (qy * qz - qw * qx)
    body_z_world_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    # Euclidean distance to world up [0, 0, 1]
    upright_distance = torch.sqrt(
        body_z_world_x * body_z_world_x
        + body_z_world_y * body_z_world_y
        + (body_z_world_z - 1.0) * (body_z_world_z - 1.0)
    )
    rew_upright = rew_scale_upright * upright_distance

    total_reward = (
        rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel + rew_upright
    )
    return total_reward
