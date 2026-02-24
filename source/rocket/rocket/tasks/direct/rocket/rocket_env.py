# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from typing import Dict, Tuple

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, quat_apply, quat_inv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg

from .rocket_env_cfg import RocketEnvCfg


class RocketEnv(DirectRLEnv):
    cfg: RocketEnvCfg

    def __init__(self, cfg: RocketEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Find joint indices by group
        self._servo_joint_ids, _ = self.robot.find_joints(self.cfg.servo_joint_names)
        self._stepper_joint_ids, _ = self.robot.find_joints(self.cfg.stepper_joint_names)
        self._joint_ids = self._servo_joint_ids + self._stepper_joint_ids
        self.imu = self.scene["imu"]

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # Find joint limits for action normalization (actions * self.joint_pos_range + self.joint_pos_mid)
        joint_limits = self.robot.data.soft_joint_pos_limits[:, self._joint_ids]  # (num_envs, num_joints, 2)
        self.joint_pos_mid = (joint_limits[..., 0] + joint_limits[..., 1]) * 0.5
        self.joint_pos_range = (joint_limits[..., 1] - joint_limits[..., 0]) * 0.5  # half-range as scale

        # Log dict for reward components and diagnostics
        self.extras["log"] = {}

        # Print out root z-height for testing purposes
        print("Resting height:", self.robot.data.root_pos_w[:, 2].mean().item())
        print(f"IMU position in world frame: {self.imu.data.pos_w.mean(dim=0)}")
        print(f"IMU orientation in world frame (quat): {self.imu.data.quat_w.mean(dim=0)}")

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
        total_reward, components = compute_rewards(
            rew_scale_alive=self.cfg.rew_scale_alive,
            rew_scale_terminated=self.cfg.rew_scale_terminated,
            rew_scale_upright=self.cfg.rew_scale_upright,
            rew_scale_joint_vel=self.cfg.rew_scale_joint_vel,
            rew_scale_energy=self.cfg.rew_scale_energy,
            rew_scale_lin_vel=self.cfg.rew_scale_lin_vel,
            quat_w=self.imu.data.quat_w,
            root_lin_vel_w=self.robot.data.root_lin_vel_w[:, :3],
            joint_vel=self.joint_vel[:, self._joint_ids],
            actions=self.actions,
            torques=self.robot.data.applied_torque[:, self._joint_ids],
            reset_terminated=self.reset_terminated,
        )
        
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

# =============================================================================
# STANDING REWARD
# =============================================================================

@torch.jit.script
def compute_standing_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_upright: float,
    rew_scale_joint_vel: float,
    rew_scale_energy: float,
    rew_scale_lin_vel: float,
    quat_w: torch.Tensor,           # (N, 4)
    root_lin_vel_w: torch.Tensor,   # (N, 3)
    joint_vel: torch.Tensor,        # (N, 6)
    actions: torch.Tensor,          # (N, 6)
    torques: torch.Tensor,          # (N, 6)
    reset_terminated: torch.Tensor, # (N,)
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # --- Alive bonus ---
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())

    # --- Termination penalty ---
    rew_termination = rew_scale_terminated * reset_terminated.float()

    # --- Upright reward ---
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

    # --- Root linear velocity penalty ---
    # Penalises ALL root motion — vertical bounce, horizontal drift, everything.
    # This is the primary anti-bounce term.
    rew_lin_vel = rew_scale_lin_vel * torch.sum(torch.square(root_lin_vel_w), dim=-1)

    # --- Joint velocity penalty ---
    rew_joint_vel = rew_scale_joint_vel * torch.sum(torch.abs(joint_vel), dim=-1)

    # --- Energy penalty ---
    rew_energy = rew_scale_energy * torch.sum(torch.square(torques), dim=-1)

    total_reward = (
        rew_alive
        + rew_termination
        + rew_upright
        + rew_lin_vel
        + rew_joint_vel
        + rew_energy
    )

    components: dict[str, torch.Tensor] = {
        "alive": rew_alive,
        "termination": rew_termination,
        "upright": rew_upright,
        "lin_vel": rew_lin_vel,
        "joint_vel": rew_joint_vel,
        "energy": rew_energy,
    }

    return total_reward, components


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_upright: float,
    rew_scale_joint_vel: float,
    rew_scale_energy: float,
    rew_scale_lin_vel: float,
    quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,   # (N, 3)
    joint_vel: torch.Tensor,
    actions: torch.Tensor,
    torques: torch.Tensor,
    reset_terminated: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    return compute_standing_rewards(
        rew_scale_alive, rew_scale_terminated, rew_scale_upright, rew_scale_joint_vel, rew_scale_energy,
        rew_scale_lin_vel, quat_w,
        root_lin_vel_w, joint_vel, actions, torques, reset_terminated,
    )
