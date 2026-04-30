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
from isaaclab.utils.math import quat_apply
from isaaclab.sensors import ContactSensor

from .rocket_env_cfg import RocketEnvCfg
from .reward_cfg import POLICIES, RewardInput
from .reward_gait_utils import compute_gait_signals


class RocketEnv(DirectRLEnv):
    cfg: RocketEnvCfg

    def __init__(self, cfg: RocketEnvCfg, render_mode: str | None = None, **kwargs):
        # Apply policy-type reward config before the env is fully initialised
        self.policy_type = cfg.policy_type
        if self.policy_type not in POLICIES:
            raise ValueError(f"Unknown policy_type '{self.policy_type}'. Available: {list(POLICIES.keys())}")
        cfg.rewards = POLICIES[self.policy_type]
        print(f"[RocketEnv] Policy type: {self.policy_type}")

        # Disable startup DR terms if flag is off (resets always remain active)
        if not cfg.enable_domain_randomization:
            cfg.events.randomize_mass = None
            cfg.events.randomize_actuator_gains = None
            cfg.events.randomize_com = None
            cfg.events.randomize_foot_friction = None
            print("[RocketEnv] Domain randomization DISABLED")
        else:
            print("[RocketEnv] Domain randomization ENABLED")

        super().__init__(cfg, render_mode, **kwargs)

        # Find joint indices by group
        self._servo_joint_ids, _        = self.robot.find_joints(self.cfg.servo_joint_names)
        self._hip_stepper_joint_ids, _  = self.robot.find_joints(self.cfg.hip_joint_names)
        self._knee_stepper_joint_ids, _ = self.robot.find_joints(self.cfg.knee_joint_names)
        self._stepper_joint_ids = self._hip_stepper_joint_ids + self._knee_stepper_joint_ids
        self._joint_ids         = self._servo_joint_ids + self._stepper_joint_ids

        # Velocity limits from actuator cfg (URDF joint limits may be unset/inf)
        # Order matches _joint_ids = servo + hip_stepper + knee_stepper
        vel_limit_list = (
            [self.cfg.robot_cfg.actuators["servos"].velocity_limit]       * len(self._servo_joint_ids) +
            [self.cfg.robot_cfg.actuators["hip_steppers"].velocity_limit]  * len(self._hip_stepper_joint_ids) +
            [self.cfg.robot_cfg.actuators["knee_steppers"].velocity_limit] * len(self._knee_stepper_joint_ids)
        )
        self.joint_vel_limits = torch.tensor(vel_limit_list, device=self.device).unsqueeze(0)  # (1, num_joints)
        self.imu = self.scene["imu"]
        self.contact_sensor_calves: ContactSensor = self.scene["contact_sensor_calves"]
        self.contact_sensor_toes: ContactSensor = self.scene["contact_sensor_toes"]

        # Toe body ids for sliding diagnostics/rewards (MDP feet_slide style).
        self._toe_body_ids, _ = self.robot.find_bodies(["Toe_L_1", "Toe_R_1"])

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.prev_joint_vel = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)

        # action ∈ [-1,1] × joint_delta_max = delta in radians; clamped to limits in _pre_physics_step
        joint_limits = self.robot.data.soft_joint_pos_limits[:, self._joint_ids]  # (num_envs, num_joints, 2)
        self.joint_delta_max = joint_limits[..., 1] - joint_limits[..., 0]  # full physical range (upper - lower)
        self._delta_target_pos = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        print(f"Joint position limits: {joint_limits[0]}")
        print(f"Joint delta max: {self.joint_delta_max[0]}")

        # Target standing pose: all joints at 0 (broadcasts over envs via shape (1, num_joints))
        self.target_standing_pose = torch.zeros(1, len(self._joint_ids), device=self.device)

        self.actions           = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.prev_actions      = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.prev_prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # consecutive steps with both feet airborne — terminates after max_airborne_steps
        self._airborne_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

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
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )
        ))
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
        # Update action history before overwriting self.actions.
        # Must be here (called once per env step), NOT in _apply_action
        # which is called decimation times and would corrupt the history.
        self.prev_prev_actions[:] = self.prev_actions
        self.prev_actions[:] = self.actions
        self.actions = actions.clone()

        # Compute delta target once per policy step so _apply_action (called decimation
        # times) just holds the same target rather than re-integrating each sub-step.
        current_pos = self.robot.data.joint_pos[:, self._joint_ids].clone()
        delta = self.actions * self.joint_delta_max
        self._delta_target_pos = (current_pos + delta).clamp(
            self.robot.data.soft_joint_pos_limits[:, self._joint_ids, 0],
            self.robot.data.soft_joint_pos_limits[:, self._joint_ids, 1],
        )
        self._delta_target_pos[:, self._servo_joint_ids] = 0.0

        self.scene.update(dt=self.physics_dt)
        self.imu.update(dt=self.physics_dt)

    def _apply_action(self) -> None:
        # Target was computed once in _pre_physics_step; just apply it each sub-step.
        self.robot.set_joint_position_target(self._delta_target_pos, joint_ids=self._joint_ids)

    def _add_obs_noise(self, x: torch.Tensor, std: float) -> torch.Tensor:
        """Add zero-mean Gaussian noise to an observation tensor. No-op if std is 0."""
        if std == 0.0:
            return x
        return x + std * torch.randn_like(x)

    def _get_observations(self) -> dict:
        imu = self.imu.data
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        n = self.cfg.obs_noise

        obs = torch.cat(
            (
                self._add_obs_noise(self.joint_pos[:, self._stepper_joint_ids], n.joint_pos_std),  # (4,) stepper only — no encoders on hip servos (R1/R2)
                self._add_obs_noise(self.joint_vel[:, self._stepper_joint_ids], n.joint_vel_std),  # (4,)
                self._add_obs_noise(imu.ang_vel_b,                              n.ang_vel_std),    # (3,)
                self._add_obs_noise(imu.lin_acc_b,                              n.lin_acc_std),    # (3,)
                self._add_obs_noise(imu.quat_w,                                 n.quat_std),       # (4,)
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        env_dt = float(self.cfg.sim.dt * self.cfg.decimation)

        toe_vel_xy = self.robot.data.body_lin_vel_w[:, self._toe_body_ids, :2]
        toe_pos_z = self.robot.data.body_pos_w[:, self._toe_body_ids, 2]
        gait = compute_gait_signals(self.contact_sensor_toes, toe_vel_xy=toe_vel_xy, toe_pos_z=toe_pos_z)

        joint_vel_ctrl = self.joint_vel[:, self._joint_ids]
        joint_acc_ctrl = (joint_vel_ctrl - self.prev_joint_vel) / max(env_dt, 1e-6)
        self.prev_joint_vel[:] = joint_vel_ctrl

        inputs = RewardInput(
            quat_w               = self.imu.data.quat_w,
            root_lin_vel_w       = self.robot.data.root_lin_vel_w[:, :3],
            joint_pos            = self.joint_pos[:, self._joint_ids],
            joint_vel            = joint_vel_ctrl,
            joint_acc            = joint_acc_ctrl,
            actions              = self.actions,
            prev_actions         = self.prev_actions,
            prev_prev_actions    = self.prev_prev_actions,
            torques              = self.robot.data.applied_torque[:, self._joint_ids],
            reset_terminated     = self.reset_terminated,
            target_standing_pose = self.target_standing_pose,
            joint_pos_target     = self._delta_target_pos,
            z_height             = self.robot.data.root_pos_w[:, 2],
            calf_forces          = self.contact_sensor_calves.data.net_forces_w,
            toe_forces           = self.contact_sensor_toes.data.net_forces_w,
            gait                 = gait,
            projected_gravity_b  = getattr(self.robot.data, "projected_gravity_b", None),
        )

        total_reward, components = self.cfg.rewards.compute(inputs)

        self.extras["log"].update({
            f"rewards/{k}": v.mean().item() for k, v in components.items()
        })

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
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

        joint_vel_exceeded = torch.any(
            torch.abs(self.joint_vel[:, self._joint_ids]) > self.joint_vel_limits, dim=-1
        )  # (N,)

        # both feet airborne for too many consecutive steps — physically impossible on real hardware
        contact_time = self.contact_sensor_toes.data.current_contact_time  # (N, 2)
        both_airborne = (contact_time[:, 0] == 0.0) & (contact_time[:, 1] == 0.0)  # (N,)
        self._airborne_steps[both_airborne] += 1
        self._airborne_steps[~both_airborne] = 0
        airborne_too_long = self._airborne_steps >= self.cfg.max_airborne_steps

        return tilted | joint_vel_exceeded | airborne_too_long, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # EventManager (configured in EventCfg) handles all reset randomization:
        # joint offsets, root state, and any startup/interval terms.
        super()._reset_idx(env_ids)
        # Clear action history so jerk/action_rate penalties are not contaminated
        # by actions from the previous episode on the first step of a new episode.
        self.prev_actions[env_ids] = 0.0
        self.prev_prev_actions[env_ids] = 0.0
        self.prev_joint_vel[env_ids] = 0.0
        self._airborne_steps[env_ids] = 0
