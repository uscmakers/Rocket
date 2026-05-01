# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward configuration and policy registry for Rocket.

Architecture:
    RewardInput       — typed container for all per-step tensor inputs
    RewardCfg   — scales + compute() method; one instance per policy
    POLICIES          — registry mapping policy_type string → RewardCfg

Adding a new policy:
    1. Add an entry to POLICIES with only the non-zero scales.
    2. Set cfg.policy_type = "<your_key>" — the env picks it up automatically.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import torch

from .reward_utils import (
    rew_upright,
    rew_flat_orientation_l2,
    rew_flat_orientation_l2_from_projected_gravity_b,
    rew_heading_vel,
    rew_forward_vel_tracking,
    rew_vertical_vel_penalty,
    rew_forward_vel_l2_deadzone,
    rew_toe_walking,
    rew_alternating_contact,
    rew_friction_cone_penalty,
    rew_joint_vel_penalty,
    rew_joint_acc_l2,
    rew_torque_penalty,
    rew_pose,
    rew_action_rate_penalty,
    rew_jerk_penalty,
    rew_both_feet_airborne,
    rew_joint_pos_tracking,
)
from .reward_gait_utils import GaitSignals, rew_feet_air_time_biped, rew_feet_slide, rew_toe_clearance_biped


# =============================================================================
# REWARD INPUT
# =============================================================================

@dataclass
class RewardInput:
    """All per-step tensor inputs required by RewardCfg.compute().

    Shapes assume N environments and J controlled joints.
    """
    quat_w:               torch.Tensor  # (N, 4)  IMU orientation (w, x, y, z)
    root_lin_vel_w:       torch.Tensor  # (N, 3)  root linear velocity in world frame
    joint_pos:            torch.Tensor  # (N, J)  controlled joint positions
    joint_vel:            torch.Tensor  # (N, J)  controlled joint velocities
    joint_acc:            torch.Tensor  # (N, J)  controlled joint accelerations (env-step)
    actions:              torch.Tensor  # (N, J)  current policy actions
    prev_actions:         torch.Tensor  # (N, J)  actions from t-1
    prev_prev_actions:    torch.Tensor  # (N, J)  actions from t-2
    torques:              torch.Tensor  # (N, J)  applied joint torques
    reset_terminated:     torch.Tensor  # (N,)    bool — episode terminated this step
    target_standing_pose: torch.Tensor  # (1, J)  reference joint pose for pose reward
    z_height:             torch.Tensor  # (N,)    root z-height in world frame
    joint_pos_target:     torch.Tensor  # (N, J)  target joint positions (_delta_target_pos)
    calf_forces:          torch.Tensor  # (N, 2, 3) net contact forces on calves
    toe_forces:           torch.Tensor  # (N, 2, 3) net contact forces on toes
    gait:                 GaitSignals | None = None
    projected_gravity_b:  torch.Tensor | None = None  # (N, 3) if available (MDP-style)


# =============================================================================
# REWARD SCALES CONFIG
# =============================================================================

@dataclass
class RewardCfg:
    """Per-policy reward scales.

    Defaults match the YAML config exactly.

    Usage:
        cfg = RewardCfg(upright=3.0, lin_vel=-1.0, action_rate=-0.2)
        total, components = cfg.compute(inputs)
    """

    # survival — always active, shared across all policies
    alive:                float = 10.0
    terminated:           float = -0.0   # YAML: -0.0

    # balance
    upright:              float = 0.0    # YAML: 0.0
    flat_orientation_l2:  float = 0.0
    # target projected_gravity XY in body frame — shifts the bowl minimum away from perfectly vertical.
    # (0, 0) = vertical; (0, -0.0664) = 3.8° forward lean (COM over support center).
    flat_orientation_l2_target_xy: tuple[float, float] = (0.0, -0.0664)

    # velocity
    lin_vel:              float = 0.0
    forward_vel:          float = 0.0
    forward_vel_track:    float = 0.0
    forward_vel_target:   float = 0.1    # YAML: 0.1
    forward_vel_sigma:    float = 0.025
    backward_vel:         float = 0.0
    lat_vel:              float = 0.0
    vertical_vel:         float = 0.0
    # standing-in-place velocity bowl (deadzone + L2)
    forward_vel_l2:       float = 0.0
    forward_vel_threshold: float = 0.02  # m/s deadzone for near-zero drift

    # contact quality
    toe_walking:          float = 0.0
    alternating_contact:  float = 0.0   # YAML: 0.0
    friction_cone:        float = 0.0

    # smoothness
    action_rate:          float = 0.0
    jerk:                 float = 0.0

    # inactive by default — enable as needed
    joint_vel:            float = 0.0
    joint_acc:            float = 0.0
    torque:               float = 0.0
    knee_torque:          float = 0.0
    target_standing_pose: float = 0.0
    height:               float = 0.0
    height_target:        float = 0.14
    height_sigma:         float = 0.01

    # gait (optional; requires ContactSensorCfg(track_air_time=True) on toes)
    feet_air_time_biped:  float = 0.0
    feet_air_time_biped_threshold_s: float = 0.25  # YAML: 0.25
    feet_slide:           float = 0.0
    toe_clearance_biped:  float = 0.0
    toe_clearance_biped_height_m: float = 0.02
    both_feet_airborne:   float = 0.0
    joint_pos_tracking:   float = 0.0

    def compute(
        self,
        inputs: RewardInput,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute total reward and per-component breakdown.

        All component functions are @torch.jit.script — GPU work is compiled.
        Returns:
            total:      (N,) total reward per environment
            components: dict of named (N,) reward tensors for logging
        """
        # --- survival ---
        alive_mask   = 1.0 - inputs.reset_terminated.float()
        rew_alive    = self.alive      * alive_mask
        rew_term     = self.terminated * inputs.reset_terminated.float()

        # --- balance ---
        rew_up = self.upright * rew_upright(inputs.quat_w)
        _flat_target = torch.tensor(self.flat_orientation_l2_target_xy, device=inputs.quat_w.device, dtype=inputs.quat_w.dtype)
        if inputs.projected_gravity_b is not None:
            flat = rew_flat_orientation_l2_from_projected_gravity_b(inputs.projected_gravity_b, _flat_target)
        else:
            flat = rew_flat_orientation_l2(inputs.quat_w, _flat_target)
        rew_flat_r = self.flat_orientation_l2 * flat

        # --- velocity ---
        forward_vel, lateral_vel = rew_heading_vel(inputs.quat_w, inputs.root_lin_vel_w)
        rew_lin_vel_r         = self.lin_vel       * torch.abs(forward_vel)
        rew_forward_vel_r     = self.forward_vel   * forward_vel
        rew_forward_vel_track_r = self.forward_vel_track * rew_forward_vel_tracking(
            forward_vel, self.forward_vel_target, self.forward_vel_sigma
        )
        rew_backward_vel_r    = self.backward_vel  * torch.relu(-forward_vel)
        rew_lat_vel_r         = self.lat_vel       * torch.square(lateral_vel)
        rew_vert_vel_r        = self.vertical_vel  * rew_vertical_vel_penalty(inputs.root_lin_vel_w)
        rew_forward_vel_l2_r  = self.forward_vel_l2 * rew_forward_vel_l2_deadzone(forward_vel, self.forward_vel_threshold)

        # --- contact quality ---
        rew_toe_r      = self.toe_walking         * rew_toe_walking(inputs.calf_forces, inputs.toe_forces)
        rew_alt_r      = self.alternating_contact  * rew_alternating_contact(inputs.toe_forces)
        if inputs.gait is not None and inputs.gait.contacts_force is not None:
            rew_friction_r = self.friction_cone * rew_friction_cone_penalty(inputs.toe_forces, inputs.gait.contacts_force)
        else:
            rew_friction_r = torch.zeros_like(rew_alive)

        # --- smoothness ---
        rew_rate_r = self.action_rate * rew_action_rate_penalty(inputs.actions, inputs.prev_actions)
        rew_jerk_r = self.jerk        * rew_jerk_penalty(inputs.actions, inputs.prev_actions, inputs.prev_prev_actions)

        # --- optional / inactive by default ---
        rew_jvel_r        = self.joint_vel            * rew_joint_vel_penalty(inputs.joint_vel)
        rew_jacc_r        = self.joint_acc            * rew_joint_acc_l2(inputs.joint_acc)
        rew_torque_r      = self.torque               * rew_torque_penalty(inputs.torques)
        rew_knee_torque_r = self.knee_torque          * rew_torque_penalty(inputs.torques[:, 4:6])
        rew_pose_r        = self.target_standing_pose * rew_pose(inputs.joint_pos, inputs.target_standing_pose)
        rew_jpos_track_r  = self.joint_pos_tracking   * rew_joint_pos_tracking(inputs.joint_pos, inputs.joint_pos_target)
        # Two-sided height penalty (matches YAML/reset e024eaf behaviour — no relu)
        height_error = inputs.z_height - self.height_target
        rew_height_r  = self.height * torch.exp(-torch.square(height_error) / (self.height_sigma * self.height_sigma))

        # --- gait (optional) ---
        if inputs.gait is not None:
            gait = inputs.gait
            rew_air_time_biped_r = self.feet_air_time_biped * rew_feet_air_time_biped(
                gait.in_contact,
                gait.current_air_time,
                gait.current_contact_time,
                threshold=self.feet_air_time_biped_threshold_s,
            )
            if gait.contacts_force is not None and gait.toe_vel_xy is not None:
                rew_slide_r = self.feet_slide * rew_feet_slide(gait.contacts_force, gait.toe_vel_xy)
            else:
                rew_slide_r = torch.zeros_like(rew_alive)
            if gait.toe_pos_z is not None:
                rew_toe_clear_r = self.toe_clearance_biped * rew_toe_clearance_biped(
                    gait.in_contact, gait.toe_pos_z, height_threshold=self.toe_clearance_biped_height_m
                )
                rew_airborne_r = self.both_feet_airborne * rew_both_feet_airborne(
                    gait.in_contact, gait.toe_pos_z
                )
            else:
                rew_toe_clear_r = torch.zeros_like(rew_alive)
                rew_airborne_r = torch.zeros_like(rew_alive)
        else:
            rew_air_time_biped_r = torch.zeros_like(rew_alive)
            rew_slide_r = torch.zeros_like(rew_alive)
            rew_toe_clear_r = torch.zeros_like(rew_alive)
            rew_airborne_r = torch.zeros_like(rew_alive)

        # --- diagnostics (per-joint-group pose error, always logged) ---
        hip_yaw_err  = torch.abs(inputs.joint_pos[:, 0:2] - inputs.target_standing_pose[:, 0:2]).mean(dim=-1)
        hip_roll_err = torch.abs(inputs.joint_pos[:, 2:4] - inputs.target_standing_pose[:, 2:4]).mean(dim=-1)
        knee_err     = torch.abs(inputs.joint_pos[:, 4:6] - inputs.target_standing_pose[:, 4:6]).mean(dim=-1)

        total = (
            rew_alive + rew_term
            + rew_up + rew_flat_r
            + rew_lin_vel_r + rew_forward_vel_r + rew_forward_vel_track_r + rew_backward_vel_r + rew_lat_vel_r + rew_vert_vel_r + rew_forward_vel_l2_r
            + rew_toe_r + rew_alt_r + rew_friction_r
            + rew_rate_r + rew_jerk_r
            + rew_jvel_r + rew_jacc_r + rew_torque_r + rew_knee_torque_r + rew_pose_r + rew_height_r + rew_jpos_track_r
            + rew_air_time_biped_r + rew_slide_r + rew_toe_clear_r + rew_airborne_r
        )

        components: dict[str, torch.Tensor] = {
            "alive":                rew_alive,
            "termination":          rew_term,
            "upright":              rew_up,
            "flat_orientation_l2":  rew_flat_r,
            "lin_vel":              rew_lin_vel_r,
            "forward_vel":          rew_forward_vel_r,
            "forward_vel_track":    rew_forward_vel_track_r,
            "backward_vel":         rew_backward_vel_r,
            "lat_vel":              rew_lat_vel_r,
            "vertical_vel":         rew_vert_vel_r,
            "forward_vel_l2":       rew_forward_vel_l2_r,
            "toe_walking":          rew_toe_r,
            "alternating_contact":  rew_alt_r,
            "friction_cone":        rew_friction_r,
            "action_rate":          rew_rate_r,
            "jerk":                 rew_jerk_r,
            "joint_vel":            rew_jvel_r,
            "joint_acc":            rew_jacc_r,
            "torque":               rew_torque_r,
            "knee_torque":          rew_knee_torque_r,
            "target_standing_pose": rew_pose_r,
            "height":               rew_height_r,
            "feet_air_time_biped":  rew_air_time_biped_r,
            "feet_slide":           rew_slide_r,
            "toe_clearance_biped":  rew_toe_clear_r,
            "both_feet_airborne":   rew_airborne_r,
            "joint_pos_tracking":   rew_jpos_track_r,
            # diagnostics
            "hip_yaw_pose_error":   hip_yaw_err,
            "hip_roll_pose_error":  hip_roll_err,
            "knee_pose_error":      knee_err,
        }

        return total, components


# =============================================================================
# POLICY REGISTRY
# Only set non-zero scales — alive/terminated are always ±10.0/0.0 by default.
# =============================================================================

POLICIES: dict[str, RewardCfg] = {

    "standing": RewardCfg(
        # uprightness & balance
        flat_orientation_l2 = -0.5,
        height              =  1.0,

        # locomotion
        forward_vel_l2      = -0.5,
        forward_vel_threshold = 0.02,

        # gait rewards
        toe_walking         =  3.0,
        feet_air_time_biped =  2.0,
        toe_clearance_biped =  1.0,
        both_feet_airborne  = -0.5,
        joint_pos_tracking  =  0.0,

        # action smoothness
        action_rate         = -0.005,
        joint_acc           = -1.25e-7,
    ),

    "walking": RewardCfg(
        flat_orientation_l2 = -0.5,
        height              =  1.0,

        forward_vel_track   =  2.0,

        toe_walking         =  3.0,
        feet_air_time_biped =  2.0,
        toe_clearance_biped =  1.0,
        both_feet_airborne  = -0.5,
        joint_pos_tracking  = -0.0,

        action_rate         = -0.005,
        joint_acc           = -1.25e-7,
    ),

}
