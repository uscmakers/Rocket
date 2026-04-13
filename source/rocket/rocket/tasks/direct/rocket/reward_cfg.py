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
    rew_vertical_vel_penalty,
    rew_toe_walking,
    rew_alternating_contact,
    rew_joint_vel_penalty,
    rew_torque_penalty,
    rew_pose,
    rew_action_rate_penalty,
    rew_jerk_penalty,
)
from .reward_gait_utils import GaitSignals, rew_feet_air_time_biped, rew_feet_slide


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
    actions:              torch.Tensor  # (N, J)  current policy actions
    prev_actions:         torch.Tensor  # (N, J)  actions from t-1
    prev_prev_actions:    torch.Tensor  # (N, J)  actions from t-2
    torques:              torch.Tensor  # (N, J)  applied joint torques
    reset_terminated:     torch.Tensor  # (N,)    bool — episode terminated this step
    target_standing_pose: torch.Tensor  # (1, J)  reference joint pose for pose reward
    z_height:             torch.Tensor  # (N,)    root z-height in world frame
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

    Defaults:
        alive      =  5.0  (always active)
        terminated = -5.0  (always active)
        everything else = 0.0

    Usage:
        cfg = RewardCfg(upright=3.0, lin_vel=-1.0, action_rate=-0.2)
        total, components = cfg.compute(inputs)
    """

    # survival — always active, shared across all policies
    alive:                float = 5.0
    terminated:           float = -5.0

    # balance
    upright:              float = 0.0
    flat_orientation_l2:  float = 0.0  # MDP-style bowl (penalty); set negative to penalize

    # velocity
    lin_vel:              float = 0.0  # penalize |forward_vel| — any horiz movement (standing)
    forward_vel:          float = 0.0  # reward signed forward vel (walking)
    lat_vel:              float = 0.0  # penalize lateral drift (squared)
    vertical_vel:         float = 0.0  # penalize vertical bouncing (squared)

    # contact quality
    toe_walking:          float = 0.0  # penalize calf ground contact
    alternating_contact:  float = 0.0  # reward alternating foot contact

    # smoothness
    action_rate:          float = 0.0  # penalize rapid command changes (squared delta)
    jerk:                 float = 0.0  # penalize second-order command changes

    # inactive by default — enable as needed
    joint_vel:            float = 0.0
    torque:               float = 0.0
    target_standing_pose: float = 0.0
    height:               float = 0.0

    # gait (optional; requires ContactSensorCfg(track_air_time=True) on toes)
    feet_air_time_biped:  float = 0.0  # single-stance shaping based on air/contact timers
    feet_air_time_biped_threshold_s: float = 0.4
    feet_slide:           float = 0.0  # penalize toe sliding when in force-threshold contact

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
        rew_up       = self.upright    * rew_upright(inputs.quat_w)
        if inputs.projected_gravity_b is not None:
            flat = rew_flat_orientation_l2_from_projected_gravity_b(inputs.projected_gravity_b)
        else:
            flat = rew_flat_orientation_l2(inputs.quat_w)
        rew_flat_r = self.flat_orientation_l2 * flat

        # --- velocity ---
        forward_vel, lateral_vel = rew_heading_vel(inputs.quat_w, inputs.root_lin_vel_w)
        rew_lin_vel_r    = self.lin_vel      * torch.abs(forward_vel)      # |fwd| penalty (standing)
        rew_forward_vel_r = self.forward_vel * forward_vel                 # signed reward (walking)
        rew_lat_vel_r    = self.lat_vel      * torch.square(lateral_vel)   # quadratic lateral penalty
        rew_vert_vel_r   = self.vertical_vel * rew_vertical_vel_penalty(inputs.root_lin_vel_w)

        # --- contact quality ---
        rew_toe_r  = self.toe_walking        * rew_toe_walking(inputs.calf_forces, inputs.toe_forces)
        rew_alt_r  = self.alternating_contact * rew_alternating_contact(inputs.toe_forces)

        # --- smoothness ---
        rew_rate_r = self.action_rate * rew_action_rate_penalty(inputs.actions, inputs.prev_actions)
        rew_jerk_r = self.jerk        * rew_jerk_penalty(inputs.actions, inputs.prev_actions, inputs.prev_prev_actions)

        # --- optional / inactive by default ---
        rew_jvel_r   = self.joint_vel            * rew_joint_vel_penalty(inputs.joint_vel)
        rew_torque_r = self.torque               * rew_torque_penalty(inputs.torques)
        rew_pose_r   = self.target_standing_pose * rew_pose(inputs.joint_pos, inputs.target_standing_pose)
        rew_height_r = self.height               * inputs.z_height

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
        else:
            rew_air_time_biped_r = torch.zeros_like(rew_alive)
            rew_slide_r = torch.zeros_like(rew_alive)

        # --- diagnostics (per-joint-group pose error, always logged) ---
        hip_yaw_err  = torch.abs(inputs.joint_pos[:, 0:2] - inputs.target_standing_pose[:, 0:2]).mean(dim=-1)
        hip_roll_err = torch.abs(inputs.joint_pos[:, 2:4] - inputs.target_standing_pose[:, 2:4]).mean(dim=-1)
        knee_err     = torch.abs(inputs.joint_pos[:, 4:6] - inputs.target_standing_pose[:, 4:6]).mean(dim=-1)

        total = (
            rew_alive + rew_term
            + rew_up + rew_flat_r
            + rew_lin_vel_r + rew_forward_vel_r + rew_lat_vel_r + rew_vert_vel_r
            + rew_toe_r + rew_alt_r
            + rew_rate_r + rew_jerk_r
            + rew_jvel_r + rew_torque_r + rew_pose_r + rew_height_r
            + rew_air_time_biped_r + rew_slide_r
        )

        components: dict[str, torch.Tensor] = {
            "alive":                rew_alive,
            "termination":          rew_term,
            "upright":              rew_up,
            "flat_orientation_l2":  rew_flat_r,
            "lin_vel":              rew_lin_vel_r,
            "forward_vel":          rew_forward_vel_r,
            "lat_vel":              rew_lat_vel_r,
            "vertical_vel":         rew_vert_vel_r,
            "toe_walking":          rew_toe_r,
            "alternating_contact":  rew_alt_r,
            "action_rate":          rew_rate_r,
            "jerk":                 rew_jerk_r,
            "joint_vel":            rew_jvel_r,
            "torque":               rew_torque_r,
            "target_standing_pose": rew_pose_r,
            "height":               rew_height_r,
            "feet_air_time_biped":  rew_air_time_biped_r,
            "feet_slide":           rew_slide_r,
            # diagnostics
            "hip_yaw_pose_error":   hip_yaw_err,
            "hip_roll_pose_error":  hip_roll_err,
            "knee_pose_error":      knee_err,
        }

        return total, components


# =============================================================================
# POLICY REGISTRY
# Only set non-zero scales — alive/terminated are always ±5.0 by default.
# =============================================================================

POLICIES: dict[str, RewardCfg] = {

    "standing": RewardCfg(
        upright             =  0.0,
        flat_orientation_l2 = -2.0,   # this is a softer tilt penalty with softer gradients closer to upright vector
        lin_vel             = -1.0,   # penalize any horizontal movement
        vertical_vel        = -0.1,
        toe_walking         =  1.0,   # penalty for calves contacting the ground (should be refactored into a penalty)
        alternating_contact =  0.0,   # temporarily off (previously rewarded one-foot loading)
        feet_air_time_biped =  2.0,
        feet_slide          = -0.2,
        action_rate         = -0.2,
        jerk                = -0.1,
    ),

    "walking": RewardCfg(
        upright             =  0.0,
        flat_orientation_l2 = -2.0,
        forward_vel         =  3.0,   # reward forward motion
        vertical_vel        = -0.1,
        toe_walking         =  1.0,
        alternating_contact =  0.0,
        feet_air_time_biped =  2.0,
        feet_slide          = -0.2,
        action_rate         = -0.002,
        jerk                = -0.001,
    ),

}
