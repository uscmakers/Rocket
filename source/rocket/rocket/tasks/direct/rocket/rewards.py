# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from typing import Dict, Tuple

import torch


# =============================================================================
# REWARD COMPONENTS
# All component functions return unscaled values; callers apply the scale.
# =============================================================================

@torch.jit.script
def rew_upright(quat_w: torch.Tensor) -> torch.Tensor:
    """exp(-distance) where distance is how far body z-axis is from world up.

    quat_w: (N, 4) in (w, x, y, z) format.
    Returns (N,) in [0, 1]; 1.0 = perfectly upright.
    """
    qw = quat_w[:, 0]
    qx = quat_w[:, 1]
    qy = quat_w[:, 2]
    qz = quat_w[:, 3]
    body_z_x = 2.0 * (qx * qz + qw * qy)
    body_z_y = 2.0 * (qy * qz - qw * qx)
    body_z_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    distance = torch.sqrt(
        body_z_x * body_z_x
        + body_z_y * body_z_y
        + (body_z_z - 1.0) * (body_z_z - 1.0)
    )
    return torch.exp(-distance)


@torch.jit.script
def rew_toe_walking(
    calf_forces: torch.Tensor,  # (N, 2, 3)
    toe_forces: torch.Tensor,   # (N, 2, 3)
) -> torch.Tensor:
    """toe contacts minus calf contacts, summed across legs.

    Returns (N,) in [-2, +2]; positive = good (toe contact), negative = bad (calf contact).
    """
    calf_contact = (torch.norm(calf_forces, dim=-1) > 1.0).float().sum(dim=-1)
    toe_contact  = (torch.norm(toe_forces,  dim=-1) > 1.0).float().sum(dim=-1)
    return toe_contact - calf_contact


@torch.jit.script
def rew_pose(
    joint_pos: torch.Tensor,         # (N, J)
    target_standing_pose: torch.Tensor,  # (1 or N, J)
) -> torch.Tensor:
    """exp(-RMS pose error). Returns (N,) in (0, 1]; 1.0 = at target pose."""
    pose_error = torch.sqrt(torch.sum(torch.square(joint_pos - target_standing_pose), dim=-1))
    return torch.exp(-pose_error)


@torch.jit.script
def rew_joint_vel_penalty(joint_vel: torch.Tensor) -> torch.Tensor:
    """Sum of absolute joint velocities. Returns (N,); caller applies negative scale."""
    return torch.sum(torch.abs(joint_vel), dim=-1)


@torch.jit.script
def rew_torque_penalty(torques: torch.Tensor) -> torch.Tensor:
    """Sum of squared torques. Returns (N,); caller applies negative scale."""
    return torch.sum(torch.square(torques), dim=-1)


@torch.jit.script
def rew_lin_vel_penalty(root_lin_vel_w: torch.Tensor) -> torch.Tensor:
    """Sum of squared root linear velocity (all axes). Returns (N,); caller applies negative scale."""
    return torch.sum(torch.square(root_lin_vel_w), dim=-1)


# =============================================================================
# STANDING REWARD
# =============================================================================

@torch.jit.script
def compute_standing_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_upright: float,
    rew_scale_target_standing_pose: float,
    rew_scale_joint_vel: float,
    rew_scale_torque: float,
    rew_scale_lin_vel: float,
    rew_scale_height: float,
    rew_scale_toe_walking: float,
    quat_w: torch.Tensor,                # (N, 4)
    root_lin_vel_w: torch.Tensor,        # (N, 3)
    joint_pos: torch.Tensor,             # (N, J)
    joint_vel: torch.Tensor,             # (N, J)
    actions: torch.Tensor,               # (N, J)
    torques: torch.Tensor,               # (N, J)
    reset_terminated: torch.Tensor,      # (N,)
    target_standing_pose: torch.Tensor,  # (1 or N, J)
    z_height: torch.Tensor,              # (N,)
    calf_forces: torch.Tensor,           # (N, 2, 3)
    toe_forces: torch.Tensor,            # (N, 2, 3)
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    alive          = 1.0 - reset_terminated.float()
    rew_alive      = rew_scale_alive      * alive
    rew_terminated = rew_scale_terminated * reset_terminated.float()
    rew_up         = rew_scale_upright    * rew_upright(quat_w)
    rew_height_r   = rew_scale_height     * z_height
    rew_lin_vel    = rew_scale_lin_vel    * rew_lin_vel_penalty(root_lin_vel_w)
    rew_jvel       = rew_scale_joint_vel  * rew_joint_vel_penalty(joint_vel)
    rew_torque_r   = rew_scale_torque     * rew_torque_penalty(torques)
    rew_pose_r     = rew_scale_target_standing_pose * rew_pose(joint_pos, target_standing_pose)
    rew_toe_r      = rew_scale_toe_walking * rew_toe_walking(calf_forces, toe_forces)

    # Per-joint-group pose error for diagnostics
    hip_yaw_pose_error  = torch.abs(joint_pos[:, 0:2] - target_standing_pose[:, 0:2]).mean(dim=-1)
    hip_roll_pose_error = torch.abs(joint_pos[:, 2:4] - target_standing_pose[:, 2:4]).mean(dim=-1)
    knee_pose_error     = torch.abs(joint_pos[:, 4:6] - target_standing_pose[:, 4:6]).mean(dim=-1)

    total_reward = (
        rew_alive + rew_terminated + rew_up + rew_lin_vel
        + rew_jvel + rew_torque_r + rew_pose_r + rew_height_r + rew_toe_r
    )

    components: dict[str, torch.Tensor] = {
        "alive":                rew_alive,
        "termination":          rew_terminated,
        "upright":              rew_up,
        "lin_vel":              rew_lin_vel,
        "joint_vel":            rew_jvel,
        "torque":               rew_torque_r,
        "target_standing_pose": rew_pose_r,
        "hip_yaw_pose_error":   hip_yaw_pose_error,
        "hip_roll_pose_error":  hip_roll_pose_error,
        "knee_pose_error":      knee_pose_error,
        "height":               rew_height_r,
        "toe_walking":          rew_toe_r,
    }

    return total_reward, components


# =============================================================================
# WALKING REWARD
# =============================================================================

@torch.jit.script
def compute_walking_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_upright: float,
    rew_scale_target_standing_pose: float,
    rew_scale_joint_vel: float,
    rew_scale_torque: float,
    rew_scale_lin_vel: float,
    rew_scale_height: float,
    rew_scale_toe_walking: float,
    quat_w: torch.Tensor,                # (N, 4)
    root_lin_vel_w: torch.Tensor,        # (N, 3)
    joint_pos: torch.Tensor,             # (N, J)
    joint_vel: torch.Tensor,             # (N, J)
    actions: torch.Tensor,               # (N, J)
    torques: torch.Tensor,               # (N, J)
    reset_terminated: torch.Tensor,      # (N,)
    target_standing_pose: torch.Tensor,  # (1 or N, J)
    z_height: torch.Tensor,              # (N,)
    calf_forces: torch.Tensor,           # (N, 2, 3)
    toe_forces: torch.Tensor,            # (N, 2, 3)
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    alive          = 1.0 - reset_terminated.float()
    rew_alive      = rew_scale_alive      * alive
    rew_terminated = rew_scale_terminated * reset_terminated.float()
    rew_up         = rew_scale_upright    * rew_upright(quat_w)
    rew_height_r   = rew_scale_height     * z_height
    rew_jvel       = rew_scale_joint_vel  * rew_joint_vel_penalty(joint_vel)
    rew_torque_r   = rew_scale_torque     * rew_torque_penalty(torques)
    rew_pose_r     = rew_scale_target_standing_pose * rew_pose(joint_pos, target_standing_pose)
    rew_toe_r      = rew_scale_toe_walking * rew_toe_walking(calf_forces, toe_forces)

    # Forward reward + lateral penalty (walking-specific lin_vel logic)
    forward_vel  = root_lin_vel_w[:, 0]
    lateral_vel  = root_lin_vel_w[:, 1]
    rew_lin_vel  = rew_scale_lin_vel * forward_vel - 0.5 * torch.square(lateral_vel)

    total_reward = (
        rew_alive + rew_terminated + rew_up + rew_lin_vel
        + rew_height_r + rew_pose_r + rew_jvel + rew_torque_r + rew_toe_r
    )

    components: dict[str, torch.Tensor] = {
        "alive":                rew_alive,
        "termination":          rew_terminated,
        "upright":              rew_up,
        "lin_vel":              rew_lin_vel,
        "forward_vel":          forward_vel,
        "lateral_vel":          lateral_vel,
        "height":               rew_height_r,
        "target_standing_pose": rew_pose_r,
        "joint_vel":            rew_jvel,
        "torque":               rew_torque_r,
        "toe_walking":          rew_toe_r,
    }

    return total_reward, components
