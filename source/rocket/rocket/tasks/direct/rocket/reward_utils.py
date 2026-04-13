# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward component functions for Rocket.

All functions are @torch.jit.script for GPU performance.
They return unscaled values — callers (RewardScalesCfg.compute) apply the scale.
"""

from __future__ import annotations
from typing import Tuple

import torch


# =============================================================================
# ORIENTATION
# =============================================================================

@torch.jit.script
def _quat_apply(quat_w: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by quaternion (w, x, y, z)."""
    qw = quat_w[:, 0:1]
    qv = quat_w[:, 1:4]
    t = 2.0 * torch.cross(qv, vec, dim=-1)
    return vec + qw * t + torch.cross(qv, t, dim=-1)


@torch.jit.script
def _quat_apply_inverse(quat_w: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse (conjugate) of quaternion (w, x, y, z)."""
    conj = torch.cat((quat_w[:, 0:1], -quat_w[:, 1:4]), dim=-1)
    return _quat_apply(conj, vec)


@torch.jit.script
def rew_upright(quat_w: torch.Tensor) -> torch.Tensor:
    """exp(-distance) where distance is how far body z-axis is from world up.

    Args:
        quat_w: (N, 4) quaternion in (w, x, y, z) format.
    Returns:
        (N,) in [0, 1]; 1.0 = perfectly upright.
    """
    qw = quat_w[:, 0]; qx = quat_w[:, 1]; qy = quat_w[:, 2]; qz = quat_w[:, 3]
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
def rew_flat_orientation_l2(quat_w: torch.Tensor) -> torch.Tensor:
    """Isaac Lab MDP-style 'flat_orientation_l2' penalty (fallback from quat).

    Prefer using `rew_flat_orientation_l2_from_projected_gravity_b` if you already have
    `asset.data.projected_gravity_b` available (matches MDP exactly).
    """
    n = quat_w.shape[0]
    g_w = torch.tensor([0.0, 0.0, -1.0], device=quat_w.device, dtype=quat_w.dtype).expand(n, 3)
    projected_gravity_b = _quat_apply_inverse(quat_w, g_w)
    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=-1)


@torch.jit.script
def rew_flat_orientation_l2_from_projected_gravity_b(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    """Isaac Lab MDP-style 'flat_orientation_l2' penalty.

    This matches the upstream computation exactly:
        sum(square(asset.data.projected_gravity_b[:, :2]))
    """
    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=-1)


# =============================================================================
# VELOCITY
# =============================================================================

@torch.jit.script
def rew_heading_vel(
    quat_w: torch.Tensor,         # (N, 4) in (w, x, y, z)
    root_lin_vel_w: torch.Tensor, # (N, 3)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose root velocity into the robot's heading frame.

    Projects body +x to the horizontal plane and renormalizes so pitch
    doesn't bleed into the forward signal.

    Returns:
        forward_vel: (N,) velocity along the robot's facing direction (signed)
        lateral_vel: (N,) velocity perpendicular to heading (left = positive)
    """
    qw = quat_w[:, 0]; qx = quat_w[:, 1]; qy = quat_w[:, 2]; qz = quat_w[:, 3]

    fwd_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    fwd_y = 2.0 * (qx * qy + qw * qz)
    fwd_norm = torch.sqrt(fwd_x * fwd_x + fwd_y * fwd_y).clamp(min=1e-6)
    fwd_x = fwd_x / fwd_norm
    fwd_y = fwd_y / fwd_norm

    vel_x = root_lin_vel_w[:, 0]
    vel_y = root_lin_vel_w[:, 1]

    forward_vel = vel_x * fwd_x + vel_y * fwd_y
    lateral_vel = -vel_x * fwd_y + vel_y * fwd_x

    return forward_vel, lateral_vel


@torch.jit.script
def rew_vertical_vel_penalty(root_lin_vel_w: torch.Tensor) -> torch.Tensor:
    """Squared vertical (z) velocity. Returns (N,); caller applies negative scale."""
    return torch.square(root_lin_vel_w[:, 2])


# =============================================================================
# CONTACT
# =============================================================================

@torch.jit.script
def rew_toe_walking(
    calf_forces: torch.Tensor,  # (N, 2, 3)
    toe_forces: torch.Tensor,   # (N, 2, 3)  # kept for API consistency
) -> torch.Tensor:
    """Penalty for calf ground contact.

    Returns (N,) <= 0; 0 = calf off ground (good), negative = calf contact (bad).
    Toe contact and airborne both return 0 — no conflict with alternating_contact.
    Caller applies a positive scale so this acts as a penalty in the total reward.
    """
    calf_mag = torch.norm(calf_forces, dim=-1)  # (N, 2)
    return -calf_mag.sum(dim=-1)                # (N,) always <= 0


@torch.jit.script
def rew_alternating_contact(
    toe_forces: torch.Tensor,  # (N, 2, 3)
) -> torch.Tensor:
    """Continuous force-imbalance reward for alternating foot contact.

    Returns (N,) in [0, 1]; 1.0 = all weight on one foot, 0.0 = equal weight.
    Normalized by total force so gradient signal is smooth throughout.
    """
    toe_mag = torch.norm(toe_forces, dim=-1)            # (N, 2)
    total   = toe_mag.sum(dim=-1).clamp(min=1e-6)       # (N,)
    return torch.abs(toe_mag[:, 0] - toe_mag[:, 1]) / total


# =============================================================================
# JOINT / ACTUATOR
# =============================================================================

@torch.jit.script
def rew_joint_vel_penalty(joint_vel: torch.Tensor) -> torch.Tensor:
    """Sum of absolute joint velocities. Returns (N,); caller applies negative scale."""
    return torch.sum(torch.abs(joint_vel), dim=-1)


@torch.jit.script
def rew_torque_penalty(torques: torch.Tensor) -> torch.Tensor:
    """Sum of squared torques. Returns (N,); caller applies negative scale."""
    return torch.sum(torch.square(torques), dim=-1)


@torch.jit.script
def rew_pose(
    joint_pos: torch.Tensor,            # (N, J)
    target_standing_pose: torch.Tensor, # (1 or N, J)
) -> torch.Tensor:
    """exp(-RMS pose error). Returns (N,) in (0, 1]; 1.0 = at target pose."""
    pose_error = torch.sqrt(torch.sum(torch.square(joint_pos - target_standing_pose), dim=-1))
    return torch.exp(-pose_error)


# =============================================================================
# SMOOTHNESS
# =============================================================================

@torch.jit.script
def rew_action_rate_penalty(
    actions: torch.Tensor,       # (N, J)
    prev_actions: torch.Tensor,  # (N, J)
) -> torch.Tensor:
    """Sum of squared action deltas. Returns (N,); caller applies negative scale."""
    return torch.sum(torch.square(actions - prev_actions), dim=-1)


@torch.jit.script
def rew_jerk_penalty(
    actions: torch.Tensor,            # (N, J) current
    prev_actions: torch.Tensor,       # (N, J) t-1
    prev_prev_actions: torch.Tensor,  # (N, J) t-2
) -> torch.Tensor:
    """Discrete jerk: sum of squared second-order action differences.

    Jerk = d²a/dt² ≈ a_t - 2*a_{t-1} + a_{t-2}.
    Returns (N,); caller applies negative scale.
    """
    return torch.sum(torch.square(actions - 2.0 * prev_actions + prev_prev_actions), dim=-1)


# =============================================================================
# DEBUG (not jit — uses print)
# =============================================================================

def rew_toe_walking_debug(
    calf_forces: torch.Tensor,  # (N, 2, 3)
    toe_forces: torch.Tensor,   # (N, 2, 3)
) -> None:
    """Print per-leg calf and toe force magnitudes for the first environment."""
    calf_mag = torch.norm(calf_forces, dim=-1)
    toe_mag  = torch.norm(toe_forces,  dim=-1)
    print(f"[toe_walking_debug] calf_mag (L, R): {calf_mag[0].tolist()}")
    print(f"[toe_walking_debug]  toe_mag (L, R): {toe_mag[0].tolist()}")
