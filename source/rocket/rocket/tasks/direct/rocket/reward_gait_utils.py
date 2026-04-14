from __future__ import annotations

from dataclasses import dataclass

import torch
from isaaclab.sensors import ContactSensor


@dataclass
class GaitSignals:
    """Contact-derived gait signals for a biped.

    All tensors are per-environment, per-foot with the last dim = 2 (L, R).
    """

    # Contact state
    in_contact: torch.Tensor            # (N, 2) bool
    # Timers (seconds)
    current_air_time: torch.Tensor      # (N, 2) float
    current_contact_time: torch.Tensor  # (N, 2) float

    # Convenience
    single_stance: torch.Tensor         # (N,) bool

    # Sliding (optional, MDP-style)
    contacts_force: torch.Tensor | None = None  # (N, 2) bool, force-threshold contact
    toe_vel_xy: torch.Tensor | None = None      # (N, 2, 2) float, toe linear velocity in world XY
    toe_pos_z: torch.Tensor | None = None       # (N, 2) float, toe height in world frame


def compute_gait_signals(
    toe_contact_sensor: ContactSensor,
    *,
    toe_vel_xy: torch.Tensor | None = None,
    toe_pos_z: torch.Tensor | None = None,
    slide_force_threshold: float = 1.0,
) -> GaitSignals:
    """Compute gait signals from a toe ContactSensor.

    This function is intentionally **not** TorchScript: it calls sensor helper
    methods and reads sensor-managed timers.

    Notes:
        - For full functionality, enable `ContactSensorCfg(track_air_time=True)`.
    """

    data = toe_contact_sensor.data

    # Isaac Lab MDP biped stepping reward relies on these timer buffers.
    # Keep this strict so behavior matches upstream (and fails loudly if misconfigured).
    missing: list[str] = []
    for attr in ("current_air_time", "current_contact_time"):
        if not hasattr(data, attr):
            missing.append(f"ContactSensor.data.{attr}")
    if missing:
        raise RuntimeError(
            "Toe ContactSensor is missing air-time tracking fields. "
            "Set ContactSensorCfg(track_air_time=True). Missing: " + ", ".join(missing)
        )

    current_air_time = data.current_air_time
    current_contact_time = data.current_contact_time
    in_contact = current_contact_time > 0.0

    single_stance = torch.sum(in_contact.to(torch.int32), dim=-1) == 1

    # MDP feet_slide uses a force-threshold contact mask based on net_forces history.
    contacts_force: torch.Tensor | None = None
    if hasattr(data, "net_forces_w_history"):
        # (N, H, B, 3) -> (N, B)
        contacts_force = (
            data.net_forces_w_history.norm(dim=-1).max(dim=1)[0] > slide_force_threshold
        )
    elif hasattr(data, "net_forces_w"):
        # (N, B, 3) -> (N, B)
        contacts_force = data.net_forces_w.norm(dim=-1) > slide_force_threshold

    return GaitSignals(
        in_contact=in_contact,
        current_air_time=current_air_time,
        current_contact_time=current_contact_time,
        single_stance=single_stance,
        contacts_force=contacts_force,
        toe_vel_xy=toe_vel_xy,
        toe_pos_z=toe_pos_z,
    )


@torch.jit.script
def rew_feet_air_time(
    first_contact: torch.Tensor,  # (N, 2) bool
    last_air_time: torch.Tensor,  # (N, 2) float
    threshold: float,
) -> torch.Tensor:
    """MDP-style feet air-time reward (touchdown event, no single-stance constraint).

    Intuition: reward a foot when it *just* touched down and it had been in the air
    longer than `threshold`.
    """
    # Matches Isaac Lab MDP: negative reward for "too short" steps (last_air_time < threshold).
    contact = first_contact.to(last_air_time.dtype)
    return torch.sum((last_air_time - threshold) * contact, dim=-1)


@torch.jit.script
def rew_feet_air_time_biped(
    in_contact: torch.Tensor,            # (N, 2) bool
    current_air_time: torch.Tensor,      # (N, 2) float
    current_contact_time: torch.Tensor,  # (N, 2) float
    threshold: float,
) -> torch.Tensor:
    """MDP-style biped air/contact-time reward with single-stance constraint.

    Rewards clean single-stance by paying up to `threshold` seconds of the
    *minimum* per-foot mode-time (contact time for the stance foot, air time for
    the swing foot) when exactly one foot is in contact.
    """
    single_stance = (torch.sum(in_contact.to(torch.int32), dim=-1) == 1)
    in_contact_f = in_contact.to(current_air_time.dtype)
    mode_time = in_contact_f * current_contact_time + (1.0 - in_contact_f) * current_air_time
    min_mode_time = torch.min(mode_time, dim=-1).values
    shaped = torch.clamp(min_mode_time, max=threshold)
    return shaped * single_stance.to(shaped.dtype)


@torch.jit.script
def rew_feet_slide(
    contacts_force: torch.Tensor,  # (N, 2) bool
    toe_vel_xy: torch.Tensor,      # (N, 2, 2) float
) -> torch.Tensor:
    """MDP-style feet sliding penalty.

    Matches Isaac Lab MDP: sum(||v_xy|| * contact_mask) over feet.
    """
    contacts_f = contacts_force.to(toe_vel_xy.dtype)
    speed = torch.norm(toe_vel_xy, dim=-1)  # (N, 2)
    return torch.sum(speed * contacts_f, dim=-1)


@torch.jit.script
def rew_toe_clearance_biped(
    in_contact: torch.Tensor,  # (N, 2) bool
    toe_pos_z: torch.Tensor,   # (N, 2) float
    height_threshold: float,
) -> torch.Tensor:
    """Reward swing-toe clearance during single-stance (biped).

    Pays only when exactly one foot is in contact. Rewards the *swing* toe being
    above `height_threshold` (meters), normalized to [0, 1] by the threshold
    for easy scaling:
        clearance = clamp(relu(z - h), max=h) / h

    Returns:
        (N,) in [0, 1]
    """
    single_stance = (torch.sum(in_contact.to(torch.int32), dim=-1) == 1)  # (N,)
    swing = (~in_contact)  # (N, 2)
    # Only consider swing toe(s) (in single-stance there should be exactly one).
    z_swing = torch.where(swing, toe_pos_z, torch.zeros_like(toe_pos_z))
    h = max(height_threshold, 1e-6)
    clearance = torch.clamp(torch.relu(z_swing - h), max=h) / h  # (N, 2) in [0, 1]
    reward = torch.max(clearance, dim=-1).values  # pick the swing toe
    return reward * single_stance.to(reward.dtype)
