"""Stepper motor actuator model for Isaac Lab.

Models MKS SERVO57C steppers driven via RS485 CR_UART position mode (function 0xFD).

Improvements over a plain PD actuator:
  - Constant-velocity slew toward target (not spring-like)
  - Acceleration ramp via max_torque_rate (no instant torque snap)
  - Speed-dependent torque cap (hyperbolic falloff above corner_speed)
  - Coulomb friction opposing motion
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.actuators.actuator_base import ActuatorBase
from isaaclab.actuators.actuator_base_cfg import ActuatorBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class StepperActuator(ActuatorBase):
    """Explicit stepper motor actuator with hardware-realistic torque model.

    Compute phases:
      1. Slew: torque = velocity_gain × (sign(err) × vel_limit − joint_vel)
      2. Accel ramp: clamp torque delta to max_torque_rate per physics step
      3. Hold: within deadband → torque = stiffness × pos_error
      4. Speed-dependent cap: effort_limit / (1 + |vel| / corner_speed)
      5. Coulomb friction: subtract coulomb_friction × sign(vel) when moving
    """

    cfg: "StepperActuatorCfg"

    def __post_init__(self):
        super().__post_init__()
        self._prev_torque: torch.Tensor | None = None

    def reset(self, env_ids: Sequence[int]):
        if self._prev_torque is not None:
            self._prev_torque[env_ids] = 0.0

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        if self._prev_torque is None:
            self._prev_torque = torch.zeros_like(joint_pos)

        pos_error = control_action.joint_positions - joint_pos

        # --- slew: constant velocity toward target ---
        desired_vel = torch.sign(pos_error) * self.velocity_limit
        raw_torque = self.cfg.velocity_gain * (desired_vel - joint_vel)

        # --- acceleration ramp: limit torque rate per physics step ---
        delta = torch.clamp(
            raw_torque - self._prev_torque,
            -self.cfg.max_torque_rate,
            self.cfg.max_torque_rate,
        )
        torque = self._prev_torque + delta

        # --- hold: within deadband, stiffness spring ---
        at_target = pos_error.abs() <= self.cfg.position_deadband
        torque[at_target] = (self.stiffness * pos_error)[at_target]

        # --- speed-dependent torque cap (hyperbolic falloff) ---
        torque_cap = self.effort_limit / (1.0 + joint_vel.abs() / self.cfg.corner_speed)
        torque = torch.clamp(torque, -torque_cap, torque_cap)

        # --- Coulomb friction (opposes motion, dead zone at near-zero vel) ---
        if self.cfg.coulomb_friction > 0.0:
            moving = joint_vel.abs() > 0.01
            torque[moving] = torque[moving] - self.cfg.coulomb_friction * torch.sign(joint_vel[moving])

        self._prev_torque = torque.clone()
        self.computed_effort = torque
        self.applied_effort = self._clip_effort(torque)

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class StepperActuatorCfg(ActuatorBaseCfg):
    """Configuration for stepper motor actuator."""

    class_type: type = StepperActuator

    velocity_gain: float = 10.0
    """Gain on velocity error during slew phase (Nm / (rad/s))."""

    position_deadband: float = 0.005
    """Position error (rad) below which motor switches to hold mode."""

    max_torque_rate: float = float("inf")
    """Max torque change per physics step (Nm/step). inf = no ramp (instant).
    Fit from hardware current ramp measurements."""

    corner_speed: float = 8.0
    """Output shaft speed (rad/s) at which available torque drops to 50% of effort_limit.
    MKS SERVO57C at 48V: motor corner ~400 RPM / 5 gear = 80 RPM output = 8.4 rad/s."""

    coulomb_friction: float = 0.0
    """Constant friction torque (Nm) opposing joint motion. 0 = disabled.
    Measure via stall-torque ramp test on hardware."""
