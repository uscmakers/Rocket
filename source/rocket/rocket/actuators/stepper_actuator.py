"""Stepper motor actuator model for Isaac Lab.

Models MKS SERVO57C steppers driven via RS485 CR_UART position mode (function 0xFD).
Real hardware sends incremental pulse commands each control step at a fixed slow speed.
This model approximates that behavior as constant-velocity slew + stiffness hold.

velocity_limit should be set to the effective output-shaft rad/s derived from hardware:
  PULSES_PER_REV=200, GEAR_RATIO=5 → 1 pulse = 2π/1000 ≈ 0.00628 rad
  Hip cap: 10 pulses/step; Knee cap: 20 pulses/step.
  Effective vel = pulses_cap * rad_per_pulse * control_Hz.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.actuators.actuator_base import ActuatorBase
from isaaclab.actuators.actuator_base_cfg import ActuatorBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


@configclass
class StepperActuatorCfg(ActuatorBaseCfg):
    """Configuration for stepper motor actuator."""

    class_type: type = None  # set below after class definition

    velocity_gain: float = 5.0
    """Gain applied to velocity error to compute torque during slew phase (Nm / (rad/s))."""

    position_deadband: float = 0.005
    """Position error (rad) below which the motor switches from slew to hold mode."""


class StepperActuator(ActuatorBase):
    """Explicit stepper motor actuator.

    Two-phase model:
      - **Slew**: |pos_error| > deadband → command constant velocity toward target,
        torque = velocity_gain × (desired_vel - actual_vel)
      - **Hold**: |pos_error| <= deadband → torque = stiffness × pos_error
        (detent-style position hold)

    All torques are clipped to effort_limit.
    """

    cfg: StepperActuatorCfg

    def reset(self, env_ids: Sequence[int]):
        pass  # no internal state to reset

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        pos_error = control_action.joint_positions - joint_pos  # (N, J)

        # --- slew phase: move at constant velocity toward target ---
        desired_vel = torch.sign(pos_error) * self.velocity_limit  # (N, J)
        torque = self.cfg.velocity_gain * (desired_vel - joint_vel)

        # --- hold phase: within deadband, switch to stiffness hold ---
        at_target = pos_error.abs() <= self.cfg.position_deadband
        torque[at_target] = (self.stiffness * pos_error)[at_target]

        self.computed_effort = torque
        self.applied_effort = self._clip_effort(torque)

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


StepperActuatorCfg.class_type = StepperActuator
