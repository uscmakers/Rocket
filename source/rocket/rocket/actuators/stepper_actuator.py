"""Stepper motor actuator model for Isaac Lab.

Models MKS SERVO57C steppers driven via RS485 CR_UART position mode (function 0xFD).
Real hardware sends incremental pulse commands each control step at a fixed slow speed.
This model approximates that behavior as constant-velocity slew + stiffness hold.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.actuators.actuator_base import ActuatorBase
from isaaclab.actuators.actuator_base_cfg import ActuatorBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class StepperActuator(ActuatorBase):
    """Explicit stepper motor actuator.

    Two-phase model:
      - **Slew**: |pos_error| > deadband → torque = velocity_gain × (desired_vel - actual_vel)
      - **Hold**: |pos_error| <= deadband → torque = stiffness × pos_error

    All torques are clipped to effort_limit.
    """

    cfg: "StepperActuatorCfg"

    def reset(self, env_ids: Sequence[int]):
        pass

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        pos_error = control_action.joint_positions - joint_pos

        desired_vel = torch.sign(pos_error) * self.velocity_limit
        torque = self.cfg.velocity_gain * (desired_vel - joint_vel)

        at_target = pos_error.abs() <= self.cfg.position_deadband
        torque[at_target] = (self.stiffness * pos_error)[at_target]

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
    """Gain applied to velocity error to compute torque during slew phase (Nm / (rad/s))."""

    position_deadband: float = 0.005
    """Position error (rad) below which the motor switches from slew to hold mode."""
