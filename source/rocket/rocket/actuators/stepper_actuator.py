"""
StepperActuator: Isaac Lab actuator modelling MKS SERVO57C stepper motor.

Physical model
--------------
This models a stepper motor driving a joint through a gearbox, where the
compliance (stiffness/damping) is treated as SERIES ELASTIC (SEA-style):
a physical spring between the motor output and the joint load. This means
stiffness and damping are NOT PD control gains — they are physical properties
of the joint's compliance (spring + damping in the drivetrain).

Velocity profile (follows MKS spec section 6.1):
  - Motor accelerates from 0 → max_vel at rate acc_rad_s2 (trapezoidal ramp).
  - Firmware updates velocity every 10ms; we apply continuously scaled by dt.
  - Motor decelerates back to 0 when within stopping distance of target.
  - Full stop at target — no velocity carry-over between commands (unlike PD).
  - If target direction reverses, decelerates to 0 first then re-accelerates.

Effort model:
  - Moving  : constant rated torque in motion direction (stepper = fixed torque).
  - Always  : SEA spring force  = stiffness × (target - joint_pos)  [compliance]
  - Always  : SEA damping force = -damping × joint_vel               [back-EMF / compliance damping]
  - Total effort clamped to effort_limit.
"""

from __future__ import annotations

import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg
from isaaclab.utils import configclass

from .stepper_params import StepperParams


class StepperActuator(ActuatorBase):
    """
    Isaac Lab actuator that models MKS SERVO57C stepper motor dynamics.

    Key differences from ImplicitActuatorCfg (PD):
      - Torque is CONSTANT while moving, not proportional to position error.
      - Full stop at target — velocity forced to 0, not carried over.
      - Velocity and acceleration are capped by hardware register values.
      - Stiffness/damping model SEA compliance, not internal control gains.
    """

    cfg: "StepperActuatorCfg"

    def __init__(
        self,
        cfg: "StepperActuatorCfg",
        joint_names: list[str],
        joint_ids: list[int],
        num_envs: int,
        device: str,
        stiffness: torch.Tensor | float = 0.0,
        damping: torch.Tensor | float = 0.0,
        armature: torch.Tensor | float = 0.0,
        friction: torch.Tensor | float = 0.0,
        **kwargs,
    ):
        # Isaac Lab versions vary in whether they pass `limits`; discard it.
        # The base class handles limits separately from the _parse_joint_parameter loop.
        kwargs.pop("limits", None)
        super().__init__(cfg, joint_names, joint_ids, num_envs, device,
                         stiffness, damping, armature, friction)

        # Convert hardware register values → SI units for the sim
        self._params = StepperParams(
            mstep=cfg.mstep,
            speed=cfg.speed,
            acc=cfg.acc,
            gear_ratio=cfg.gear_ratio,
            rated_torque_nm=cfg.rated_torque_nm,
        )

        # Max joint velocity (rad/s) derived from speed register + gear ratio
        self._max_vel = self._params.speed_to_joint_rad_s()

        # Joint acceleration (rad/s²) derived from acc register + gear ratio
        # The MKS firmware ramps speed 100 times/sec; we apply continuously.
        self._acc_rad_s2 = self._params.acc_to_joint_rad_s2()

        # SEA compliance — physical spring/damper between motor shaft and joint.
        # stiffness: spring stiffness (Nm/rad), referenced to motor_setpoint (not final target)
        # damping:   viscous damping (Nm·s/rad)
        self._stiffness = cfg.stiffness
        self._damping   = cfg.damping

        # Deadband: setpoint errors below this are treated as "at target" (rad)
        self._deadband = cfg.deadband_rad

        # _cmd_vel: velocity of the motor's internal setpoint (rad/s), ramped trapezoidally.
        self._cmd_vel = torch.zeros(num_envs, self.num_joints, device=device)

        # _motor_setpoint: the motor's current internal position reference (rad).
        # Integrates _cmd_vel each physics step — this is what the PD tracks against actual joint pos.
        self._motor_setpoint = torch.zeros(num_envs, self.num_joints, device=device)

        print(f"[StepperActuator] joints={joint_names}")
        print(f"  {self._params.summary()}")
        print(f"  SEA stiffness={self._stiffness} Nm/rad  damping={self._damping} Nm·s/rad")

    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor) -> None:
        self._cmd_vel[env_ids] = 0.0
        self._motor_setpoint[env_ids] = 0.0

    # ------------------------------------------------------------------

    def compute(
        self,
        control_action,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ):
        """
        Compute effort to apply each physics sub-step.

        Called at physics_dt (200 Hz / 0.005s), not at policy rate (10 Hz).
        The velocity ramp is applied continuously every sub-step, scaled by dt,
        which correctly approximates the hardware's 10ms update loop.
        """

        dt: float = sim_utils.SimulationContext.instance().get_physics_dt()

        target_pos = control_action.joint_positions  # (num_envs, num_joints)

        # ----------------------------------------------------------------
        # Trapezoidal velocity profile — drives the internal motor setpoint.
        # The profile generator works on setpoint distance, not actual joint
        # distance. This matches how the real firmware operates: it plans a
        # smooth trajectory independent of the actual encoder position.
        # ----------------------------------------------------------------

        setpoint_err = target_pos - self._motor_setpoint   # remaining setpoint travel
        direction     = torch.sign(setpoint_err)
        at_target     = setpoint_err.abs() < self._deadband

        # Stopping distance: v² / (2a) — how far the setpoint travels if we
        # start decelerating right now. When this equals remaining distance,
        # begin deceleration so the setpoint arrives at target with zero velocity.
        stopping_dist = (self._cmd_vel.abs() ** 2) / (2.0 * self._acc_rad_s2 + 1e-8)
        wrong_direction = (~at_target) & (self._cmd_vel.abs() > 1e-4) & (torch.sign(self._cmd_vel) != direction)
        needs_decel   = (setpoint_err.abs() <= stopping_dist) | wrong_direction

        target_cmd_vel = torch.where(
            at_target,
            torch.zeros_like(self._cmd_vel),
            torch.where(needs_decel, torch.zeros_like(self._cmd_vel), direction * self._max_vel),
        )

        # Acc ramp applied every physics substep (200Hz), scaled by dt.
        # Total acceleration per second is identical to 100Hz firing — smoother interpolation.
        acc_step = self._acc_rad_s2 * dt
        diff = target_cmd_vel - self._cmd_vel
        self._cmd_vel = (self._cmd_vel + diff.clamp(-acc_step, acc_step)).clamp(-self._max_vel, self._max_vel)

        # Advance motor setpoint along the trapezoidal profile.
        # Clamp so the setpoint never overshoots the target.
        self._motor_setpoint = (self._motor_setpoint + self._cmd_vel * dt).clamp(
            torch.minimum(target_pos, self._motor_setpoint),
            torch.maximum(target_pos, self._motor_setpoint),
        )

        # ----------------------------------------------------------------
        # SEA effort — spring between motor output shaft and joint.
        # The motor shaft follows the trapezoidal setpoint; the SEA spring
        # transmits force from shaft to joint. When the motor is moving,
        # the spring reference is the current shaft position (motor_setpoint).
        # When stopped at target, this naturally becomes the holding spring.
        # ----------------------------------------------------------------

        sea_spring  =  self._stiffness * (self._motor_setpoint - joint_pos)
        sea_damping = -self._damping   * joint_vel
        self.computed_effort = (sea_spring + sea_damping).clamp(
            -self.effort_limit, self.effort_limit
        )
        self.applied_effort = self.computed_effort.clone()

        # Isaac Lab expects compute() to return the control_action with efforts filled in.
        # Setting positions/velocities to None tells PhysX to use effort control only.
        control_action.joint_positions = None
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort
        return control_action


# ---------------------------------------------------------------------------
# Configuration dataclass — define AFTER StepperActuator so class_type works
# ---------------------------------------------------------------------------

@configclass
class StepperActuatorCfg(ActuatorBaseCfg):
    """
    Config for StepperActuator.

    Hardware registers (mstep, speed, acc) are the same values you set
    on the physical motor via the display menu or RS485 commands.
    The actuator converts them to SI units internally.

    stiffness / damping are SEA-style physical joint compliance parameters,
    NOT PD control gains. They model the spring and damping in the drivetrain
    between the motor output shaft and the joint load.
    """

    class_type: type = StepperActuator

    # --- RS485 hardware register values (dimensionless, match real motor) ---
    mstep: int = 1              # Microstepping setting on the motor (default: 1)
    speed: int = 10             # Speed register (0-1600) — set to match real motor
    acc: int = 2                # Acceleration register (0-32) — set to match real motor
    gear_ratio: float = 5.0     # Physical gearbox ratio
    rated_torque_nm: float = 1.5  # Motor shaft rated torque (Nm) from datasheet

    # --- SEA compliance (physical spring/damper between motor shaft and joint) ---
    stiffness: float = 80.0     # Nm/rad — drivetrain spring stiffness
    damping: float = 0.01        # Nm·s/rad — viscous damping (tune to suppress oscillation)
    armature: float = 7.5e-4    # rotor inertia reflected to joint: J_rotor(3e-5) × gear_ratio²(25)
    friction: float = 0.0       # joint friction (Nm)

    # --- Isaac Lab required fields ---
    effort_limit: float = MISSING    # Set per joint group (hip vs knee)
    velocity_limit: float = MISSING  # Set per joint group; should match speed_to_joint_rad_s()

    # Deadband: errors below this are treated as "at target", motor holds position
    deadband_rad: float = 0.002      # ~0.1 degrees

    # Rate at which the MKS firmware applies the acc increment (from spec: every 10ms)
    acc_update_hz: float = 100.0     # Hz — do not change unless firmware changes
