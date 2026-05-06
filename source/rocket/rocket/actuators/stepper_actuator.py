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
        stiffness: torch.Tensor | float,
        damping: torch.Tensor | float,
        armature: torch.Tensor | float,
        friction: torch.Tensor | float,
        limits: torch.Tensor | None,
    ):
        super().__init__(cfg, joint_names, joint_ids, num_envs, device,
                         stiffness, damping, armature, friction, limits)

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

        # Constant rated torque at output shaft while stepping (Nm)
        # This is motor holding torque × gear_ratio — does NOT scale with error.
        self._rated_torque = self._params.output_torque_nm()

        # SEA stiffness/damping — physical joint compliance, NOT PD gains.
        # stiffness: spring between motor output and joint (Nm/rad)
        # damping:   damping in that spring / back-EMF (Nm·s/rad)
        self._stiffness = cfg.stiffness
        self._damping = cfg.damping

        # Deadband: position errors below this are treated as "at target" (rad)
        self._deadband = cfg.deadband_rad

        # MKS firmware updates the acceleration ramp every 10ms (100Hz).
        # We track accumulated time to fire acc updates at the correct hardware rate
        # rather than every physics sub-step (200Hz), matching real behaviour.
        self._acc_update_dt: float = 1.0 / cfg.acc_update_hz
        self._acc_time_accum = torch.zeros(num_envs, self.num_joints, device=device)

        # Internal state: tracked commanded velocity per (env, joint)
        # This implements the hardware velocity ramp across physics sub-steps.
        self._cmd_vel = torch.zeros(num_envs, self.num_joints, device=device)

        print(f"[StepperActuator] joints={joint_names}")
        print(f"  {self._params.summary()}")
        print(f"  SEA stiffness={self._stiffness} Nm/rad  damping={self._damping} Nm·s/rad")

    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset internal velocity and acc accumulator for terminated/reset environments."""
        self._cmd_vel[env_ids] = 0.0
        self._acc_time_accum[env_ids] = 0.0

    # ------------------------------------------------------------------

    def compute(
        self,
        control_action,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> None:
        """
        Compute effort to apply each physics sub-step.

        Called at physics_dt (200 Hz / 0.005s), not at policy rate (10 Hz).
        The velocity ramp is applied continuously every sub-step, scaled by dt,
        which correctly approximates the hardware's 10ms update loop.
        """

        # Physics sub-step dt (0.005s at 200 Hz)
        dt: float = sim_utils.SimulationContext.instance().get_physics_dt()

        target_pos = control_action.joint_positions       # (num_envs, num_joints)
        error = target_pos - joint_pos                    # positive = need to move forward

        # Direction the motor needs to turn to reach target
        direction = torch.sign(error)

        # Are we close enough to consider "at target"?
        at_target = error.abs() < self._deadband

        # ----------------------------------------------------------------
        # Velocity ramp — trapezoidal profile matching MKS spec section 6.1
        # ----------------------------------------------------------------

        # Stopping distance: how far the joint travels to decelerate from
        # current speed to 0 at the configured acc rate.
        # Physics: d = v² / (2 × a)
        stopping_dist = (self._cmd_vel.abs() ** 2) / (2.0 * self._acc_rad_s2 + 1e-8)

        # Decelerate if:
        #   (a) close enough to stop exactly at target, OR
        #   (b) target is now in the opposite direction (need to stop before reversing)
        wrong_direction = (~at_target) & (self._cmd_vel.abs() > 1e-4) & (torch.sign(self._cmd_vel) != direction)
        needs_decel = (error.abs() <= stopping_dist) | wrong_direction

        # Desired commanded velocity:
        #   - at target       → 0 (stopped)
        #   - needs decel     → 0 (ramp down toward 0)
        #   - otherwise       → max_vel in direction of error (ramp up)
        target_cmd_vel = torch.where(
            at_target,
            torch.zeros_like(self._cmd_vel),
            torch.where(
                needs_decel,
                torch.zeros_like(self._cmd_vel),
                direction * self._max_vel,
            )
        )

        # Ramp _cmd_vel toward target_cmd_vel at the hardware acc rate (10ms / 100Hz).
        # Accumulate physics sub-step time; only fire an acc increment when 10ms has
        # elapsed, matching the MKS firmware's internal update loop exactly.
        self._acc_time_accum += dt
        acc_fired = self._acc_time_accum >= self._acc_update_dt
        acc_step = torch.where(
            acc_fired,
            torch.full_like(self._cmd_vel, self._acc_rad_s2 * self._acc_update_dt),
            torch.zeros_like(self._cmd_vel),
        )
        self._acc_time_accum = torch.where(
            acc_fired, torch.zeros_like(self._acc_time_accum), self._acc_time_accum
        )
        diff = target_cmd_vel - self._cmd_vel
        self._cmd_vel = (self._cmd_vel + diff.clamp(-acc_step, acc_step)).clamp(
            -self._max_vel, self._max_vel
        )

        # ----------------------------------------------------------------
        # Effort computation
        # ----------------------------------------------------------------

        moving = self._cmd_vel.abs() > 1e-4

        # Stepping torque: constant rated torque while the motor is moving.
        # This is the fundamental stepper behaviour — torque does NOT scale
        # with position error, unlike a PD controller.
        stepping_effort = torch.where(
            moving,
            torch.sign(self._cmd_vel) * self._rated_torque,
            torch.zeros_like(self._cmd_vel),
        )

        # SEA spring: physical compliance between motor output and joint.
        # Always active — models the joint's spring-like resistance to displacement.
        # stiffness=80-100 Nm/rad represents the drivetrain compliance, NOT kp.
        sea_spring = self._stiffness * error

        # SEA damping: physical damping in the compliant element + back-EMF.
        # Always active — resists velocity through the joint spring.
        # damping=0.01 Nm·s/rad is very light, representing minimal back-EMF.
        sea_damping = -self._damping * joint_vel

        # Total effort = stepping + compliance, clamped to hardware torque limit
        self.computed_effort = (stepping_effort + sea_spring + sea_damping).clamp(
            -self.effort_limit, self.effort_limit
        )
        self.applied_effort = self.computed_effort.clone()


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

    # --- SEA compliance parameters (physical, NOT control gains) ---
    # stiffness: spring stiffness of joint compliance (Nm/rad)
    #   Models the physical spring-like resistance of the drivetrain.
    #   Higher → stiffer joint, less flex between motor and load.
    #   Recommended: 80-100 Nm/rad for a rigid-ish stepper drivetrain.
    stiffness: float = 90.0     # Nm/rad — SEA spring (NOT kp)

    # damping: damping coefficient of joint compliance (Nm·s/rad)
    #   Models back-EMF and physical damping in the drivetrain spring.
    #   Very small for steppers — back-EMF is minimal.
    #   Recommended: 0.01 Nm·s/rad
    damping: float = 0.01       # Nm·s/rad — back-EMF + compliance damping (NOT kd)

    # --- Isaac Lab required fields ---
    effort_limit: float = MISSING    # Set per joint group (hip vs knee)
    velocity_limit: float = MISSING  # Set per joint group; should match speed_to_joint_rad_s()

    # Deadband: errors below this are treated as "at target", motor holds position
    deadband_rad: float = 0.002      # ~0.1 degrees

    # Rate at which the MKS firmware applies the acc increment (from spec: every 10ms)
    acc_update_hz: float = 100.0     # Hz — do not change unless firmware changes
