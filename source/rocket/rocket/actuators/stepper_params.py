"""
StepperParams: converts MKS SERVO57C hardware register values to SI units.

The MKS SERVO57C uses dimensionless integers for speed (0-1600) and
acceleration (0-32) that are sent over RS485. This class converts those
hardware values into physical units (rad/s, rad/s²) at the OUTPUT SHAFT
(after the gearbox), which is what Isaac Lab physics needs.

Conversion chain:
  speed register  →  motor shaft RPM  →  joint RPM  →  joint rad/s
  acc register    →  speed units/s    →  motor RPM/s →  joint rad/s²

All "joint" values are at the output shaft after gear_ratio reduction.
"""

import math
from dataclasses import dataclass


@dataclass
class StepperParams:
    """
    Hardware parameters for one MKS SERVO57C stepper + gearbox assembly.

    Fields match exactly what you set/send on the real hardware:
      mstep       — microstepping set in motor menu (default 1 = full step)
      speed       — RS485 speed register value (0-1600, dimensionless)
      acc         — RS485 acceleration register value (0-32, dimensionless)
      gear_ratio  — physical gearbox reduction (e.g. 5.0 means 5:1)
      full_steps_per_rev — always 200 for a standard NEMA stepper
      rated_torque_nm    — motor shaft holding torque from motor datasheet (Nm)
    """

    mstep: int = 1                  # Microstepping: 1,2,4,8,16,32,64,128,256
    speed: int = 10                 # Speed register sent over RS485 (0-1600)
    acc: int = 2                    # Acceleration register sent over RS485 (0-32)
    gear_ratio: float = 5.0         # Gearbox output ratio (motor turns / joint turn)
    full_steps_per_rev: int = 200   # NEMA standard: 200 full steps = 1 motor revolution
    rated_torque_nm: float = 1.5    # Motor shaft rated torque (Nm) — NEMA 23 typical

    # -------------------------------------------------------------------------
    # SPEED CONVERSIONS
    # From MKS spec section 6.1: Vrpm = (speed × 6000) / (Mstep × 200)
    # This gives MOTOR shaft RPM. We then divide by gear_ratio for joint RPM,
    # then convert RPM → rad/s via × 2π / 60.
    # -------------------------------------------------------------------------

    def speed_to_motor_rpm(self) -> float:
        """
        RS485 speed register → motor shaft RPM (before gearbox).

        Formula from MKS spec section 6.1:
            Vrpm = (speed × 6000) / (Mstep × 200)

        Example: speed=10, mstep=1 → (10 × 6000) / (1 × 200) = 300 RPM
        """
        return (self.speed * 6000.0) / (self.mstep * self.full_steps_per_rev)

    def speed_to_joint_rpm(self) -> float:
        """
        RS485 speed register → joint output RPM (after gearbox).

        Gearbox divides speed by gear_ratio (and multiplies torque).
        Example: 300 RPM motor / 5 gear = 60 RPM at joint
        """
        return self.speed_to_motor_rpm() / self.gear_ratio

    def speed_to_joint_rad_s(self) -> float:
        """
        RS485 speed register → joint angular velocity in rad/s.

        This is the value Isaac Lab physics uses for velocity limits.
        Example: 60 RPM × 2π / 60 = 6.28 rad/s
        """
        return self.speed_to_joint_rpm() * 2.0 * math.pi / 60.0

    def joint_rad_s_to_speed(self, rad_s: float) -> int:
        """
        Joint rad/s → RS485 speed register value (reverse conversion).

        Useful for choosing a speed register value to match a desired
        real-world joint velocity.
        """
        rpm_joint = rad_s * 60.0 / (2.0 * math.pi)
        rpm_motor = rpm_joint * self.gear_ratio
        speed = (rpm_motor * self.mstep * self.full_steps_per_rev) / 6000.0
        return int(round(speed))

    # -------------------------------------------------------------------------
    # ACCELERATION CONVERSIONS
    # From MKS spec section 6.1: the firmware updates speed every 10ms,
    # adding `acc` speed units per update.
    #   → 100 updates/sec × acc speed_units = acc × 100 speed_units/sec
    # Then convert speed_units/sec → motor RPM/sec → joint RPM/sec → rad/s²
    # -------------------------------------------------------------------------

    def acc_to_speed_units_per_s(self) -> float:
        """
        RS485 acc register → speed units added per second.

        The MKS firmware runs its acceleration loop at 100 Hz (every 10ms),
        adding `acc` speed register units each time.
        Example: acc=2 → 2 × 100 = 200 speed units/sec
        """
        return self.acc * 100.0

    def acc_to_motor_rpm_per_s(self) -> float:
        """
        RS485 acc register → motor shaft acceleration (RPM/s).

        Converts speed_units/sec → RPM/sec using the same scale factor
        as the speed conversion: RPM = speed_units × 6000 / (Mstep × 200)
        Example: 200 speed_units/s × 6000/(1×200) = 6000 RPM/s
        """
        return self.acc_to_speed_units_per_s() * 6000.0 / (self.mstep * self.full_steps_per_rev)

    def acc_to_joint_rad_s2(self) -> float:
        """
        RS485 acc register → joint angular acceleration (rad/s²).

        This is the value used in the Isaac Lab actuator's velocity ramp.
        Chain: acc_reg → motor RPM/s → joint RPM/s → joint rad/s²
        Example: acc=2, mstep=1, gear=5
            → 6000 RPM/s motor → 1200 RPM/s joint → 125.7 rad/s²
        """
        return self.acc_to_motor_rpm_per_s() / self.gear_ratio * 2.0 * math.pi / 60.0

    def joint_rad_s2_to_acc(self, rad_s2: float) -> int:
        """
        Joint rad/s² → RS485 acc register value (reverse conversion).

        Clamps to valid range 0-32.
        """
        rpm_s_joint = rad_s2 * 60.0 / (2.0 * math.pi)
        rpm_s_motor = rpm_s_joint * self.gear_ratio
        speed_units_per_s = rpm_s_motor * self.mstep * self.full_steps_per_rev / 6000.0
        acc = speed_units_per_s / 100.0
        return int(max(0, min(32, round(acc))))

    # -------------------------------------------------------------------------
    # TORQUE
    # -------------------------------------------------------------------------

    def output_torque_nm(self) -> float:
        """
        Rated torque at the joint output shaft (Nm).

        Gearbox amplifies torque by gear_ratio (opposite of speed reduction).
        Example: 1.5 Nm motor × 5 gear = 7.5 Nm at joint
        """
        return self.rated_torque_nm * self.gear_ratio

    def summary(self) -> str:
        return (
            f"StepperParams(mstep={self.mstep}, speed={self.speed}, acc={self.acc}, gear={self.gear_ratio})\n"
            f"  max joint velocity : {self.speed_to_joint_rad_s():.4f} rad/s  "
            f"  (motor: {self.speed_to_motor_rpm():.1f} RPM)\n"
            f"  max joint accel    : {self.acc_to_joint_rad_s2():.4f} rad/s²\n"
            f"  output torque      : {self.output_torque_nm():.4f} Nm"
        )
