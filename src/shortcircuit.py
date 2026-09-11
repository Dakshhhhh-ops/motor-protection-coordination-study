"""
Three-phase short-circuit calculation to IEC 60909-0:2016.

Method
------
The equivalent voltage source at the fault location:

    I_k'' = c * U_n / (sqrt(3) * |Z_k|)                        [clause 6.2]
    i_p   = kappa * sqrt(2) * I_k''                            [clause 8]
    kappa = 1.02 + 0.98 * exp(-3 R/X)

Two distinct duties are computed, and the difference between them is not
cosmetic -- they are used for opposite purposes:

  MAXIMUM   c = 1.05, conductors at 20 degC, motor contribution INCLUDED.
            Sizes equipment: breaking capacity, busbar withstand, peak
            electrodynamic stress.

  MINIMUM   c = 0.95, conductors at 90 degC, motor contribution EXCLUDED
            (the motors may be stopped when the fault occurs).
            Verifies protection SENSITIVITY: the instantaneous element must
            still operate on the weakest credible fault.

Two modelling decisions worth defending
---------------------------------------
1. Transformer impedance correction factor K_T (clause 6.3.3):

       K_T = 0.95 * c_max / (1 + 0.6 * x_T)

   applied to the transformer impedance. This accounts for the fact that the
   nameplate impedance is a rated-condition quantity while the fault occurs at
   the actual operating flux and tap. For this unit K_T = 0.969, which raises
   the bus fault current by about 3 per cent. Omitting it would be
   non-conservative for equipment rating.

2. For a fault at motor k's terminals, motor k's own back-feed does NOT flow
   through motor k's protective device -- it flows from the machine into the
   fault, on the load side of the device. Only the network and the OTHER
   motors contribute to the current the device measures. Including the motor's
   own contribution here would overstate the current available to operate its
   instantaneous element, which is exactly the wrong direction for a
   sensitivity check. The total current at the fault point, including the
   local machine, is reported separately because that is what the cable and
   the arc energy see.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from constants import (
    C_MAX_LV,
    C_MIN_LV,
    MOTOR_R_OVER_X_LV,
    SQRT3,
    TEMP_COLD_C,
    TEMP_XLPE_OPERATING_C,
    kappa,
)
from models import Motor, Plant, parallel


# ===========================================================================
# Motor short-circuit impedance
# ===========================================================================


def motor_impedance(motor: Motor) -> complex:
    """Subtransient short-circuit impedance of an asynchronous motor.

    IEC 60909-0 clause 6.6:

        Z_M = U_rM^2 / (I_LR/I_rM * S_rM)      with  S_rM = P_shaft/(eta*pf)

    i.e. the machine looks like a source behind an impedance that would draw
    exactly its locked-rotor current at rated voltage. For LV motors the
    standard fixes the angle with R_M/X_M = 0.42, hence

        X_M = Z_M / sqrt(1 + 0.42^2),   R_M = 0.42 * X_M

    which corresponds to kappa_M = 1.3.
    """
    s_va = motor.s_rated_kva * 1000.0
    z_mag = (motor.voltage_v**2) / (motor.lrc_multiple * s_va)
    x = z_mag / math.sqrt(1.0 + MOTOR_R_OVER_X_LV**2)
    r = MOTOR_R_OVER_X_LV * x
    return complex(r, x)


def motor_branch_impedance(motor: Motor, temp_c: float) -> complex:
    """Motor impedance plus its supply cable, as seen from the bus.

    IEC 60909-0 permits lumping LV motors at the bus, but the cable data is
    available so it is included: it reduces the motor contribution at the bus
    and is the more accurate (and for a sensitivity check, the safer) model.
    """
    return motor_impedance(motor) + motor.cable.z(temp_c)


def transformer_correction_factor(plant: Plant, c_max: float = C_MAX_LV) -> float:
    """K_T per IEC 60909-0 clause 6.3.3:  K_T = 0.95 c_max / (1 + 0.6 x_T)."""
    x_t_pu = plant.transformer.x_pct / 100.0
    return 0.95 * c_max / (1.0 + 0.6 * x_t_pu)


# ===========================================================================
# Results
# ===========================================================================


@dataclass(frozen=True)
class FaultResult:
    """One three-phase fault calculation."""

    location: str
    ik_a: float
    ip_a: float
    kappa: float
    z: complex
    c: float
    temp_c: float
    includes_motors: bool
    #: Current supplied through the protective device at this location.
    #: Equals ik_a everywhere except a motor terminal, where the local
    #: machine's own back-feed is excluded from the device current.
    device_ik_a: float = 0.0
    #: Total current at the fault point including any local motor back-feed.
    total_ik_a: float = 0.0
    network_ik_a: float = 0.0
    motor_ik_a: float = 0.0

    @property
    def ik_ka(self) -> float:
        return self.ik_a / 1000.0

    @property
    def ip_ka(self) -> float:
        return self.ip_a / 1000.0

    @property
    def r(self) -> float:
        return self.z.real

    @property
    def x(self) -> float:
        return self.z.imag

    @property
    def x_over_r(self) -> float:
        return self.z.imag / self.z.real if self.z.real else math.inf

    @property
    def duty(self) -> str:
        return "maximum" if self.c > 1.0 else "minimum"


@dataclass
class ShortCircuitStudy:
    """Complete set of fault results for the bus and every motor terminal."""

    bus_max: FaultResult
    bus_min: FaultResult
    bus_max_no_motors: FaultResult
    terminal_max: dict[str, FaultResult] = field(default_factory=dict)
    terminal_min: dict[str, FaultResult] = field(default_factory=dict)
    kt: float = 1.0

    @property
    def motor_contribution_pct(self) -> float:
        """Motor back-feed as a percentage uplift on the network-only value."""
        base = self.bus_max_no_motors.ik_a
        if base <= 0:
            return 0.0
        return 100.0 * (self.bus_max.ik_a - base) / base


# ===========================================================================
# Calculation
# ===========================================================================


def _ik(c: float, u_n: float, z: complex) -> float:
    """I_k'' = c U_n / (sqrt(3) |Z|)."""
    return c * u_n / (SQRT3 * abs(z))


def _source_impedance(plant: Plant, c_max: float, apply_kt: bool) -> complex:
    z_tx = plant.transformer.z
    if apply_kt:
        z_tx = z_tx * transformer_correction_factor(plant, c_max)
    return plant.source.z(plant.nominal_v) + z_tx


def bus_fault(
    plant: Plant,
    maximum: bool = True,
    include_motors: bool = True,
    apply_kt: bool = True,
) -> FaultResult:
    """Three-phase fault on the 415 V bus itself."""
    c = C_MAX_LV if maximum else C_MIN_LV
    temp = TEMP_COLD_C if maximum else TEMP_XLPE_OPERATING_C
    u_n = plant.nominal_v

    z_net = _source_impedance(plant, C_MAX_LV, apply_kt)
    i_net = _ik(c, u_n, z_net)

    if include_motors and plant.motors:
        branches = [motor_branch_impedance(m, temp) for m in plant.motors]
        z_mot = parallel(*branches)
        z_total = parallel(z_net, z_mot)
        i_mot = _ik(c, u_n, z_mot)
    else:
        z_total = z_net
        i_mot = 0.0

    ik = _ik(c, u_n, z_total)
    k = kappa(z_total.real, z_total.imag)
    return FaultResult(
        location=f"{plant.meta.get('bus_designation', 'Bus')} busbar",
        ik_a=ik,
        ip_a=k * math.sqrt(2.0) * ik,
        kappa=k,
        z=z_total,
        c=c,
        temp_c=temp,
        includes_motors=include_motors and bool(plant.motors),
        device_ik_a=ik,
        total_ik_a=ik,
        network_ik_a=i_net,
        motor_ik_a=i_mot,
    )


def terminal_fault(
    plant: Plant,
    motor: Motor,
    maximum: bool = True,
    include_motors: bool = True,
    apply_kt: bool = True,
) -> FaultResult:
    """Three-phase fault at one motor's terminals.

    device_ik_a is the current through that motor's own protective device:
    network plus the OTHER motors, through this motor's cable. total_ik_a adds
    the faulted machine's own back-feed, which the cable sees but the device
    does not.
    """
    c = C_MAX_LV if maximum else C_MIN_LV
    temp = TEMP_COLD_C if maximum else TEMP_XLPE_OPERATING_C
    u_n = plant.nominal_v

    z_net = _source_impedance(plant, C_MAX_LV, apply_kt)

    others: Sequence[Motor] = [m for m in plant.motors if m.tag != motor.tag]
    if include_motors and others:
        z_src = parallel(z_net, parallel(*[motor_branch_impedance(m, temp) for m in others]))
    else:
        z_src = z_net

    z_through = z_src + motor.cable.z(temp)
    i_device = _ik(c, u_n, z_through)

    if include_motors:
        i_local = _ik(c, u_n, motor_impedance(motor))
        z_total = parallel(z_through, motor_impedance(motor))
    else:
        i_local = 0.0
        z_total = z_through

    i_total = _ik(c, u_n, z_total)
    k = kappa(z_total.real, z_total.imag)

    return FaultResult(
        location=f"{motor.tag} terminals",
        ik_a=i_device,
        ip_a=k * math.sqrt(2.0) * i_total,
        kappa=k,
        z=z_through,
        c=c,
        temp_c=temp,
        includes_motors=include_motors,
        device_ik_a=i_device,
        total_ik_a=i_total,
        network_ik_a=_ik(c, u_n, z_net + motor.cable.z(temp)),
        motor_ik_a=i_local,
    )


def run_study(plant: Plant, apply_kt: bool = True) -> ShortCircuitStudy:
    """Full short-circuit study: bus and every motor terminal, max and min."""
    study = ShortCircuitStudy(
        bus_max=bus_fault(plant, maximum=True, include_motors=True, apply_kt=apply_kt),
        bus_min=bus_fault(plant, maximum=False, include_motors=False, apply_kt=apply_kt),
        bus_max_no_motors=bus_fault(plant, maximum=True, include_motors=False, apply_kt=apply_kt),
        kt=transformer_correction_factor(plant) if apply_kt else 1.0,
    )
    for m in plant.motors:
        study.terminal_max[m.tag] = terminal_fault(
            plant, m, maximum=True, include_motors=True, apply_kt=apply_kt
        )
        study.terminal_min[m.tag] = terminal_fault(
            plant, m, maximum=False, include_motors=False, apply_kt=apply_kt
        )
    return study
