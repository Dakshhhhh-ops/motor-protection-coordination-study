"""
Standard constants, coefficients and their citations.

Every magic number used anywhere in this study lives here, next to the clause,
table or textbook reference it came from. Nothing numeric is hard-coded
elsewhere in the source tree.

Values tagged VENDOR are NOT from a standard. They are representative
manufacturer figures and must be replaced with data from the actual equipment
datasheet / transformer test certificate before a study is issued for
construction. They are reported as such so the assumption stays visible
rather than buried in code.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Electrical constants
# ---------------------------------------------------------------------------

SQRT3 = math.sqrt(3.0)

#: Resistivity of annealed copper at 20 degC, mOhm.mm^2/m.
#: IEC 60228 (conductors of insulated cables), nominal value 17.241.
RHO_CU_20 = 17.241

#: Temperature coefficient of resistance for copper, per kelvin.
#: IEC 60228 / IEC 60287-1-1.
ALPHA_CU = 0.00393

#: Reference conductor temperatures, degC.
#: IEC 60909-0 requires the cold (20 degC) resistance for MAXIMUM fault
#: current (equipment rating duty) and the operating temperature for MINIMUM
#: fault current (protection sensitivity duty).
TEMP_COLD_C = 20.0
TEMP_XLPE_OPERATING_C = 90.0  # XLPE continuous conductor rating, IEC 60502-1

# ---------------------------------------------------------------------------
# Short-circuit calculation -- IEC 60909-0:2016
# ---------------------------------------------------------------------------

#: Voltage factor c for LV systems, IEC 60909-0 Table 1.
#: c_max = 1.05 for LV networks with a voltage tolerance of +6 per cent
#: (use 1.10 where the tolerance is +10 per cent). c_min = 0.95 for LV.
C_MAX_LV = 1.05
C_MIN_LV = 0.95

#: Peak factor kappa coefficients, IEC 60909-0 clause 8:
#:     kappa = 1.02 + 0.98 * exp(-3 R/X)
#: bounded by 1.02 (pure resistance) and 2.00 (pure reactance).
KAPPA_A = 1.02
KAPPA_B = 0.98
KAPPA_EXP = 3.0

#: LV asynchronous motor short-circuit impedance angle, IEC 60909-0
#: clause 6.6: R_M / X_M = 0.42 for low-voltage motors, which yields
#: kappa_M = 1.3. Hence X_M = Z_M / sqrt(1 + 0.42^2).
MOTOR_R_OVER_X_LV = 0.42


def kappa(r: float, x: float) -> float:
    """Peak short-circuit factor, IEC 60909-0 clause 8.

    kappa = 1.02 + 0.98 * exp(-3 R/X)
    """
    if x <= 0.0:
        return KAPPA_A
    return KAPPA_A + KAPPA_B * math.exp(-KAPPA_EXP * r / x)


# ---------------------------------------------------------------------------
# Motor thermal overload relay -- IEC 60947-4-1 / IEC 60255-149
# ---------------------------------------------------------------------------

#: IEC 60947-4-1 Table 2 defines the trip class as the tripping time, from the
#: COLD state, at 7.2 x the set current I_e. Class 10 => 4 s < t <= 10 s,
#: Class 20 => 6 s < t <= 20 s, Class 30 => 9 s < t <= 30 s. The class number
#: is the upper bound of the band and is the value used to anchor the curve.
TRIP_CLASS_ANCHOR_MULTIPLE = 7.2

#: Trip classes considered by the setting search.
TRIP_CLASSES = (10, 20, 30)

#: Thermal pickup multiples of motor FLC considered by the setting search.
#: IEC 60947-4-1 allows the overload relay to be set between 100 and 115 per
#: cent of rated motor current for a machine with a 1.15 service factor.
#: Below 100 per cent it would trip on rated load; above 115 per cent it
#: would not protect the winding insulation.
PICKUP_MULTIPLES = (1.05, 1.10, 1.15)

#: Thermal state bands, in per-unit of the trip threshold (theta = 1.0 trips).
#: Digital motor relays conventionally alarm on "thermal capacity used" at
#: about 90 per cent, so 0.90 is adopted as the boundary between an acceptable
#: and a marginal start.
THETA_TRIP = 1.0
THETA_MARGINAL = 0.90


def thermal_tau(trip_class: int) -> float:
    """Thermal replica time constant implied by an IEC 60947-4-1 trip class.

    The class is defined as the cold tripping time at 7.2 x I_e. Substituting
    that point into the IEC 60255-149 thermal replica with theta_0 = 0:

        C = tau * ln( 7.2^2 / (7.2^2 - 1) )

    so   tau = C / 0.019484 = 51.32 * C.

    Class 10 -> 513 s, Class 20 -> 1026 s, Class 30 -> 1540 s -- all
    physically sensible thermal time constants for an LV induction motor.
    tau is therefore DERIVED from the standard, not assumed.
    """
    m2 = TRIP_CLASS_ANCHOR_MULTIPLE**2
    return trip_class / math.log(m2 / (m2 - 1.0))


# ---------------------------------------------------------------------------
# Locked-rotor / stall protection -- IEEE Std 242 Ch. 9
# ---------------------------------------------------------------------------

#: Locked-rotor (51LR) definite-time element pickup, as a fraction of the
#: nameplate locked-rotor current. 0.75 sits safely below LRC so a stall is
#: always detected, and far above any credible running overload.
LOCKED_ROTOR_PICKUP_FRACTION = 0.75

#: Coordination margins on the stall timer. The setting must be at least k1 x
#: the time the current actually dwells above pickup during a healthy start,
#: and at most the hot locked-rotor withstand divided by k2.
STALL_MARGIN_ABOVE_START = 1.25
STALL_MARGIN_BELOW_WITHSTAND = 1.25

# ---------------------------------------------------------------------------
# Instantaneous short-circuit element -- IEEE Std 242 Ch. 9
# ---------------------------------------------------------------------------

#: Minimum instantaneous setting as a multiple of locked-rotor current. The
#: first-cycle asymmetrical (DC-offset) component of motor inrush is seen by
#: an rms-sensing device as roughly 1.7 x the symmetrical LRC.
INSTANTANEOUS_MIN_MULTIPLE = 1.7

#: Adopted setting multiple. 1.8 x LRC gives margin over the asymmetrical
#: inrush without sacrificing fault sensitivity.
INSTANTANEOUS_SET_MULTIPLE = 1.8

#: Sensitivity margin: the minimum fault current at the protected point must
#: exceed the instantaneous setting by this factor (IEC 60364-4-41 principle
#: of assured operation).
INSTANTANEOUS_SENSITIVITY_MARGIN = 1.25

#: Total clearing time of an MCCB magnetic trip, seconds. VENDOR.
MCCB_INSTANTANEOUS_CLEARING_S = 0.03

#: Total clearing time of an ACB instantaneous element, seconds. VENDOR.
ACB_INSTANTANEOUS_CLEARING_S = 0.02

# ---------------------------------------------------------------------------
# Selectivity -- IEEE Std 242 Ch. 15
# ---------------------------------------------------------------------------

#: Coordination time interval. IEEE Std 242 recommends 0.2-0.4 s for
#: electromechanical relays and 0.1-0.2 s for static / digital trip units.
#: 0.15 s is adopted because every device modelled here is electronic.
CTI_S = 0.15

#: Alternative multiplicative margin, used in the inverse-time region where an
#: additive CTI is meaningless because both curves are steep.
CTI_RATIO = 1.3

# ---------------------------------------------------------------------------
# Motor starting / voltage dip -- IEEE Std 399 Ch. 9
# ---------------------------------------------------------------------------

#: Locked-rotor power factor. IEEE Std 242 / Std 399 give 0.20-0.30 for LV
#: squirrel-cage machines; 0.25 is adopted.
LOCKED_ROTOR_PF = 0.25

#: Star-delta line-current ratio. In star each winding sees U/sqrt(3) so the
#: winding current falls to 1/sqrt(3); in delta the line current is sqrt(3) x
#: the winding current whereas in star it equals it. The net line-current
#: ratio is therefore 1/3, and the equivalent per-phase impedance seen from
#: the line is 3x the delta value. Torque, going as V^2, also falls to 1/3.
STAR_DELTA_CURRENT_RATIO = 1.0 / 3.0
STAR_DELTA_IMPEDANCE_RATIO = 3.0

# ---------------------------------------------------------------------------
# Cable checks -- IEC 60364-5-52 / IEC 60364-5-54
# ---------------------------------------------------------------------------

#: Adiabatic short-circuit constant k for a copper conductor with XLPE
#: insulation, 90 degC initial to 250 degC final. IEC 60364-5-54 / IEC 60949.
K_ADIABATIC_CU_XLPE = 143.0

#: Ambient derating factor for XLPE at 40 degC in air (tabulated ampacities
#: are at 30 degC). IEC 60364-5-52 Annex B.
K_AMBIENT_40C_XLPE = 0.91

#: Grouping factor for circuits bunched on a perforated tray.
#: IEC 60364-5-52 Annex B. Depends on the actual routing -- VENDOR/design.
K_GROUPING = 0.85

#: Maximum permitted steady-state voltage drop for motor circuits, per cent.
#: IEC 60364-5-52 Annex G recommends 5 per cent total from the origin of the
#: installation to the point of use.
MAX_VOLTAGE_DROP_PCT = 5.0

# ---------------------------------------------------------------------------
# Damage curve validity
# ---------------------------------------------------------------------------

#: The motor thermal damage curve used here is the constant-I^2*t
#: extrapolation of the nameplate locked-rotor withstand point. That line is
#: only physically meaningful in the ACCELERATING region, where heating is
#: rotor-limited. Below roughly 3 x FLC the limit becomes stator- and
#: insulation-limited, governed by the machine's service factor and insulation
#: class, which cannot be derived from nameplate locked-rotor data. The
#: relay-below-damage-curve check is therefore applied only above this
#: multiple, and the report states so explicitly.
DAMAGE_CURVE_MIN_MULTIPLE = 3.0
