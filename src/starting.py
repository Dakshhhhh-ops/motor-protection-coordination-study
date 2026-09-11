"""
Motor starting / voltage dip study, per IEEE Std 399 Ch. 9.

Network model
-------------
At standstill the machine is a fixed series impedance (slip = 1), so a start
is a linear circuit problem, not a dynamic one:

        Z_source                Z_cable
    o------[==]-----+------------[==]-------- motor
    |               |                          |
  V_nom          Z_load                      Z_LR
    |        (running bus load)                |
    o---------------+---------------------------

    Z_branch = Z_cable + Z_LR
    Z_par    = Z_load || Z_branch        (Z_load omitted when starting alone)

    V_bus   / V_nom = |Z_par| / |Z_source + Z_par|
    V_motor / V_bus = |Z_LR|  / |Z_branch|
    I_start         = (V_bus/sqrt(3)) / |Z_branch|

Everything is solved in complex arithmetic and magnitudes are taken last.
Using |Z| sums instead would overstate the dip, because the network is
inductive (X/R about 5) while the locked rotor is nearly so (pf 0.25) but the
cable is predominantly resistive.

Why the achieved starting current is below nameplate
---------------------------------------------------
The nameplate locked-rotor current is quoted at rated terminal voltage. The
supply and cable impedance drop means the motor never sees rated voltage
during its own start, so the current actually drawn is lower. The study
reports both, because the nameplate figure is the right one for protection
settings (it is the largest current a relay could see) while the achieved
figure is the right one for the dip itself.

Conductor temperature
---------------------
The cable resistance is taken at 90 degC. A hot conductor is 27.5 per cent
more resistive, which deepens the calculated dip -- the conservative choice
for a voltage-adequacy check, and the realistic one for a motor restarted
after running at full load.

Star-delta
----------
The star stage is modelled with Z_LR x 3 (see constants for the derivation of
the factor). Two consequences are checked separately:

  * Current and dip are one third of DOL, so the dip criteria are easily met.
  * Torque is also one third, and torque goes as V^2, so the available
    accelerating torque in star is  (V_motor/V_rated)^2 / 3  of the rated DOL
    value. This is the real constraint on a star-delta start and is reported.

The delta-stage transition is bounded above by the DOL result for the same
motor, because the impedance is identical; the DOL row therefore doubles as a
conservative check on the transition, and no separate case is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from constants import (
    MAX_VOLTAGE_DROP_PCT,
    SQRT3,
    STAR_DELTA_IMPEDANCE_RATIO,
    TEMP_XLPE_OPERATING_C,
)
from models import Motor, Plant, parallel
from protection import FAIL, INFO, MARGINAL, NA, PASS, Check, worst

DELTA = "delta"
STAR = "star"

#: Label shown in the report for each connection.
METHOD_LABEL = {DELTA: "DOL (full winding)", STAR: "Star-delta, star stage"}


@dataclass
class StartingResult:
    """One starting case: a motor, a connection, and a loading condition."""

    motor_tag: str
    connection: str
    others_running: bool

    z_source: complex
    z_cable: complex
    z_lr: complex
    z_load: complex | None

    v_bus_pu: float
    v_motor_pu: float
    i_start_a: float
    i_nameplate_a: float
    torque_pu: float

    checks: list[Check] = field(default_factory=list)

    @property
    def case(self) -> str:
        loading = "all others running" if self.others_running else "starting alone"
        return f"{METHOD_LABEL[self.connection]}, {loading}"

    @property
    def bus_dip_pct(self) -> float:
        return 100.0 * (1.0 - self.v_bus_pu)

    @property
    def motor_dip_pct(self) -> float:
        return 100.0 * (1.0 - self.v_motor_pu)

    @property
    def verdict(self) -> str:
        return worst(c.verdict for c in self.checks)


def evaluate(
    plant: Plant,
    motor: Motor,
    connection: str = DELTA,
    others_running: bool = False,
) -> StartingResult:
    """Solve one starting case and check it against the voltage criteria."""
    if connection not in (DELTA, STAR):
        raise ValueError("connection must be 'delta' (DOL) or 'star'")

    z_s = plant.z_source
    z_c = motor.cable.z(TEMP_XLPE_OPERATING_C)
    z_lr = motor.z_lr(connection)
    z_branch = z_c + z_lr

    z_load = plant.load_impedance(exclude_tags=[motor.tag]) if others_running else None
    z_par = parallel(z_load, z_branch) if z_load is not None else z_branch

    v_bus_pu = abs(z_par) / abs(z_s + z_par)
    v_motor_pu = v_bus_pu * abs(z_lr) / abs(z_branch)

    v_bus_phase = v_bus_pu * plant.nominal_v / SQRT3
    i_start = v_bus_phase / abs(z_branch)

    i_nameplate = motor.i_lr_a / (
        STAR_DELTA_IMPEDANCE_RATIO if connection == STAR else 1.0
    )

    # Torque relative to rated-voltage DOL torque. T goes as V^2, and a star
    # connection develops one third of the delta torque at the same terminal
    # voltage.
    torque_pu = v_motor_pu**2 / (STAR_DELTA_IMPEDANCE_RATIO if connection == STAR else 1.0)

    result = StartingResult(
        motor_tag=motor.tag,
        connection=connection,
        others_running=others_running,
        z_source=z_s,
        z_cable=z_c,
        z_lr=z_lr,
        z_load=z_load,
        v_bus_pu=v_bus_pu,
        v_motor_pu=v_motor_pu,
        i_start_a=i_start,
        i_nameplate_a=i_nameplate,
        torque_pu=torque_pu,
    )
    result.checks = _checks(plant, motor, result)
    return result


def _checks(plant: Plant, motor: Motor, r: StartingResult) -> list[Check]:
    v_motor_min = plant.criteria.get("v_motor_min_pu", 0.80)
    v_bus_min = plant.criteria.get("v_bus_min_pu", 0.85)

    checks = [
        Check(
            "C10",
            "Motor terminal voltage during start",
            f"V_motor >= {v_motor_min * 100:.0f} per cent",
            f"{r.v_motor_pu * 100:.1f} per cent",
            PASS if r.v_motor_pu >= v_motor_min else FAIL,
            "IEEE Std 399 Ch. 9",
            "Torque goes as V^2, so this limit fixes the accelerating torque available.",
        ),
        Check(
            "C11",
            "Bus voltage during start",
            f"V_bus >= {v_bus_min * 100:.0f} per cent",
            f"{r.v_bus_pu * 100:.1f} per cent (dip {r.bus_dip_pct:.1f} per cent)",
            PASS if r.v_bus_pu >= v_bus_min else FAIL,
            "IEC 60947-4-1",
            "AC-3 contactors must hold in at 85 per cent of Uc; a deeper dip risks "
            "dropping out every other starter on the board.",
        ),
    ]
    return checks


# ===========================================================================
# Steady-state voltage drop
# ===========================================================================


def voltage_drop(motor: Motor) -> tuple[float, float]:
    """Steady-state running voltage drop in the motor cable.

    IEC 60364-5-52 Annex G:

        dV = sqrt(3) * I * (R cos(phi) + X sin(phi))

    evaluated at the running current and running power factor, with the
    conductor at its 90 degC operating temperature. Returns (volts, per cent).
    """
    phi = math.acos(motor.pf)
    r = motor.cable.r_ohm(TEMP_XLPE_OPERATING_C)
    x = motor.cable.x_ohm()
    dv = SQRT3 * motor.flc_a * (r * math.cos(phi) + x * math.sin(phi))
    return dv, 100.0 * dv / motor.voltage_v


# ===========================================================================
# Whole-plant study
# ===========================================================================


@dataclass
class StartingStudy:
    """All starting cases for the whole bus."""

    #: (tag, connection, others_running) -> result
    cases: dict[tuple[str, str, bool], StartingResult] = field(default_factory=dict)
    #: tag -> (volts, per cent)
    voltage_drop: dict[str, tuple[float, float]] = field(default_factory=dict)
    largest_motor_tag: str = ""
    recommendations: dict[str, str] = field(default_factory=dict)

    def get(self, tag: str, connection: str, others_running: bool) -> StartingResult:
        return self.cases[(tag, connection, others_running)]

    def as_designed(self, plant: Plant, tag: str, others_running: bool) -> StartingResult:
        """The case matching the starting method actually specified."""
        motor = plant.motor(tag)
        conn = STAR if motor.is_star_delta else DELTA
        return self.get(tag, conn, others_running)

    @property
    def verdict(self) -> str:
        return worst(r.verdict for r in self.cases.values())


def run_study(plant: Plant) -> StartingStudy:
    """Every motor, both starting methods, alone and with the bus loaded.

    Both methods are evaluated for every motor -- including motors specified as
    star-delta -- so the choice of starting method is justified by calculation
    rather than by convention. The DOL row for a star-delta motor is also the
    conservative bound on its star-to-delta transition.
    """
    study = StartingStudy(largest_motor_tag=plant.largest_motor.tag)

    for m in plant.motors:
        for conn in (DELTA, STAR):
            for others in (False, True):
                study.cases[(m.tag, conn, others)] = evaluate(plant, m, conn, others)
        study.voltage_drop[m.tag] = voltage_drop(m)

    for m in plant.motors:
        study.recommendations[m.tag] = _recommend(plant, study, m)

    return study


def critical_transformer_kva(
    plant: Plant,
    motor: Motor,
    connection: str = DELTA,
    others_running: bool = True,
    limit_pu: float | None = None,
) -> float | None:
    """Transformer rating at which this start would just reach the voltage limit.

    Answers the question a passing result leaves open: how much margin is
    there? The transformer impedance scales as 1/S for a fixed impedance
    voltage, so shrinking the transformer is the cleanest single-parameter way
    to weaken the source. Everything else -- cable, motor, running load -- is
    held constant and the rating is found by bisection.

    Both u_k and X/R are held constant while the rating is scaled, which is
    what a smaller transformer of the same family actually looks like: its load
    loss falls roughly in proportion to its rating, so R per unit stays put. It
    is emphatically NOT the same as keeping the absolute load loss fixed, which
    would drive R per cent up as the rating falls and rotate the impedance
    angle.

    This matters because on an LV bus the criterion frequently cannot bind at
    all: the motor's own locked-rotor impedance is an order of magnitude larger
    than the source impedance, so the terminal voltage is dominated by the
    motor itself. Reporting the critical rating turns "it passed" into "it
    passes until the transformer is this small", which is the defensible form
    of the statement.

    Returns None if the limit is unreachable within a 10 kVA to 100 MVA search
    range, which is itself the finding.
    """
    if limit_pu is None:
        limit_pu = plant.criteria.get("v_motor_min_pu", 0.80)

    z_c = motor.cable.z(TEMP_XLPE_OPERATING_C)
    z_lr = motor.z_lr(connection)
    z_branch = z_c + z_lr
    z_load = plant.load_impedance(exclude_tags=[motor.tag]) if others_running else None
    z_par = parallel(z_load, z_branch) if z_load is not None else z_branch

    z_utility = plant.source.z(plant.nominal_v)
    z_tx_rated = plant.transformer.z
    kva_rated = plant.transformer.kva

    def v_motor(kva: float) -> float:
        # Z_T = (u_k/100) * V^2 / S, so scaling the rating scales Z by 1/S.
        z_s = z_utility + z_tx_rated * (kva_rated / kva)
        return (abs(z_par) / abs(z_s + z_par)) * abs(z_lr) / abs(z_branch)

    lo, hi = 10.0, 100_000.0
    if v_motor(hi) < limit_pu or v_motor(lo) > limit_pu:
        return None
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if v_motor(mid) < limit_pu:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def _recommend(plant: Plant, study: StartingStudy, motor: Motor) -> str:
    """Starting-method recommendation, driven by the computed dip results."""
    v_motor_min = plant.criteria.get("v_motor_min_pu", 0.80)
    v_bus_min = plant.criteria.get("v_bus_min_pu", 0.85)
    max_vd = plant.criteria.get("max_voltage_drop_pct", MAX_VOLTAGE_DROP_PCT)

    dol = study.get(motor.tag, DELTA, True)
    star = study.get(motor.tag, STAR, True)
    _, vd_pct = study.voltage_drop[motor.tag]

    dol_ok = dol.v_motor_pu >= v_motor_min and dol.v_bus_pu >= v_bus_min
    star_ok = star.v_motor_pu >= v_motor_min and star.v_bus_pu >= v_bus_min

    parts: list[str] = []

    if dol_ok:
        parts.append(
            f"DOL is electrically acceptable with all other motors running: "
            f"{dol.v_motor_pu * 100:.1f} per cent at the terminals and "
            f"{dol.v_bus_pu * 100:.1f} per cent on the bus, against limits of "
            f"{v_motor_min * 100:.0f} and {v_bus_min * 100:.0f} per cent."
        )
        if motor.is_star_delta:
            parts.append(
                f"Star-delta is nevertheless specified. It is not needed to satisfy the "
                f"voltage criteria, so the justification is mechanical and thermal rather "
                f"than electrical: it cuts the starting current from "
                f"{dol.i_start_a:.0f} A to {star.i_start_a:.0f} A, which reduces the shaft "
                f"and coupling torque transient, cuts the I^2*t deposited in the rotor "
                f"during run-up by a factor of about nine, and eases the duty on the "
                f"upstream contactor. The cost is that only "
                f"{star.torque_pu * 100:.0f} per cent of rated DOL torque is available in "
                f"star, so the driven load must accelerate on that torque before the "
                f"changeover."
            )
        else:
            parts.append("DOL is confirmed as the correct starting method.")
    elif star_ok:
        parts.append(
            f"DOL is NOT acceptable: {dol.v_motor_pu * 100:.1f} per cent at the terminals "
            f"and {dol.v_bus_pu * 100:.1f} per cent on the bus. Star-delta satisfies both "
            f"criteria ({star.v_motor_pu * 100:.1f} per cent and "
            f"{star.v_bus_pu * 100:.1f} per cent) and is recommended, subject to confirming "
            f"that the load accelerates on the {star.torque_pu * 100:.0f} per cent of rated "
            f"torque available in star."
        )
    else:
        parts.append(
            f"NEITHER method satisfies the voltage criteria with the bus loaded "
            f"(DOL {dol.v_motor_pu * 100:.1f} per cent, star-delta "
            f"{star.v_motor_pu * 100:.1f} per cent at the terminals). A closed-transition "
            f"reduced-voltage starter, a soft starter with a current-limit setting, or a "
            f"variable-speed drive is required. Increasing the cable cross-section from "
            f"{motor.cable.size_mm2:g} mm2 would recover part of the drop, and the "
            f"remainder can only come from a stiffer source."
        )

    if vd_pct > max_vd:
        parts.append(
            f"Separately, the steady-state running voltage drop is {vd_pct:.2f} per cent, "
            f"which exceeds the {max_vd:.0f} per cent limit of IEC 60364-5-52 Annex G; the "
            f"cable must be increased regardless of the starting method."
        )

    return " ".join(parts)
