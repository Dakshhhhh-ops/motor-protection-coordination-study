"""
Supporting checks: cable adequacy, switchboard rating, transformer loading and
incomer behaviour during motor starting.

These are the checks that a protection study is incomplete without, because a
perfectly coordinated relay setting is worthless if the cable cannot carry the
load current, the switchboard cannot withstand the fault, or the incomer trips
every time the largest motor starts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from constants import (
    INSTANTANEOUS_MIN_MULTIPLE,
    K_ADIABATIC_CU_XLPE,
    K_AMBIENT_40C_XLPE,
    K_GROUPING,
    MAX_VOLTAGE_DROP_PCT,
    SQRT3,
    TEMP_XLPE_OPERATING_C,
)
from models import Cable, Motor, Plant, parallel
from protection import FAIL, INFO, MARGINAL, NA, PASS, Check, MotorSettings, worst
from shortcircuit import ShortCircuitStudy
from starting import DELTA, STAR, StartingStudy


# ===========================================================================
# Cable checks
# ===========================================================================


def derated_ampacity(cable: Cable) -> float:
    """Installed current-carrying capacity after derating.

    IEC 60364-5-52:

        I_z = I_z,tabulated * k_ambient * k_grouping * runs

    k_ambient = 0.91 corrects the 30 degC tabulated value to a 40 degC ambient
    for XLPE; k_grouping = 0.85 allows for circuits bunched on a tray. Both
    depend on the actual installation and must be confirmed against the
    project cable schedule.
    """
    return cable.ctype.iz_base_a * K_AMBIENT_40C_XLPE * K_GROUPING * cable.runs


def adiabatic_withstand_s(cable: Cable, ik_a: float) -> float:
    """Short-circuit withstand time of the conductor.

    IEC 60364-5-54 adiabatic equation, rearranged for time:

        S >= I * sqrt(t) / k     ->     t = (k * S / I)^2

    with k = 143 for a copper conductor in XLPE heating from its 90 degC
    operating temperature to the 250 degC limit. Adiabatic means no heat
    escapes the conductor, which is valid for clearing times under about 5 s.

    Parallel runs share the current, so each run carries I/runs.
    """
    if ik_a <= 0:
        return math.inf
    i_per_run = ik_a / cable.runs
    return (K_ADIABATIC_CU_XLPE * cable.size_mm2 / i_per_run) ** 2


def cable_checks(
    plant: Plant,
    motor: Motor,
    settings: MotorSettings,
    sc: ShortCircuitStudy,
    starting: StartingStudy,
) -> list[Check]:
    """C12-C14: ampacity, steady-state voltage drop, short-circuit withstand."""
    iz = derated_ampacity(motor.cable)
    _, vd_pct = starting.voltage_drop[motor.tag]
    max_vd = plant.criteria.get("max_voltage_drop_pct", MAX_VOLTAGE_DROP_PCT)

    # The cable's worst thermal duty is a fault at its LOAD end, where the
    # whole length carries the through-fault current. A fault at the source
    # end passes through almost no cable.
    ik = sc.terminal_max[motor.tag].device_ik_a
    t_withstand = adiabatic_withstand_s(motor.cable, ik)
    t_clear = settings.inst_clearing_s

    utilisation = 100.0 * motor.flc_a / iz

    return [
        Check(
            "C12",
            "Cable current-carrying capacity",
            "I_z,derated >= I_FLC",
            f"{iz:.0f} A vs {motor.flc_a:.1f} A ({utilisation:.0f} per cent utilised)",
            PASS if iz >= motor.flc_a else FAIL,
            "IEC 60364-5-52",
            f"{motor.cable.ctype.iz_base_a:.0f} A tabulated x {K_AMBIENT_40C_XLPE} ambient "
            f"x {K_GROUPING} grouping x {motor.cable.runs} run(s).",
        ),
        Check(
            "C13",
            "Steady-state voltage drop",
            f"dV <= {max_vd:.0f} per cent",
            f"{vd_pct:.2f} per cent",
            PASS if vd_pct <= max_vd else FAIL,
            "IEC 60364-5-52 Annex G",
        ),
        Check(
            "C14",
            "Cable short-circuit withstand",
            "t_withstand >= device clearing time",
            f"{t_withstand:.2f} s vs {t_clear:.2f} s at {ik:.0f} A",
            PASS if t_withstand >= t_clear else FAIL,
            "IEC 60364-5-54",
            f"Adiabatic, k = {K_ADIABATIC_CU_XLPE:.0f} for copper in XLPE, 90 to 250 degC.",
        ),
    ]


def recommend_cable_size(plant: Plant, motor: Motor) -> float | None:
    """Smallest library cross-section whose derated ampacity carries the motor.

    Used to justify the cable schedule: every size selected in plant.yaml
    should equal this value, otherwise the report says why it does not.
    """
    for size in plant.cable_library.sizes_ascending():
        trial = Cable(plant.cable_library.get(size), motor.cable.length_m, motor.cable.runs)
        if derated_ampacity(trial) >= motor.flc_a:
            return size
    return None


# ===========================================================================
# Plant-level checks
# ===========================================================================


@dataclass
class PlantChecks:
    checks: list[Check] = field(default_factory=list)
    worst_start_bus_current_a: float = 0.0
    worst_start_tag: str = ""

    @property
    def verdict(self) -> str:
        return worst(c.verdict for c in self.checks)


def _bus_current_during_start(plant: Plant, motor: Motor, connection: str) -> float:
    """Total current drawn from the source while one motor starts, others running.

    Solved from the same network as the dip study, so the running load and the
    starting branch are combined with their correct phase angles rather than
    having their magnitudes added.
    """
    z_s = plant.z_source
    z_branch = motor.cable.z(TEMP_XLPE_OPERATING_C) + motor.z_lr(connection)
    z_load = plant.load_impedance(exclude_tags=[motor.tag])
    z_par = parallel(z_load, z_branch) if z_load is not None else z_branch
    return (plant.nominal_v / SQRT3) / abs(z_s + z_par)


def plant_checks(
    plant: Plant, sc: ShortCircuitStudy, starting: StartingStudy
) -> PlantChecks:
    """C15-C18: switchboard rating, transformer loading, incomer vs starting."""
    out = PlantChecks()

    # -- C15 switchboard short-circuit withstand ---------------------------
    rating = plant.bus.rated_short_time_withstand_ka
    ik_bus = sc.bus_max.ik_ka
    margin = 100.0 * (rating - ik_bus) / rating
    v15 = PASS if ik_bus <= rating else FAIL
    if v15 == PASS and margin < 10.0:
        v15 = MARGINAL
    out.checks.append(
        Check(
            "C15",
            "Switchboard short-circuit withstand",
            f"I_cw >= I_k,max = {ik_bus:.2f} kA",
            f"{rating:.0f} kA rated, {margin:.0f} per cent margin",
            v15,
            "IEC 61439-1 / IEC 60909-0",
            f"Maximum bus fault includes the {sc.motor_contribution_pct:.0f} per cent uplift "
            f"from motor back-feed ({sc.bus_max_no_motors.ik_ka:.2f} kA network only). Peak "
            f"i_p = {sc.bus_max.ip_ka:.1f} kA at kappa = {sc.bus_max.kappa:.2f} sets the "
            f"electrodynamic duty on the busbar supports.",
        )
    )

    # -- C16 transformer loading ------------------------------------------
    limit = plant.criteria.get("transformer_max_loading_pct", 80.0)
    loading = plant.transformer_loading_pct
    out.checks.append(
        Check(
            "C16",
            "Transformer loading",
            f"loading <= {limit:.0f} per cent of rating",
            f"{loading:.1f} per cent ({plant.diversified_kva:.0f} kVA of "
            f"{plant.transformer.kva:.0f} kVA at pf {plant.diversified_pf:.3f})",
            PASS if loading <= limit else FAIL,
            "Design criterion",
            f"Connected {plant.connected_kva:.0f} kVA, diversified at "
            f"{plant.motor_diversity_factor:.2f} on the motor group. Complex addition, so "
            f"the differing power factors of the motor group and the static load are "
            f"respected.",
        )
    )

    # -- C17 incomer rides through the worst motor start -------------------
    worst_i = 0.0
    worst_tag = ""
    worst_motor = None
    for m in plant.motors:
        conn = STAR if m.is_star_delta else DELTA
        i_bus = _bus_current_during_start(plant, m, conn)
        if i_bus > worst_i:
            worst_i, worst_tag, worst_motor = i_bus, m.tag, m
    out.worst_start_bus_current_a = worst_i
    out.worst_start_tag = worst_tag

    assert worst_motor is not None
    t_feeder = plant.feeder.trip_time_s(worst_i)
    t_start = worst_motor.t_start_s
    if math.isinf(t_feeder):
        v17, val = PASS, f"{worst_i:.0f} A is below the {plant.feeder.ir_a:.0f} A long-time pickup"
    else:
        ratio = t_feeder / t_start
        v17 = PASS if ratio >= 2.0 else (MARGINAL if ratio >= 1.2 else FAIL)
        val = (
            f"{worst_i:.0f} A for {t_start:.1f} s; incomer would trip in {t_feeder:.1f} s "
            f"(ratio {ratio:.1f})"
        )
    out.checks.append(
        Check(
            "C17",
            "Incomer does not trip on the worst-case motor start",
            "t_feeder at the starting current >= 2 x start time",
            val,
            v17,
            "IEC 60947-2",
            f"Worst case is {worst_tag} starting with every other motor running and the "
            f"static load connected, solved as a single network so the running load and the "
            f"starting branch combine with their correct phase angles.",
        )
    )

    # -- C18 incomer short-time above the bus inrush -----------------------
    required = INSTANTANEOUS_MIN_MULTIPLE * worst_i
    out.checks.append(
        Check(
            "C18",
            "Incomer short-time pickup above the bus inrush",
            f"I_sd >= {INSTANTANEOUS_MIN_MULTIPLE} x worst starting current",
            f"{plant.feeder.isd_a:.0f} A vs {required:.0f} A required "
            f"({plant.feeder.isd_a / worst_i:.1f} x the {worst_i:.0f} A start)",
            PASS if plant.feeder.isd_a >= required else FAIL,
            "IEEE Std 242 Ch. 9",
            "The 1.7 factor covers the asymmetrical first-cycle offset that an rms-sensing "
            "trip unit responds to.",
        )
    )

    # -- C19 cable schedule justification ---------------------------------
    oversized = []
    for m in plant.motors:
        rec = recommend_cable_size(plant, m)
        if rec is not None and rec != m.cable.size_mm2:
            oversized.append(f"{m.tag} {m.cable.size_mm2:g} mm2 vs {rec:g} mm2 minimum")
    out.checks.append(
        Check(
            "C19",
            "Cable cross-sections match the calculated minimum",
            "selected size == smallest size meeting the derated ampacity",
            "all match" if not oversized else "; ".join(oversized),
            PASS if not oversized else INFO,
            "IEC 60364-5-52",
            "An oversized cable is not a fault. It is flagged so the reason (voltage drop, "
            "future duty, standardisation of stock) is recorded rather than assumed.",
        )
    )

    return out
