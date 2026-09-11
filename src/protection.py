"""
Motor protection setting design and coordination verification.

Device model per motor feeder (the standard IEC motor starter):

    MCCB (magnetic trip)  ->  contactor  ->  thermal overload relay  ->  motor
          ANSI 50                              ANSI 49/51 + 51LR

A contactor cannot interrupt a 33 kA fault, so the MCCB provides the
short-circuit protection and the overload relay provides the thermal
protection. This is type-2 coordination in the sense of IEC 60947-4-1 and it
is why the "instantaneous" element belongs to the MCCB, not the relay.

Three elements are set for each motor:

  49/51  Thermal overload.  IEC 60255-149 single-time-constant thermal
         replica, with tau derived from the IEC 60947-4-1 trip class.
  51LR   Locked-rotor / stall.  Definite time, set inside a window bounded
         below by the healthy start and above by the hot stall withstand.
  50     Instantaneous short circuit.  Definite time, set above the
         asymmetrical inrush and below the minimum terminal fault current.

Why the start-up check is an integration, not a curve overlay
-------------------------------------------------------------
The naive check is "is the relay curve above the starting current point".
That is wrong for a two-stage start, because it compares a single steady
current against a curve while the real relay integrates heat over a changing
current. A star-delta start spends most of its time at I_LR/3, which deposits
one ninth of the heat per second, and a curve overlay cannot see that. So the
thermal state is integrated stage by stage:

    dtheta/dt = [ (I/I_p)^2 - theta ] / tau
    theta(t)  = theta_inf + (theta_0 - theta_inf) * exp(-t/tau),
    theta_inf = (I/I_p)^2

and the start passes only if theta never reaches 1.0. This is what a digital
motor relay actually computes, and it is the quantity a relay displays as
"thermal capacity used".

The binding case is the HOT RESTART. A motor that has been running at full
load sits at theta_0 = (I_FLC/I_p)^2, which for a 110 per cent pickup is 0.83
-- 83 per cent of the thermal budget is already spent before the start begins.
Both the hot and the cold case are therefore computed and reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from constants import (
    CTI_RATIO,
    CTI_S,
    DAMAGE_CURVE_MIN_MULTIPLE,
    INSTANTANEOUS_MIN_MULTIPLE,
    INSTANTANEOUS_SENSITIVITY_MARGIN,
    INSTANTANEOUS_SET_MULTIPLE,
    LOCKED_ROTOR_PICKUP_FRACTION,
    MCCB_INSTANTANEOUS_CLEARING_S,
    PICKUP_MULTIPLES,
    STALL_MARGIN_ABOVE_START,
    STALL_MARGIN_BELOW_WITHSTAND,
    STAR_DELTA_CURRENT_RATIO,
    THETA_MARGINAL,
    THETA_TRIP,
    TRIP_CLASSES,
    thermal_tau,
)
from models import Motor, Plant
from shortcircuit import ShortCircuitStudy

INF = math.inf

PASS = "PASS"
MARGINAL = "MARGINAL"
FAIL = "FAIL"
INFO = "INFO"
NA = "N/A"

#: Verdict severity ordering, worst last.
_SEVERITY = {INFO: 0, NA: 0, PASS: 1, MARGINAL: 2, FAIL: 3}


def worst(verdicts: Iterable[str]) -> str:
    out = PASS
    for v in verdicts:
        if _SEVERITY.get(v, 0) > _SEVERITY.get(out, 0):
            out = v
    return out


# ===========================================================================
# Check record
# ===========================================================================


@dataclass(frozen=True)
class Check:
    """One numbered acceptance check with its computed value and verdict."""

    id: str
    name: str
    criterion: str
    value: str
    verdict: str
    reference: str = ""
    note: str = ""


# ===========================================================================
# Thermal replica -- IEC 60255-149
# ===========================================================================


def thermal_trip_time(i_a: float, pickup_a: float, tau_s: float, theta_0: float = 0.0) -> float:
    """Time to trip at a constant current, from an initial thermal state.

                       (I/I_p)^2 - theta_0
        t = tau * ln  ---------------------
                        (I/I_p)^2 - 1

    theta_0 = 0 gives the cold curve; theta_0 = (I_FLC/I_p)^2 the hot curve.
    Returns infinity below pickup, where the replica never reaches the
    threshold.
    """
    if pickup_a <= 0 or tau_s <= 0:
        raise ValueError("pickup and tau must be positive")
    m2 = (i_a / pickup_a) ** 2
    if m2 <= 1.0:
        return INF
    numerator = m2 - theta_0
    denominator = m2 - 1.0
    if numerator <= 0.0:
        return INF
    return tau_s * math.log(numerator / denominator)


def thermal_state(
    i_a: float, pickup_a: float, tau_s: float, theta_0: float, duration_s: float
) -> float:
    """Thermal state after holding a constant current for a given time.

        theta(t) = theta_inf + (theta_0 - theta_inf) e^(-t/tau),
        theta_inf = (I/I_p)^2

    This is the closed-form solution of the first-order thermal replica, so no
    numerical integration error is introduced.
    """
    theta_inf = (i_a / pickup_a) ** 2
    return theta_inf + (theta_0 - theta_inf) * math.exp(-duration_s / tau_s)


# ===========================================================================
# Start-up thermal simulation
# ===========================================================================


@dataclass
class StartSimulation:
    """Result of integrating the thermal replica through a motor start."""

    condition: str  # "hot restart" or "cold start"
    theta_0: float
    theta_max: float
    trips: bool
    trip_time_s: float | None
    stage_results: list[dict] = field(default_factory=list)
    trace_t: list[float] = field(default_factory=list)
    trace_theta: list[float] = field(default_factory=list)

    @property
    def headroom_pct(self) -> float:
        """Unused thermal capacity at the end of the start, per cent."""
        return 100.0 * max(0.0, THETA_TRIP - self.theta_max)

    @property
    def verdict(self) -> str:
        if self.trips or self.theta_max >= THETA_TRIP:
            return FAIL
        if self.theta_max > THETA_MARGINAL:
            return MARGINAL
        return PASS


def simulate_start(
    motor: Motor,
    pickup_a: float,
    tau_s: float,
    theta_0: float,
    condition: str,
    trace_points: int = 60,
) -> StartSimulation:
    """Integrate the thermal state through every stage of a healthy start.

    The current profile is taken on a nameplate basis (rated voltage at the
    terminals), which is conservative: with the real supply impedance the
    achieved starting current is lower, so less heat is deposited.
    """
    sim = StartSimulation(
        condition=condition, theta_0=theta_0, theta_max=theta_0, trips=False, trip_time_s=None
    )
    theta = theta_0
    t_elapsed = 0.0
    sim.trace_t.append(0.0)
    sim.trace_theta.append(theta)

    for current, duration, label in motor.starting_stages():
        # Would the replica reach theta = 1 inside this stage?
        t_to_trip = thermal_trip_time(current, pickup_a, tau_s, theta)
        trips_here = t_to_trip <= duration

        for k in range(1, trace_points + 1):
            dt = duration * k / trace_points
            sim.trace_t.append(t_elapsed + dt)
            sim.trace_theta.append(thermal_state(current, pickup_a, tau_s, theta, dt))

        theta_end = thermal_state(current, pickup_a, tau_s, theta, duration)
        sim.stage_results.append(
            {
                "label": label,
                "current_a": current,
                "multiple_of_pickup": current / pickup_a,
                "duration_s": duration,
                "theta_start": theta,
                "theta_end": theta_end,
                "trips": trips_here,
                "time_to_trip_s": t_to_trip if trips_here else None,
            }
        )

        if trips_here and not sim.trips:
            sim.trips = True
            sim.trip_time_s = t_elapsed + t_to_trip

        theta = theta_end
        t_elapsed += duration
        sim.theta_max = max(sim.theta_max, theta)

    if sim.trips:
        sim.theta_max = max(sim.theta_max, THETA_TRIP)
    return sim


# ===========================================================================
# Motor damage curve
# ===========================================================================


def damage_time(motor: Motor, i_a: float, hot: bool = True) -> float:
    """Motor thermal (stall) withstand time at a given current.

        t_damage(I) = t_stall * (I_LR / I)^2

    the constant-I^2*t extrapolation of the nameplate locked-rotor withstand
    point (IEEE Std 242 Ch. 9 motor thermal limit curve). Only physically
    meaningful in the accelerating region -- see
    constants.DAMAGE_CURVE_MIN_MULTIPLE.
    """
    if i_a <= 0:
        return INF
    t_stall = motor.t_stall_hot_s if hot else motor.t_stall_cold_s
    return t_stall * (motor.i_lr_a / i_a) ** 2


# ===========================================================================
# Settings
# ===========================================================================


@dataclass
class MotorSettings:
    """Computed protection settings for one motor, with the reasoning."""

    motor: Motor

    # 49/51 thermal overload
    pickup_multiple: float
    pickup_a: float
    trip_class: int
    tau_s: float

    # 51LR locked rotor
    lr_pickup_a: float
    lr_window_s: tuple[float, float]
    lr_time_s: float | None
    lr_dwell_s: float

    # 50 instantaneous
    inst_pickup_a: float
    inst_clearing_s: float

    # thermal simulations
    sim_hot: StartSimulation
    sim_cold: StartSimulation

    # MCCB selection
    mccb_frame_a: float | None
    mccb_icu_ka: float | None

    reasoning: dict[str, str] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)
    as_specified: bool = True
    recommendation: str = ""

    # -- device curve ------------------------------------------------------

    def thermal_time(self, i_a: float, hot: bool = True) -> float:
        theta_0 = self.theta_hot if hot else 0.0
        return thermal_trip_time(i_a, self.pickup_a, self.tau_s, theta_0)

    @property
    def theta_hot(self) -> float:
        """Thermal state of a motor running at rated load, (I_FLC/I_p)^2."""
        return (self.motor.flc_a / self.pickup_a) ** 2

    def locked_rotor_time(self, i_a: float) -> float:
        if self.lr_time_s is None or i_a < self.lr_pickup_a:
            return INF
        return self.lr_time_s

    def instantaneous_time(self, i_a: float) -> float:
        return self.inst_clearing_s if i_a >= self.inst_pickup_a else INF

    def device_time(self, i_a: float, hot: bool = True) -> float:
        """Composite clearing time: the fastest element that has picked up."""
        return min(
            self.thermal_time(i_a, hot=hot),
            self.locked_rotor_time(i_a),
            self.instantaneous_time(i_a),
        )

    @property
    def verdict(self) -> str:
        return worst(c.verdict for c in self.checks)


# ===========================================================================
# Setting design
# ===========================================================================


def _locked_rotor_window(motor: Motor, lr_pickup_a: float) -> tuple[float, float, float]:
    """Feasible window for the stall timer, and the dwell time that sets it.

    Lower bound  k1 * t_dwell        must not trip on a healthy start
    Upper bound  t_stall_hot / k2    must trip before the rotor is damaged

    t_dwell is the time the current actually stays above the 51LR pickup. For
    a star-delta start the star-stage current is I_LR/3, well below the 0.75 x
    I_LR pickup, so the timer only arms during the delta stage and the window
    is much wider than the total start time would suggest.
    """
    dwell = motor.time_above(lr_pickup_a)
    lower = STALL_MARGIN_ABOVE_START * dwell
    upper = motor.t_stall_hot_s / STALL_MARGIN_BELOW_WITHSTAND
    return lower, upper, dwell


def _damage_margin(motor: Motor, pickup_a: float, tau_s: float, theta_hot: float) -> tuple[float, float]:
    """Worst-case ratio of relay time to damage time in the accelerating region.

    Swept from DAMAGE_CURVE_MIN_MULTIPLE x FLC up to locked-rotor current.
    Returns (worst_ratio, current_at_worst). A ratio below 1.0 means the relay
    is slower than the motor can withstand at that current.
    """
    i_lo = max(DAMAGE_CURVE_MIN_MULTIPLE * motor.flc_a, pickup_a * 1.001)
    i_hi = motor.i_lr_a
    if i_hi <= i_lo:
        return 0.0, i_hi
    worst_ratio = 0.0
    worst_i = i_lo
    for i_a in np.geomspace(i_lo, i_hi, 120):
        t_relay = thermal_trip_time(float(i_a), pickup_a, tau_s, theta_hot)
        t_dmg = damage_time(motor, float(i_a), hot=True)
        if t_dmg <= 0:
            continue
        ratio = t_relay / t_dmg
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_i = float(i_a)
    return worst_ratio, worst_i


def _star_stall_margin(
    motor: Motor, pickup_a: float, tau_s: float, theta_hot: float
) -> float:
    """Ratio of thermal trip time to allowed time for a stall in the star stage.

    Only meaningful for a star-delta starter. The star-stage current sits below
    the 51LR pickup, so a rotor that fails to break away during the star stage
    is invisible to the stall timer and must be caught by the thermal element
    alone, before the rotor's I^2*t allowance at that current is used up:

        t_allowed = t_stall_hot * (I_LR / I_star)^2 = 9 * t_stall_hot

    Returns the ratio t_thermal / t_allowed; below 1.0 passes. Returns 0.0 for a
    DOL machine, where the check does not apply.
    """
    if not motor.is_star_delta:
        return 0.0
    i_star = motor.i_lr_a * STAR_DELTA_CURRENT_RATIO
    t_thermal = thermal_trip_time(i_star, pickup_a, tau_s, theta_hot)
    t_allowed = motor.t_stall_hot_s * (motor.i_lr_a / i_star) ** 2
    if t_allowed <= 0:
        return math.inf
    return t_thermal / t_allowed


def search_thermal_setting(motor: Motor) -> tuple[int, float, str]:
    """Independent search for the best thermal overload setting.

    Preference order, and why:
      1. Lowest trip class that survives the start. A lower class is a faster
         curve, which is better rotor protection during a stall -- the duty
         that actually destroys motors.
      2. Lowest pickup multiple at that class. A lower pickup is more
         sensitive to sustained small overloads.

    "Survives the start" means the hot restart does not trip AND leaves more
    than 10 per cent thermal headroom (theta_max <= 0.90), AND the relay stays
    faster than the damage curve throughout the accelerating region, AND -- for
    a star-delta machine -- the thermal element still covers a stall stuck in
    the star stage, which the 51LR element cannot see.

    That last constraint pulls against the first: raising the pickup buys
    thermal headroom for the start but slows the element down, and on a
    high-inertia star-delta drive the two requirements can leave no setting
    satisfying both with margin. The search says so rather than recommending a
    setting that trades one failure for another.
    """
    # (theta_max, trip_class, pickup) for settings that break no other check ...
    best_clean: tuple[float, int, float] | None = None
    # ... and the same allowing a star-stage stall violation, as a last resort.
    best_any: tuple[float, int, float] | None = None

    for trip_class in TRIP_CLASSES:
        tau = thermal_tau(trip_class)
        for kp in PICKUP_MULTIPLES:
            pickup = kp * motor.flc_a
            theta_hot = (motor.flc_a / pickup) ** 2
            sim = simulate_start(motor, pickup, tau, theta_hot, "hot restart", trace_points=1)
            ratio, _ = _damage_margin(motor, pickup, tau, theta_hot)
            if ratio >= 1.0:  # slower than the motor can withstand; never acceptable
                continue
            stall_ratio = _star_stall_margin(motor, pickup, tau, theta_hot)
            stall_ok = stall_ratio < 1.0

            if not sim.trips and sim.theta_max <= THETA_MARGINAL and stall_ok:
                extra = (
                    f" A stall held in the star stage is still cleared by the thermal "
                    f"element at {stall_ratio * 100:.0f} per cent of the rotor's I^2*t "
                    f"allowance."
                    if motor.is_star_delta
                    else ""
                )
                return (
                    trip_class,
                    kp,
                    f"Class {trip_class} at {kp * 100:.0f} per cent FLC is the fastest "
                    f"setting that rides through a hot restart with margin "
                    f"(theta_max = {sim.theta_max:.3f}) while staying inside the motor "
                    f"damage curve (worst relay/damage ratio {ratio:.2f}).{extra}",
                )

            key = (sim.theta_max, trip_class, kp)
            if best_any is None or key[0] < best_any[0]:
                best_any = key
            if stall_ok and (best_clean is None or key[0] < best_clean[0]):
                best_clean = key

    if best_clean is not None:
        theta_max, trip_class, kp = best_clean
        return (
            trip_class,
            kp,
            f"No setting achieves a 10 per cent thermal margin on a hot restart. "
            f"Class {trip_class} at {kp * 100:.0f} per cent FLC is the best available "
            f"(theta_max = {theta_max:.3f}) without compromising any other element. A "
            f"restart-inhibit timer is required so the machine cannot be restarted from "
            f"the fully hot state, and the permitted starts per hour must be confirmed "
            f"with the manufacturer.",
        )

    if best_any is not None:
        theta_max, trip_class, kp = best_any
        return (
            trip_class,
            kp,
            f"No setting satisfies the start-up and star-stage stall requirements at "
            f"once: raising the pickup enough to survive a hot restart slows the thermal "
            f"element past the rotor withstand for a stall held in star. Class "
            f"{trip_class} at {kp * 100:.0f} per cent FLC (theta_max = {theta_max:.3f}) "
            f"is the least-bad compromise, but the real fix is a dedicated motor "
            f"protection relay with an independent start-supervision element, or a soft "
            f"starter that removes the star-delta transition altogether.",
        )

    return (
        TRIP_CLASSES[-1],
        PICKUP_MULTIPLES[-1],
        "No setting satisfies both the start-up and the damage-curve constraint; "
        "the motor, the starting method or the driven load must change.",
    )


def _select_mccb(plant: Plant, motor: Motor, inst_pickup_a: float, ik_bus_ka: float):
    """Smallest MCCB frame that carries the motor and can break the bus fault."""
    candidates = [
        f
        for f in plant.mccb_frames
        if float(f["frame_a"]) >= motor.flc_a
        and float(f["frame_a"]) * 12.0 >= inst_pickup_a
        and float(f["icu_ka"]) >= ik_bus_ka
    ]
    if not candidates:
        return None, None
    chosen = min(candidates, key=lambda f: float(f["frame_a"]))
    return float(chosen["frame_a"]), float(chosen["icu_ka"])


def design_settings(
    plant: Plant,
    motor: Motor,
    sc: ShortCircuitStudy,
    use_recommended: bool = False,
) -> MotorSettings:
    """Compute and justify the complete protection setting set for one motor."""

    rec_class, rec_pickup, rec_reason = search_thermal_setting(motor)

    if use_recommended:
        trip_class, pickup_multiple = rec_class, rec_pickup
        as_specified = trip_class == motor.trip_class and math.isclose(
            pickup_multiple, motor.ol_pickup_multiple, rel_tol=1e-9
        )
    else:
        trip_class, pickup_multiple = motor.trip_class, motor.ol_pickup_multiple
        as_specified = True

    pickup_a = pickup_multiple * motor.flc_a
    tau = thermal_tau(trip_class)
    theta_hot = (motor.flc_a / pickup_a) ** 2

    sim_hot = simulate_start(motor, pickup_a, tau, theta_hot, "hot restart")
    sim_cold = simulate_start(motor, pickup_a, tau, 0.0, "cold start")

    lr_pickup = LOCKED_ROTOR_PICKUP_FRACTION * motor.i_lr_a
    lr_lo, lr_hi, dwell = _locked_rotor_window(motor, lr_pickup)
    lr_time = math.sqrt(lr_lo * lr_hi) if lr_lo <= lr_hi else None

    inst_pickup = INSTANTANEOUS_SET_MULTIPLE * motor.i_lr_a
    ik_min = sc.terminal_min[motor.tag].device_ik_a
    frame, icu = _select_mccb(plant, motor, inst_pickup, sc.bus_max.ik_ka)

    settings = MotorSettings(
        motor=motor,
        pickup_multiple=pickup_multiple,
        pickup_a=pickup_a,
        trip_class=trip_class,
        tau_s=tau,
        lr_pickup_a=lr_pickup,
        lr_window_s=(lr_lo, lr_hi),
        lr_time_s=lr_time,
        lr_dwell_s=dwell,
        inst_pickup_a=inst_pickup,
        inst_clearing_s=MCCB_INSTANTANEOUS_CLEARING_S,
        sim_hot=sim_hot,
        sim_cold=sim_cold,
        mccb_frame_a=frame,
        mccb_icu_ka=icu,
        as_specified=as_specified,
        recommendation=rec_reason,
    )

    settings.reasoning = _build_reasoning(motor, settings, sc, rec_class, rec_pickup)
    settings.checks = _build_checks(plant, motor, settings, sc, rec_class, rec_pickup)
    return settings


def _build_reasoning(
    motor: Motor,
    s: MotorSettings,
    sc: ShortCircuitStudy,
    rec_class: int,
    rec_pickup: float,
) -> dict[str, str]:
    """Plain-language justification for every number, for the report."""
    ik_min = sc.terminal_min[motor.tag].device_ik_a
    r: dict[str, str] = {}

    r["Thermal pickup"] = (
        f"{s.pickup_a:.0f} A = {s.pickup_multiple * 100:.0f} per cent of the "
        f"{motor.flc_a:.1f} A full-load current. IEC 60947-4-1 permits 100-115 per cent "
        f"for a machine with a 1.15 service factor: below 100 per cent the relay would "
        f"trip on rated load, above 115 per cent it would not protect the winding "
        f"insulation. Running at rated load therefore leaves the replica at "
        f"theta = {s.theta_hot:.3f}, i.e. {s.theta_hot * 100:.0f} per cent of thermal "
        f"capacity used before any start begins."
    )

    r["Trip class"] = (
        f"Class {s.trip_class}, giving a thermal time constant "
        f"tau = {s.tau_s:.0f} s derived from the IEC 60947-4-1 definition of the class "
        f"(tripping time from cold at 7.2 x I_e). tau is not assumed: "
        f"tau = {s.trip_class} / ln(7.2^2/(7.2^2-1)) = 51.32 x {s.trip_class}."
    )

    stages = ", ".join(
        f"{d['label']} {d['current_a']:.0f} A for {d['duration_s']:.1f} s "
        f"(theta {d['theta_start']:.3f} -> {d['theta_end']:.3f})"
        for d in s.sim_hot.stage_results
    )
    if s.sim_hot.trips:
        outcome = (
            f"Peak theta reaches the 1.000 trip threshold {s.sim_hot.trip_time_s:.2f} s into "
            f"a {motor.t_start_s:g} s start, so THE RELAY TRIPS on every hot restart. From "
            f"cold the peak is only {s.sim_cold.theta_max:.3f}, which is why this fault "
            f"survives commissioning and appears later as a machine that will not restart "
            f"while warm."
        )
    else:
        outcome = (
            f"Peak theta = {s.sim_hot.theta_max:.3f} against a trip threshold of 1.000, "
            f"leaving {s.sim_hot.headroom_pct:.0f} per cent headroom. From cold the peak is "
            f"only {s.sim_cold.theta_max:.3f}."
        )
    r["Start-up ride-through"] = (
        f"Thermal state integrated through the start from the hot condition: {stages}. "
        f"{outcome} The hot restart is the binding case."
    )

    if s.lr_time_s is not None:
        r["Locked-rotor pickup and time"] = (
            f"Pickup {s.lr_pickup_a:.0f} A = {LOCKED_ROTOR_PICKUP_FRACTION:.2f} x the "
            f"{motor.i_lr_a:.0f} A locked-rotor current, which is far above any credible "
            f"running overload and safely below LRC so a stall is always detected. "
            f"During a healthy start the current dwells above that pickup for "
            f"{s.lr_dwell_s:.1f} s"
            + (
                " (the delta stage only - the star-stage current of "
                f"{motor.i_lr_a * STAR_DELTA_CURRENT_RATIO:.0f} A is below pickup, so the "
                "stall timer does not arm during the star stage)"
                if motor.is_star_delta
                else " (the whole start)"
            )
            + f". The setting must therefore exceed {s.lr_window_s[0]:.1f} s "
            f"({STALL_MARGIN_ABOVE_START} x dwell) and stay below "
            f"{s.lr_window_s[1]:.1f} s (the {motor.t_stall_hot_s:.0f} s hot stall withstand "
            f"divided by {STALL_MARGIN_BELOW_WITHSTAND}). Set to "
            f"{s.lr_time_s:.1f} s, the geometric mean of the window, which balances the "
            f"margin on both sides."
        )
    else:
        r["Locked-rotor pickup and time"] = (
            f"NO FEASIBLE SETTING. The current dwells above the {s.lr_pickup_a:.0f} A pickup "
            f"for {s.lr_dwell_s:.1f} s during a healthy start, so the timer must exceed "
            f"{s.lr_window_s[0]:.1f} s; but the {motor.t_stall_hot_s:.0f} s hot stall "
            f"withstand caps it at {s.lr_window_s[1]:.1f} s. The window is empty. The "
            f"standard remedy (IEEE Std 242 Ch. 9) is to supervise the stall timer with a "
            f"zero-speed switch on the shaft, so a long healthy run-up is distinguished "
            f"from a genuine stall by shaft rotation rather than by time alone."
        )

    r["Instantaneous pickup"] = (
        f"{s.inst_pickup_a:.0f} A = {INSTANTANEOUS_SET_MULTIPLE} x the {motor.i_lr_a:.0f} A "
        f"locked-rotor current. The lower bound is "
        f"{INSTANTANEOUS_MIN_MULTIPLE} x LRC = {INSTANTANEOUS_MIN_MULTIPLE * motor.i_lr_a:.0f} A, "
        f"because the first-cycle asymmetrical (DC offset) component of inrush is seen by an "
        f"rms-sensing device as roughly 1.7 x the symmetrical LRC. The upper bound is the "
        f"minimum fault current at the terminals, {ik_min:.0f} A, divided by the "
        f"{INSTANTANEOUS_SENSITIVITY_MARGIN} sensitivity margin = "
        f"{ik_min / INSTANTANEOUS_SENSITIVITY_MARGIN:.0f} A. "
        f"The setting sits {ik_min / s.inst_pickup_a:.1f} x below the minimum fault current."
    )

    if s.mccb_frame_a is not None:
        r["MCCB selection"] = (
            f"{s.mccb_frame_a:.0f} A frame, Icu {s.mccb_icu_ka:.0f} kA. The frame carries the "
            f"{motor.flc_a:.1f} A full-load current and the breaking capacity exceeds the "
            f"{sc.bus_max.ik_ka:.1f} kA maximum bus fault, which is the duty for a fault "
            f"immediately downstream of the device. Note that the contactor and overload "
            f"relay cannot interrupt this current -- the MCCB is the short-circuit "
            f"protective device (IEC 60947-4-1 type-2 coordination)."
        )

    if (s.trip_class, s.pickup_multiple) != (rec_class, rec_pickup):
        r["Setting search"] = (
            f"Specified as Class {motor.trip_class} at "
            f"{motor.ol_pickup_multiple * 100:.0f} per cent FLC. Independent search "
            f"recommends Class {rec_class} at {rec_pickup * 100:.0f} per cent FLC instead: "
            f"{s.recommendation}"
        )
    else:
        r["Setting search"] = (
            f"The independent setting search confirms the specified values. {s.recommendation}"
        )

    return r


def _build_checks(
    plant: Plant,
    motor: Motor,
    s: MotorSettings,
    sc: ShortCircuitStudy,
    rec_class: int,
    rec_pickup: float,
) -> list[Check]:
    checks: list[Check] = []
    ik_min = sc.terminal_min[motor.tag].device_ik_a
    ik_bus = sc.bus_max.ik_ka

    # -- C1 overload pickup band ------------------------------------------
    in_band = 1.00 <= s.pickup_multiple <= 1.15
    checks.append(
        Check(
            "C1",
            "Thermal overload pickup within the permitted band",
            "1.00 <= I_p / I_FLC <= 1.15",
            f"{s.pickup_multiple:.2f}",
            PASS if in_band else FAIL,
            "IEC 60947-4-1",
        )
    )

    # -- C2 no trip during start ------------------------------------------
    hv = s.sim_hot.verdict
    note = ""
    if s.sim_hot.trips:
        note = (
            f"Relay trips {s.sim_hot.trip_time_s:.2f} s into a {motor.t_start_s:.1f} s start. "
            f"Recommended setting: Class {rec_class} at {rec_pickup * 100:.0f} per cent FLC."
        )
    elif hv == MARGINAL:
        note = (
            "Rides through, but with under 10 per cent thermal headroom on a hot restart. "
            "Fit a restart-inhibit timer so the machine cannot be restarted from the fully "
            "hot state, and confirm the permitted starts per hour with the manufacturer."
        )
    checks.append(
        Check(
            "C2",
            "No overload trip during a healthy start (hot restart, integrated)",
            f"theta_max < 1.00 (>{THETA_MARGINAL:.2f} is marginal)",
            f"theta_max = {s.sim_hot.theta_max:.3f} hot, {s.sim_cold.theta_max:.3f} cold",
            hv,
            "IEC 60255-149",
            note,
        )
    )

    # -- C3 relay below damage curve --------------------------------------
    ratio, i_at = _damage_margin(motor, s.pickup_a, s.tau_s, s.theta_hot)
    checks.append(
        Check(
            "C3",
            "Relay curve below the motor thermal damage curve",
            f"t_relay / t_damage < 1.0 for I >= {DAMAGE_CURVE_MIN_MULTIPLE:.0f} x FLC",
            f"worst ratio {ratio:.2f} at {i_at:.0f} A",
            PASS if ratio < 1.0 else FAIL,
            "IEEE Std 242 Ch. 9",
            "Checked only in the accelerating region: the constant-I^2*t extrapolation of "
            "the nameplate locked-rotor withstand is rotor-limited and is not valid at low "
            "overloads, where the limit is stator- and insulation-limited and cannot be "
            "derived from nameplate data.",
        )
    )

    # -- C4 stall timer window --------------------------------------------
    lo, hi = s.lr_window_s
    if s.lr_time_s is not None:
        v4 = PASS if (hi / lo) >= 1.3 else MARGINAL
        note4 = "" if v4 == PASS else "Window is narrow; confirm the start and stall times with the manufacturer."
    else:
        v4 = FAIL
        note4 = (
            "Empty window: the healthy start lasts longer than the rotor can withstand a "
            "stall. Supervise the stall timer with a zero-speed switch (ANSI 14) so that "
            "rotation, not elapsed time, distinguishes a run-up from a stall."
        )
    checks.append(
        Check(
            "C4",
            "Locked-rotor timer has a feasible coordination window",
            f"{STALL_MARGIN_ABOVE_START} x dwell <= t_LR <= t_stall_hot / {STALL_MARGIN_BELOW_WITHSTAND}",
            f"[{lo:.1f}, {hi:.1f}] s"
            + (f", set {s.lr_time_s:.1f} s" if s.lr_time_s is not None else ", EMPTY"),
            v4,
            "IEEE Std 242 Ch. 9",
            note4,
        )
    )

    # -- C5 stall during the star stage -----------------------------------
    if motor.is_star_delta:
        i_star = motor.i_lr_a * STAR_DELTA_CURRENT_RATIO
        t_th = thermal_trip_time(i_star, s.pickup_a, s.tau_s, s.theta_hot)
        # I^2*t equivalence: at one third of LRC the rotor withstands 9x longer.
        t_allow = motor.t_stall_hot_s * (motor.i_lr_a / i_star) ** 2
        ok = t_th <= t_allow
        checks.append(
            Check(
                "C5",
                "Stall stuck in the star stage is covered by the thermal element",
                "t_thermal(I_star) <= t_stall_hot x (I_LR/I_star)^2",
                f"{t_th:.0f} s vs {t_allow:.0f} s allowed",
                PASS if ok else FAIL,
                "I^2*t equivalence, IEEE Std 242 Ch. 9",
                f"The star-stage current ({i_star:.0f} A) is below the 51LR pickup "
                f"({s.lr_pickup_a:.0f} A), so a rotor stalled during the star stage is not "
                f"seen by the stall timer and must be caught by the thermal element alone.",
            )
        )
    else:
        checks.append(
            Check(
                "C5",
                "Stall stuck in the star stage is covered by the thermal element",
                "star-delta starters only",
                "not applicable (DOL)",
                NA,
                "IEEE Std 242 Ch. 9",
            )
        )

    # -- C6 instantaneous above asymmetrical inrush -----------------------
    mult = s.inst_pickup_a / motor.i_lr_a
    checks.append(
        Check(
            "C6",
            "Instantaneous set above the asymmetrical inrush",
            f"I_inst >= {INSTANTANEOUS_MIN_MULTIPLE} x I_LR",
            f"{mult:.2f} x I_LR = {s.inst_pickup_a:.0f} A",
            PASS if mult >= INSTANTANEOUS_MIN_MULTIPLE else FAIL,
            "IEEE Std 242 Ch. 9",
        )
    )

    # -- C7 instantaneous sensitivity -------------------------------------
    sens = ik_min / s.inst_pickup_a
    checks.append(
        Check(
            "C7",
            "Instantaneous sensitive to the minimum terminal fault",
            f"I_k,min / I_inst >= {INSTANTANEOUS_SENSITIVITY_MARGIN}",
            f"{sens:.2f} ({ik_min:.0f} A / {s.inst_pickup_a:.0f} A)",
            PASS if sens >= INSTANTANEOUS_SENSITIVITY_MARGIN else FAIL,
            "IEC 60364-4-41",
            "Minimum fault uses c = 0.95, conductors at 90 degC, and no motor contribution.",
        )
    )

    # -- C8 breaking capacity ---------------------------------------------
    if s.mccb_icu_ka is None:
        checks.append(
            Check(
                "C8",
                "Short-circuit protective device breaking capacity",
                f"Icu >= {ik_bus:.1f} kA",
                "no suitable frame in the library",
                FAIL,
                "IEC 60947-2",
            )
        )
    else:
        checks.append(
            Check(
                "C8",
                "Short-circuit protective device breaking capacity",
                f"Icu >= {ik_bus:.1f} kA (maximum bus fault)",
                f"{s.mccb_icu_ka:.0f} kA on a {s.mccb_frame_a:.0f} A frame",
                PASS if s.mccb_icu_ka >= ik_bus else FAIL,
                "IEC 60947-2",
            )
        )

    # -- C9 selectivity with the incomer ----------------------------------
    checks.append(check_selectivity(plant, s, sc))
    return checks


# ===========================================================================
# Selectivity
# ===========================================================================


def check_selectivity(plant: Plant, s: MotorSettings, sc: ShortCircuitStudy) -> Check:
    """Verify the incomer is slower than the motor device across the fault range.

    Swept logarithmically from the overload pickup to the maximum bus fault
    current. Selectivity is accepted at a given current if EITHER

        t_feeder - t_motor >= CTI          (definite-time region)
        t_feeder / t_motor >= CTI_RATIO    (inverse-time region)

    Both forms are needed: an additive interval is meaningless where both
    curves are steep, and a multiplicative one is meaningless at the 30 ms
    timescale of an instantaneous trip. IEEE Std 242 Ch. 15 recommends
    0.2-0.4 s for electromechanical relays and 0.1-0.2 s for the electronic
    trip units modelled here.
    """
    i_lo = s.pickup_a * 1.01
    i_hi = sc.bus_max.ik_a
    violations: list[tuple[float, float, float]] = []
    worst_gap = INF

    for i_a in np.geomspace(i_lo, i_hi, 400):
        t_m = s.device_time(float(i_a), hot=True)
        if not math.isfinite(t_m):
            continue
        t_f = plant.feeder.trip_time_s(float(i_a))
        if not math.isfinite(t_f):
            continue
        additive_ok = (t_f - t_m) >= CTI_S
        ratio_ok = t_f >= CTI_RATIO * t_m
        if not (additive_ok or ratio_ok):
            violations.append((float(i_a), t_m, t_f))
        worst_gap = min(worst_gap, t_f - t_m)

    if violations:
        i_bad, tm, tf = violations[0]
        return Check(
            "C9",
            "Selective with the incoming feeder breaker",
            f"t_feeder - t_motor >= {CTI_S} s, or t_feeder / t_motor >= {CTI_RATIO}",
            f"{len(violations)} violating currents, first at {i_bad:.0f} A "
            f"(motor {tm:.3f} s, feeder {tf:.3f} s)",
            FAIL,
            "IEEE Std 242 Ch. 15",
            "The incomer would trip with, or before, the motor device -- a motor-branch "
            "fault would black out the whole board.",
        )

    gap_txt = "feeder never operates in the swept range" if not math.isfinite(worst_gap) else f"smallest time gap {worst_gap:.3f} s"
    return Check(
        "C9",
        "Selective with the incoming feeder breaker",
        f"t_feeder - t_motor >= {CTI_S} s, or t_feeder / t_motor >= {CTI_RATIO}",
        f"selective over {i_lo:.0f} A to {i_hi:.0f} A; {gap_txt}",
        PASS,
        "IEEE Std 242 Ch. 15",
    )


# ===========================================================================
# Whole-plant driver
# ===========================================================================


def design_all(
    plant: Plant, sc: ShortCircuitStudy, use_recommended: bool = False
) -> dict[str, MotorSettings]:
    return {
        m.tag: design_settings(plant, m, sc, use_recommended=use_recommended)
        for m in plant.motors
    }
