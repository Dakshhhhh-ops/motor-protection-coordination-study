"""Thermal replica, trip classes, stall timing and instantaneous settings."""

from __future__ import annotations

import math

import pytest

from constants import (
    INSTANTANEOUS_MIN_MULTIPLE,
    LOCKED_ROTOR_PICKUP_FRACTION,
    PICKUP_MULTIPLES,
    STAR_DELTA_CURRENT_RATIO,
    THETA_MARGINAL,
    TRIP_CLASSES,
    thermal_tau,
)
import protection as pr
import shortcircuit as sc


@pytest.fixture(scope="module")
def study(plant):
    return sc.run_study(plant)


@pytest.fixture(scope="module")
def settings(plant, study):
    return pr.design_all(plant, study)


# ---------------------------------------------------------------------------
# Trip class and the thermal replica
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trip_class, expected", [(10, 513.4), (20, 1026.8), (30, 1540.2)])
def test_tau_derived_from_trip_class(trip_class, expected):
    assert thermal_tau(trip_class) == pytest.approx(expected, abs=0.2)
    assert thermal_tau(trip_class) == pytest.approx(51.32 * trip_class, rel=1e-3)


@pytest.mark.parametrize("trip_class", TRIP_CLASSES)
def test_curve_reproduces_its_own_trip_class(trip_class):
    """The defining property: from cold, at 7.2 x I_e, the curve trips in C seconds.

    This is the IEC 60947-4-1 definition of the class. If the curve did not
    reproduce it, tau would be an invented number rather than a derived one.
    """
    pickup = 100.0
    t = pr.thermal_trip_time(7.2 * pickup, pickup, thermal_tau(trip_class), theta_0=0.0)
    assert t == pytest.approx(float(trip_class), rel=1e-9)


@pytest.mark.parametrize("trip_class", TRIP_CLASSES)
def test_trip_class_lands_inside_the_iec_band(trip_class):
    """IEC 60947-4-1 Table 2 bands at 7.2 x I_e, from cold."""
    bands = {10: (4.0, 10.0), 20: (6.0, 20.0), 30: (9.0, 30.0)}
    lo, hi = bands[trip_class]
    t = pr.thermal_trip_time(7.2 * 100.0, 100.0, thermal_tau(trip_class), 0.0)
    assert lo < t <= hi


def test_no_trip_at_or_below_pickup():
    tau = thermal_tau(10)
    assert pr.thermal_trip_time(99.0, 100.0, tau) == math.inf
    assert pr.thermal_trip_time(100.0, 100.0, tau) == math.inf


def test_trip_time_diverges_as_current_approaches_pickup():
    """The curve has a vertical asymptote at pickup, approached logarithmically.

    t = tau ln[m^2/(m^2 - 1)] grows without bound as m -> 1, but only as
    ln(1/eps), so the growth is slow: at 1.0001 x pickup the trip time is still
    only about 9 tau. Asserting a fixed number of seconds here would be testing
    the arbitrary choice of test current, not the physics.
    """
    tau = thermal_tau(10)
    currents = [101.0, 100.1, 100.01, 100.001]
    times = [pr.thermal_trip_time(i, 100.0, tau) for i in currents]

    assert all(a < b for a, b in zip(times, times[1:]))
    assert times[-1] > 10.0 * tau
    assert pr.thermal_trip_time(100.0, 100.0, tau) == math.inf


def test_trip_time_falls_monotonically_with_current():
    tau = thermal_tau(20)
    times = [pr.thermal_trip_time(i, 100.0, tau) for i in (150, 300, 600, 1200)]
    assert all(a > b for a, b in zip(times, times[1:]))


def test_hot_curve_is_always_faster_than_cold():
    tau = thermal_tau(20)
    for i in (200.0, 400.0, 800.0):
        hot = pr.thermal_trip_time(i, 100.0, tau, theta_0=0.826)
        cold = pr.thermal_trip_time(i, 100.0, tau, theta_0=0.0)
        assert hot < cold


def test_thermal_state_starts_and_saturates_correctly():
    tau, pickup, current = 1000.0, 100.0, 300.0
    theta_inf = (current / pickup) ** 2
    assert pr.thermal_state(current, pickup, tau, 0.0, 0.0) == pytest.approx(0.0)
    assert pr.thermal_state(current, pickup, tau, 0.0, 1.0e6) == pytest.approx(theta_inf)


def test_running_at_rated_load_settles_at_the_expected_state():
    """A motor at I_FLC with a 1.10 pickup sits at (1/1.10)^2 = 0.826."""
    flc, pickup = 100.0, 110.0
    steady = pr.thermal_state(flc, pickup, thermal_tau(10), 0.0, 1.0e7)
    assert steady == pytest.approx((flc / pickup) ** 2, rel=1e-6)
    assert steady == pytest.approx(0.826, abs=0.001)


def test_integration_agrees_with_the_closed_form(plant):
    """Stepping the state to theta = 1 must reproduce the curve's trip time.

    The simulation and the curve are two views of the same differential
    equation; if they disagree, one of them is wrong.
    """
    tau, pickup, current = thermal_tau(10), 100.0, 500.0
    theta_0 = 0.3
    t_curve = pr.thermal_trip_time(current, pickup, tau, theta_0)
    theta_at_t = pr.thermal_state(current, pickup, tau, theta_0, t_curve)
    assert theta_at_t == pytest.approx(1.0, rel=1e-9)


def test_state_is_continuous_across_a_split_interval():
    """theta(2t) must equal stepping twice by t."""
    tau, pickup, current = 800.0, 100.0, 250.0
    one_step = pr.thermal_state(current, pickup, tau, 0.2, 10.0)
    two_steps = pr.thermal_state(
        current, pickup, tau, pr.thermal_state(current, pickup, tau, 0.2, 5.0), 5.0
    )
    assert one_step == pytest.approx(two_steps, rel=1e-12)


# ---------------------------------------------------------------------------
# Start simulation
# ---------------------------------------------------------------------------


def test_simulation_starts_from_the_given_state(plant):
    m = plant.motor("M4")
    sim = pr.simulate_start(m, 1.1 * m.flc_a, thermal_tau(10), 0.5, "test")
    assert sim.trace_theta[0] == pytest.approx(0.5)
    assert sim.theta_0 == 0.5


def test_simulation_covers_the_whole_start(plant):
    for m in plant.motors:
        sim = pr.simulate_start(m, 1.1 * m.flc_a, thermal_tau(20), 0.0, "test")
        assert sim.trace_t[-1] == pytest.approx(m.t_start_s)
        assert len(sim.stage_results) == len(m.starting_stages())


def test_hot_restart_is_the_binding_case(plant, settings):
    """Every motor must be hotter after a hot restart than after a cold one."""
    for m in plant.motors:
        s = settings[m.tag]
        assert s.sim_hot.theta_max > s.sim_cold.theta_max


def test_star_delta_deposits_far_less_heat_than_dol(plant):
    """The whole point of the integration: one third the current, one ninth the heat."""
    m = plant.motor("M1")
    assert m.is_star_delta
    pickup, tau = 1.1 * m.flc_a, thermal_tau(20)

    star_delta = pr.simulate_start(m, pickup, tau, 0.0, "cold")

    # Same machine and the same total run-up time, but held at full LRC.
    import dataclasses

    as_dol = dataclasses.replace(m, start_method="dol", t_star_s=None)
    dol = pr.simulate_start(as_dol, pickup, tau, 0.0, "cold")

    assert star_delta.theta_max < dol.theta_max


def test_known_failing_setting_is_detected(plant, settings):
    """M4 on Class 10 at 110 per cent must trip on a hot restart."""
    s = settings["M4"]
    assert s.trip_class == 10
    assert s.sim_hot.trips is True
    assert s.sim_hot.theta_max >= 1.0
    assert s.sim_hot.verdict == pr.FAIL
    # But it survives a cold start, which is why the distinction matters.
    assert s.sim_cold.trips is False


def test_verdict_bands(plant):
    m = plant.motor("M4")
    tau = thermal_tau(10)
    good = pr.simulate_start(m, 1.1 * m.flc_a, tau, 0.0, "cold")
    bad = pr.simulate_start(m, 1.1 * m.flc_a, tau, 0.826, "hot")
    assert good.verdict == pr.PASS
    assert bad.verdict == pr.FAIL
    assert 0.0 <= good.theta_max <= THETA_MARGINAL


# ---------------------------------------------------------------------------
# Damage curve
# ---------------------------------------------------------------------------


def test_damage_curve_passes_through_the_nameplate_point(plant):
    for m in plant.motors:
        assert pr.damage_time(m, m.i_lr_a, hot=True) == pytest.approx(m.t_stall_hot_s)
        assert pr.damage_time(m, m.i_lr_a, hot=False) == pytest.approx(m.t_stall_cold_s)


def test_damage_curve_is_constant_i_squared_t(plant):
    m = plant.motor("M1")
    t1 = pr.damage_time(m, m.i_lr_a)
    t2 = pr.damage_time(m, 2.0 * m.i_lr_a)
    assert t2 == pytest.approx(t1 / 4.0)


def test_cold_withstand_exceeds_hot(plant):
    for m in plant.motors:
        assert pr.damage_time(m, m.i_lr_a, hot=False) > pr.damage_time(m, m.i_lr_a, hot=True)


# ---------------------------------------------------------------------------
# Locked-rotor element
# ---------------------------------------------------------------------------


def test_locked_rotor_pickup_sits_between_overload_and_lrc(plant, settings):
    for m in plant.motors:
        s = settings[m.tag]
        assert s.lr_pickup_a == pytest.approx(LOCKED_ROTOR_PICKUP_FRACTION * m.i_lr_a)
        assert s.pickup_a < s.lr_pickup_a < m.i_lr_a


def test_stall_timer_sits_inside_its_window(plant, settings):
    for m in plant.motors:
        s = settings[m.tag]
        if s.lr_time_s is not None:
            lo, hi = s.lr_window_s
            assert lo <= s.lr_time_s <= hi


def test_stall_timer_clears_before_the_hot_withstand(plant, settings):
    for m in plant.motors:
        s = settings[m.tag]
        if s.lr_time_s is not None:
            assert s.lr_time_s < m.t_stall_hot_s


def test_star_delta_widens_the_stall_window(plant):
    """Only the delta stage arms the timer, so the lower bound drops."""
    import dataclasses

    m = plant.motor("M1")
    pickup = LOCKED_ROTOR_PICKUP_FRACTION * m.i_lr_a
    lo_sd, hi_sd, dwell_sd = pr._locked_rotor_window(m, pickup)

    as_dol = dataclasses.replace(m, start_method="dol", t_star_s=None)
    lo_dol, hi_dol, dwell_dol = pr._locked_rotor_window(as_dol, pickup)

    assert dwell_sd < dwell_dol
    assert lo_sd < lo_dol
    assert hi_sd == hi_dol  # the withstand is unchanged


def test_empty_stall_window_yields_no_setting(plant, make_plant):
    """A run-up longer than the rotor withstand must be reported, not fudged."""
    from conftest import motor_dict

    def mutate(data):
        m = motor_dict(data, "M4")
        m["t_start_s"] = 12.0
        m["t_stall_hot_s"] = 10.0
        m["t_stall_cold_s"] = 20.0

    broken = make_plant(mutate)
    study = sc.run_study(broken)
    s = pr.design_settings(broken, broken.motor("M4"), study)
    assert s.lr_time_s is None
    c4 = next(c for c in s.checks if c.id == "C4")
    assert c4.verdict == pr.FAIL
    assert "zero-speed switch" in c4.note


# ---------------------------------------------------------------------------
# Instantaneous element
# ---------------------------------------------------------------------------


def test_instantaneous_above_asymmetrical_inrush(plant, settings):
    for m in plant.motors:
        s = settings[m.tag]
        assert s.inst_pickup_a >= INSTANTANEOUS_MIN_MULTIPLE * m.i_lr_a


def test_instantaneous_below_minimum_terminal_fault(plant, settings, study):
    for m in plant.motors:
        s = settings[m.tag]
        assert s.inst_pickup_a < study.terminal_min[m.tag].device_ik_a


def test_device_curve_takes_the_fastest_element(plant, settings):
    for m in plant.motors:
        s = settings[m.tag]
        i = s.inst_pickup_a * 2.0
        assert s.device_time(i) == pytest.approx(s.inst_clearing_s)
        # Just below the instantaneous, a slower element must govern.
        assert s.device_time(s.inst_pickup_a * 0.99) > s.inst_clearing_s


def test_device_curve_is_monotonically_decreasing(plant, settings):
    import numpy as np

    for m in plant.motors:
        s = settings[m.tag]
        times = [s.device_time(float(i)) for i in np.geomspace(s.pickup_a * 1.01, 5.0e4, 200)]
        finite = [t for t in times if math.isfinite(t)]
        assert all(a >= b - 1e-12 for a, b in zip(finite, finite[1:]))


# ---------------------------------------------------------------------------
# Setting search
# ---------------------------------------------------------------------------


def test_search_returns_a_legal_setting(plant):
    for m in plant.motors:
        cls, kp, reason = pr.search_thermal_setting(m)
        assert cls in TRIP_CLASSES
        assert kp in PICKUP_MULTIPLES
        assert reason


def test_search_fixes_the_known_failures(plant, study):
    """Applying the searched settings must clear every start-up failure."""
    recommended = pr.design_all(plant, study, use_recommended=True)
    for tag, s in recommended.items():
        assert not s.sim_hot.trips, f"{tag} still trips on a hot restart"
        c2 = next(c for c in s.checks if c.id == "C2")
        assert c2.verdict != pr.FAIL


def test_search_does_not_break_the_star_stage_stall_check(plant, study):
    """Raising the pickup must not slow the element past the star-stall limit."""
    recommended = pr.design_all(plant, study, use_recommended=True)
    for tag, s in recommended.items():
        c5 = next(c for c in s.checks if c.id == "C5")
        assert c5.verdict != pr.FAIL, f"{tag} C5 broken by the recommended setting"


def test_star_stall_margin_is_zero_for_dol(plant):
    m = plant.motor("M4")
    assert pr._star_stall_margin(m, 1.1 * m.flc_a, thermal_tau(10), 0.8) == 0.0


def test_star_stall_margin_grows_with_pickup(plant):
    """A higher pickup means a slower element and less stall margin."""
    m = plant.motor("M3")
    tau = thermal_tau(30)
    low = pr._star_stall_margin(m, 1.05 * m.flc_a, tau, (1 / 1.05) ** 2)
    high = pr._star_stall_margin(m, 1.15 * m.flc_a, tau, (1 / 1.15) ** 2)
    assert high > low


# ---------------------------------------------------------------------------
# Selectivity
# ---------------------------------------------------------------------------


def test_base_case_is_selective(plant, settings):
    for m in plant.motors:
        c9 = next(c for c in settings[m.tag].checks if c.id == "C9")
        assert c9.verdict == pr.PASS


def test_selectivity_fails_when_the_incomer_instantaneous_is_enabled(make_plant):
    """Turn on the incomer's instantaneous element and selectivity must break."""
    def mutate(data):
        data["feeder_breaker"]["ii_a"] = 2000.0

    broken = make_plant(mutate)
    study = sc.run_study(broken)
    settings = pr.design_all(broken, study)
    verdicts = [
        next(c for c in settings[m.tag].checks if c.id == "C9").verdict
        for m in broken.motors
    ]
    assert pr.FAIL in verdicts


def test_selectivity_fails_with_no_short_time_delay(make_plant):
    def mutate(data):
        data["feeder_breaker"]["tsd_s"] = 0.01
        data["feeder_breaker"]["isd_a"] = 3000.0

    broken = make_plant(mutate)
    study = sc.run_study(broken)
    settings = pr.design_all(broken, study)
    verdicts = [
        next(c for c in settings[m.tag].checks if c.id == "C9").verdict
        for m in broken.motors
    ]
    assert pr.FAIL in verdicts


# ---------------------------------------------------------------------------
# Verdict aggregation
# ---------------------------------------------------------------------------


def test_worst_picks_the_most_severe():
    assert pr.worst([pr.PASS, pr.PASS]) == pr.PASS
    assert pr.worst([pr.PASS, pr.MARGINAL]) == pr.MARGINAL
    assert pr.worst([pr.MARGINAL, pr.FAIL]) == pr.FAIL
    assert pr.worst([pr.NA, pr.PASS]) == pr.PASS
    assert pr.worst([]) == pr.PASS
