"""Motor starting and voltage dip study."""

from __future__ import annotations

import dataclasses
import math

import pytest

from constants import SQRT3, STAR_DELTA_IMPEDANCE_RATIO, TEMP_XLPE_OPERATING_C
from models import Cable, parallel
import starting as stg


@pytest.fixture(scope="module")
def study(plant):
    return stg.run_study(plant)


# ---------------------------------------------------------------------------
# Limiting cases, where the answer is known without any calculation
# ---------------------------------------------------------------------------


def test_zero_impedance_source_holds_the_bus_at_nominal(make_plant):
    """An infinitely stiff source cannot dip. V_bus must be exactly 1.0."""
    plant = make_plant(
        lambda d: (
            d["source"].update(fault_mva=1.0e12),
            d["transformer"].update(uk_pct=1.0e-9, load_loss_kw=1.0e-12),
        )
    )
    r = stg.evaluate(plant, plant.motor("M1"), stg.DELTA, others_running=True)
    assert r.v_bus_pu == pytest.approx(1.0, abs=1e-6)


def test_zero_length_cable_puts_the_motor_at_bus_voltage(plant):
    """With no cable there is nothing between the bus and the terminals."""
    m = plant.motor("M5")
    short = dataclasses.replace(m, cable=Cable(m.cable.ctype, 1e-9, 1))
    r = stg.evaluate(plant, short, stg.DELTA, others_running=False)
    assert r.v_motor_pu == pytest.approx(r.v_bus_pu, abs=1e-6)


def test_cable_always_drops_voltage(plant, study):
    for m in plant.motors:
        for conn in (stg.DELTA, stg.STAR):
            r = study.get(m.tag, conn, False)
            assert r.v_motor_pu < r.v_bus_pu


def test_voltages_are_physical(plant, study):
    for r in study.cases.values():
        assert 0.0 < r.v_motor_pu < r.v_bus_pu <= 1.0


# ---------------------------------------------------------------------------
# Star-delta
# ---------------------------------------------------------------------------


def test_star_delta_draws_one_third_of_nameplate_current(plant, study):
    for m in plant.motors:
        dol = study.get(m.tag, stg.DELTA, False)
        star = study.get(m.tag, stg.STAR, False)
        assert star.i_nameplate_a == pytest.approx(dol.i_nameplate_a / 3.0, rel=1e-12)


def test_star_delta_dips_less_than_dol(plant, study):
    for m in plant.motors:
        for others in (False, True):
            dol = study.get(m.tag, stg.DELTA, others)
            star = study.get(m.tag, stg.STAR, others)
            assert star.v_bus_pu > dol.v_bus_pu
            assert star.v_motor_pu > dol.v_motor_pu
            assert star.i_start_a < dol.i_start_a


def test_star_delta_torque_is_one_third_at_equal_voltage(plant):
    """T goes as V^2, and star develops one third of delta torque."""
    m = plant.motor("M1")
    dol = stg.evaluate(plant, m, stg.DELTA, False)
    star = stg.evaluate(plant, m, stg.STAR, False)
    assert dol.torque_pu == pytest.approx(dol.v_motor_pu**2, rel=1e-12)
    assert star.torque_pu == pytest.approx(star.v_motor_pu**2 / 3.0, rel=1e-12)


def test_star_stage_torque_is_around_thirty_per_cent(plant, study):
    """The real constraint on a star-delta start, and worth pinning down."""
    for m in plant.motors:
        star = study.get(m.tag, stg.STAR, True)
        assert 0.25 < star.torque_pu < 0.35


def test_star_impedance_is_three_times_delta(plant, study):
    for m in plant.motors:
        dol = study.get(m.tag, stg.DELTA, False)
        star = study.get(m.tag, stg.STAR, False)
        assert abs(star.z_lr) == pytest.approx(
            STAR_DELTA_IMPEDANCE_RATIO * abs(dol.z_lr), rel=1e-12
        )


# ---------------------------------------------------------------------------
# Monotonicity: the direction of every effect must be right
# ---------------------------------------------------------------------------


def test_longer_cable_deepens_the_terminal_dip(plant):
    m = plant.motor("M5")
    results = []
    for length in (25.0, 100.0, 400.0):
        variant = dataclasses.replace(m, cable=Cable(m.cable.ctype, length, 1))
        results.append(stg.evaluate(plant, variant, stg.DELTA, False).v_motor_pu)
    assert results[0] > results[1] > results[2]


def test_bigger_cable_lifts_the_terminal_voltage(plant):
    m = plant.motor("M5")
    lib = plant.cable_library
    small = dataclasses.replace(m, cable=Cable(lib.get(16), 100.0, 1))
    large = dataclasses.replace(m, cable=Cable(lib.get(120), 100.0, 1))
    assert (
        stg.evaluate(plant, large, stg.DELTA, False).v_motor_pu
        > stg.evaluate(plant, small, stg.DELTA, False).v_motor_pu
    )


def test_running_load_deepens_the_dip(plant, study):
    for m in plant.motors:
        for conn in (stg.DELTA, stg.STAR):
            alone = study.get(m.tag, conn, False)
            loaded = study.get(m.tag, conn, True)
            assert loaded.v_bus_pu < alone.v_bus_pu
            assert loaded.v_motor_pu < alone.v_motor_pu


def test_weaker_source_deepens_the_dip(plant, make_plant):
    weak = make_plant(lambda d: d["transformer"].update(kva=400.0))
    assert (
        stg.evaluate(weak, weak.motor("M1"), stg.DELTA, True).v_bus_pu
        < stg.evaluate(plant, plant.motor("M1"), stg.DELTA, True).v_bus_pu
    )


def test_higher_lrc_deepens_the_dip(plant):
    m = plant.motor("M4")
    stiff = dataclasses.replace(m, lrc_multiple=5.0)
    heavy = dataclasses.replace(m, lrc_multiple=8.0)
    assert (
        stg.evaluate(plant, heavy, stg.DELTA, False).v_bus_pu
        < stg.evaluate(plant, stiff, stg.DELTA, False).v_bus_pu
    )


# ---------------------------------------------------------------------------
# Network solution consistency
# ---------------------------------------------------------------------------


def test_achieved_current_is_below_nameplate(plant, study):
    """Supply impedance means the motor never sees rated volts during its start."""
    for r in study.cases.values():
        assert r.i_start_a < r.i_nameplate_a


def test_starting_current_follows_ohms_law(plant, study):
    """I = V_bus_phase / |Z_branch| must hold exactly."""
    for m in plant.motors:
        for conn in (stg.DELTA, stg.STAR):
            r = study.get(m.tag, conn, True)
            z_branch = r.z_cable + r.z_lr
            expected = (r.v_bus_pu * plant.nominal_v / SQRT3) / abs(z_branch)
            assert r.i_start_a == pytest.approx(expected, rel=1e-12)


def test_voltage_divider_is_consistent(plant, study):
    for m in plant.motors:
        r = study.get(m.tag, stg.DELTA, False)
        z_branch = r.z_cable + r.z_lr
        expected_bus = abs(z_branch) / abs(r.z_source + z_branch)
        assert r.v_bus_pu == pytest.approx(expected_bus, rel=1e-12)
        assert r.v_motor_pu == pytest.approx(
            expected_bus * abs(r.z_lr) / abs(z_branch), rel=1e-12
        )


def test_complex_arithmetic_is_not_magnitude_addition(plant):
    """Adding magnitudes instead of phasors would understate the voltage.

    Guards the single most likely modelling error in a dip study.
    """
    m = plant.motor("M1")
    r = stg.evaluate(plant, m, stg.DELTA, False)
    z_branch = r.z_cable + r.z_lr
    naive = abs(z_branch) / (abs(r.z_source) + abs(z_branch))
    assert r.v_bus_pu > naive


# ---------------------------------------------------------------------------
# Steady-state voltage drop
# ---------------------------------------------------------------------------


def test_voltage_drop_formula(plant):
    for m in plant.motors:
        volts, pct = stg.voltage_drop(m)
        phi = math.acos(m.pf)
        expected = SQRT3 * m.flc_a * (
            m.cable.r_ohm(TEMP_XLPE_OPERATING_C) * math.cos(phi)
            + m.cable.x_ohm() * math.sin(phi)
        )
        assert volts == pytest.approx(expected, rel=1e-12)
        assert pct == pytest.approx(100.0 * volts / m.voltage_v)


def test_voltage_drop_within_limits_in_the_base_case(plant, study):
    for m in plant.motors:
        assert study.voltage_drop[m.tag][1] < 5.0


# ---------------------------------------------------------------------------
# Criteria and verdicts
# ---------------------------------------------------------------------------


def test_base_case_passes_every_voltage_criterion(plant, study):
    assert study.verdict == "PASS"


def test_criteria_actually_bite(make_plant):
    """Raise the limit above what the plant achieves and C10 must fail."""
    strict = make_plant(lambda d: d["criteria"].update(v_motor_min_pu=0.99))
    r = stg.evaluate(strict, strict.motor("M1"), stg.DELTA, True)
    c10 = next(c for c in r.checks if c.id == "C10")
    assert c10.verdict == "FAIL"


def test_bus_criterion_bites(make_plant):
    strict = make_plant(lambda d: d["criteria"].update(v_bus_min_pu=0.99))
    r = stg.evaluate(strict, strict.motor("M2"), stg.DELTA, True)
    c11 = next(c for c in r.checks if c.id == "C11")
    assert c11.verdict == "FAIL"


def test_recommendation_changes_when_dol_fails(make_plant):
    strict = make_plant(lambda d: d["criteria"].update(v_motor_min_pu=0.95))
    s = stg.run_study(strict)
    text = s.recommendations["M1"]
    assert "DOL is NOT acceptable" in text or "NEITHER method" in text


def test_recommendation_explains_a_redundant_star_delta(plant, study):
    """Where DOL would pass, the report must say why star-delta is still chosen."""
    text = study.recommendations["M1"]
    assert "DOL is electrically acceptable" in text
    assert "mechanical and thermal" in text


# ---------------------------------------------------------------------------
# Critical source strength
# ---------------------------------------------------------------------------


def test_critical_transformer_rating_is_self_consistent(plant):
    """At the returned rating, the terminal voltage must equal the limit."""
    limit = plant.criteria["v_motor_min_pu"]
    for m in plant.motors:
        kva = stg.critical_transformer_kva(plant, m, stg.DELTA, True, limit)
        assert kva is not None

        import copy

        # The search holds u_k AND X/R constant while scaling the rating, so
        # the rebuilt transformer must scale its load loss with the rating too;
        # keeping the absolute loss fixed would inflate R per cent at a smaller
        # rating and change the impedance angle.
        scaled = copy.copy(plant)
        scaled.transformer = dataclasses.replace(
            plant.transformer,
            kva=kva,
            load_loss_kw=plant.transformer.load_loss_kw * kva / plant.transformer.kva,
        )
        assert scaled.transformer.r_pct == pytest.approx(plant.transformer.r_pct)

        r = stg.evaluate(scaled, m, stg.DELTA, True)
        assert r.v_motor_pu == pytest.approx(limit, abs=1e-4)


def test_critical_rating_is_below_the_installed_rating(plant):
    """Every motor passes, so every critical rating must be smaller than fitted."""
    for m in plant.motors:
        kva = stg.critical_transformer_kva(plant, m, stg.DELTA, True)
        assert kva < plant.transformer.kva


def test_critical_rating_rises_with_a_longer_cable(plant):
    """A longer cable means the limit binds at a larger transformer."""
    m = plant.motor("M5")
    short = dataclasses.replace(m, cable=Cable(m.cable.ctype, 20.0, 1))
    long = dataclasses.replace(m, cable=Cable(m.cable.ctype, 200.0, 1))
    assert stg.critical_transformer_kva(plant, long) > stg.critical_transformer_kva(plant, short)


def test_unreachable_limit_returns_none(plant):
    """A limit no source weakness can breach must be reported as unreachable."""
    assert stg.critical_transformer_kva(plant, plant.motor("M5"), limit_pu=1e-6) is None
