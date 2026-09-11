"""Plant model: ratings, impedances, cable data and input validation."""

from __future__ import annotations

import math

import pytest

from conftest import CABLES_YAML, PLANT_YAML, motor_dict
from constants import ALPHA_CU, SQRT3, STAR_DELTA_IMPEDANCE_RATIO
from models import Cable, CableLibrary, Motor, Plant, Transformer, parallel


# ---------------------------------------------------------------------------
# Full-load current
# ---------------------------------------------------------------------------


def test_flc_matches_the_formula(plant):
    """I_FLC = P / (sqrt(3) V eta pf), computed independently."""
    for m in plant.motors:
        expected = (m.kw * 1000.0) / (SQRT3 * m.voltage_v * m.efficiency * m.pf)
        assert m.flc_a == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "tag, expected_flc",
    [("M1", 330.0), ("M2", 221.7), ("M3", 153.7), ("M4", 96.3), ("M5", 66.2)],
)
def test_flc_hand_calculated_values(plant, tag, expected_flc):
    """Guards the arithmetic against a silent regression."""
    assert plant.motor(tag).flc_a == pytest.approx(expected_flc, abs=0.1)


def test_flc_within_catalogue_tolerance(plant):
    """Computed FLC should land near catalogue values for 415 V 4-pole machines.

    Catalogue figures for these ratings, used as an independent sanity check on
    the efficiency and power factor data rather than as an input.
    """
    catalogue = {"M1": 330.0, "M2": 225.0, "M3": 156.0, "M4": 98.0, "M5": 68.0}
    for tag, cat in catalogue.items():
        computed = plant.motor(tag).flc_a
        assert abs(computed - cat) / cat < 0.04, f"{tag}: {computed:.1f} vs catalogue {cat}"


def test_locked_rotor_current_is_a_multiple_of_flc(plant):
    for m in plant.motors:
        assert m.i_lr_a == pytest.approx(m.flc_a * m.lrc_multiple)


def test_apparent_power_consistent_with_flc(plant):
    """S = sqrt(3) V I must agree with S = P/(eta pf)."""
    for m in plant.motors:
        from_current = SQRT3 * m.voltage_v * m.flc_a / 1000.0
        assert from_current == pytest.approx(m.s_rated_kva, rel=1e-12)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


def test_transformer_base_impedance_and_flc(plant):
    t = plant.transformer
    assert t.z_base_ohm == pytest.approx(t.lv_v**2 / (t.kva * 1000.0))
    assert t.flc_a == pytest.approx(t.kva * 1000.0 / (SQRT3 * t.lv_v))
    assert t.flc_a == pytest.approx(1391.2, abs=0.5)


def test_transformer_resistance_derived_from_load_loss(plant):
    t = plant.transformer
    assert t.r_pct == pytest.approx(100.0 * t.load_loss_kw / t.kva)
    # X follows from the impedance triangle, never assumed.
    assert t.x_pct == pytest.approx(math.sqrt(t.uk_pct**2 - t.r_pct**2))
    assert abs(t.z) == pytest.approx((t.uk_pct / 100.0) * t.z_base_ohm, rel=1e-12)


def test_transformer_x_over_r_is_plausible(plant):
    """A 1000 kVA distribution transformer sits around X/R = 4 to 7."""
    assert 3.0 < plant.transformer.x_over_r < 8.0


def test_transformer_rejects_impossible_loss():
    """Load loss implying R >= u_k is not physical and must be refused."""
    with pytest.raises(ValueError, match="not less than"):
        Transformer(kva=1000.0, hv_kv=11.0, lv_v=415.0, uk_pct=1.0, load_loss_kw=50.0)


def test_source_impedance_scales_with_fault_level(plant):
    z_250 = plant.source.z(415.0)
    from models import Source

    z_125 = Source(fault_mva=125.0, x_over_r=10.0).z(415.0)
    assert abs(z_125) == pytest.approx(2.0 * abs(z_250), rel=1e-12)


# ---------------------------------------------------------------------------
# Cables
# ---------------------------------------------------------------------------


def test_cable_temperature_correction(plant):
    """R_90 / R_20 = 1 + alpha * 70 = 1.2751 for copper."""
    ratio = 1.0 + ALPHA_CU * 70.0
    for m in plant.motors:
        c = m.cable
        assert c.r_ohm(90.0) / c.r_ohm(20.0) == pytest.approx(ratio, rel=1e-12)
    assert ratio == pytest.approx(1.2751, abs=1e-4)


def test_cable_impedance_scales_with_length(plant):
    lib = plant.cable_library
    short = Cable(lib.get(95), 50.0)
    long = Cable(lib.get(95), 150.0)
    assert long.r_ohm(20.0) == pytest.approx(3.0 * short.r_ohm(20.0), rel=1e-12)
    assert long.x_ohm() == pytest.approx(3.0 * short.x_ohm(), rel=1e-12)


def test_parallel_runs_halve_impedance(plant):
    lib = plant.cable_library
    one = Cable(lib.get(120), 60.0, runs=1)
    two = Cable(lib.get(120), 60.0, runs=2)
    assert two.r_ohm(20.0) == pytest.approx(one.r_ohm(20.0) / 2.0, rel=1e-12)
    assert two.x_ohm() == pytest.approx(one.x_ohm() / 2.0, rel=1e-12)


def test_cable_reactance_is_temperature_independent(plant):
    c = plant.motor("M5").cable
    assert c.z(20.0).imag == pytest.approx(c.z(90.0).imag)


def test_cable_library_covers_the_declared_range(plant):
    sizes = plant.cable_library.sizes_ascending()
    assert sizes[0] == 4.0 and sizes[-1] == 240.0
    assert len(sizes) == 13
    # Resistance must fall monotonically as cross-section rises.
    r = [plant.cable_library.get(s).r20_ohm_km for s in sizes]
    assert all(a > b for a, b in zip(r, r[1:]))


def test_unknown_cable_size_is_rejected(plant):
    with pytest.raises(KeyError, match="not in the cable library"):
        plant.cable_library.get(300)


def test_cable_rejects_bad_geometry(plant):
    lib = plant.cable_library
    with pytest.raises(ValueError):
        Cable(lib.get(25), -10.0)
    with pytest.raises(ValueError):
        Cable(lib.get(25), 10.0, runs=0)


# ---------------------------------------------------------------------------
# Star-delta
# ---------------------------------------------------------------------------


def test_star_equivalent_impedance_is_three_times_delta(plant):
    for m in plant.motors:
        assert abs(m.z_lr("star")) == pytest.approx(
            STAR_DELTA_IMPEDANCE_RATIO * abs(m.z_lr("delta")), rel=1e-12
        )


def test_locked_rotor_impedance_draws_locked_rotor_current(plant):
    """|Z_LR| must be exactly the impedance that draws I_LR at rated volts."""
    for m in plant.motors:
        i = m.voltage_v / (SQRT3 * abs(m.z_lr("delta")))
        assert i == pytest.approx(m.i_lr_a, rel=1e-12)


def test_star_delta_initial_current_is_one_third(plant):
    m = plant.motor("M1")
    assert m.is_star_delta
    assert m.i_start_initial_a == pytest.approx(m.i_lr_a / 3.0, rel=1e-12)


def test_dol_initial_current_is_full_lrc(plant):
    m = plant.motor("M4")
    assert not m.is_star_delta
    assert m.i_start_initial_a == pytest.approx(m.i_lr_a)


def test_starting_stages_durations_sum_to_start_time(plant):
    for m in plant.motors:
        total = sum(d for _, d, _ in m.starting_stages())
        assert total == pytest.approx(m.t_start_s)


def test_dwell_above_pickup_excludes_the_star_stage(plant):
    """The key star-delta effect: the stall timer only arms in delta."""
    m = plant.motor("M1")
    pickup = 0.75 * m.i_lr_a
    assert m.time_above(pickup) == pytest.approx(m.t_delta_s)
    assert m.time_above(pickup) < m.t_start_s

    dol = plant.motor("M4")
    assert dol.time_above(0.75 * dol.i_lr_a) == pytest.approx(dol.t_start_s)


def test_dwell_above_zero_is_the_whole_start(plant):
    for m in plant.motors:
        assert m.time_above(0.0) == pytest.approx(m.t_start_s)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _motor(**overrides) -> dict:
    base = dict(
        tag="MX", service="test", kw=55.0, voltage_v=415.0, efficiency=0.93, pf=0.85,
        lrc_multiple=6.5, start_method="dol", t_start_s=4.0, t_stall_hot_s=10.0,
        t_stall_cold_s=20.0, trip_class=10, ol_pickup_multiple=1.10,
    )
    base.update(overrides)
    return base


def test_motor_rejects_unknown_start_method(plant):
    cable = plant.motor("M4").cable
    with pytest.raises(ValueError, match="start_method"):
        Motor(cable=cable, **_motor(start_method="soft-start"))


def test_star_delta_requires_a_star_duration(plant):
    cable = plant.motor("M4").cable
    with pytest.raises(ValueError, match="t_star_s"):
        Motor(cable=cable, **_motor(start_method="star-delta"))


def test_star_stage_must_be_shorter_than_the_start(plant):
    cable = plant.motor("M4").cable
    with pytest.raises(ValueError, match="must be greater than zero"):
        Motor(cable=cable, **_motor(start_method="star-delta", t_star_s=9.0, t_start_s=4.0))


def test_cold_stall_cannot_be_shorter_than_hot(plant):
    cable = plant.motor("M4").cable
    with pytest.raises(ValueError, match="cold stall"):
        Motor(cable=cable, **_motor(t_stall_hot_s=20.0, t_stall_cold_s=10.0))


def test_duplicate_motor_tags_are_rejected(make_plant):
    def mutate(data):
        data["motors"][1]["tag"] = data["motors"][0]["tag"]

    with pytest.raises(ValueError, match="unique"):
        make_plant(mutate)


# ---------------------------------------------------------------------------
# Loading and network helpers
# ---------------------------------------------------------------------------


def test_parallel_of_equal_impedances_halves(plant):
    z = complex(0.2, 0.1)
    assert parallel(z, z) == pytest.approx(z / 2.0)


def test_parallel_is_order_independent():
    a, b, c = complex(0.1, 0.2), complex(0.3, 0.05), complex(1.0, 1.0)
    assert parallel(a, b, c) == pytest.approx(parallel(c, a, b))


def test_load_impedance_excludes_named_motors(plant):
    everything = plant.load_impedance()
    without_m1 = plant.load_impedance(exclude_tags=["M1"])
    # Removing load raises the equivalent impedance.
    assert abs(without_m1) > abs(everything)


def test_diversified_kva_uses_complex_addition(plant):
    """Complex addition must give less than the arithmetic sum of kVA."""
    arithmetic = plant.static_load.kva + plant.motor_diversity_factor * sum(
        m.s_rated_kva for m in plant.motors
    )
    assert plant.diversified_kva < arithmetic
    assert 0.80 < plant.diversified_pf < 0.95


def test_transformer_loading_is_consistent(plant):
    expected = 100.0 * plant.diversified_kva / plant.transformer.kva
    assert plant.transformer_loading_pct == pytest.approx(expected)


def test_largest_motor_is_identified(plant):
    assert plant.largest_motor.tag == "M1"


def test_z_source_is_utility_plus_transformer(plant):
    assert plant.z_source == pytest.approx(
        plant.source.z(plant.nominal_v) + plant.transformer.z
    )


# ---------------------------------------------------------------------------
# Scenario overrides
# ---------------------------------------------------------------------------


def test_scenario_overrides_are_applied():
    base = Plant.from_yaml(PLANT_YAML, CABLES_YAML)
    weak = Plant.from_yaml(PLANT_YAML, CABLES_YAML, scenario="weak_source")
    assert weak.transformer.kva == 630.0
    assert weak.source.fault_mva == 100.0
    assert abs(weak.z_source) > abs(base.z_source)
    assert weak.scenario == "weak_source"
    assert weak.scenario_description


def test_scenario_can_override_a_motor_by_tag():
    dol = Plant.from_yaml(PLANT_YAML, CABLES_YAML, scenario="m1_dol")
    assert not dol.motor("M1").is_star_delta
    # Other motors must be untouched.
    assert dol.motor("M3").is_star_delta


def test_unknown_scenario_is_rejected():
    with pytest.raises(KeyError, match="not defined"):
        Plant.from_yaml(PLANT_YAML, CABLES_YAML, scenario="does_not_exist")


def test_override_of_a_nonexistent_path_is_rejected(make_plant):
    def mutate(data):
        data["scenarios"]["bad"] = {"overrides": {"transformer.not_a_field": 1}}

    with pytest.raises(KeyError, match="does not exist"):
        make_plant(mutate, scenario="bad")


def test_override_of_unknown_motor_tag_is_rejected(make_plant):
    def mutate(data):
        data["scenarios"]["bad"] = {"overrides": {"motors.M9.t_start_s": 1}}

    with pytest.raises(KeyError, match="unknown motor tag"):
        make_plant(mutate, scenario="bad")
