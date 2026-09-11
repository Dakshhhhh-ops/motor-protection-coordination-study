"""Short-circuit calculation to IEC 60909-0."""

from __future__ import annotations

import math

import pytest

from constants import C_MAX_LV, C_MIN_LV, MOTOR_R_OVER_X_LV, SQRT3, kappa
import shortcircuit as sc
from models import Cable


# ---------------------------------------------------------------------------
# First-principles anchors
# ---------------------------------------------------------------------------


def test_infinite_source_bus_fault_equals_flc_over_z_pu(make_plant):
    """The textbook check: with an infinite source, I_k = I_FLC / Z_pu.

    A 1000 kVA transformer at 5 per cent gives 1391.2 / 0.05 = 27 824 A. The
    voltage factor and the K_T correction are switched off so the comparison is
    against the raw textbook result rather than the IEC-corrected one.
    """
    plant = make_plant(lambda d: d["source"].update(fault_mva=1.0e9))
    result = sc.bus_fault(plant, maximum=True, include_motors=False, apply_kt=False)
    textbook = plant.transformer.flc_a / (plant.transformer.uk_pct / 100.0)
    assert result.ik_a == pytest.approx(C_MAX_LV * textbook, rel=1e-3)


def test_ik_formula_is_c_un_over_root3_z(plant):
    r = sc.bus_fault(plant, maximum=True, include_motors=False)
    assert r.ik_a == pytest.approx(r.c * plant.nominal_v / (SQRT3 * abs(r.z)), rel=1e-12)


def test_peak_current_uses_kappa_and_root2(plant):
    r = sc.bus_fault(plant)
    assert r.ip_a == pytest.approx(r.kappa * math.sqrt(2.0) * r.ik_a, rel=1e-9)


@pytest.mark.parametrize("r_over_x", [0.01, 0.1, 0.42, 1.0, 10.0])
def test_kappa_stays_inside_its_physical_bounds(r_over_x):
    k = kappa(r_over_x, 1.0)
    assert 1.02 <= k <= 2.0


def test_kappa_limits():
    assert kappa(0.0, 1.0) == pytest.approx(2.0)       # purely inductive
    assert kappa(1.0, 0.0) == pytest.approx(1.02)      # purely resistive
    # IEC 60909-0 gives kappa = 1.3 for LV motors at R/X = 0.42.
    assert kappa(MOTOR_R_OVER_X_LV, 1.0) == pytest.approx(1.3, abs=0.02)


def test_transformer_correction_factor_formula(plant):
    kt = sc.transformer_correction_factor(plant)
    x_t = plant.transformer.x_pct / 100.0
    assert kt == pytest.approx(0.95 * C_MAX_LV / (1.0 + 0.6 * x_t), rel=1e-12)
    assert 0.9 < kt < 1.0


def test_correction_factor_raises_the_fault_current(plant):
    """K_T scales the transformer impedance only, not the series source impedance.

    So the current rises by less than 1/K_T: the utility impedance is
    unscaled and dilutes the effect. Getting this the wrong way round would
    silently overstate every fault current in the study.
    """
    with_kt = sc.bus_fault(plant, include_motors=False, apply_kt=True)
    without = sc.bus_fault(plant, include_motors=False, apply_kt=False)
    kt = sc.transformer_correction_factor(plant)

    assert with_kt.ik_a > without.ik_a
    assert with_kt.ik_a / without.ik_a < 1.0 / kt
    # The exact relationship is the ratio of the total impedance magnitudes.
    assert with_kt.ik_a / without.ik_a == pytest.approx(
        abs(without.z) / abs(with_kt.z), rel=1e-12
    )


def test_correction_factor_equals_one_over_kt_with_a_stiff_source(make_plant):
    """Remove the utility impedance and the dilution disappears."""
    plant = make_plant(lambda d: d["source"].update(fault_mva=1.0e9))
    kt = sc.transformer_correction_factor(plant)
    with_kt = sc.bus_fault(plant, include_motors=False, apply_kt=True)
    without = sc.bus_fault(plant, include_motors=False, apply_kt=False)
    assert with_kt.ik_a / without.ik_a == pytest.approx(1.0 / kt, rel=1e-6)


# ---------------------------------------------------------------------------
# Motor impedance and contribution
# ---------------------------------------------------------------------------


def test_motor_impedance_draws_locked_rotor_current(plant):
    """Z_M is by definition the impedance drawing I_LR at rated voltage."""
    for m in plant.motors:
        z = sc.motor_impedance(m)
        i = m.voltage_v / (SQRT3 * abs(z))
        assert i == pytest.approx(m.i_lr_a, rel=1e-9)


def test_motor_impedance_angle_follows_the_standard(plant):
    for m in plant.motors:
        z = sc.motor_impedance(m)
        assert z.real / z.imag == pytest.approx(MOTOR_R_OVER_X_LV, rel=1e-12)


def test_motor_contribution_increases_the_bus_fault(plant):
    with_motors = sc.bus_fault(plant, include_motors=True)
    without = sc.bus_fault(plant, include_motors=False)
    assert with_motors.ik_a > without.ik_a
    # For a bus of this composition the uplift is a material 15-25 per cent,
    # not a rounding effect.
    uplift = (with_motors.ik_a - without.ik_a) / without.ik_a
    assert 0.10 < uplift < 0.35


def test_study_reports_the_motor_uplift(plant):
    study = sc.run_study(plant)
    expected = 100.0 * (study.bus_max.ik_a - study.bus_max_no_motors.ik_a) / study.bus_max_no_motors.ik_a
    assert study.motor_contribution_pct == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Maximum vs minimum duty
# ---------------------------------------------------------------------------


def test_minimum_is_always_below_maximum(plant):
    study = sc.run_study(plant)
    assert study.bus_min.ik_a < study.bus_max.ik_a
    for tag in study.terminal_max:
        assert study.terminal_min[tag].device_ik_a < study.terminal_max[tag].device_ik_a


def test_duty_uses_the_right_voltage_factors(plant):
    assert sc.bus_fault(plant, maximum=True).c == C_MAX_LV
    assert sc.bus_fault(plant, maximum=False).c == C_MIN_LV


def test_duty_uses_the_right_conductor_temperature(plant):
    study = sc.run_study(plant)
    assert study.terminal_max["M5"].temp_c == 20.0
    assert study.terminal_min["M5"].temp_c == 90.0


def test_minimum_duty_excludes_motor_contribution(plant):
    study = sc.run_study(plant)
    assert study.bus_min.includes_motors is False
    assert study.bus_min.motor_ik_a == 0.0


# ---------------------------------------------------------------------------
# Network topology
# ---------------------------------------------------------------------------


def test_fault_current_falls_along_the_cable(plant):
    """A terminal fault must always be weaker than a bus fault."""
    study = sc.run_study(plant)
    for m in plant.motors:
        assert study.terminal_max[m.tag].device_ik_a < study.bus_max.ik_a


def test_longer_cable_gives_a_weaker_fault(plant, make_plant):
    from conftest import motor_dict

    def mutate(data):
        motor_dict(data, "M5")["cable"]["length_m"] = 300.0

    longer = make_plant(mutate)
    base_ik = sc.terminal_fault(plant, plant.motor("M5")).device_ik_a
    long_ik = sc.terminal_fault(longer, longer.motor("M5")).device_ik_a
    assert long_ik < base_ik


def test_bigger_cable_gives_a_stronger_fault(plant, make_plant):
    from conftest import motor_dict

    def mutate(data):
        motor_dict(data, "M5")["cable"]["size_mm2"] = 240

    bigger = make_plant(mutate)
    assert (
        sc.terminal_fault(bigger, bigger.motor("M5")).device_ik_a
        > sc.terminal_fault(plant, plant.motor("M5")).device_ik_a
    )


def test_terminal_fault_excludes_the_local_machine_from_device_current(plant):
    """The faulted motor back-feeds the fault, not through its own device."""
    study = sc.run_study(plant)
    for m in plant.motors:
        r = study.terminal_max[m.tag]
        assert r.device_ik_a < r.total_ik_a
        assert r.motor_ik_a > 0.0


def test_local_backfeed_matches_the_motor_impedance(plant):
    study = sc.run_study(plant)
    for m in plant.motors:
        r = study.terminal_max[m.tag]
        expected = r.c * plant.nominal_v / (SQRT3 * abs(sc.motor_impedance(m)))
        assert r.motor_ik_a == pytest.approx(expected, rel=1e-9)


def test_ordering_of_terminal_faults_follows_cable_impedance(plant):
    """The motor with the highest cable impedance must have the weakest fault."""
    study = sc.run_study(plant)
    by_z = sorted(plant.motors, key=lambda m: abs(m.cable.z(20.0)))
    by_ik = sorted(plant.motors, key=lambda m: -study.terminal_max[m.tag].device_ik_a)
    assert [m.tag for m in by_z] == [m.tag for m in by_ik]


# ---------------------------------------------------------------------------
# Regression anchors from the base case
# ---------------------------------------------------------------------------


def test_base_case_headline_values(plant):
    """Hand-checked values for the documented base case."""
    study = sc.run_study(plant)
    assert study.bus_max_no_motors.ik_ka == pytest.approx(27.9, abs=0.2)
    assert study.bus_max.ik_ka == pytest.approx(33.3, abs=0.2)
    assert study.bus_max.ip_ka == pytest.approx(70.1, abs=0.5)
    assert study.terminal_min["M5"].device_ik_a == pytest.approx(2368, rel=0.01)


def test_weak_source_reduces_every_fault_current(plant):
    from conftest import CABLES_YAML, PLANT_YAML
    from models import Plant

    weak = Plant.from_yaml(PLANT_YAML, CABLES_YAML, scenario="weak_source")
    assert sc.run_study(weak).bus_max.ik_a < sc.run_study(plant).bus_max.ik_a
