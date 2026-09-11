"""
Acceptance checks C1-C19, end-to-end runs, and the report pipeline.

The governing principle here: a check that cannot fail is worthless. Every
numbered check is therefore exercised twice - once against the base case, where
its verdict is known, and once against a plant deliberately broken in the one
way that check exists to catch.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

import checks as ck
from conftest import CABLES_YAML, PLANT_YAML, motor_dict
from constants import K_ADIABATIC_CU_XLPE, K_AMBIENT_40C_XLPE, K_GROUPING
import protection as pr
import report as rp
import shortcircuit as sc
import starting as stg

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def results_for(plant) -> rp.StudyResults:
    """Run the whole study for a plant, without rendering figures."""
    study = sc.run_study(plant)
    settings = pr.design_all(plant, study)
    start = stg.run_study(plant)
    return rp.StudyResults(
        plant=plant,
        sc=study,
        settings=settings,
        starting=start,
        plant_checks=ck.plant_checks(plant, study, start),
        cable_checks={
            m.tag: ck.cable_checks(plant, m, settings[m.tag], study, start)
            for m in plant.motors
        },
        figures={},
    )


def verdict_of(results: rp.StudyResults, check_id: str, tag: str | None = None) -> str:
    """Most severe verdict recorded for one check id, optionally for one item.

    Deliberately not `protection.worst`, which treats INFO and N/A as less
    severe than PASS and so can never return either of them. Here they must be
    distinguishable, because C5 is N/A for DOL motors and C19 reports INFO.
    """
    hits = [
        c.verdict
        for t, c in results.all_checks()
        if c.id == check_id and (tag is None or t == tag)
    ]
    assert hits, f"check {check_id} was never evaluated"
    for verdict in (pr.FAIL, pr.MARGINAL, pr.INFO):
        if verdict in hits:
            return verdict
    return pr.NA if all(h == pr.NA for h in hits) else pr.PASS


# ---------------------------------------------------------------------------
# Base case: every check is evaluated, and none is silently absent
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base(plant):
    return results_for(plant)


def test_every_check_id_is_present(base):
    ids = {c.id for _, c in base.all_checks()}
    expected = {f"C{n}" for n in range(1, 20)}
    assert expected <= ids, f"missing checks: {sorted(expected - ids)}"


def test_every_check_has_a_reference_and_criterion(base):
    for tag, c in base.all_checks():
        assert c.criterion, f"{tag}/{c.id} has no criterion"
        assert c.value, f"{tag}/{c.id} has no calculated value"
        assert c.verdict in (pr.PASS, pr.MARGINAL, pr.FAIL, pr.INFO, pr.NA)
        if c.verdict != pr.NA:
            assert c.reference, f"{tag}/{c.id} cites no standard"


def test_base_case_findings_are_the_documented_ones(base):
    """The base case is an audit of a real starting design, and it finds things."""
    fails = [(t, c.id) for t, c in base.findings() if c.verdict == pr.FAIL]
    assert ("M4", "C2") in fails
    assert ("M5", "C2") in fails
    assert base.overall == pr.FAIL


def test_recommended_settings_clear_every_failure(plant):
    study = sc.run_study(plant)
    settings = pr.design_all(plant, study, use_recommended=True)
    start = stg.run_study(plant)
    res = rp.StudyResults(
        plant=plant, sc=study, settings=settings, starting=start,
        plant_checks=ck.plant_checks(plant, study, start),
        cable_checks={
            m.tag: ck.cable_checks(plant, m, settings[m.tag], study, start)
            for m in plant.motors
        },
        figures={},
    )
    assert not [c for _, c in res.findings() if c.verdict == pr.FAIL]


def test_counts_add_up(base):
    counts = base.counts()
    assert sum(counts.values()) == len(base.all_checks())


# ---------------------------------------------------------------------------
# C1  overload pickup band
# ---------------------------------------------------------------------------


def test_c1_passes_in_the_base_case(base):
    assert verdict_of(base, "C1") == pr.PASS


def test_c1_fails_outside_the_permitted_band(make_plant):
    def mutate(data):
        motor_dict(data, "M2")["ol_pickup_multiple"] = 1.60

    assert verdict_of(results_for(make_plant(mutate)), "C1", "M2") == pr.FAIL


# ---------------------------------------------------------------------------
# C2  no trip during a healthy start
# ---------------------------------------------------------------------------


def test_c2_fails_for_the_known_bad_setting(base):
    assert verdict_of(base, "C2", "M4") == pr.FAIL


def test_c2_passes_with_a_generous_class(make_plant):
    def mutate(data):
        m = motor_dict(data, "M4")
        m["trip_class"] = 20
        m["ol_pickup_multiple"] = 1.15

    assert verdict_of(results_for(make_plant(mutate)), "C2", "M4") != pr.FAIL


def test_c2_fails_when_the_start_is_made_far_too_long(make_plant):
    def mutate(data):
        m = motor_dict(data, "M1")
        m["t_star_s"] = 20.0
        m["t_start_s"] = 40.0
        m["t_stall_hot_s"] = 60.0
        m["t_stall_cold_s"] = 90.0

    assert verdict_of(results_for(make_plant(mutate)), "C2", "M1") == pr.FAIL


# ---------------------------------------------------------------------------
# C3  relay below the damage curve
# ---------------------------------------------------------------------------


def test_c3_passes_in_the_base_case(base):
    assert verdict_of(base, "C3") == pr.PASS


def test_c3_fails_when_the_rotor_withstand_is_tiny(make_plant):
    """A motor that cannot withstand a stall for long outruns any relay."""
    def mutate(data):
        m = motor_dict(data, "M2")
        m["t_stall_hot_s"] = 1.2
        m["t_stall_cold_s"] = 2.0
        m["t_start_s"] = 0.8

    assert verdict_of(results_for(make_plant(mutate)), "C3", "M2") == pr.FAIL


# ---------------------------------------------------------------------------
# C4  stall timer window
# ---------------------------------------------------------------------------


def test_c4_passes_in_the_base_case(base):
    assert verdict_of(base, "C4") in (pr.PASS, pr.MARGINAL)


def test_c4_fails_when_the_run_up_outlasts_the_rotor(make_plant):
    def mutate(data):
        m = motor_dict(data, "M3")
        m["t_star_s"] = 12.0
        m["t_start_s"] = 22.0
        m["t_stall_hot_s"] = 15.0

    res = results_for(make_plant(mutate))
    assert verdict_of(res, "C4", "M3") == pr.FAIL
    note = next(c.note for t, c in res.all_checks() if c.id == "C4" and t == "M3")
    assert "zero-speed switch" in note


# ---------------------------------------------------------------------------
# C5  stall held in the star stage
# ---------------------------------------------------------------------------


def test_c5_is_not_applicable_to_dol_motors(base):
    for tag in ("M2", "M4", "M5"):
        assert verdict_of(base, "C5", tag) == pr.NA


def test_c5_applies_to_star_delta_motors(base):
    for tag in ("M1", "M3"):
        assert verdict_of(base, "C5", tag) in (pr.PASS, pr.FAIL)


def test_c5_fails_when_the_thermal_element_is_too_slow(make_plant):
    def mutate(data):
        m = motor_dict(data, "M3")
        m["trip_class"] = 30
        m["ol_pickup_multiple"] = 1.15
        m["t_stall_hot_s"] = 12.0

    assert verdict_of(results_for(make_plant(mutate)), "C5", "M3") == pr.FAIL


# ---------------------------------------------------------------------------
# C6, C7  instantaneous element
# ---------------------------------------------------------------------------


def test_c6_and_c7_pass_in_the_base_case(base):
    assert verdict_of(base, "C6") == pr.PASS
    assert verdict_of(base, "C7") == pr.PASS


def test_c7_fails_on_a_very_long_weak_feeder(make_plant):
    """Push the terminal fault down until the instantaneous cannot see it."""
    def mutate(data):
        c = motor_dict(data, "M5")["cable"]
        c["size_mm2"] = 4
        c["length_m"] = 400.0

    assert verdict_of(results_for(make_plant(mutate)), "C7", "M5") == pr.FAIL


# ---------------------------------------------------------------------------
# C8  breaking capacity
# ---------------------------------------------------------------------------


def test_c8_passes_in_the_base_case(base):
    assert verdict_of(base, "C8") == pr.PASS


def test_c8_fails_when_no_frame_can_break_the_fault(make_plant):
    def mutate(data):
        data["mccb_frames"] = [{"frame_a": 630, "icu_ka": 10}]

    assert verdict_of(results_for(make_plant(mutate)), "C8") == pr.FAIL


# ---------------------------------------------------------------------------
# C9  selectivity
# ---------------------------------------------------------------------------


def test_c9_passes_in_the_base_case(base):
    assert verdict_of(base, "C9") == pr.PASS


def test_c9_fails_with_a_low_incomer_instantaneous(make_plant):
    def mutate(data):
        data["feeder_breaker"]["ii_a"] = 1500.0

    assert verdict_of(results_for(make_plant(mutate)), "C9") == pr.FAIL


# ---------------------------------------------------------------------------
# C10, C11  starting voltage
# ---------------------------------------------------------------------------


def test_c10_and_c11_pass_in_the_base_case(base):
    assert verdict_of(base, "C10") == pr.PASS
    assert verdict_of(base, "C11") == pr.PASS


def test_c10_fails_on_a_grossly_undersized_transformer(make_plant):
    def mutate(data):
        data["transformer"].update(kva=150.0, load_loss_kw=2.0, uk_pct=4.0)
        data["source"]["fault_mva"] = 20.0

    assert verdict_of(results_for(make_plant(mutate)), "C10") == pr.FAIL


# ---------------------------------------------------------------------------
# C12, C13, C14  cable checks
# ---------------------------------------------------------------------------


def test_cable_checks_pass_in_the_base_case(base):
    for cid in ("C12", "C13", "C14"):
        assert verdict_of(base, cid) == pr.PASS


def test_c12_fails_on_an_undersized_cable(make_plant):
    def mutate(data):
        motor_dict(data, "M1")["cable"]["size_mm2"] = 25

    assert verdict_of(results_for(make_plant(mutate)), "C12", "M1") == pr.FAIL


def test_c13_fails_on_an_excessively_long_cable(make_plant):
    def mutate(data):
        motor_dict(data, "M5")["cable"]["length_m"] = 600.0

    assert verdict_of(results_for(make_plant(mutate)), "C13", "M5") == pr.FAIL


def test_c14_fails_when_the_conductor_cannot_survive_the_fault(make_plant):
    def mutate(data):
        motor_dict(data, "M1")["cable"]["size_mm2"] = 4
        motor_dict(data, "M1")["cable"]["length_m"] = 2.0

    assert verdict_of(results_for(make_plant(mutate)), "C14", "M1") == pr.FAIL


def test_ampacity_derating_formula(plant):
    for m in plant.motors:
        c = m.cable
        expected = c.ctype.iz_base_a * K_AMBIENT_40C_XLPE * K_GROUPING * c.runs
        assert ck.derated_ampacity(c) == pytest.approx(expected)


def test_adiabatic_formula(plant):
    c = plant.motor("M1").cable
    i = 20000.0
    assert ck.adiabatic_withstand_s(c, i) == pytest.approx(
        (K_ADIABATIC_CU_XLPE * c.size_mm2 / i) ** 2
    )


def test_adiabatic_shares_current_between_parallel_runs(plant):
    from models import Cable

    lib = plant.cable_library
    one = Cable(lib.get(120), 60.0, runs=1)
    two = Cable(lib.get(120), 60.0, runs=2)
    assert ck.adiabatic_withstand_s(two, 20000.0) == pytest.approx(
        4.0 * ck.adiabatic_withstand_s(one, 20000.0)
    )


def test_recommended_cable_size_matches_the_schedule(plant):
    """Every cable in the plant data must be the calculated minimum."""
    for m in plant.motors:
        assert ck.recommend_cable_size(plant, m) == m.cable.size_mm2


# ---------------------------------------------------------------------------
# C15-C19  plant level
# ---------------------------------------------------------------------------


def test_c15_is_marginal_in_the_base_case(base):
    """33.3 kA against a 36 kA board leaves only 7 per cent - a real finding."""
    assert verdict_of(base, "C15") == pr.MARGINAL


def test_c15_fails_on_an_underrated_board(make_plant):
    def mutate(data):
        data["bus"]["rated_short_time_withstand_ka"] = 25.0

    assert verdict_of(results_for(make_plant(mutate)), "C15") == pr.FAIL


def test_c15_passes_with_a_50ka_board(make_plant):
    def mutate(data):
        data["bus"]["rated_short_time_withstand_ka"] = 50.0

    assert verdict_of(results_for(make_plant(mutate)), "C15") == pr.PASS


def test_c16_fails_on_an_overloaded_transformer(make_plant):
    def mutate(data):
        data["transformer"].update(kva=400.0, load_loss_kw=4.6)

    assert verdict_of(results_for(make_plant(mutate)), "C16") == pr.FAIL


def test_c17_fails_when_the_incomer_would_trip_on_a_start(make_plant):
    def mutate(data):
        data["feeder_breaker"].update(ir_a=300.0, tr_s=0.5, isd_a=1000.0)

    assert verdict_of(results_for(make_plant(mutate)), "C17") == pr.FAIL


def test_c18_fails_when_short_time_sits_below_the_inrush(make_plant):
    def mutate(data):
        data["feeder_breaker"]["isd_a"] = 1500.0

    assert verdict_of(results_for(make_plant(mutate)), "C18") == pr.FAIL


def test_c19_flags_an_oversized_cable(make_plant):
    def mutate(data):
        motor_dict(data, "M5")["cable"]["size_mm2"] = 240

    assert verdict_of(results_for(make_plant(mutate)), "C19") == pr.INFO


def test_worst_start_is_identified(plant, base):
    assert base.plant_checks.worst_start_tag in {m.tag for m in plant.motors}
    assert base.plant_checks.worst_start_bus_current_a > plant.transformer.flc_a


# ---------------------------------------------------------------------------
# Report pipeline
# ---------------------------------------------------------------------------


def test_markdown_report_is_complete(base, tmp_path):
    md = rp.build_markdown(base)
    for heading in (
        "# 415 V Industrial Plant Bus",
        "## Executive summary",
        "## 1. Plant description",
        "## 2. Basis of calculation",
        "## 3. Input data",
        "## 4. Short-circuit study",
        "## 5. Motor protection settings",
        "## 6. Combined bus coordination",
        "## 7. Motor starting and voltage dip study",
        "## 8. Supporting checks",
        "## 9. Findings and recommended actions",
        "## Appendix A. Formula reference",
        "## Appendix B. Check index",
    ):
        assert heading in md, f"missing section: {heading}"


def test_report_mentions_every_motor_and_check(base):
    md = rp.build_markdown(base)
    for m in base.plant.motors:
        assert m.tag in md
        assert m.service in md
    for n in range(1, 20):
        assert f"C{n}" in md


def test_report_surfaces_the_failures(base):
    md = rp.build_markdown(base)
    assert "FAIL" in md
    assert "Findings requiring action" in md


def test_markdown_tables_are_well_formed(base):
    """Every table row must have the same number of cells.

    Counts UNESCAPED pipes only. Several formulas carry impedance magnitude
    bars such as |Z_k|, and the parallel operator ||; unescaped, those split
    into extra cells and wreck the table, so the escaping is what this is
    really testing.
    """
    md = rp.build_markdown(base)
    for block in md.split("\n\n"):
        lines = [l for l in block.split("\n") if l.startswith("|")]
        if len(lines) < 2:
            continue
        widths = {len(rp._split_row(line)) for line in lines}
        assert len(widths) == 1, f"ragged table {widths}:\n{block[:400]}"


def test_pipes_inside_cells_are_escaped(base):
    """Guards the escaping directly, so a regression cannot hide."""
    md = rp.build_markdown(base)
    table_lines = [
        l for l in md.split("\n")
        if l.startswith("|") and not re.match(r"^\|[\s\-|]+\|$", l)
    ]
    assert any(r"\|" in l for l in table_lines), "expected escaped pipes in the formula table"
    for line in table_lines:
        for cell in rp._split_row(line):
            assert "\\" not in cell.replace("\\\\", ""), f"unescaped backslash in {cell!r}"


def test_html_is_self_contained(base, tmp_path):
    """Every referenced figure must end up embedded, not linked."""
    figures = tmp_path / "figures"
    figures.mkdir()
    # A minimal valid 1x1 PNG, enough to exercise the base64 embedding.
    import base64 as _b64

    png = _b64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    for name in (
        "single_line_diagram", "tcc_bus_combined", "thermal_state", "voltage_dip",
        *(f"tcc_{m.tag}" for m in base.plant.motors),
    ):
        (figures / f"{name}.png").write_bytes(png)

    md = rp.build_markdown(base)
    html = rp.markdown_to_html(md, tmp_path, "test")

    assert html.startswith("<!doctype html>")
    assert "<table>" in html and "</html>" in html
    assert html.count("data:image/png;base64") == md.count("![")
    assert 'src="figures/' not in html


def test_html_preserves_escaped_pipes_in_cells(base, tmp_path):
    """The parallel operator must survive the round trip into a table cell."""
    html = rp.markdown_to_html(rp.build_markdown(base), tmp_path, "test")
    assert "Z_load || Z_branch" in html
    assert r"\|" not in html


def test_csv_exports_are_written(base, tmp_path):
    paths = rp.write_report(base, tmp_path)
    assert paths["markdown"].exists()
    assert paths["html"].exists()
    csvs = [p for k, p in paths.items() if k.startswith("csv_")]
    assert len(csvs) >= 7
    for p in csvs:
        assert p.exists() and p.stat().st_size > 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_runs_and_reports_failure(tmp_path):
    """The base case has failures, so the CLI must exit non-zero."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "main.py"),
         "--outdir", str(tmp_path), "--no-figures", "--settings", "specified"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 1, proc.stderr
    assert "overall: FAIL" in proc.stdout
    assert (tmp_path / "report.md").exists()


def test_cli_succeeds_with_recommended_settings(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "main.py"),
         "--outdir", str(tmp_path), "--no-figures", "--settings", "recommended"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "overall: FAIL" not in proc.stdout


def test_cli_lists_scenarios():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--list-scenarios"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0
    for name in ("weak_source", "m1_dol", "worst_case_start"):
        assert name in proc.stdout


def test_cli_rejects_an_unknown_scenario(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "main.py"),
         "--scenario", "nope", "--outdir", str(tmp_path), "--no-figures"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 2
    assert "error" in proc.stderr.lower()


@pytest.mark.parametrize("scenario", ["weak_source", "m1_dol", "worst_case_start"])
def test_every_scenario_runs(scenario, tmp_path):
    from models import Plant

    res = results_for(Plant.from_yaml(PLANT_YAML, CABLES_YAML, scenario=scenario))
    md = rp.build_markdown(res)
    assert scenario in md
    assert res.overall in (pr.PASS, pr.MARGINAL, pr.FAIL)
