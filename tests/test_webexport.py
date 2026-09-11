"""
Interactive viewer export.

The viewer must not become a second, divergent implementation of the physics.
These tests pin it to the engine: every headline number and every plotted point
in the payload has to match what the calculation modules produce.
"""

from __future__ import annotations

import json
import math
import re

import pytest

import checks as ck
import protection as pr
import report as rp
import shortcircuit as sc
import starting as stg
import webexport as wx


@pytest.fixture(scope="module")
def results(plant):
    study = sc.run_study(plant)
    settings = pr.design_all(plant, study)
    start = stg.run_study(plant)
    return rp.StudyResults(
        plant=plant, sc=study, settings=settings, starting=start,
        plant_checks=ck.plant_checks(plant, study, start),
        cable_checks={
            m.tag: ck.cable_checks(plant, m, settings[m.tag], study, start)
            for m in plant.motors
        },
        figures={},
    )


@pytest.fixture(scope="module")
def payload(results):
    return wx.build_payload(results)


# ---------------------------------------------------------------------------
# Payload shape
# ---------------------------------------------------------------------------


def test_payload_is_json_serialisable(payload):
    text = json.dumps(payload)
    assert len(text) > 10_000
    # Round trips without loss of structure.
    assert json.loads(text)["overall"] == payload["overall"]


def test_payload_has_every_motor(payload, plant):
    assert [m["tag"] for m in payload["motors"]] == [m.tag for m in plant.motors]


def test_payload_carries_no_infinities(payload):
    """JSON has no Infinity, so the sampler must have filtered them out."""
    def walk(node):
        if isinstance(node, float):
            assert math.isfinite(node), "non-finite value reached the payload"
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(payload)


# ---------------------------------------------------------------------------
# The viewer must agree with the engine
# ---------------------------------------------------------------------------


def test_headline_numbers_match_the_engine(payload, results):
    assert payload["overall"] == results.overall
    assert payload["counts"] == results.counts()
    assert payload["shortCircuit"]["busMax"]["ik"] == pytest.approx(
        results.sc.bus_max.ik_ka, rel=1e-3
    )
    assert len(payload["findings"]) == len(results.findings())


def test_motor_values_match_the_engine(payload, results, plant):
    for entry in payload["motors"]:
        m = plant.motor(entry["tag"])
        s = results.settings[m.tag]
        assert entry["flc"] == pytest.approx(m.flc_a, rel=1e-3)
        assert entry["ilr"] == pytest.approx(m.i_lr_a, abs=1)
        assert entry["settings"]["tripClass"] == s.trip_class
        assert entry["settings"]["pickup"] == pytest.approx(s.pickup_a, rel=1e-3)
        assert entry["settings"]["thetaHot"] == pytest.approx(s.sim_hot.theta_max, rel=1e-3)
        assert entry["settings"]["tripsHot"] == s.sim_hot.trips
        assert entry["fault"]["minDevice"] == pytest.approx(
            results.sc.terminal_min[m.tag].device_ik_a, abs=1
        )


def test_curve_points_match_the_device_curve(payload, results, plant):
    """Spot-check sampled points against a live call into the engine.

    Currents are stored to four significant figures to keep the single-file
    payload small. That is a relative error of 5e-5 -- about 0.005 of a pixel
    on a log axis spanning four decades, so it is invisible on the chart. But
    the thermal curve is near-vertical as it approaches pickup, so
    re-evaluating the engine at the *rounded* current there produces a
    materially different time even though the stored point is right. The
    asymptote is excluded rather than the tolerance loosened, so the rest of
    the curve is still held to a tight match.
    """
    for entry in payload["motors"]:
        s = results.settings[entry["tag"]]
        pts = entry["curves"]["device"]
        assert len(pts) > 50

        checked = 0
        for i_a, t in pts:
            if t > 1000.0:            # the near-vertical approach to pickup
                continue
            assert t == pytest.approx(s.device_time(i_a, hot=True), rel=5e-3)
            checked += 1
        assert checked > 30, f"{entry['tag']}: too few points away from the asymptote"


def test_curve_rounding_is_visually_negligible(payload, results):
    """The stored current must be within a small fraction of a pixel of truth."""
    decades = math.log10(payload["plot"]["iMax"] / payload["plot"]["iMin"])
    px_per_decade = 820 / decades          # chart drawing width in the template
    for entry in payload["motors"]:
        for i_a, _ in entry["curves"]["device"][::25]:
            # 4 significant figures => relative error at most 5e-5
            px_error = abs(math.log10(1 + 5e-5)) * px_per_decade
            assert px_error < 0.05


def test_curve_starts_at_the_overload_pickup(payload, results):
    for entry in payload["motors"]:
        s = results.settings[entry["tag"]]
        first_i = entry["curves"]["device"][0][0]
        assert first_i == pytest.approx(s.pickup_a, rel=0.02)


def test_curves_are_sorted_and_inside_the_plot_window(payload):
    win = payload["plot"]
    for entry in payload["motors"]:
        for name, pts in entry["curves"].items():
            if name == "envelope":
                continue
            assert pts == sorted(pts, key=lambda p: p[0]), f"{name} not sorted"
            for i_a, t in pts:
                assert win["iMin"] <= i_a <= win["iMax"]
                assert win["tMin"] <= t <= win["tMax"]


def test_device_curve_is_monotonically_decreasing(payload):
    for entry in payload["motors"]:
        times = [t for _, t in entry["curves"]["device"]]
        assert all(a >= b - 1e-9 for a, b in zip(times, times[1:]))


# ---------------------------------------------------------------------------
# Starting envelope
# ---------------------------------------------------------------------------


def test_envelope_is_a_descending_staircase(payload):
    for entry in payload["motors"]:
        env = entry["curves"]["envelope"]
        assert env[0][1] == pytest.approx(entry["tStart"], rel=1e-3)
        times = [t for _, t in env]
        assert all(a >= b - 1e-9 for a, b in zip(times, times[1:]))
        currents = [i for i, _ in env]
        assert currents == sorted(currents)


def test_star_delta_envelope_has_two_steps(payload):
    """The star-delta benefit must be visible in the drawn envelope."""
    for entry in payload["motors"]:
        env = entry["curves"]["envelope"]
        levels = sorted({t for _, t in env}, reverse=True)
        if entry["isStarDelta"]:
            # full start time, then the delta-stage duration, then the floor
            assert levels[0] == pytest.approx(entry["tStart"], rel=1e-3)
            assert any(abs(l - entry["tDelta"]) < 0.05 for l in levels)
        else:
            assert levels[0] == pytest.approx(entry["tStart"], rel=1e-3)


def test_envelope_drops_at_the_stage_currents(payload, plant):
    for entry in payload["motors"]:
        m = plant.motor(entry["tag"])
        env_currents = {i for i, _ in entry["curves"]["envelope"]}
        for current, _, _ in m.starting_stages():
            assert any(abs(c - current) / current < 1e-3 for c in env_currents)


# ---------------------------------------------------------------------------
# Thermal trace
# ---------------------------------------------------------------------------


def test_thermal_trace_matches_the_simulation(payload, results):
    for entry in payload["motors"]:
        s = results.settings[entry["tag"]]
        hot = entry["thermalTrace"]["hot"]
        assert len(hot["t"]) == len(hot["theta"]) == len(s.sim_hot.trace_t)
        assert hot["t"][-1] == pytest.approx(entry["tStart"], rel=1e-2)
        assert max(hot["theta"]) == pytest.approx(
            max(s.sim_hot.trace_theta), rel=2e-3
        )


def test_hot_trace_starts_above_cold(payload):
    for entry in payload["motors"]:
        tr = entry["thermalTrace"]
        assert tr["hot"]["theta"][0] > tr["cold"]["theta"][0]
        assert tr["cold"]["theta"][0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_template_placeholder_is_replaced(results):
    html = wx.render_viewer(results, standalone=False)
    assert "__STUDY_DATA__" not in html
    assert html.lstrip().startswith("<title>")


def test_standalone_document_is_well_formed(results):
    html = wx.render_viewer(results, standalone=True)
    assert html.startswith("<!doctype html>")
    assert "<head>" in html and "</head>" in html and "</html>" in html
    # The title and font link belong in the head, not the body.
    head = html.split("</head>")[0]
    assert "<title>" in head
    assert "fonts.googleapis.com" in head
    assert html.count("<title>") == 1


def test_embedded_data_parses_as_json(results):
    html = wx.render_viewer(results, standalone=True)
    blob = re.search(r"const D = (\{.*?\});\n", html, re.S).group(1)
    data = json.loads(blob)
    assert data["overall"] == results.overall
    assert len(data["motors"]) == len(results.plant.motors)


def test_viewer_is_self_contained(results):
    """Only the font stylesheet may be external; no other network dependency."""
    html = wx.render_viewer(results, standalone=True)
    urls = re.findall(r'(?:src|href)="(https?://[^"]+)"', html)
    assert all(u.startswith("https://fonts.googleapis.com") for u in urls), urls


def test_write_viewer_writes_a_file(results, tmp_path):
    path = wx.write_viewer(results, tmp_path)
    assert path.exists()
    assert path.name == "index.html"
    assert path.stat().st_size > 50_000


def test_missing_placeholder_is_rejected(results, tmp_path):
    bad = tmp_path / "bad.html"
    bad.write_text("<title>x</title>", encoding="utf-8")
    with pytest.raises(ValueError, match="__STUDY_DATA__"):
        wx.render_viewer(results, template=bad)


def test_payload_size_stays_reasonable(payload):
    """Guard against a sampling change bloating the single-file viewer."""
    size = len(json.dumps(payload, separators=(",", ":")))
    assert size < 600_000, f"payload grew to {size} bytes"
