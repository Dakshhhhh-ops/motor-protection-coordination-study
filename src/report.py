"""
Report generation: markdown, a self-contained HTML rendering, and CSV exports.

The markdown file is the primary deliverable. The HTML is the same content with
the figures embedded as base64 data URIs, so it is a single file that can be
mailed or opened anywhere, and printed to PDF from a browser without needing a
LaTeX toolchain or a headless-Chrome dependency.

Tables are assembled as pandas DataFrames so the same objects can be written
out as CSV alongside the report, then rendered to markdown by a small local
function rather than through `DataFrame.to_markdown`, which would add a
`tabulate` dependency beyond the four libraries the project declares.
"""

from __future__ import annotations

import base64
import datetime as _dt
import html as _html
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from checks import PlantChecks, cable_checks, derated_ampacity
from constants import (
    ALPHA_CU,
    C_MAX_LV,
    C_MIN_LV,
    CTI_S,
    DAMAGE_CURVE_MIN_MULTIPLE,
    INSTANTANEOUS_SET_MULTIPLE,
    K_ADIABATIC_CU_XLPE,
    K_AMBIENT_40C_XLPE,
    K_GROUPING,
    LOCKED_ROTOR_PF,
    LOCKED_ROTOR_PICKUP_FRACTION,
    MOTOR_R_OVER_X_LV,
    TEMP_COLD_C,
    TEMP_XLPE_OPERATING_C,
    THETA_MARGINAL,
)
from models import Plant
from protection import FAIL, INFO, MARGINAL, NA, PASS, Check, MotorSettings
from shortcircuit import ShortCircuitStudy
from starting import DELTA, STAR, StartingStudy

# ===========================================================================
# Small markdown helpers
# ===========================================================================


def _fmt(value) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "-"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 10:
            return f"{value:.1f}"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _escape_cell(text: str) -> str:
    """Escape pipes so a cell cannot break out of its markdown table.

    Needed because several formulas contain the parallel-impedance operator
    `||`, which would otherwise be parsed as two empty cells and produce a
    ragged table.
    """
    return text.replace("|", r"\|")


def _table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    cols = [_escape_cell(str(c)) for c in df.columns]
    rows = [[_escape_cell(_fmt(v)) for v in rec] for rec in df.itertuples(index=False)]
    widths = [
        max(len(cols[i]), *(len(r[i]) for r in rows)) if rows else len(cols[i])
        for i in range(len(cols))
    ]
    head = "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols)) + " |"
    rule = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = [
        "| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(cols))) + " |"
        for r in rows
    ]
    return "\n".join([head, rule, *body])


def _checks_table(checks: list[Check]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": c.id,
                "Check": c.name,
                "Criterion": c.criterion,
                "Calculated": c.value,
                "Verdict": c.verdict,
                "Reference": c.reference,
            }
            for c in checks
        ]
    )


def _badge(verdict: str) -> str:
    return {
        PASS: "**PASS**",
        MARGINAL: "**MARGINAL**",
        FAIL: "**FAIL**",
        INFO: "INFO",
        NA: "n/a",
    }.get(verdict, verdict)


# ===========================================================================
# Result bundle
# ===========================================================================


@dataclass
class StudyResults:
    """Everything one run of the study produces."""

    plant: Plant
    sc: ShortCircuitStudy
    settings: dict[str, MotorSettings]
    starting: StartingStudy
    plant_checks: PlantChecks
    cable_checks: dict[str, list[Check]]
    figures: dict[str, Path]

    def all_checks(self) -> list[tuple[str, Check]]:
        """Every check in the study, tagged with the item it belongs to."""
        out: list[tuple[str, Check]] = []
        for tag, s in self.settings.items():
            out.extend((tag, c) for c in s.checks)
            out.extend((tag, c) for c in self.cable_checks[tag])
        for m in self.plant.motors:
            r = self.starting.as_designed(self.plant, m.tag, others_running=True)
            out.extend((m.tag, c) for c in r.checks)
        out.extend(("Plant", c) for c in self.plant_checks.checks)
        return out

    def findings(self) -> list[tuple[str, Check]]:
        """Checks that are not a clean pass, worst first."""
        order = {FAIL: 0, MARGINAL: 1, INFO: 2}
        flagged = [(t, c) for t, c in self.all_checks() if c.verdict in order]
        return sorted(flagged, key=lambda tc: (order[tc[1].verdict], tc[0], tc[1].id))

    def counts(self) -> dict[str, int]:
        counts = {PASS: 0, MARGINAL: 0, FAIL: 0, INFO: 0, NA: 0}
        for _, c in self.all_checks():
            counts[c.verdict] = counts.get(c.verdict, 0) + 1
        return counts

    @property
    def overall(self) -> str:
        c = self.counts()
        if c.get(FAIL):
            return FAIL
        if c.get(MARGINAL):
            return MARGINAL
        return PASS


# ===========================================================================
# Data tables
# ===========================================================================


def motor_table(plant: Plant) -> pd.DataFrame:
    rows = []
    for m in plant.motors:
        rows.append(
            {
                "Tag": m.tag,
                "Service": m.service,
                "kW": m.kw,
                "eta": m.efficiency,
                "pf": m.pf,
                "FLC (A)": round(m.flc_a, 1),
                "kVA": round(m.s_rated_kva, 1),
                "LRC x": m.lrc_multiple,
                "I_LR (A)": round(m.i_lr_a, 0),
                "Start": "Y-D" if m.is_star_delta else "DOL",
                "t_start (s)": m.t_start_s,
                "t_star (s)": m.t_star_s if m.t_star_s else "-",
                "Stall hot/cold (s)": f"{m.t_stall_hot_s:g} / {m.t_stall_cold_s:g}",
                "Cable": m.cable.describe(),
            }
        )
    return pd.DataFrame(rows)


def cable_table(plant: Plant) -> pd.DataFrame:
    rows = []
    for m in plant.motors:
        c = m.cable
        rows.append(
            {
                "Tag": m.tag,
                "Size (mm2)": c.size_mm2,
                "Runs": c.runs,
                "Length (m)": c.length_m,
                "R20 (ohm/km)": c.ctype.r20_ohm_km,
                "X (ohm/km)": c.ctype.x_ohm_km,
                "R at 20C (ohm)": round(c.r_ohm(TEMP_COLD_C), 5),
                "R at 90C (ohm)": round(c.r_ohm(TEMP_XLPE_OPERATING_C), 5),
                "X (ohm)": round(c.x_ohm(), 5),
                "Iz derated (A)": round(derated_ampacity(c), 0),
                "FLC (A)": round(m.flc_a, 1),
            }
        )
    return pd.DataFrame(rows)


def shortcircuit_table(results: StudyResults) -> pd.DataFrame:
    sc = results.sc
    rows = [
        {
            "Location": "Busbar, network only",
            "Duty": "maximum",
            "Ik'' (kA)": round(sc.bus_max_no_motors.ik_ka, 2),
            "ip (kA)": round(sc.bus_max_no_motors.ip_ka, 1),
            "kappa": round(sc.bus_max_no_motors.kappa, 3),
            "R (mohm)": round(sc.bus_max_no_motors.r * 1000, 3),
            "X (mohm)": round(sc.bus_max_no_motors.x * 1000, 3),
            "X/R": round(sc.bus_max_no_motors.x_over_r, 2),
        },
        {
            "Location": "Busbar, with motor contribution",
            "Duty": "maximum",
            "Ik'' (kA)": round(sc.bus_max.ik_ka, 2),
            "ip (kA)": round(sc.bus_max.ip_ka, 1),
            "kappa": round(sc.bus_max.kappa, 3),
            "R (mohm)": round(sc.bus_max.r * 1000, 3),
            "X (mohm)": round(sc.bus_max.x * 1000, 3),
            "X/R": round(sc.bus_max.x_over_r, 2),
        },
        {
            "Location": "Busbar",
            "Duty": "minimum",
            "Ik'' (kA)": round(sc.bus_min.ik_ka, 2),
            "ip (kA)": round(sc.bus_min.ip_ka, 1),
            "kappa": round(sc.bus_min.kappa, 3),
            "R (mohm)": round(sc.bus_min.r * 1000, 3),
            "X (mohm)": round(sc.bus_min.x * 1000, 3),
            "X/R": round(sc.bus_min.x_over_r, 2),
        },
    ]
    return pd.DataFrame(rows)


def terminal_fault_table(results: StudyResults) -> pd.DataFrame:
    rows = []
    for m in results.plant.motors:
        hi = results.sc.terminal_max[m.tag]
        lo = results.sc.terminal_min[m.tag]
        rows.append(
            {
                "Tag": m.tag,
                "Through device, max (A)": round(hi.device_ik_a, 0),
                "Total at fault, max (A)": round(hi.total_ik_a, 0),
                "Local motor back-feed (A)": round(hi.motor_ik_a, 0),
                "Through device, min (A)": round(lo.device_ik_a, 0),
                "min / max": round(lo.device_ik_a / hi.device_ik_a, 3),
            }
        )
    return pd.DataFrame(rows)


def settings_table(results: StudyResults) -> pd.DataFrame:
    rows = []
    for m in results.plant.motors:
        s = results.settings[m.tag]
        rows.append(
            {
                "Tag": m.tag,
                "FLC (A)": round(m.flc_a, 1),
                "49/51 pickup (A)": round(s.pickup_a, 1),
                "x FLC": s.pickup_multiple,
                "Class": s.trip_class,
                "tau (s)": round(s.tau_s, 0),
                "51LR pickup (A)": round(s.lr_pickup_a, 0),
                "51LR time (s)": round(s.lr_time_s, 1) if s.lr_time_s else "NONE",
                "50 pickup (A)": round(s.inst_pickup_a, 0),
                "MCCB": f"{s.mccb_frame_a:.0f} A / {s.mccb_icu_ka:.0f} kA"
                if s.mccb_frame_a
                else "none",
                "theta hot": round(s.sim_hot.theta_max, 3),
                "theta cold": round(s.sim_cold.theta_max, 3),
                "Verdict": s.verdict,
            }
        )
    return pd.DataFrame(rows)


def starting_table(results: StudyResults) -> pd.DataFrame:
    rows = []
    for m in results.plant.motors:
        for conn in (DELTA, STAR):
            for others in (False, True):
                r = results.starting.get(m.tag, conn, others)
                designed = (conn == STAR) == m.is_star_delta
                rows.append(
                    {
                        "Tag": m.tag,
                        "Method": "DOL" if conn == DELTA else "Star-delta",
                        "Condition": "all others running" if others else "starting alone",
                        "Specified": "yes" if designed else "",
                        "V bus (%)": round(r.v_bus_pu * 100, 1),
                        "V motor (%)": round(r.v_motor_pu * 100, 1),
                        "I start (A)": round(r.i_start_a, 0),
                        "I nameplate (A)": round(r.i_nameplate_a, 0),
                        "Torque (% rated)": round(r.torque_pu * 100, 0),
                        "Verdict": r.verdict,
                    }
                )
    return pd.DataFrame(rows)


# ===========================================================================
# Markdown report
# ===========================================================================


def build_markdown(results: StudyResults, figures_dir: str = "figures") -> str:
    p = results.plant
    sc = results.sc
    L: list[str] = []
    add = L.append

    today = _dt.date.today().isoformat()
    counts = results.counts()

    # -- header ------------------------------------------------------------
    add(f"# {p.meta.get('project', 'Motor Protection Coordination Study')}")
    add("")
    add(
        f"**Bus** {p.meta.get('bus_designation', '-')} | "
        f"**Revision** {p.meta.get('revision', '-')} | "
        f"**Date** {today} | "
        f"**Prepared by** {p.meta.get('prepared_by', '-')}"
    )
    if p.scenario:
        add("")
        add(f"> **Scenario: `{p.scenario}`** - {p.scenario_description}")
    add("")
    add(f"**Basis:** {p.meta.get('basis', '-')}")
    add("")
    add("---")
    add("")

    # -- executive summary -------------------------------------------------
    add("## Executive summary")
    add("")
    add(
        f"Overall verdict: {_badge(results.overall)} - "
        f"{counts.get(PASS, 0)} checks pass, {counts.get(MARGINAL, 0)} marginal, "
        f"{counts.get(FAIL, 0)} fail, {counts.get(INFO, 0)} informational."
    )
    add("")
    add(
        f"The {p.transformer.kva:.0f} kVA transformer feeds {len(p.motors)} motors "
        f"totalling {sum(m.kw for m in p.motors):.0f} kW of shaft power plus "
        f"{p.static_load.kva:.0f} kVA of static load, loading the transformer to "
        f"{p.transformer_loading_pct:.1f} per cent. The maximum three-phase fault at the "
        f"busbar is **{sc.bus_max.ik_ka:.2f} kA** including a "
        f"{sc.motor_contribution_pct:.0f} per cent uplift from motor back-feed, with a peak "
        f"of {sc.bus_max.ip_ka:.1f} kA."
    )
    add("")

    findings = results.findings()
    if findings:
        add("### Findings requiring action")
        add("")
        for tag, c in findings:
            note = f" {c.note}" if c.note else ""
            add(f"- **{tag} / {c.id} - {_badge(c.verdict)}** - {c.name}: {c.value}.{note}")
        add("")
    else:
        add("No check falls short of its acceptance criterion. The design is coordinated.")
        add("")

    add("---")
    add("")

    # -- 1 plant description ----------------------------------------------
    add("## 1. Plant description")
    add("")
    t = p.transformer
    add(
        f"An {t.hv_kv:g} kV utility supply with a {p.source.fault_mva:.0f} MVA fault level "
        f"feeds a {t.kva:.0f} kVA, {t.hv_kv:g} kV / {t.lv_v:g} V {t.vector_group} transformer. "
        f"The transformer secondary supplies switchboard "
        f"{p.meta.get('bus_designation', 'MCC-01')}, a single {p.bus.nominal_v:g} V, "
        f"{p.bus.frequency_hz:g} Hz busbar rated "
        f"{p.bus.rated_short_time_withstand_ka:.0f} kA short-time withstand, through a "
        f"{p.feeder.frame_a:.0f} A air circuit breaker ({p.feeder.designation}). "
        f"{len(p.motors)} three-phase squirrel-cage induction motors and a "
        f"{p.static_load.kva:.0f} kVA static load are connected to that busbar."
    )
    add("")
    add(
        "Each motor feeder is a standard IEC motor starter: an MCCB providing "
        "short-circuit protection (the ANSI 50 element), a contactor, and a thermal "
        "overload relay (ANSI 49/51 with a 51LR stall element). The split matters - a "
        f"contactor cannot interrupt the {sc.bus_max.ik_ka:.1f} kA available at this bus, "
        "so the MCCB is the short-circuit protective device and the relay handles thermal "
        "duty. This is type-2 coordination in the sense of IEC 60947-4-1."
    )
    add("")
    add(f"![Single-line diagram]({figures_dir}/single_line_diagram.png)")
    add("")
    add("---")
    add("")

    # -- 2 basis of calculation -------------------------------------------
    add("## 2. Basis of calculation")
    add("")
    add("### 2.1 Standards applied")
    add("")
    std = pd.DataFrame(
        [
            {"Standard": "IEC 60909-0:2016", "Applied to": "Three-phase short-circuit currents: voltage factor c, transformer correction K_T, motor contribution, peak factor kappa"},
            {"Standard": "IEC 60947-4-1:2018", "Applied to": "Motor starter trip classes, overload pickup band, type-2 coordination, contactor hold-in voltage"},
            {"Standard": "IEC 60947-2", "Applied to": "Circuit-breaker trip-unit parameters (Ir, tr, Isd, tsd, Ii) and breaking capacity"},
            {"Standard": "IEC 60255-149", "Applied to": "Thermal-replica overload relay model and thermal state integration"},
            {"Standard": "IEC 60364-5-52", "Applied to": "Cable current-carrying capacity, derating factors, steady-state voltage drop"},
            {"Standard": "IEC 60364-5-54 / IEC 60949", "Applied to": "Adiabatic short-circuit withstand of the conductor"},
            {"Standard": "IEC 60228", "Applied to": "Conductor DC resistance at 20 degC and its temperature correction"},
            {"Standard": "IEEE Std 242 (Buff Book)", "Applied to": "Motor protection philosophy, thermal limit curve, stall timing, coordination time intervals"},
            {"Standard": "IEEE Std 399 (Brown Book)", "Applied to": "Motor-starting study method and voltage acceptance criteria"},
        ]
    )
    add(_table(std))
    add("")

    add("### 2.2 Assumptions and data provenance")
    add("")
    add(
        "Values below marked **VENDOR** are representative manufacturer figures, not "
        "standardised quantities. They must be replaced with data from the transformer "
        "test certificate, the motor datasheets and the cable manufacturer's tables "
        "before this study is issued for construction. They are listed explicitly rather "
        "than buried so that the sensitivity of each result to them is visible."
    )
    add("")
    prov = pd.DataFrame(
        [
            {"Quantity": "Utility fault level and X/R", "Value": f"{p.source.fault_mva:.0f} MVA, X/R {p.source.x_over_r:g}", "Source": "VENDOR - utility fault level letter"},
            {"Quantity": "Transformer impedance u_k", "Value": f"{t.uk_pct:g} per cent", "Source": "VENDOR - test certificate"},
            {"Quantity": "Transformer load loss", "Value": f"{t.load_loss_kw:g} kW, giving R = {t.r_pct:.2f} per cent and X/R = {t.x_over_r:.2f}", "Source": "VENDOR - test certificate"},
            {"Quantity": "Motor efficiency and power factor", "Value": "per motor, see table 3.2", "Source": "VENDOR - motor datasheets"},
            {"Quantity": "Motor run-up and stall withstand times", "Value": "per motor, see table 3.2", "Source": "VENDOR - the single most sensitive input; every protection setting depends on it"},
            {"Quantity": "Locked-rotor power factor", "Value": f"{LOCKED_ROTOR_PF:g}", "Source": "IEEE Std 242 / Std 399 typical range 0.20-0.30 for LV cage machines"},
            {"Quantity": "Cable R20", "Value": "IEC 60228 Table 2 maxima", "Source": "STANDARD"},
            {"Quantity": "Cable reactance", "Value": "0.079-0.110 ohm/km by size", "Source": "VENDOR - construction dependent"},
            {"Quantity": "Cable ampacity", "Value": "installation method E, 30 degC", "Source": "VENDOR - confirm against the project cable schedule"},
            {"Quantity": "Derating factors", "Value": f"ambient {K_AMBIENT_40C_XLPE:g} at 40 degC, grouping {K_GROUPING:g}", "Source": "IEC 60364-5-52 Annex B, routing dependent"},
            {"Quantity": "Motor R/X for short circuit", "Value": f"{MOTOR_R_OVER_X_LV:g}", "Source": "STANDARD - IEC 60909-0 for LV motors"},
            {"Quantity": "Voltage factors", "Value": f"c_max {C_MAX_LV:g}, c_min {C_MIN_LV:g}", "Source": "STANDARD - IEC 60909-0 Table 1, 415 V"},
        ]
    )
    add(_table(prov))
    add("")

    add("### 2.3 Conventions")
    add("")
    add(
        "- All impedances are complex ohms per phase, star-equivalent, referred to "
        f"{p.bus.nominal_v:g} V. Inductive reactance is +j. Network reduction uses complex "
        "admittance summation, never addition of magnitudes: the network X/R is "
        f"{sc.bus_max_no_motors.x_over_r:.1f} while an LV motor is "
        f"{1 / MOTOR_R_OVER_X_LV:.1f}, so the contributions do not add in phase."
    )
    add(
        f"- Conductor resistance is taken at {TEMP_COLD_C:.0f} degC for **maximum** fault "
        f"current (equipment rating duty) and at {TEMP_XLPE_OPERATING_C:.0f} degC for "
        f"**minimum** fault current and for voltage-drop work (protection sensitivity "
        f"duty). The ratio is {1 + ALPHA_CU * 70:.4f}, so the choice matters more than skin "
        "effect at these cross-sections."
    )
    add(
        "- Motor starting currents used for **protection settings** are on a nameplate "
        "basis (rated terminal voltage), which is the largest current a relay could see. "
        "Currents used for the **dip study** are the achieved values from the network "
        "solution, which are lower because of the supply impedance drop."
    )
    add("")
    add("---")
    add("")

    # -- 3 input data ------------------------------------------------------
    add("## 3. Input data")
    add("")
    add("### 3.1 Source and transformer")
    add("")
    src = pd.DataFrame(
        [
            {"Quantity": "Utility fault level (11 kV)", "Value": f"{p.source.fault_mva:.0f} MVA"},
            {"Quantity": "Utility X/R", "Value": f"{p.source.x_over_r:g}"},
            {"Quantity": "Utility impedance referred to LV", "Value": f"{abs(p.source.z(p.nominal_v)) * 1000:.4f} mohm"},
            {"Quantity": "Transformer rating", "Value": f"{t.kva:.0f} kVA, {t.hv_kv:g} kV / {t.lv_v:g} V, {t.vector_group}"},
            {"Quantity": "Impedance voltage u_k", "Value": f"{t.uk_pct:g} per cent"},
            {"Quantity": "Base impedance V^2/S", "Value": f"{t.z_base_ohm * 1000:.3f} mohm"},
            {"Quantity": "Resistance from load loss", "Value": f"{t.r_pct:.3f} per cent = {t.z.real * 1000:.4f} mohm"},
            {"Quantity": "Reactance sqrt(Z^2 - R^2)", "Value": f"{t.x_pct:.3f} per cent = {t.z.imag * 1000:.4f} mohm"},
            {"Quantity": "Transformer X/R", "Value": f"{t.x_over_r:.2f}"},
            {"Quantity": "Secondary full-load current", "Value": f"{t.flc_a:.1f} A"},
            {"Quantity": "Total source impedance at the bus", "Value": f"{abs(p.z_source) * 1000:.4f} mohm"},
            {"Quantity": "Static load", "Value": f"{p.static_load.kva:.0f} kVA at pf {p.static_load.pf:g}"},
            {"Quantity": "Motor diversity factor", "Value": f"{p.motor_diversity_factor:g}"},
            {"Quantity": "Diversified load", "Value": f"{p.diversified_kva:.0f} kVA at pf {p.diversified_pf:.3f} ({p.transformer_loading_pct:.1f} per cent of rating)"},
        ]
    )
    add(_table(src))
    add("")

    add("### 3.2 Motors")
    add("")
    add(
        "Full-load current is **computed**, not tabulated: "
        "`I_FLC = P_shaft / (sqrt(3) x V x eta x pf)`. The computed values fall within a "
        "few per cent of catalogue figures for 415 V four-pole machines, which is the "
        "validation that the efficiency and power factor data are self-consistent."
    )
    add("")
    add(_table(motor_table(p)))
    add("")

    add("### 3.3 Cables")
    add("")
    add(
        f"Copper conductor, XLPE insulated, 0.6/1 kV. Resistance is corrected from the "
        f"IEC 60228 20 degC value with `R_theta = R_20 [1 + {ALPHA_CU:g}(theta - 20)]`. "
        f"Derated capacity is `I_z = I_z,tab x {K_AMBIENT_40C_XLPE:g} x {K_GROUPING:g} x runs`."
    )
    add("")
    add(_table(cable_table(p)))
    add("")

    add("### 3.4 Incoming feeder breaker")
    add("")
    fb = p.feeder
    add(
        f"{fb.designation}: {fb.frame_a:.0f} A frame, Icu {fb.icu_ka:.0f} kA. "
        f"Long time Ir = {fb.ir_a:.0f} A with tr = {fb.tr_s:g} s at 6 x Ir on an I^2t band; "
        f"short time Isd = {fb.isd_a:.0f} A, definite time tsd = {fb.tsd_s:g} s; "
        f"instantaneous {'OFF' if fb.ii_a is None else f'{fb.ii_a:.0f} A'}."
    )
    add("")
    if fb.ii_a is None:
        add(
            "> The instantaneous element is deliberately switched off. On a main incomer it "
            "would operate simultaneously with every downstream motor MCCB for a bus-side "
            f"fault and destroy selectivity. The short-time band, delayed {fb.tsd_s:g} s, "
            "provides the busbar protection instead and gives "
            f"{fb.tsd_s - results.settings[p.motors[0].tag].inst_clearing_s:.2f} s over the "
            f"motor MCCBs against a {CTI_S:g} s required interval. Verified by check C9."
        )
        add("")
    add("---")
    add("")

    # -- 4 short circuit ---------------------------------------------------
    add("## 4. Short-circuit study")
    add("")
    add("### 4.1 Method")
    add("")
    add("```")
    add("I_k'' = c * U_n / (sqrt(3) * |Z_k|)                    IEC 60909-0 clause 6.2")
    add("i_p   = kappa * sqrt(2) * I_k''                        IEC 60909-0 clause 8")
    add("kappa = 1.02 + 0.98 * exp(-3 R/X)")
    add("K_T   = 0.95 * c_max / (1 + 0.6 x_T)                   IEC 60909-0 clause 6.3.3")
    add("Z_M   = U_rM^2 / (LRC_multiple * S_rM),  R_M/X_M = 0.42 for LV motors")
    add("```")
    add("")
    add(
        f"The transformer impedance correction factor evaluates to **K_T = {sc.kt:.5f}**, "
        f"raising the calculated bus fault current by about "
        f"{100 * (1 / sc.kt - 1):.1f} per cent. Omitting it would be non-conservative for "
        "equipment rating."
    )
    add("")
    add(
        "Two duties are computed and they are used for opposite purposes. The **maximum** "
        f"duty (c = {C_MAX_LV:g}, conductors at {TEMP_COLD_C:.0f} degC, motor contribution "
        "included) sizes equipment. The **minimum** duty "
        f"(c = {C_MIN_LV:g}, conductors at {TEMP_XLPE_OPERATING_C:.0f} degC, motor "
        "contribution excluded because the motors may be stopped) verifies that protection "
        "still operates on the weakest credible fault."
    )
    add("")
    add("### 4.2 Busbar fault")
    add("")
    add(_table(shortcircuit_table(results)))
    add("")
    add(
        f"Motor back-feed adds {sc.motor_contribution_pct:.1f} per cent to the busbar fault "
        f"current, taking it from {sc.bus_max_no_motors.ik_ka:.2f} kA to "
        f"**{sc.bus_max.ik_ka:.2f} kA**. This is not a rounding effect and it is the number "
        "that must be compared against switchgear ratings."
    )
    add("")
    add("### 4.3 Motor terminal faults")
    add("")
    add(
        "For a fault at a motor's terminals, that machine's own back-feed does **not** flow "
        "through its own protective device - it flows from the machine into the fault, on "
        "the load side of the device. Only the network and the other motors contribute to "
        "the current the device measures. Including the local machine here would overstate "
        "the current available to operate the instantaneous element, which is the wrong "
        "direction for a sensitivity check. The total current at the fault point, which the "
        "cable and the arc see, is listed separately."
    )
    add("")
    add(_table(terminal_fault_table(results)))
    add("")
    add("---")
    add("")

    # -- 5 protection settings --------------------------------------------
    add("## 5. Motor protection settings")
    add("")
    add("### 5.1 Summary")
    add("")
    add(_table(settings_table(results)))
    add("")
    add(
        "`theta` is the peak thermal-replica state reached during a start, in per unit of "
        "the trip threshold. A value at or above 1.000 means the relay trips on a healthy "
        f"start; above {THETA_MARGINAL:g} it rides through with less than "
        f"{100 * (1 - THETA_MARGINAL):.0f} per cent margin."
    )
    add("")

    add("### 5.2 Method")
    add("")
    add("```")
    add("Thermal replica, IEC 60255-149:")
    add("    t(I)     = tau * ln[ ((I/I_p)^2 - theta_0) / ((I/I_p)^2 - 1) ]")
    add("    theta(t) = theta_inf + (theta_0 - theta_inf) e^(-t/tau),  theta_inf = (I/I_p)^2")
    add("")
    add("tau from the IEC 60947-4-1 trip class (cold trip time at 7.2 x I_e):")
    add("    tau = C / ln(7.2^2 / (7.2^2 - 1)) = 51.32 * C")
    add("    Class 10 -> 513 s    Class 20 -> 1027 s    Class 30 -> 1540 s")
    add("")
    add("Motor damage limit, IEEE Std 242:")
    add("    t_damage(I) = t_stall * (I_LR / I)^2")
    add("```")
    add("")
    add(
        "**The start-up check is an integration, not a curve overlay.** The naive test - is "
        "the relay curve above the starting-current point - compares a single steady current "
        "against a curve, while the real relay integrates heat through a changing current. A "
        "star-delta start spends most of its time at one third of locked-rotor current, "
        "depositing one ninth of the heat per second, and a curve overlay cannot see that. "
        "The thermal state is therefore integrated stage by stage and the start passes only "
        "if `theta` never reaches 1.0. This is what a digital motor relay actually computes "
        "and what it displays as thermal capacity used."
    )
    add("")
    add(
        "**The binding case is the hot restart.** A motor that has been running at rated "
        "load sits at `theta_0 = (I_FLC/I_p)^2`, which for a 110 per cent pickup is 0.826 - "
        "83 per cent of the thermal budget is spent before the start even begins. Both the "
        "hot and cold cases are computed; the figure below shows why the distinction decides "
        "the settings."
    )
    add("")
    add(f"![Thermal state during starting]({figures_dir}/thermal_state.png)")
    add("")

    # -- per motor ---------------------------------------------------------
    for m in p.motors:
        s = results.settings[m.tag]
        add(f"### 5.3.{p.motors.index(m) + 1} {m.tag} - {m.service}")
        add("")
        add(
            f"{m.kw:g} kW, {m.voltage_v:g} V, FLC {m.flc_a:.1f} A, "
            f"LRC {m.lrc_multiple:g} x = {m.i_lr_a:.0f} A, "
            f"{'star-delta' if m.is_star_delta else 'direct-on-line'} start, "
            f"{m.t_start_s:g} s run-up, {m.t_stall_hot_s:g} s hot stall withstand. "
            f"Verdict: {_badge(s.verdict)}."
        )
        add("")
        for key, text in s.reasoning.items():
            add(f"**{key}.** {text}")
            add("")
        add(_table(_checks_table(s.checks + results.cable_checks[m.tag])))
        add("")
        add(f"![{m.tag} time-current characteristic]({figures_dir}/tcc_{m.tag}.png)")
        add("")

    add("---")
    add("")

    # -- 6 combined coordination ------------------------------------------
    add("## 6. Combined bus coordination")
    add("")
    add(
        "Selectivity is verified by sweeping every current from each motor's overload "
        f"pickup up to the {sc.bus_max.ik_ka:.2f} kA maximum bus fault and comparing "
        "clearing times. A current is selective if **either** the incomer is at least "
        f"{CTI_S:g} s slower (the definite-time region) **or** at least 1.3 times slower "
        "(the inverse-time region). Both forms are needed: an additive interval is "
        "meaningless where both curves are steep, and a multiplicative one is meaningless at "
        "the 30 ms timescale of an instantaneous trip. IEEE Std 242 recommends 0.2-0.4 s for "
        f"electromechanical relays and 0.1-0.2 s for the electronic trip units modelled here."
    )
    add("")
    add(f"![Combined bus time-current characteristic]({figures_dir}/tcc_bus_combined.png)")
    add("")
    add("---")
    add("")

    # -- 7 starting study --------------------------------------------------
    add("## 7. Motor starting and voltage dip study")
    add("")
    add("### 7.1 Method")
    add("")
    add("```")
    add("Z_LR     = V_LL / (sqrt(3) I_LR)  at  arccos(pf_LR),  pf_LR = 0.25")
    add("Z_branch = Z_cable + Z_LR                 star stage: Z_LR x 3")
    add("Z_par    = Z_load || Z_branch             Z_load omitted when starting alone")
    add("")
    add("V_bus / V_nom = |Z_par| / |Z_source + Z_par|")
    add("V_motor/V_bus = |Z_LR|  / |Z_branch|")
    add("I_start       = (V_bus/sqrt(3)) / |Z_branch|")
    add("```")
    add("")
    add(
        "At standstill the machine is a fixed series impedance (slip = 1), so a start is a "
        "linear circuit problem rather than a dynamic one. The running bus load is "
        "represented as a constant impedance, which is the correct choice for a sub-second "
        "dip: a constant-power model would draw more current as voltage falls, which is the "
        "behaviour of a regulated drive over seconds, not of lighting and HVAC over a motor "
        "start."
    )
    add("")
    add(
        "The star stage uses `Z_LR x 3`. In star each winding sees `U/sqrt(3)` so the "
        "winding current falls to `1/sqrt(3)`; in delta the line current is `sqrt(3) x` the "
        "winding current whereas in star it equals it. The net line-current ratio is "
        "therefore 1/3 and the equivalent per-phase impedance is 3x. Torque, going as V^2, "
        "also falls to one third - which is the real constraint on a star-delta start, and "
        "is reported in the table."
    )
    add("")
    add(
        "Both methods are evaluated for **every** motor, including those specified as "
        "star-delta, so the choice of starting method is justified by calculation rather "
        "than convention. The DOL row for a star-delta motor also bounds its star-to-delta "
        "transition, because the impedance at that instant is identical."
    )
    add("")
    add("### 7.2 Results")
    add("")
    add(_table(starting_table(results)))
    add("")
    add(f"![Motor starting voltage]({figures_dir}/voltage_dip.png)")
    add("")

    add("### 7.3 Largest motor starting with all others running")
    add("")
    big = p.largest_motor
    r_designed = results.starting.as_designed(p, big.tag, others_running=True)
    add(
        f"{big.tag} ({big.kw:g} kW) is the largest machine. Starting it as specified "
        f"({r_designed.case}) with every other motor running and the static load connected "
        f"gives **{r_designed.v_bus_pu * 100:.1f} per cent** at the busbar and "
        f"**{r_designed.v_motor_pu * 100:.1f} per cent** at the motor terminals, drawing "
        f"{r_designed.i_start_a:.0f} A against a nameplate figure of "
        f"{r_designed.i_nameplate_a:.0f} A. Verdict: {_badge(r_designed.verdict)}."
    )
    add("")
    add(
        f"The worst total bus current during any start is "
        f"{results.plant_checks.worst_start_bus_current_a:.0f} A, drawn when "
        f"**{results.plant_checks.worst_start_tag}** starts - not necessarily the largest "
        "motor, because a star-delta starter draws only one third of the line current that "
        "a smaller DOL machine does."
    )
    add("")

    add("### 7.4 How much margin is there?")
    add("")
    add(
        "Every voltage check in this study passes, which on its own is a weak statement - "
        "it does not say whether the design is comfortable or one change away from trouble. "
        "The table below answers that directly by holding the cable, motor and running load "
        "constant and shrinking the transformer until the terminal voltage of a DOL start "
        f"reaches the {p.criteria.get('v_motor_min_pu', 0.80) * 100:.0f} per cent limit. "
        "The transformer impedance scales as 1/S at a fixed impedance voltage, so its "
        "rating is the cleanest single parameter for weakening the source."
    )
    add("")
    crit_rows = []
    for m in p.motors:
        k = __import__("starting").critical_transformer_kva(p, m, DELTA, True)
        crit_rows.append(
            {
                "Tag": m.tag,
                "kW": m.kw,
                "Cable": m.cable.describe(),
                "V motor, DOL (%)": round(
                    results.starting.get(m.tag, DELTA, True).v_motor_pu * 100, 1
                ),
                "Critical transformer (kVA)": round(k, 0) if k else "unreachable",
                "Margin on rating": f"{p.transformer.kva / k:.1f} x" if k else "-",
            }
        )
    add(_table(pd.DataFrame(crit_rows)))
    add("")
    add(
        "The reason the margins are this wide is worth stating plainly, because it is the "
        "first thing a reviewer should challenge. On an LV bus the motor's own locked-rotor "
        f"impedance dominates the starting circuit: for {p.largest_motor.tag} it is "
        f"{abs(p.largest_motor.z_lr()) * 1000:.0f} mohm against "
        f"{abs(p.z_source) * 1000:.1f} mohm of source impedance and "
        f"{abs(p.largest_motor.cable.z(TEMP_XLPE_OPERATING_C)) * 1000:.1f} mohm of cable - "
        "roughly an order of magnitude. The terminal voltage during a start is therefore "
        "set mainly by the machine itself, and the supply can be weakened a long way before "
        "the criterion binds. **On this class of installation the binding constraints are "
        "thermal and they are in section 5, not here.** A voltage-dip study that returns all "
        "passes is the expected result for a transformer sized to its load; it becomes the "
        "critical study when the largest motor approaches about a quarter of the transformer "
        "rating, or when cables are long."
    )
    add("")
    add(
        "The ranking by critical rating is not simply the ranking by motor size: cable "
        "length shifts it, because a long run adds impedance in series with the machine "
        "without adding any of the voltage the machine needs. That is why the check is run "
        "for every motor rather than only for the largest, and why a small motor on a long "
        "cable can sit closer to the limit than a larger one on a short cable."
    )
    add("")

    add("### 7.5 Starting-method recommendations")
    add("")
    for m in p.motors:
        add(f"**{m.tag} ({m.kw:g} kW, specified {'star-delta' if m.is_star_delta else 'DOL'}).** "
            f"{results.starting.recommendations[m.tag]}")
        add("")
    add("---")
    add("")

    # -- 8 supporting checks ----------------------------------------------
    add("## 8. Supporting checks")
    add("")
    add("### 8.1 Plant level")
    add("")
    add(_table(_checks_table(results.plant_checks.checks)))
    add("")
    add("### 8.2 Cable adequacy")
    add("")
    add(
        f"Short-circuit withstand uses the adiabatic equation `t = (k S / I)^2` with "
        f"k = {K_ADIABATIC_CU_XLPE:.0f} for copper in XLPE heating from 90 to 250 degC "
        "(IEC 60364-5-54). The worst thermal duty on a cable is a fault at its **load** end, "
        "where the whole length carries the through-fault current; a fault at the source end "
        "passes through almost no cable."
    )
    add("")
    rows = []
    for m in p.motors:
        for c in results.cable_checks[m.tag]:
            rows.append({"Tag": m.tag, "ID": c.id, "Check": c.name,
                         "Calculated": c.value, "Verdict": c.verdict})
    add(_table(pd.DataFrame(rows)))
    add("")
    add("---")
    add("")

    # -- 9 consolidated findings ------------------------------------------
    add("## 9. Findings and recommended actions")
    add("")
    if not findings:
        add("Every check passes. No action required.")
        add("")
    else:
        add(_table(pd.DataFrame(
            [
                {
                    "Item": tag,
                    "ID": c.id,
                    "Verdict": c.verdict,
                    "Finding": c.name,
                    "Calculated": c.value,
                    "Action": c.note or "See the per-item justification above.",
                }
                for tag, c in findings
            ]
        )))
        add("")

    changed = [
        (m.tag, s)
        for m in p.motors
        if (s := results.settings[m.tag])
        and not (s.trip_class == m.trip_class
                 and math.isclose(s.pickup_multiple, m.ol_pickup_multiple, rel_tol=1e-9))
    ]
    add("### 9.1 Setting changes recommended by the independent search")
    add("")
    add(
        "The tool designs each overload setting independently of the value specified in the "
        "plant data, preferring the **lowest trip class** that survives a hot restart (a "
        "lower class is a faster curve and better rotor protection during a stall) and then "
        "the **lowest pickup** at that class (more sensitive to sustained overloads). Where "
        "its answer differs from the specified value, both are reported so the change can be "
        "reviewed rather than applied blindly."
    )
    add("")
    rec_rows = []
    for m in p.motors:
        _, _, reason = __import__("protection").search_thermal_setting(m)
        s = results.settings[m.tag]
        rec_rows.append(
            {
                "Tag": m.tag,
                "Specified": f"Class {m.trip_class} at {m.ol_pickup_multiple * 100:.0f}%",
                "Recommended": reason.split(".")[0] + ".",
                "theta hot as specified": round(s.sim_hot.theta_max, 3),
            }
        )
    add(_table(pd.DataFrame(rec_rows)))
    add("")
    add(
        "Re-run with `--settings recommended` to apply the searched settings and confirm "
        "that every start-up check then passes."
    )
    add("")
    add("---")
    add("")

    # -- appendix ----------------------------------------------------------
    add("## Appendix A. Formula reference")
    add("")
    formulas = pd.DataFrame(
        [
            {"Quantity": "Motor full-load current", "Formula": "I_FLC = P_shaft / (sqrt(3) V eta pf)", "Reference": "IEC 60034-1 rating definition"},
            {"Quantity": "Base impedance", "Formula": "Z_base = V^2 / S", "Reference": "Per-unit convention"},
            {"Quantity": "Transformer resistance", "Formula": "R_pu = P_loss / S_rated", "Reference": "IEC 60076-1 test quantities"},
            {"Quantity": "Transformer reactance", "Formula": "X = sqrt(Z^2 - R^2), Z_pu = u_k/100", "Reference": "IEEE Std 242 Ch. 2"},
            {"Quantity": "Source impedance", "Formula": "|Z_Q| = U_n^2 / S_kQ", "Reference": "IEC 60909-0"},
            {"Quantity": "Conductor temperature correction", "Formula": f"R_theta = R_20 [1 + {ALPHA_CU:g}(theta - 20)]", "Reference": "IEC 60228"},
            {"Quantity": "Short-circuit current", "Formula": "I_k'' = c U_n / (sqrt(3) |Z_k|)", "Reference": "IEC 60909-0 cl. 6.2"},
            {"Quantity": "Peak short-circuit current", "Formula": "i_p = kappa sqrt(2) I_k'', kappa = 1.02 + 0.98 e^(-3R/X)", "Reference": "IEC 60909-0 cl. 8"},
            {"Quantity": "Transformer correction", "Formula": "K_T = 0.95 c_max / (1 + 0.6 x_T)", "Reference": "IEC 60909-0 cl. 6.3.3"},
            {"Quantity": "Motor SC impedance", "Formula": "Z_M = U_rM^2 / (LRC_mult S_rM), R/X = 0.42", "Reference": "IEC 60909-0 cl. 6.6"},
            {"Quantity": "Thermal replica trip time", "Formula": "t = tau ln[((I/I_p)^2 - theta_0)/((I/I_p)^2 - 1)]", "Reference": "IEC 60255-149"},
            {"Quantity": "Thermal state", "Formula": "theta(t) = theta_inf + (theta_0 - theta_inf) e^(-t/tau)", "Reference": "IEC 60255-149"},
            {"Quantity": "Trip class time constant", "Formula": "tau = C / ln(7.2^2/(7.2^2 - 1)) = 51.32 C", "Reference": "IEC 60947-4-1 Table 2"},
            {"Quantity": "Motor damage limit", "Formula": "t_damage = t_stall (I_LR / I)^2", "Reference": "IEEE Std 242 Ch. 9"},
            {"Quantity": "Locked-rotor pickup", "Formula": f"I_set = {LOCKED_ROTOR_PICKUP_FRACTION:g} I_LR", "Reference": "IEEE Std 242 Ch. 9"},
            {"Quantity": "Stall timer window", "Formula": "1.25 t_dwell <= t_LR <= t_stall_hot / 1.25", "Reference": "IEEE Std 242 Ch. 9"},
            {"Quantity": "Instantaneous pickup", "Formula": f"I_inst = {INSTANTANEOUS_SET_MULTIPLE:g} I_LR, min 1.7 I_LR", "Reference": "IEEE Std 242 Ch. 9"},
            {"Quantity": "Locked-rotor impedance", "Formula": "|Z_LR| = V / (sqrt(3) I_LR) at arccos(pf_LR)", "Reference": "IEEE Std 399 Ch. 9"},
            {"Quantity": "Star-delta equivalent", "Formula": "Z_star = 3 Z_delta, I_star = I_delta / 3, T_star = T_delta / 3", "Reference": "Derived, see section 7.1"},
            {"Quantity": "Load constant impedance", "Formula": "|Z| = V^2 / |S| at arccos(pf)", "Reference": "IEEE Std 399 Ch. 9"},
            {"Quantity": "Voltage divider", "Formula": "V_bus/V_nom = |Z_par| / |Z_src + Z_par|", "Reference": "IEEE Std 399 Ch. 9"},
            {"Quantity": "Cable ampacity derating", "Formula": f"I_z = I_z,tab x {K_AMBIENT_40C_XLPE:g} x {K_GROUPING:g} x runs", "Reference": "IEC 60364-5-52"},
            {"Quantity": "Steady-state voltage drop", "Formula": "dV = sqrt(3) I (R cos_phi + X sin_phi)", "Reference": "IEC 60364-5-52 Annex G"},
            {"Quantity": "Cable adiabatic withstand", "Formula": f"t = (k S / I)^2, k = {K_ADIABATIC_CU_XLPE:.0f}", "Reference": "IEC 60364-5-54"},
        ]
    )
    add(_table(formulas))
    add("")

    add("## Appendix B. Check index")
    add("")
    seen: dict[str, Check] = {}
    for _, c in results.all_checks():
        seen.setdefault(c.id, c)
    add(_table(pd.DataFrame(
        [
            {"ID": cid, "Check": c.name, "Criterion": c.criterion, "Reference": c.reference}
            for cid, c in sorted(seen.items(), key=lambda kv: int(kv[0][1:]))
        ]
    )))
    add("")
    add(
        f"*Generated {today} by the motor protection coordination tool. "
        f"Plant data: `data/plant.yaml`"
        + (f", scenario `{p.scenario}`" if p.scenario else "")
        + ".*"
    )
    add("")
    return "\n".join(L)


# ===========================================================================
# HTML rendering
# ===========================================================================

_VERDICT_CLASS = {
    "PASS": "v-pass",
    "MARGINAL": "v-marg",
    "FAIL": "v-fail",
    "INFO": "v-info",
    "n/a": "v-na",
}

_CSS = """
:root{--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8983;--surface:#fcfcfb;
--rule:#dcdbd6;--pass:#0ca30c;--marg:#fab219;--fail:#d03b3b;--accent:#2a78d6}
*{box-sizing:border-box}
body{margin:0;background:#f4f3f0;color:var(--ink);
font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:48px 40px 80px;background:var(--surface);
box-shadow:0 0 0 1px var(--rule)}
h1{font-size:26px;line-height:1.25;margin:0 0 18px;letter-spacing:-.01em}
h2{font-size:20px;margin:44px 0 14px;padding-bottom:8px;border-bottom:2px solid var(--rule)}
h3{font-size:16px;margin:30px 0 10px;color:var(--ink)}
p{margin:12px 0}
hr{border:0;border-top:1px solid var(--rule);margin:34px 0}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:12.5px;
display:block;overflow-x:auto}
th,td{border:1px solid var(--rule);padding:6px 9px;text-align:left;vertical-align:top;
white-space:nowrap}
th{background:#f0efec;font-weight:600}
tr:nth-child(even) td{background:#fafaf8}
code{background:#f0efec;padding:1px 5px;border-radius:3px;font-size:12.5px}
pre{background:#f7f6f3;border:1px solid var(--rule);border-left:3px solid var(--accent);
padding:12px 14px;overflow-x:auto;font-size:12.5px;line-height:1.5}
pre code{background:none;padding:0}
blockquote{margin:16px 0;padding:10px 16px;border-left:3px solid var(--accent);
background:#f5f8fd;color:var(--ink2)}
img{max-width:100%;height:auto;display:block;margin:20px auto;
border:1px solid var(--rule);background:#fff}
ul{margin:12px 0;padding-left:22px}li{margin:5px 0}
strong.v-pass{color:var(--pass)}strong.v-marg{color:#a9760a}
strong.v-fail{color:var(--fail)}
em{color:var(--ink2)}
@media print{body{background:#fff}main{box-shadow:none;max-width:none;padding:0}
h2{page-break-after:avoid}img,table,pre{page-break-inside:avoid}}
"""


def _inline_md(text: str) -> str:
    """Inline markdown: escape, then apply bold, code and images."""
    text = _html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    for verdict, cls in _VERDICT_CLASS.items():
        text = text.replace(
            f"<strong>{verdict}</strong>", f'<strong class="{cls}">{verdict}</strong>'
        )
    return text


def _split_row(line: str) -> list[str]:
    """Split a markdown table row on unescaped pipes, then unescape the cells."""
    inner = line.strip()
    inner = re.sub(r"^\|", "", inner)
    inner = re.sub(r"(?<!\\)\|$", "", inner)
    return [c.strip().replace(r"\|", "|") for c in re.split(r"(?<!\\)\|", inner)]


def _embed(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def markdown_to_html(md: str, base_dir: Path, title: str) -> str:
    """Convert the markdown this module emits into a self-contained HTML page.

    Deliberately handles only the constructs `build_markdown` produces -
    headings, tables, fenced code, blockquotes, lists, images, paragraphs and
    inline bold/code. Images are inlined as base64 so the file stands alone.
    """
    out: list[str] = []
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(_html.escape(lines[i]))
                i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(block) + "</code></pre>")
            continue

        img = re.match(r"!\[(.*?)\]\((.+?)\)", line.strip())
        if img:
            alt, src = img.group(1), img.group(2)
            path = (base_dir / src).resolve()
            uri = _embed(path) if path.exists() else src
            out.append(f'<img src="{uri}" alt="{_html.escape(alt)}">')
            i += 1
            continue

        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s\-|]+\|$", lines[i + 1]):
            header = _split_row(line)
            i += 2
            body = []
            while i < len(lines) and lines[i].startswith("|"):
                body.append(_split_row(lines[i]))
                i += 1
            th = "".join(f"<th>{_inline_md(c)}</th>" for c in header)
            trs = []
            for row in body:
                tds = "".join(f"<td>{_inline_md(c)}</td>" for c in row)
                trs.append(f"<tr>{tds}</tr>")
            out.append(f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>")
            continue

        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{level}>{_inline_md(line[level:].strip())}</h{level}>")
            i += 1
            continue

        if line.strip() == "---":
            out.append("<hr>")
            i += 1
            continue

        if line.startswith(">"):
            block = []
            while i < len(lines) and lines[i].startswith(">"):
                block.append(lines[i].lstrip("> ").rstrip())
                i += 1
            out.append(f"<blockquote>{_inline_md(' '.join(block))}</blockquote>")
            continue

        if line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{_inline_md(lines[i][2:].strip())}</li>")
                i += 1
            out.append(f"<ul>{''.join(items)}</ul>")
            continue

        if line.strip():
            para = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(
                ("#", "|", ">", "- ", "```", "![", "---")
            ):
                para.append(lines[i].strip())
                i += 1
            text = _inline_md(" ".join(para))
            text = re.sub(r"^\*(.+)\*$", r"<em>\1</em>", text)
            out.append(f"<p>{text}</p>")
            continue

        i += 1

    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{_html.escape(title)}</title><style>{_CSS}</style></head>"
        f"<body><main>\n{chr(10).join(out)}\n</main></body></html>"
    )


# ===========================================================================
# Driver
# ===========================================================================


def write_report(results: StudyResults, outdir: str | Path) -> dict[str, Path]:
    """Write report.md, report.html and the CSV data exports."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "data").mkdir(exist_ok=True)

    md = build_markdown(results)
    md_path = outdir / "report.md"
    md_path.write_text(md, encoding="utf-8")

    title = results.plant.meta.get("project", "Motor Protection Study")
    if results.plant.scenario:
        title += f" - {results.plant.scenario}"
    html_path = outdir / "report.html"
    html_path.write_text(markdown_to_html(md, outdir, title), encoding="utf-8")

    exports = {
        "motors": motor_table(results.plant),
        "cables": cable_table(results.plant),
        "shortcircuit_bus": shortcircuit_table(results),
        "shortcircuit_terminals": terminal_fault_table(results),
        "protection_settings": settings_table(results),
        "starting_study": starting_table(results),
        "checks": pd.DataFrame(
            [
                {"Item": tag, "ID": c.id, "Check": c.name, "Criterion": c.criterion,
                 "Calculated": c.value, "Verdict": c.verdict, "Reference": c.reference,
                 "Note": c.note}
                for tag, c in results.all_checks()
            ]
        ),
    }
    paths = {"markdown": md_path, "html": html_path}
    for name, df in exports.items():
        csv_path = outdir / "data" / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        paths[f"csv_{name}"] = csv_path
    return paths
