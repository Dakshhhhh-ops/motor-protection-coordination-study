"""
Time-current characteristic plots, voltage-dip charts and the single-line diagram.

Plotting conventions follow protection-engineering practice: log-log axes,
current on x, time on y, decade major gridlines with 2-9 minor lines, and a
tall aspect ratio so the whole 10 A to 100 kA by 10 ms to 10 000 s field is
legible on one sheet -- the same layout as a manufacturer's TCC sheet.

Colour is assigned by role, not by taste:
  * Motor identity in the combined bus plot uses a fixed categorical order, so
    M1 is the same colour on every figure it appears on.
  * The damage limit uses the reserved "critical" status colour, because it
    means danger rather than identity.
  * The relay and the upstream breaker are boundaries, not peer series, so the
    relay is a single fixed hue on every per-motor sheet and the incomer is a
    near-black dashed line.
The categorical order was checked with a colour-vision-deficiency validator on
the adjacent-pair list; every curve also carries a direct label, so identity is
never conveyed by colour alone.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

from checks import derated_ampacity
from constants import (
    DAMAGE_CURVE_MIN_MULTIPLE,
    INSTANTANEOUS_MIN_MULTIPLE,
    THETA_TRIP,
)
from models import Motor, Plant
from protection import MotorSettings, damage_time
from shortcircuit import ShortCircuitStudy
from starting import DELTA, STAR, StartingStudy

# ---------------------------------------------------------------------------
# Palette and style
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#dcdbd6"
GRID_MINOR = "#eceae5"

#: Fixed categorical order. Motor k always gets slot k, on every figure.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

RELAY = "#2a78d6"      # the protective device curve (boundary, fixed hue)
DAMAGE = "#d03b3b"     # reserved status: critical
FEEDER = INK           # upstream boundary, rendered near-black dashed
ENVELOPE_FILL = "#e8eef7"
ENVELOPE_EDGE = "#6f8fb8"

I_MIN, I_MAX = 10.0, 1.0e5
T_MIN, T_MAX = 0.01, 1.0e4

LW = 1.8
LW_HEAVY = 2.4


def _style_axes(ax, title: str, subtitle: str = "") -> None:
    """Apply the shared TCC sheet styling."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(I_MIN, I_MAX)
    ax.set_ylim(T_MIN, T_MAX)
    ax.set_xlabel("Current (A, rms symmetrical at 415 V)", fontsize=9, color=INK_2)
    ax.set_ylabel("Time (s)", fontsize=9, color=INK_2)

    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.grid(which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(which="minor", color=GRID_MINOR, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, which="both")

    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=24 if subtitle else 8)
    if subtitle:
        ax.text(
            0.0, 1.006, subtitle, transform=ax.transAxes,
            fontsize=8.5, color=INK_2, ha="left", va="bottom",
        )


def _figure(width: float = 8.4, height: float = 9.6):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return p


def _curve(func, i_lo: float, i_hi: float, n: int = 600):
    """Sample a trip-time function over a current range, dropping non-finite points."""
    xs, ys = [], []
    for i_a in np.geomspace(max(i_lo, I_MIN), min(i_hi, I_MAX), n):
        t = func(float(i_a))
        if math.isfinite(t) and T_MIN <= t <= T_MAX:
            xs.append(float(i_a))
            ys.append(t)
    return xs, ys


def _label(ax, x: float, y: float, text: str, color: str, ha: str = "left", va: str = "bottom"):
    """Direct label with a surface halo, so it stays readable over gridlines."""
    ax.annotate(
        text, xy=(x, y), fontsize=8, color=color, ha=ha, va=va, weight="bold",
        path_effects=[
            matplotlib.patheffects.withStroke(linewidth=2.6, foreground=SURFACE)
        ],
    )


import matplotlib.patheffects  # noqa: E402  (needed by _label)


# ===========================================================================
# Starting envelope
# ===========================================================================


def _starting_envelope(motor: Motor) -> tuple[list[float], list[float]]:
    """Step curve of dwell-time-above-current during a healthy start.

    At current I the envelope value is the total time the motor spends at or
    above I while accelerating. The relay curve must lie ABOVE this envelope
    everywhere, otherwise it trips on a healthy start.

    This representation is what makes a star-delta start legible on a TCC: the
    envelope drops to the delta-stage duration alone once the current exceeds
    I_LR/3, instead of showing the full start time at full locked-rotor current.
    """
    stages = sorted(motor.starting_stages(), key=lambda s: s[0])
    xs = [I_MIN]
    ys = [motor.t_start_s]
    remaining = motor.t_start_s
    for current, duration, _ in stages:
        xs.append(current)
        remaining -= duration
        ys.append(max(remaining, 0.0))
    return xs, ys


# ===========================================================================
# Per-motor TCC
# ===========================================================================


def plot_motor_tcc(
    plant: Plant,
    motor: Motor,
    settings: MotorSettings,
    sc: ShortCircuitStudy,
    path: str | Path,
) -> Path:
    """Full coordination sheet for one motor."""
    fig, ax = _figure()
    ik_min = sc.terminal_min[motor.tag].device_ik_a
    ik_max = sc.terminal_max[motor.tag].device_ik_a
    ik_bus = sc.bus_max.ik_a

    verdict = settings.verdict
    _style_axes(
        ax,
        f"{motor.tag}  -  {motor.service}",
        f"{motor.kw:g} kW, {motor.voltage_v:g} V, FLC {motor.flc_a:.1f} A, "
        f"{'star-delta' if motor.is_star_delta else 'DOL'} start   |   "
        f"overload Class {settings.trip_class} at {settings.pickup_multiple * 100:.0f}% FLC   |   "
        f"verdict: {verdict}",
    )

    # -- starting envelope (region the relay must stay above) --------------
    ex, ey = _starting_envelope(motor)
    ax.fill_between(
        ex, T_MIN, ey, step="post", color=ENVELOPE_FILL, zorder=1,
        edgecolor=ENVELOPE_EDGE, linewidth=1.2,
    )
    ax.step(ex, ey, where="post", color=ENVELOPE_EDGE, linewidth=1.2, zorder=2)

    # Asymmetrical first-cycle inrush, half a cycle at 50 Hz.
    ax.plot(
        [INSTANTANEOUS_MIN_MULTIPLE * motor.i_lr_a], [0.01],
        marker="v", markersize=8, color=ENVELOPE_EDGE, zorder=6,
    )

    # -- motor thermal damage limits ---------------------------------------
    i_dmg_lo = DAMAGE_CURVE_MIN_MULTIPLE * motor.flc_a
    for hot, ls, name in ((True, "-", "hot"), (False, "--", "cold")):
        dx, dy = _curve(lambda i, h=hot: damage_time(motor, i, hot=h), i_dmg_lo, I_MAX)
        ax.plot(dx, dy, ls, color=DAMAGE, linewidth=LW, zorder=5)
        if dx:
            _label(ax, dx[0] * 1.05, dy[0], f"Damage limit ({name})", DAMAGE, va="top")

    # -- protective device -------------------------------------------------
    cx, cy = _curve(lambda i: settings.device_time(i, hot=True), settings.pickup_a * 1.001, I_MAX)
    ax.plot(cx, cy, "-", color=RELAY, linewidth=LW_HEAVY, zorder=7,
            solid_capstyle="round")
    hx, hy = _curve(lambda i: settings.thermal_time(i, hot=False), settings.pickup_a * 1.001, I_MAX)
    ax.plot(hx, hy, "--", color=RELAY, linewidth=1.2, zorder=6, alpha=0.85)
    if cx:
        mid = len(cx) // 3
        _label(ax, cx[mid], cy[mid] * 1.35, "Relay + MCCB (hot)", RELAY)
    if hx:
        _label(ax, hx[len(hx) // 2], hy[len(hx) // 2] * 1.3, "thermal (cold)", RELAY, va="bottom")

    # -- upstream incomer --------------------------------------------------
    fx, fy = _curve(plant.feeder.trip_time_s, plant.feeder.ir_a * 1.001, I_MAX)
    ax.plot(fx, fy, color=FEEDER, linewidth=LW, linestyle=(0, (7, 3)), zorder=4)
    if fx:
        _label(ax, fx[len(fx) // 4], fy[len(fx) // 4] * 1.3, plant.feeder.designation, FEEDER)

    # -- reference currents ------------------------------------------------
    for x, text, color in (
        (motor.flc_a, f"FLC {motor.flc_a:.0f} A", INK_MUTED),
        (motor.i_lr_a, f"LRC {motor.i_lr_a:.0f} A", INK_MUTED),
        (ik_min, f"Ik,min {ik_min / 1000:.1f} kA", INK_2),
        (ik_bus, f"Ik,bus {ik_bus / 1000:.1f} kA", INK_2),
    ):
        ax.axvline(x, color=color, linewidth=0.9, linestyle=":", zorder=3)
        ax.annotate(
            text, xy=(x, T_MAX * 0.55), fontsize=7.5, color=color,
            rotation=90, ha="right", va="top",
            path_effects=[matplotlib.patheffects.withStroke(linewidth=2.4, foreground=SURFACE)],
        )

    handles = [
        mpatches.Patch(facecolor=ENVELOPE_FILL, edgecolor=ENVELOPE_EDGE,
                       label="Starting envelope (dwell above current)"),
        Line2D([], [], color=RELAY, lw=LW_HEAVY, label="Overload relay + MCCB, hot"),
        Line2D([], [], color=RELAY, lw=1.2, ls="--", label="Thermal element, cold"),
        Line2D([], [], color=DAMAGE, lw=LW, label="Motor damage limit, hot"),
        Line2D([], [], color=DAMAGE, lw=LW, ls="--", label="Motor damage limit, cold"),
        Line2D([], [], color=FEEDER, lw=LW, ls=(0, (7, 3)), label=f"{plant.feeder.designation}"),
        Line2D([], [], color=ENVELOPE_EDGE, marker="v", ls="", label="Asymmetrical inrush, 1.7 x LRC"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=7.8, framealpha=0.96,
                    facecolor=SURFACE, edgecolor=GRID)
    leg.set_zorder(10)

    fig.text(
        0.0, -0.012,
        f"Damage limit is plotted only above {DAMAGE_CURVE_MIN_MULTIPLE:.0f} x FLC: the "
        "constant-I²t extrapolation of the nameplate locked-rotor withstand is rotor-limited "
        "and is not valid at low\noverloads, where the limit is stator- and insulation-limited "
        "and cannot be derived from nameplate data. The relay curve must lie above the "
        "starting envelope,\nbelow both damage limits, and below the incomer curve.",
        fontsize=7.4, color=INK_2, ha="left", va="top",
    )
    return _save(fig, path)


# ===========================================================================
# Combined bus TCC
# ===========================================================================


def plot_bus_tcc(
    plant: Plant,
    settings: dict[str, MotorSettings],
    sc: ShortCircuitStudy,
    path: str | Path,
) -> Path:
    """One sheet showing every motor feeder against the incomer."""
    fig, ax = _figure(width=9.0, height=9.6)
    _style_axes(
        ax,
        f"{plant.meta.get('bus_designation', 'Bus')}  -  combined bus coordination",
        f"All motor feeders and the {plant.feeder.designation}, "
        f"415 V, maximum bus fault {sc.bus_max.ik_ka:.1f} kA",
    )

    for idx, m in enumerate(plant.motors):
        colour = SERIES[idx % len(SERIES)]
        s = settings[m.tag]
        cx, cy = _curve(lambda i, s=s: s.device_time(i, hot=True), s.pickup_a * 1.001, I_MAX)
        ax.plot(cx, cy, color=colour, linewidth=LW, zorder=6)
        if cx:
            # Stagger the direct labels down the sheet, one decade band per
            # series, so five labels on five converging curves never collide.
            target_t = 900.0 / (3.2**idx)
            k = min(range(len(cy)), key=lambda j: abs(math.log(cy[j]) - math.log(target_t)))
            _label(ax, cx[k] * 1.06, cy[k], f"{m.tag} ({m.kw:g} kW)", colour, va="center")

    fx, fy = _curve(plant.feeder.trip_time_s, plant.feeder.ir_a * 1.001, I_MAX)
    ax.plot(fx, fy, color=FEEDER, linewidth=LW_HEAVY, linestyle=(0, (7, 3)), zorder=7)
    if fx:
        _label(ax, fx[len(fx) // 5], fy[len(fx) // 5] * 1.35, plant.feeder.designation, FEEDER)

    ax.axvline(plant.transformer.flc_a, color=INK_MUTED, lw=0.9, ls=":", zorder=3)
    ax.annotate(
        f"Transformer FLC {plant.transformer.flc_a:.0f} A",
        xy=(plant.transformer.flc_a, T_MAX * 0.55), fontsize=7.5, color=INK_MUTED,
        rotation=90, ha="right", va="top",
        path_effects=[matplotlib.patheffects.withStroke(linewidth=2.4, foreground=SURFACE)],
    )
    ax.axvline(sc.bus_max.ik_a, color=DAMAGE, lw=1.1, ls=":", zorder=3)
    ax.annotate(
        f"Ik,max {sc.bus_max.ik_ka:.1f} kA", xy=(sc.bus_max.ik_a, T_MAX * 0.55),
        fontsize=7.5, color=DAMAGE, rotation=90, ha="right", va="top",
        path_effects=[matplotlib.patheffects.withStroke(linewidth=2.4, foreground=SURFACE)],
    )

    handles = [
        Line2D([], [], color=SERIES[i % len(SERIES)], lw=LW,
               label=f"{m.tag}  {m.service} ({m.kw:g} kW)")
        for i, m in enumerate(plant.motors)
    ]
    handles.append(
        Line2D([], [], color=FEEDER, lw=LW_HEAVY, ls=(0, (7, 3)),
               label=f"{plant.feeder.designation}  Ir {plant.feeder.ir_a:.0f} A, "
                     f"Isd {plant.feeder.isd_a:.0f} A / {plant.feeder.tsd_s:g} s")
    )
    leg = ax.legend(handles=handles, loc="lower left", fontsize=7.8, framealpha=0.96,
                    facecolor=SURFACE, edgecolor=GRID)
    leg.set_zorder(10)
    return _save(fig, path)


# ===========================================================================
# Thermal state during starting
# ===========================================================================


def plot_thermal_state(
    plant: Plant, settings: dict[str, MotorSettings], path: str | Path
) -> Path:
    """Thermal replica state through a start, hot restart against cold start.

    This is the figure that shows why the hot restart is the binding case: every
    motor begins a hot restart with most of its thermal budget already spent.
    """
    fig, (ax_hot, ax_cold) = plt.subplots(
        1, 2, figsize=(11.0, 4.6), dpi=150, sharey=True
    )
    fig.patch.set_facecolor(SURFACE)

    for ax, cond in ((ax_hot, "hot restart"), (ax_cold, "cold start")):
        ax.set_facecolor(SURFACE)
        for idx, m in enumerate(plant.motors):
            s = settings[m.tag]
            sim = s.sim_hot if cond == "hot restart" else s.sim_cold
            colour = SERIES[idx % len(SERIES)]
            ax.plot(sim.trace_t, sim.trace_theta, color=colour, linewidth=LW, zorder=5)
            if sim.trace_t:
                _label(ax, sim.trace_t[-1], sim.trace_theta[-1], f" {m.tag}", colour, va="center")
        ax.axhline(THETA_TRIP, color=DAMAGE, linewidth=LW, zorder=6)
        ax.annotate(
            "trip threshold  θ = 1.0", xy=(0.02, THETA_TRIP * 1.03),
            xycoords=("axes fraction", "data"), fontsize=8, color=DAMAGE, va="bottom",
            path_effects=[matplotlib.patheffects.withStroke(linewidth=2.6, foreground=SURFACE)],
        )
        ax.set_xlabel("Time from energisation (s)", fontsize=9, color=INK_2)
        ax.set_title(cond.capitalize(), fontsize=10.5, color=INK, loc="left")
        ax.grid(color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 1.25)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8)

    ax_hot.set_ylabel("Thermal state θ (per unit of trip threshold)", fontsize=9, color=INK_2)
    fig.suptitle(
        "Thermal replica state during starting  -  IEC 60255-149 model",
        fontsize=12, color=INK, x=0.008, ha="left", y=1.02,
    )
    fig.text(
        0.008, -0.06,
        "A motor at rated load already sits at θ = (I_FLC/I_p)² = 0.83 for a 110 per cent pickup, so a hot restart "
        "begins with 83 per cent of the\nthermal budget spent. Curves reaching θ = 1.0 trip on a healthy start.",
        fontsize=8, color=INK_2, ha="left",
    )
    return _save(fig, path)


# ===========================================================================
# Voltage dip
# ===========================================================================


def plot_voltage_dip(plant: Plant, study: StartingStudy, path: str | Path) -> Path:
    """Terminal and bus voltage for each motor and starting method.

    A dot plot against the two acceptance thresholds, rather than grouped bars:
    the question is "does each case clear its limit", and position against a
    reference line answers that far more directly than bar length.
    """
    motors = list(plant.motors)
    n = len(motors)
    fig, (ax_m, ax_b) = plt.subplots(
        1, 2, figsize=(11.4, 0.72 * n + 2.6), dpi=150, sharey=True
    )
    fig.patch.set_facecolor(SURFACE)

    v_motor_min = plant.criteria.get("v_motor_min_pu", 0.80) * 100
    v_bus_min = plant.criteria.get("v_bus_min_pu", 0.85) * 100

    specs = (
        (ax_m, "v_motor_pu", v_motor_min, "Motor terminal voltage", "IEEE Std 399 limit"),
        (ax_b, "v_bus_pu", v_bus_min, "Bus voltage", "IEC 60947-4-1 contactor limit"),
    )

    for ax, attr, limit, title, limit_name in specs:
        ax.set_facecolor(SURFACE)
        for row, m in enumerate(motors):
            y = n - 1 - row
            dol = study.get(m.tag, DELTA, True)
            star = study.get(m.tag, STAR, True)
            v_dol = getattr(dol, attr) * 100
            v_star = getattr(star, attr) * 100
            ax.plot([v_dol, v_star], [y, y], color=GRID, linewidth=2.0, zorder=2,
                    solid_capstyle="round")
            ax.plot([v_dol], [y], marker="o", markersize=9, color=SERIES[0],
                    markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=5)
            ax.plot([v_star], [y], marker="D", markersize=8, color=SERIES[2],
                    markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=5)
            # Both values are labelled: the aqua marker sits below 3:1 contrast
            # on this surface, so the number carries the reading, not the hue.
            for value, dx in ((v_dol, -1), (v_star, 1)):
                ax.annotate(
                    f"{value:.1f}", xy=(value, y), xytext=(6 * dx, -13),
                    textcoords="offset points", fontsize=7.5, color=INK_2,
                    ha="center", zorder=6,
                    path_effects=[matplotlib.patheffects.withStroke(
                        linewidth=2.4, foreground=SURFACE)],
                )

        ax.axvline(limit, color=DAMAGE, linewidth=LW, zorder=4)
        ax.annotate(
            f"{limit_name}  {limit:.0f}%", xy=(limit - 0.6, -0.55), fontsize=7.8,
            color=DAMAGE, ha="right", va="center",
            path_effects=[matplotlib.patheffects.withStroke(linewidth=2.6, foreground=SURFACE)],
        )
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"{m.tag}  {m.kw:g} kW" for m in reversed(motors)], fontsize=9)
        ax.set_xlabel("Voltage (per cent of nominal)", fontsize=9, color=INK_2)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_xlim(min(70, limit - 6), 102)
        ax.set_ylim(-0.7, n - 0.2)
        ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8)

    handles = [
        Line2D([], [], marker="o", ls="", markersize=9, color=SERIES[0],
               label="DOL (full winding)"),
        Line2D([], [], marker="D", ls="", markersize=8, color=SERIES[2],
               label="Star-delta, star stage"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5,
               framealpha=0.0, bbox_to_anchor=(0.5, -0.055))

    fig.suptitle(
        "Motor starting voltage  -  each motor starting with all others running",
        fontsize=12, color=INK, x=0.008, ha="left", y=1.0,
    )
    return _save(fig, path)


# ===========================================================================
# Single-line diagram
# ===========================================================================


def plot_sld(
    plant: Plant,
    settings: dict[str, MotorSettings],
    sc: ShortCircuitStudy,
    path: str | Path,
) -> Path:
    """Single-line diagram with the calculated fault levels annotated."""
    n = len(plant.motors)
    # Motor columns occupy 1.95 units each; 2.4 further units keep the static
    # load clear of the last motor feeder.
    width = 1.95 * n + 2.4
    x_src = (1.95 * n) * 0.5
    bus_y = 4.6
    bus_x0, bus_x1 = 0.45, width - 0.45

    fig, ax = plt.subplots(figsize=(0.80 * (width + 1.4), 0.80 * 9.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_axis_off()
    # Equal aspect so circles are circles and the arcs read as contactors.
    ax.set_aspect("equal", adjustable="box")

    def wire(x0, y0, x1, y1, lw=1.6, color=INK):
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, solid_capstyle="round",
                zorder=4)

    def text(x, y, s, size=8.2, color=INK, ha="center", va="center", weight="normal"):
        ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, weight=weight, zorder=6)

    # -- utility source ----------------------------------------------------
    text(x_src, 8.15, f"11 kV utility  -  {plant.source.fault_mva:.0f} MVA fault level, "
                      f"X/R {plant.source.x_over_r:.0f}", size=9, weight="bold")
    wire(x_src, 7.85, x_src, 7.25)

    # -- transformer: two overlapping circles ------------------------------
    for dy in (0.0, -0.42):
        ax.add_patch(mpatches.Circle((x_src, 6.85 + dy), 0.34, fill=False,
                                     edgecolor=INK, linewidth=1.6, zorder=5))
    t = plant.transformer
    text(x_src + 0.62, 6.66,
         f"{t.kva:.0f} kVA  {t.hv_kv:g} kV / {t.lv_v:g} V  {t.vector_group}\n"
         f"u_k {t.uk_pct:g}%   X/R {t.x_over_r:.2f}   FLC {t.flc_a:.0f} A",
         size=8, ha="left")
    wire(x_src, 6.01, x_src, 5.42)

    # -- incomer ACB -------------------------------------------------------
    fb = plant.feeder
    ax.add_patch(mpatches.Rectangle((x_src - 0.30, 5.05), 0.60, 0.42, fill=False,
                                    edgecolor=INK, linewidth=1.8, zorder=5))
    text(x_src + 0.52, 5.26,
         f"{fb.designation}   {fb.frame_a:.0f} A, Icu {fb.icu_ka:.0f} kA\n"
         f"Ir {fb.ir_a:.0f} A / tr {fb.tr_s:g} s   "
         f"Isd {fb.isd_a:.0f} A / tsd {fb.tsd_s:g} s   "
         f"Ii {'OFF' if fb.ii_a is None else f'{fb.ii_a:.0f} A'}",
         size=8, ha="left")
    wire(x_src, 5.05, x_src, bus_y)

    # -- busbar ------------------------------------------------------------
    ax.plot([bus_x0, bus_x1], [bus_y, bus_y], color=INK, linewidth=4.6,
            solid_capstyle="butt", zorder=4)
    text(bus_x0, bus_y + 0.30,
         f"{plant.meta.get('bus_designation', 'Bus')}   415 V, 50 Hz, "
         f"{plant.bus.rated_short_time_withstand_ka:.0f} kA Icw", size=9, ha="left",
         weight="bold")
    text(bus_x1, bus_y + 0.30,
         f"Ik'' = {sc.bus_max.ik_ka:.2f} kA   ip = {sc.bus_max.ip_ka:.1f} kA",
         size=8.5, ha="right", color=DAMAGE)

    # -- motor feeders -----------------------------------------------------
    for idx, m in enumerate(plant.motors):
        s = settings[m.tag]
        x = 0.95 + idx * 1.95
        colour = SERIES[idx % len(SERIES)]
        wire(x, bus_y, x, 3.88, color=colour)

        # MCCB
        ax.add_patch(mpatches.Rectangle((x - 0.19, 3.54), 0.38, 0.34, fill=False,
                                        edgecolor=colour, linewidth=1.6, zorder=5))
        text(x + 0.26, 3.71, f"50\n{s.inst_pickup_a:.0f} A", size=6.6, ha="left", color=INK_2)
        wire(x, 3.54, x, 3.18, color=colour)

        # contactor: an open arc
        ax.add_patch(mpatches.Arc((x, 3.02), 0.34, 0.34, theta1=20, theta2=160,
                                  edgecolor=colour, linewidth=1.6, zorder=5))
        text(x + 0.26, 3.02, "K", size=6.6, ha="left", color=INK_2)
        wire(x, 2.86, x, 2.52, color=colour)

        # overload relay
        ax.add_patch(mpatches.Rectangle((x - 0.19, 2.18), 0.38, 0.34, fill=False,
                                        edgecolor=colour, linewidth=1.6, zorder=5))
        text(x + 0.26, 2.35, f"49/51\nCl {s.trip_class}", size=6.6, ha="left", color=INK_2)
        wire(x, 2.18, x, 1.62, color=colour)

        # cable annotation
        run_prefix = "" if m.cable.runs == 1 else f"{m.cable.runs} x "
        text(x + 0.13, 1.90,
             f"{run_prefix}{m.cable.size_mm2:g} mm²\n{m.cable.length_m:g} m",
             size=6.8, ha="left", color=INK_MUTED)

        # motor
        ax.add_patch(mpatches.Circle((x, 1.28), 0.34, fill=False, edgecolor=colour,
                                     linewidth=1.8, zorder=5))
        text(x, 1.28, "M", size=10, color=colour, weight="bold")
        text(x, 0.72,
             f"{m.tag}\n{m.kw:g} kW\n{m.flc_a:.0f} A\n"
             f"{'Y-Δ' if m.is_star_delta else 'DOL'}",
             size=7.4, va="top")
        text(x, -0.18, f"Ik {sc.terminal_max[m.tag].device_ik_a / 1000:.1f} kA",
             size=6.8, va="top", color=DAMAGE)

    # -- static load -------------------------------------------------------
    x_load = bus_x1 - 0.35
    wire(x_load, bus_y, x_load, 3.95)
    ax.add_patch(mpatches.Polygon(
        [[x_load - 0.26, 3.95], [x_load + 0.26, 3.95], [x_load, 3.50]],
        closed=True, fill=False, edgecolor=INK_2, linewidth=1.5, zorder=5))
    text(x_load, 3.30, f"Static load\n{plant.static_load.kva:.0f} kVA\npf {plant.static_load.pf:g}",
         size=7.4, va="top", color=INK_2)

    ax.set_xlim(-0.5, width + 0.9)
    ax.set_ylim(-0.9, 8.6)
    title = f"{plant.meta.get('project', 'Plant')}  -  single-line diagram"
    if plant.scenario:
        title += f"   [scenario: {plant.scenario}]"
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=12)
    return _save(fig, path)


# ===========================================================================
# Driver
# ===========================================================================


def generate_all(
    plant: Plant,
    settings: dict[str, MotorSettings],
    sc: ShortCircuitStudy,
    starting: StartingStudy,
    outdir: str | Path,
) -> dict[str, Path]:
    """Render every figure and return a map of name -> path."""
    outdir = Path(outdir)
    figures: dict[str, Path] = {
        "sld": plot_sld(plant, settings, sc, outdir / "single_line_diagram.png"),
        "bus_tcc": plot_bus_tcc(plant, settings, sc, outdir / "tcc_bus_combined.png"),
        "thermal": plot_thermal_state(plant, settings, outdir / "thermal_state.png"),
        "dip": plot_voltage_dip(plant, starting, outdir / "voltage_dip.png"),
    }
    for m in plant.motors:
        figures[f"tcc_{m.tag}"] = plot_motor_tcc(
            plant, m, settings[m.tag], sc, outdir / f"tcc_{m.tag}.png"
        )
    return figures
