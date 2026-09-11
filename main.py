#!/usr/bin/env python3
"""
Motor Protection Coordination & Starting Study - command line entry point.

    python main.py                                 base case, settings as specified
    python main.py --settings recommended          apply the tool's own setting search
    python main.py --scenario weak_source          run a sensitivity case
    python main.py --list-scenarios                show what is available

Outputs land in --outdir (default ./output):
    report.md        the study
    report.html      the same content, self-contained; print to PDF from a browser
    figures/*.png    single-line diagram, TCC sheets, thermal and voltage-dip charts
    data/*.csv       every result table, for onward use in a spreadsheet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

import checks as checks_mod  # noqa: E402
import plots  # noqa: E402
import protection  # noqa: E402
import report as report_mod  # noqa: E402
import shortcircuit  # noqa: E402
import starting  # noqa: E402
from models import Plant  # noqa: E402
from protection import FAIL, MARGINAL, PASS  # noqa: E402

EXIT_OK = 0
EXIT_MARGINAL = 0  # a marginal result is a finding, not a tool failure
EXIT_FAIL = 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="415 V motor protection coordination and starting study.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--plant", default=str(ROOT / "data" / "plant.yaml"),
                   help="plant definition YAML (default: data/plant.yaml)")
    p.add_argument("--cables", default=str(ROOT / "data" / "cables.yaml"),
                   help="cable library YAML (default: data/cables.yaml)")
    p.add_argument("--outdir", default=str(ROOT / "output"),
                   help="output directory (default: ./output)")
    p.add_argument("--scenario", default=None,
                   help="named sensitivity scenario from the plant YAML")
    p.add_argument("--settings", choices=("specified", "recommended"), default="specified",
                   help="use the trip classes and pickups given in the plant data "
                        "(default), or the values chosen by the independent setting search")
    p.add_argument("--no-figures", action="store_true",
                   help="skip figure rendering (much faster; report images will be missing)")
    p.add_argument("--no-viewer", action="store_true",
                   help="skip the interactive HTML viewer")
    p.add_argument("--list-scenarios", action="store_true",
                   help="list the scenarios defined in the plant YAML and exit")
    p.add_argument("--quiet", action="store_true", help="suppress the console summary")
    return p


def list_scenarios(plant_path: str) -> int:
    raw = yaml.safe_load(Path(plant_path).read_text(encoding="utf-8"))
    scenarios = raw.get("scenarios") or {}
    if not scenarios:
        print("No scenarios defined.")
        return EXIT_OK
    print(f"Scenarios defined in {plant_path}:\n")
    for name, spec in scenarios.items():
        desc = " ".join((spec.get("description") or "").split())
        print(f"  {name}")
        print(f"      {desc}")
        for k, v in (spec.get("overrides") or {}).items():
            print(f"        {k} = {v}")
        print()
    return EXIT_OK


def run(args: argparse.Namespace) -> tuple[report_mod.StudyResults, dict[str, Path]]:
    """Execute the full study and write every output."""
    plant = Plant.from_yaml(args.plant, args.cables, scenario=args.scenario)

    sc = shortcircuit.run_study(plant)
    settings = protection.design_all(
        plant, sc, use_recommended=(args.settings == "recommended")
    )
    start_study = starting.run_study(plant)
    plant_chk = checks_mod.plant_checks(plant, sc, start_study)
    cable_chk = {
        m.tag: checks_mod.cable_checks(plant, m, settings[m.tag], sc, start_study)
        for m in plant.motors
    }

    outdir = Path(args.outdir)
    figures: dict[str, Path] = {}
    if not args.no_figures:
        figures = plots.generate_all(plant, settings, sc, start_study, outdir / "figures")

    results = report_mod.StudyResults(
        plant=plant,
        sc=sc,
        settings=settings,
        starting=start_study,
        plant_checks=plant_chk,
        cable_checks=cable_chk,
        figures=figures,
    )
    paths = report_mod.write_report(results, outdir)

    if not args.no_viewer:
        import webexport

        paths["viewer"] = webexport.write_viewer(
            results, outdir / "viewer", settings_mode=args.settings
        )
    return results, paths


def print_summary(results: report_mod.StudyResults, paths: dict[str, Path], args) -> None:
    p = results.plant
    sc = results.sc
    counts = results.counts()
    rule = "=" * 78

    print(rule)
    print(p.meta.get("project", "Motor protection study"))
    if p.scenario:
        print(f"scenario: {p.scenario}")
    print(f"settings: as {args.settings}")
    print(rule)
    print(f"Transformer     {p.transformer.kva:.0f} kVA, u_k {p.transformer.uk_pct:g}%, "
          f"X/R {p.transformer.x_over_r:.2f}, FLC {p.transformer.flc_a:.0f} A")
    print(f"Loading         {p.diversified_kva:.0f} kVA "
          f"({p.transformer_loading_pct:.1f}% of rating) at pf {p.diversified_pf:.3f}")
    print(f"Bus fault       Ik'' {sc.bus_max.ik_ka:.2f} kA max "
          f"({sc.bus_max_no_motors.ik_ka:.2f} kA network + "
          f"{sc.motor_contribution_pct:.0f}% motor back-feed), "
          f"ip {sc.bus_max.ip_ka:.1f} kA")
    print()

    header = (f"{'Tag':<5}{'kW':>6}{'FLC':>8}{'Class':>7}{'xFLC':>7}"
              f"{'51LR':>8}{'50 (A)':>9}{'th.hot':>8}{'Vmot%':>8}  Verdict")
    print(header)
    print("-" * 78)
    for m in p.motors:
        s = results.settings[m.tag]
        dip = results.starting.as_designed(p, m.tag, others_running=True)
        lr = f"{s.lr_time_s:.1f}s" if s.lr_time_s else "NONE"
        worst = protection.worst(
            [s.verdict, dip.verdict] + [c.verdict for c in results.cable_checks[m.tag]]
        )
        print(f"{m.tag:<5}{m.kw:>6.0f}{m.flc_a:>8.1f}{s.trip_class:>7}"
              f"{s.pickup_multiple:>7.2f}{lr:>8}{s.inst_pickup_a:>9.0f}"
              f"{s.sim_hot.theta_max:>8.3f}{dip.v_motor_pu * 100:>8.1f}  {worst}")
    print("-" * 78)

    findings = results.findings()
    if findings:
        print(f"\n{len(findings)} finding(s):\n")
        for tag, c in findings:
            print(f"  [{c.verdict:<8}] {tag:<6} {c.id}  {c.name}")
            print(f"             {c.value}")
    else:
        print("\nNo findings. Every check passes.")

    print(f"\n{counts.get(PASS, 0)} pass / {counts.get(MARGINAL, 0)} marginal / "
          f"{counts.get(FAIL, 0)} fail / {counts.get('INFO', 0)} info"
          f"   ->  overall: {results.overall}")
    print()
    print(f"Report   {paths['markdown']}")
    print(f"HTML     {paths['html']}   (open and print to PDF for a PDF copy)")
    if "viewer" in paths:
        print(f"Viewer   {paths['viewer']}   (interactive study, open in a browser)")
    if results.figures:
        print(f"Figures  {len(results.figures)} PNGs in "
              f"{Path(args.outdir) / 'figures'}")
    print(f"Data     {len([k for k in paths if k.startswith('csv_')])} CSV exports in "
          f"{Path(args.outdir) / 'data'}")
    print(rule)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_scenarios:
        return list_scenarios(args.plant)

    try:
        results, paths = run(args)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print_summary(results, paths, args)

    return EXIT_FAIL if results.overall == FAIL else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
