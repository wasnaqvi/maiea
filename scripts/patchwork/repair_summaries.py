#!/usr/bin/env python
"""Rebuild damaged Patchwork target rollups from surviving products.

The 2026-09-04 contamination-only run replaced 16 target summaries with
empty ``visits`` dictionaries. The white fits, anomaly reports, spectra,
and reduction manifests survived. This script reconstructs the rollups
without rerunning juliet.

Dry-run by default. ``--write`` creates ``patchwork_summary.json.pre-repair``
and atomically replaces the damaged summary.

    python scripts/patchwork/repair_summaries.py ~/Patchwork_Data
    python scripts/patchwork/repair_summaries.py ~/Patchwork_Data --write
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aster_toolkit.data_reduction.contamination import (  # noqa: E402
    step_events_for_regressors,
)
from aster_toolkit.data_reduction.juliet import read_spectrum_csv  # noqa: E402
from aster_toolkit.data_reduction.survey import (  # noqa: E402
    _expand_indexed_values,
    _write_json_atomic,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _scan_steps(target_dir: Path, visit: str, detector: str) -> list[dict[str, Any]]:
    report_path = target_dir / "fits" / visit / "anomalies" / "anomaly_report.json"
    residual_path = (target_dir / "fits" / visit / detector.lower() / "pass1"
                     / "white_lightcurve_residuals.npz")
    if not report_path.exists() or not residual_path.exists():
        return []
    report = _read_json(report_path)
    with np.load(residual_path) as residuals:
        n_total = int(np.asarray(residuals["n_total"]))
        full = _expand_indexed_values(
            residuals["residual"], residuals["index"], n_total
        )
    return step_events_for_regressors(
        report.get("events", []), detector, residual=full
    )


def _fit_entry(target_dir: Path, summary_path: Path) -> dict[str, Any]:
    white = _read_json(summary_path)
    visit = summary_path.parents[1].name
    detector = summary_path.parent.name.upper()
    spectrum_path = summary_path.parent / "spectro" / "transmission_spectrum.csv"
    spectrum = read_spectrum_csv(spectrum_path) if spectrum_path.exists() else None
    scan_steps = _scan_steps(target_dir, visit, detector)
    spot_events = [
        event for event in (white.get("spot_handling") or {}).get("events", [])
        if event.get("confirmed") and event.get("kind") in (
            "spot_crossing", "facula_crossing"
        )
    ]
    spot_diagnostics = [
        event for event in (white.get("spot_handling") or {}).get("events", [])
        if event.get("kind") != "step"
    ]
    spot_handling = dict(white.get("spot_handling") or {})
    spot_handling.update({
        "method": "gaussian regressor" if spot_events else "none detected",
        "n_masked": 0,
        "frac_masked": 0.0,
        "events": spot_diagnostics,
    })
    tilt_handling = dict(white.get("tilt_handling") or {})
    tilt_handling["events"] = list(tilt_handling.get("events") or []) + scan_steps
    tilt_handling["n_events"] = len(tilt_handling["events"])
    tilt_handling["n_transition_masked"] = int(white.get("n_masked_total") or 0)
    entry = {
        "white_depth_ppm": white.get("depth_ppm"),
        "white_rms_ppm": white.get("residual_rms_ppm"),
        "fit_version": white.get("fit_version"),
        "tilt_events": tilt_handling["n_events"],
        "scan_steps_corrected": scan_steps,
        "spot_crossings_modelled": spot_events,
        "tilt_handling": tilt_handling,
        "spot_handling": spot_handling,
        "ld_source": white.get("ld_source"),
        "rednoise": white.get("rednoise"),
        "depth_check": white.get("depth_check"),
    }
    pass1_path = summary_path.parent / "pass1" / "white_fit_summary.json"
    if pass1_path.exists():
        pass1 = _read_json(pass1_path)
        entry["white_depth_ppm_pass1"] = pass1.get("depth_ppm")
        entry["rednoise_pass1"] = pass1.get("rednoise")
    if spectrum is not None:
        entry.update({
            "n_channels": int(np.isfinite(spectrum["depth_ppm"]).sum()),
            "median_depth_err_ppm": float(np.nanmedian(spectrum["depth_err_ppm"])),
            "median_channel_rms_ppm": float(np.nanmedian(spectrum["rms_ppm"])),
            "spectrum_csv": str(spectrum_path),
        })
    return {key: value for key, value in entry.items() if value is not None}


def rebuild_summary(target_dir: Path) -> tuple[dict[str, Any], list[str]]:
    path = target_dir / "patchwork_summary.json"
    prior = _read_json(path) if path.exists() else {}
    rebuilt = dict(prior)
    visits = dict(rebuilt.get("visits") or {})
    notes = []

    for manifest_path in sorted((target_dir / "reductions").glob("*/reduction_manifest.json")):
        visit = manifest_path.parent.name
        reduction = _read_json(manifest_path)
        visits.setdefault(visit, {})["reduction"] = {
            detector: bool(record.get("success") and record.get("spectra_fullres"))
            for detector, record in (reduction.get("detectors") or {}).items()
        }

    white_paths = sorted(Path(path) for path in glob.glob(str(
        target_dir / "fits" / "*" / "nrs*" / "white_fit_summary.json"
    )))
    versions = set()
    for white_path in white_paths:
        visit = white_path.parents[1].name
        detector = white_path.parent.name.upper()
        fit = _fit_entry(target_dir, white_path)
        visits.setdefault(visit, {}).setdefault("fits", {})[detector] = fit
        if fit.get("fit_version"):
            versions.add(fit["fit_version"])
        anomaly_path = target_dir / "fits" / visit / "anomalies" / "anomaly_report.json"
        if anomaly_path.exists():
            anomaly = _read_json(anomaly_path)
            visits[visit]["anomaly_scan"] = {
                "contam_version": anomaly.get("contam_version"),
                "n_events": len(anomaly.get("events", [])),
                "n_confirmed": sum(bool(event.get("confirmed"))
                                   for event in anomaly.get("events", [])),
                "masked_integrations": anomaly.get("masked_integrations", {}),
                "report": str(anomaly_path),
            }

    rebuilt["visits"] = visits
    if len(versions) == 1:
        rebuilt["fit_version"] = versions.pop()
    elif versions:
        rebuilt["fit_version"] = "mixed:" + ",".join(sorted(versions))

    combined = {}
    inferred_resolution = []
    for spectrum_path in sorted((target_dir / "combined").glob(
            "combined_*_transmission_spectrum.csv")):
        detector = "NRS2" if "nrs2" in spectrum_path.name.lower() else "NRS1"
        spectrum = read_spectrum_csv(spectrum_path)
        inferred_resolution.extend(
            (spectrum["wave"] / (2.0 * spectrum["wave_err"])).tolist()
        )
        n_visits = sum(
            detector in (visit.get("fits") or {}) for visit in visits.values()
        )
        combined[detector] = {
            "n_visits": n_visits,
            "csv": str(spectrum_path),
            "median_depth_err_ppm": float(np.nanmedian(spectrum["depth_err_ppm"])),
            "median_channel_rms_ppm": float(np.nanmedian(spectrum["rms_ppm"])),
        }
    if combined:
        rebuilt["combined"] = combined
    if inferred_resolution:
        resolution = float(np.nanmedian(inferred_resolution))
        rebuilt["resolution"] = int(5 * round(resolution / 5.0))

    if white_paths:
        first = _read_json(white_paths[0])
        rebuilt["planet_name"] = first.get("planet_name") or rebuilt.get("planet_name")
        rebuilt["program"] = first.get("program") or rebuilt.get("program", "")
    notes.append(
        "Rebuilt from surviving products; priors_source and priors_override_used "
        "cannot be recovered from pre-repair per-detector summaries."
    )
    rebuilt["summary_repair"] = {
        "source": "scripts/patchwork/repair_summaries.py",
        "notes": notes,
    }
    return rebuilt, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", help="Directory containing target output directories.")
    parser.add_argument("--write", action="store_true",
                        help="Back up and replace each patchwork_summary.json.")
    args = parser.parse_args()
    root = Path(os.path.expanduser(args.root))
    targets = sorted(path for path in root.iterdir() if path.is_dir())
    repaired = 0
    for target_dir in targets:
        if not list((target_dir / "fits").glob("*/nrs*/white_fit_summary.json")):
            continue
        summary, _ = rebuild_summary(target_dir)
        n_fits = sum(len((visit.get("fits") or {}))
                     for visit in summary.get("visits", {}).values())
        print(f"{target_dir.name}: {len(summary['visits'])} visit(s), "
              f"{n_fits} detector fit(s)")
        if args.write:
            path = target_dir / "patchwork_summary.json"
            backup = path.with_name(path.name + ".pre-repair")
            if backup.exists():
                current = _read_json(path) if path.exists() else {}
                repaired_before = ((current.get("summary_repair") or {}).get("source")
                                   == "scripts/patchwork/repair_summaries.py")
                if not repaired_before:
                    raise FileExistsError(f"Backup already exists: {backup}")
            elif path.exists():
                shutil.copy2(path, backup)
            _write_json_atomic(path, summary)
        repaired += 1
    action = "repaired" if args.write else "would repair"
    print(f"{action} {repaired} target summary file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
