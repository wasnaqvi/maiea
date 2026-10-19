"""Patchwork survey driver — one command from uncals to transmission spectrum.

Ties the whole Patchwork chain together for one target:

    uncals (mast.py download tree)
      -> segment inventory check           (exotedrf.inspect_uncal_directory)
      -> exoTEDRF Stages 1-3 per visit     (exotedrf.run_reduction)
      -> Stage 4 lightcurves + tilt scan   (lightcurves.py)
      -> juliet white fit, pass 1          (juliet.py, frozen v1.5)
      -> Stage 5.5 occulted-spot scan      (contamination.py)
      -> masked white refit + channels     (juliet.py)
      -> visit combination + metrics       (juliet.py Stage 6)
      -> Stage 6.5 unocculted-spot check   (contamination.py)

driven by a small per-target JSON manifest:

    {
      "planet_name": "GJ 9827 d",
      "planet_letter": "d",
      "visits": {
        "o010": {"raw_root": "/path/GJ_9827_d", "visit_prefix": "jw04098010001"},
        "o091": {"raw_root": "/path/GJ_9827_d", "visit_prefix": "jw04098091001"}
      },
      "stellar": {"st_teff": 4340, "st_logg": 4.66, "st_met": -0.26}  // optional
    }

A visit entry may instead be a plain directory string when that directory
already contains only that visit's uncal segments. ``stellar`` overrides
the NASA Exoplanet Archive values (useful offline on compute nodes).

Runs identically on the laptop (subset smoke tests) and on DRAC's Fir
cluster — ``write_fir_slurm_script`` emits the sbatch file; the CLI
(``python -m aster_toolkit.data_reduction.survey``) is what the job runs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    try:  # shared base + field helpers; see aster_toolkit/tool_base.py
        from aster_toolkit.tool_base import (AsterBaseTool as BaseTool, OptionalRuntimeField, RuntimeField, StateField)
    except ModuleNotFoundError as _exc:  # loaded by file path, outside the package
        if _exc.name != "aster_toolkit":
            raise
        import sys as _sys, pathlib as _pl
        _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
        from aster_toolkit.tool_base import (AsterBaseTool as BaseTool, OptionalRuntimeField, RuntimeField, StateField)
except ModuleNotFoundError:
    class BaseTool:
        """Fallback that keeps the driver importable without Orchestral."""

    def RuntimeField(default=None, description=None):
        return default

    def StateField(default=None, description=None):
        return default

    def OptionalRuntimeField(description=None, **kwargs):
        return None

from .exotedrf import (
    CRDS_CONTEXT,
    inspect_uncal_directory,
    run_reduction,
)
from .lightcurves import (
    DEFAULT_RESOLUTION,
    ramp_regressors,
    step_regressors,
    tilt_transition_keep_mask,
)
from .juliet import (
    PATCHWORK_FIT_VERSION,
    figure_title,
    combine_visit_spectra,
    detector_offset_ppm,
    fetch_transit_priors,
    fit_transmission_spectrum,
    fit_white_lightcurve_joint,
    prepare_visit_fit_inputs,
    write_spectrum_csv,
    _plot_combined,
    _require_celerite,
)
from .contamination import (
    PATCHWORK_CONTAM_VERSION,
    anomaly_keep_mask,
    detect_lightcurve_anomalies,
    match_detector_anomalies,
    plot_anomaly_diagnostic,
    remap_anomaly_report,
    retrieve_contamination,
    spot_crossing_regressors,
    step_events_for_regressors,
    summarize_anomalies,
    write_contamination_report,
)

DETECTORS = ("NRS1", "NRS2")
# "fit" is a compound step: white pass 1 -> Stage 5.5 anomaly scan ->
# masked white refit -> spectroscopic channels. The scan cannot be a
# separate top-level step because it sits *between* two fits of the same
# visit and both must see the same integrations. "contamination" (Stage
# 6.5) is separate: it runs on the combined spectrum, needs no
# refitting, and is cheap enough to iterate on.
ALL_STEPS = ("inspect", "reduce", "fit", "combine", "contamination")


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def load_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open() as fh:
        manifest = json.load(fh)
    for key in ("planet_name", "visits"):
        if key not in manifest:
            raise ValueError(f"Manifest missing required key '{key}'.")
    return manifest


def _load_existing_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open() as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Existing Patchwork summary is unreadable: {path}. Refusing to "
            "overwrite the target's science record; repair or move that file first."
        ) from exc
    if not isinstance(summary, dict):
        raise RuntimeError(
            f"Existing Patchwork summary is not a JSON object: {path}. "
            "Refusing to overwrite it."
        )
    return summary


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _known_overlapping_transits(manifest: dict[str, Any]) -> list[str]:
    overlaps: set[str] = set()
    for visit, spec in manifest.get("visits", {}).items():
        if isinstance(spec, dict):
            candidates = {str(p) for p in spec.get("transit_candidates", [])}
            if len(candidates) > 1:
                overlaps.update(candidates)
        if (manifest.get("planet_name") == "TOI-270 b"
                and visit == "o017"
                and "4098" in str(manifest.get("program", ""))):
            overlaps.update(("TOI-270 b", "TOI-270 d"))
    return sorted(overlaps)


def _expand_indexed_values(values: np.ndarray, index: np.ndarray,
                           n_total: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    index = np.asarray(index, dtype=int)
    if values.size != index.size:
        raise ValueError(
            f"Indexed series has {values.size} values but {index.size} indices."
        )
    if index.size and (index.min() < 0 or index.max() >= int(n_total)):
        raise ValueError("Indexed series contains an out-of-range integration.")
    if np.unique(index).size != index.size:
        raise ValueError("Indexed series contains duplicate integrations.")
    expanded = np.full(int(n_total), np.nan)
    expanded[index] = values
    return expanded


def stage_visit_uncals(
    visit_spec: str | dict[str, str],
    staging_dir: str | os.PathLike[str],
) -> str:
    """Resolve one manifest visit entry into a directory of that visit's
    uncal segments.

    A plain string is used unchanged. A ``{raw_root, visit_prefix}`` dict
    (needed when several visits share one download tree) is resolved by
    recursive glob and symlinked into ``staging_dir`` so exoTEDRF sees
    exactly one observation.
    """
    if isinstance(visit_spec, str):
        return visit_spec

    raw_root = visit_spec["raw_root"]
    prefix = visit_spec["visit_prefix"]
    files = sorted(glob.glob(
        os.path.join(raw_root, "**", f"{prefix}*_uncal.fits"), recursive=True
    ))
    # Target-acquisition exposures (e.g. 02101) are not TSO science.
    sci_tag = visit_spec.get("sci_tag", "04102")
    sci = [f for f in files if f"_{sci_tag}_" in os.path.basename(f)] or files
    if not sci:
        raise FileNotFoundError(
            f"No uncal files matching {prefix}* under {raw_root}."
        )
    # The same visit can exist under two directories (a named folder and
    # an obsid folder). Deduplicate by filename, keeping the largest copy
    # so a truncated download never wins over a complete one.
    best: dict[str, str] = {}
    for f in sci:
        name = os.path.basename(f)
        if name not in best or os.path.getsize(f) > os.path.getsize(best[name]):
            best[name] = f

    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    for name, f in sorted(best.items()):
        link = staging / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.abspath(f))
    return str(staging)


def existing_fit_products(fit_dir: str | os.PathLike[str]) -> list[str]:
    """Posterior files from a previous juliet run under ``fit_dir``.

    juliet RELOADS posteriors from ``out_folder`` instead of refitting when it
    finds them. That makes the fit step silently non-idempotent: after any
    change to the data, ephemeris, or priors, a rerun returns the PREVIOUS
    answers and "completes" in minutes. Detect it rather than trust it.
    """
    return sorted(glob.glob(os.path.join(str(fit_dir), "**", "*.pkl"),
                            recursive=True))


def run_patchwork_target(
    manifest: dict[str, Any],
    output_root: str | os.PathLike[str],
    *,
    steps: tuple[str, ...] | list[str] = ALL_STEPS,
    detectors: tuple[str, ...] = DETECTORS,
    force_refit: bool = False,
    log=print,
) -> dict[str, Any]:
    """Run the Patchwork chain for one target. Restartable: each step
    reads the previous step's on-disk products, so rerunning with a
    subset of ``steps`` continues where an earlier run stopped (exoTEDRF
    itself skips already-produced stage outputs unless force_redo).

    Returns (and writes) ``patchwork_summary.json`` under
    ``{output_root}/{target_slug}/`` — the per-target Patchwork log.
    """
    planet = manifest["planet_name"]
    letter = manifest.get("planet_letter", planet.split()[-1])
    program = manifest.get("program", "")   # e.g. "GO 4098"; used in figures
    slug = _slug(planet)
    root = Path(output_root) / slug
    root.mkdir(parents=True, exist_ok=True)

    # Manifest-cached priors (written by discover.py): the offline
    # fallback for compute nodes, and the expected-depth cross-check.
    cached_priors = manifest.get("priors")

    # Spectral binning resolution. Frozen survey-wide at R=100 by default
    # (see DEFAULT_RESOLUTION), but settable per target so a run can be
    # made FINER when the products need to support several binnings.
    # Rebinning only ever merges channels, so a coarser spectrum can be
    # derived from a finer fit for free (scripts/patchwork/rebin_spectra.py)
    # while the reverse needs a refit. Fitting once at R=200 therefore
    # yields R=200, R=100 and R=50; fitting at R=100 yields only 100 and
    # coarser. The cost is roughly linear in channel count: ~115 channels
    # at R=200 against ~58 at R=100, so about double the fit walltime.
    resolution = float(manifest.get("resolution") or DEFAULT_RESOLUTION)
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError(f"Manifest resolution must be positive; got {resolution!r}.")
    if resolution != DEFAULT_RESOLUTION:
        log(f"[{slug}] NON-DEFAULT binning R={resolution:.0f} "
            f"(survey default {DEFAULT_RESOLUTION}); products are not "
            "channel-for-channel comparable with default-R targets until "
            "rebinned to a common resolution.")

    stellar = manifest.get("stellar")
    if stellar is None and {"reduce", "fit", "contamination"} & set(steps):
        try:
            priors = fetch_transit_priors(planet)
        except Exception:
            if not cached_priors:
                raise
            priors = cached_priors
        stellar = {k: priors[k] for k in ("st_teff", "st_logg", "st_met")}

    # MERGE with any existing summary rather than starting empty. This
    # file is the per-target record, and a run of a step SUBSET only
    # populates the blocks belonging to those steps -- so a fresh dict
    # silently deletes everything the other steps had written. Running
    # `STEPS=contamination` on 2026-09-04 destroyed the fit record
    # (depths, beta, depth_check, scan_steps_corrected, spot_handling)
    # for all 16 fitted targets, leaving "visits": {} behind, because
    # the contamination step never touches per-visit fits.
    prior_path = root / "patchwork_summary.json"
    summary = _load_existing_summary(prior_path)
    summary.setdefault("visits", {})
    summary.update({
        "planet_name": planet,
        "steps": list(steps),
    })
    if program:
        summary["program"] = program
    else:
        summary.setdefault("program", "")
    if "fit" in steps:
        summary["fit_version"] = PATCHWORK_FIT_VERSION
        summary["resolution"] = resolution

    visit_dirs: dict[str, str] = {}
    for vname, spec in manifest["visits"].items():
        visit_dirs[vname] = stage_visit_uncals(spec, root / "uncals" / vname)

    # -- inspect ------------------------------------------------------
    if "inspect" in steps:
        for vname, vdir in visit_dirs.items():
            report = inspect_uncal_directory(vdir)
            incomplete = [
                det for det, e in report["detectors"].items() if not e["complete"]
            ]
            summary["visits"].setdefault(vname, {})["segments_complete"] = (
                not incomplete
            )
            log(f"[{slug}] {vname}: "
                + ("segments complete"
                   if not incomplete else f"INCOMPLETE segments on {incomplete}"))

    # -- reduce -------------------------------------------------------
    if "reduce" in steps:
        # A reduction that exits 0 but writes no Stage 3 product is a
        # FAILURE, not a success: exoTEDRF's run_DMS.py can die after
        # Stage 2 (e.g. a config key the installed branch requires) and
        # still leave the job looking COMPLETED. Collect every failure,
        # then raise — so the SLURM job exits nonzero and the campaign
        # cannot advance to the fit step on empty inputs.
        failures: list[str] = []
        for vname, vdir in visit_dirs.items():
            log(f"[{slug}] reducing {vname} (Stages 1-3, both detectors)...")
            m = run_reduction(
                vdir,
                root / "reductions" / vname,
                detectors=list(detectors),
                st_teff=stellar.get("st_teff"),
                st_logg=stellar.get("st_logg"),
                st_met=stellar.get("st_met"),
                planet_letter=letter,
            )
            entry = {}
            for det, e in m["detectors"].items():
                produced = bool(e.get("spectra_fullres"))
                entry[det] = e["success"] and produced
                if not entry[det]:
                    failures.append(
                        f"{vname} {det}: exit {e['returncode']}, "
                        f"{len(e.get('spectra_fullres') or [])} Stage 3 "
                        f"product(s) — see {e['log']}"
                    )
            summary["visits"].setdefault(vname, {})["reduction"] = entry

        if failures:
            summary_path = root / "patchwork_summary.json"
            _write_json_atomic(summary_path, summary)
            raise RuntimeError(
                f"[{slug}] reduction failed for {len(failures)} "
                f"visit x detector combination(s):\n  "
                + "\n  ".join(failures)
                + "\nThe last lines of the reduction.log name the cause. "
                  "Do NOT submit the fit step until this is resolved."
            )

    # -- fit ----------------------------------------------------------
    if "fit" in steps:
        from .exotedrf import find_stage3_products
        import shutil

        overlapping = _known_overlapping_transits(manifest)
        if len(overlapping) > 1:
            raise RuntimeError(
                f"{planet} has overlapping transits in this visit: "
                f"{', '.join(overlapping)}. Patchwork currently fits one planet, "
                "so a single-planet fit would absorb the deeper companion transit "
                "and assign the wrong depth/spectrum. Use a multi-planet transit "
                "model before fitting this visit."
            )

        # juliet reloads existing posteriors instead of refitting, so a rerun
        # after any change silently returns the old answers. Refuse, or clear.
        stale = existing_fit_products(root / "fits")
        if stale and force_refit:
            log(f"[{slug}] clearing {len(stale)} previous posterior file(s) "
                "before refitting.")
            shutil.rmtree(root / "fits", ignore_errors=True)
        elif stale:
            raise RuntimeError(
                f"{len(stale)} posterior file(s) from a previous fit exist "
                f"under {root / 'fits'} (e.g. {stale[0]}).\n"
                "juliet would RELOAD those instead of refitting, so any change "
                "to the ephemeris, priors, or lightcurves would be silently "
                "ignored and you would get the previous answers back.\n"
                "Pass force_refit=True (CLI: --force-refit) to delete them and "
                "fit properly, or use a fresh output_root."
            )

        # v1.6 needs celerite for the GP. Check before any sampling, not
        # after pass 1 has already spent its walltime.
        _require_celerite()

        for vname in visit_dirs:
            # -- pass 1: one unmasked joint white fit --------------------
            # The Stage 5.5 spot scan needs a clean transit model to
            # subtract, and it needs BOTH detectors before it can confirm
            # anything (a bump in one detector alone is instrumental).
            # So the detectors are fitted once unmasked, then scanned
            # together, then refitted with the corrections. Since v1.6
            # both fits are joint, so the scan never subtracts a transit
            # whose timing one detector got wrong on its own.
            preps: dict[str, dict[str, Any]] = {}
            for det in detectors:
                det_dir = root / "reductions" / vname / det.lower()
                products = [
                    p for p in find_stage3_products(det_dir)
                    if det.lower() in os.path.basename(p).lower()
                ] or find_stage3_products(det_dir)
                if not products:
                    log(f"[{slug}] {vname} {det}: no Stage 3 spectra — skipping fit.")
                    continue
                if len(products) > 1:
                    log(f"[{slug}] {vname} {det}: {len(products)} Stage 3 "
                        f"products found; using {products[0]}")
                log(f"[{slug}] preparing {vname} {det} lightcurves...")
                prep = prepare_visit_fit_inputs(
                    products[0], planet,
                    instrument=det, reduction_dir=str(det_dir),
                    fallback_priors=cached_priors,
                    # Manifest-level ephemeris correction. Only for a
                    # target whose archive ephemeris demonstrably misses
                    # a transit that IS in the data; every use is stamped
                    # into priors_source. See prepare_visit_fit_inputs.
                    priors_override=manifest.get("priors_override"),
                    resolution=resolution,
                )
                preps[det] = prep
                tr = prep["tilt_report"] or {}
                if tr.get("events"):
                    log(f"[{slug}] {vname} {det}: {len(tr['events'])} tilt "
                        "event(s) — fitted as Heaviside steps: "
                        + "; ".join(
                            f"index {e['index']} "
                            f"({e['amplitude_ppm']:+.0f} ppm"
                            + (", IN TRANSIT" if e.get("in_transit") else "")
                            + f", {'+'.join(e['sources'])})"
                            for e in tr["events"]))
                if tr.get("rate_warning"):
                    log(f"[{slug}] {vname} {det}: WARNING {tr['rate_warning']}")
            if not preps:
                continue

            # v1.6: ONE white fit for both detectors (shared orbit), no
            # GP in pass 1 -- a GP would soak up the very bumps the
            # Stage 5.5 scan has to find in these residuals.
            log(f"[{slug}] fitting {vname} white light, pass 1 "
                f"({'+'.join(preps)} joint, no GP)...")
            pass1 = fit_white_lightcurve_joint(
                {det: p["lc"] for det, p in preps.items()},
                next(iter(preps.values()))["priors"],
                {det: root / "fits" / vname / det.lower() / "pass1"
                 for det in preps},
                joint_dir=root / "fits" / vname / "joint" / "pass1",
                regressors={det: p["regressors"] for det, p in preps.items()},
                regressor_names={det: p["regressor_names"]
                                 for det, p in preps.items()},
                ld={det: p["ld"] for det, p in preps.items()},
                keep_masks={det: p["tilt_keep_mask"] for det, p in preps.items()},
                gp=False, program=program, visit=vname,
                tilt_reports={det: p["tilt_report"] for det, p in preps.items()},
                n_tilt_masked={det: p["n_tilt_masked"] for det, p in preps.items()},
            )

            # -- Stage 5.5: scan pass-1 residuals, cross-confirm --------
            anomaly_dir = root / "fits" / vname / "anomalies"
            anomaly_dir.mkdir(parents=True, exist_ok=True)
            reports, series = {}, {}
            for det, prep in preps.items():
                npz = np.load(Path(pass1[det]["output_dir"])
                              / "white_lightcurve_residuals.npz")
                original_index = np.asarray(npz["index"], dtype=int)
                residual_full = _expand_indexed_values(
                    npz["residual"], original_index, prep["lc"]["time"].size
                )
                series[det] = {"time": npz["time"], "residual": npz["residual"],
                               "residual_full": residual_full,
                               "oot_mask": npz["oot_mask"]}
                reports[det] = detect_lightcurve_anomalies(
                    npz["time"], npz["residual"], oot_mask=npz["oot_mask"],
                    detector=det,
                )
                # Pass 1 was fitted with the tilt transition mask, so the
                # npz arrays are compressed and the scan's indices count
                # the compressed series. Map them back to original
                # integration numbers before building any mask.
                reports[det] = remap_anomaly_report(
                    reports[det], npz["index"], prep["lc"]["time"].size)
            merged = match_detector_anomalies(reports)
            # The spot mask is combined with the tilt transition mask, so
            # pass 2 sees both corrections at once and the spectroscopic
            # channels can be checked against a single total.
            # v1.4: confirmed crossings are MODELLED with a Gaussian
            # (Radica et al. 2026) rather than masked, so no integrations
            # are dropped for them and the amplitude is free per channel.
            # anomaly_keep_mask is called with an empty kinds tuple so it
            # masks nothing while remaining the single place that decides
            # what a mask would have covered.
            masks = {det: preps[det]["tilt_keep_mask"] for det in preps}
            scan_payload = {
                "contam_version": PATCHWORK_CONTAM_VERSION,
                "planet_name": planet,
                "visit": vname,
                "detectors": reports,
                "events": [
                    {k: v for k, v in m.items() if k != "detectors"}
                    | {"detectors": {d: {"index_start": e["index_start"],
                                         "index_end": e["index_end"],
                                         "amplitude_ppm": e["amplitude_ppm"],
                                         "peak_sigma": e["peak_sigma"]}
                                     for d, e in m["detectors"].items()}}
                    for m in merged
                ],
                "masked_integrations": {d: int((~m).sum()) for d, m in masks.items()},
                "mask_indices": {d: np.flatnonzero(~m).tolist()
                                 for d, m in masks.items()},
            }
            with (anomaly_dir / "anomaly_report.json").open("w") as fh:
                json.dump(scan_payload, fh, indent=2)
            try:
                plot_anomaly_diagnostic(
                    reports, series, merged, anomaly_dir,
                    title=figure_title(planet, program=program, visit=vname,
                                       suffix="Stage 5.5 anomaly scan"))
            except Exception as exc:  # a figure must never fail a fit
                log(f"[{slug}] {vname}: anomaly figure failed ({exc}).")
            if merged:
                log(f"[{slug}] {vname}: Stage 5.5 scan —\n"
                    + summarize_anomalies(merged))
            else:
                log(f"[{slug}] {vname}: Stage 5.5 scan — no anomalies.")
            summary["visits"].setdefault(vname, {})["anomaly_scan"] = {
                "contam_version": PATCHWORK_CONTAM_VERSION,
                "n_events": len(merged),
                "n_confirmed": sum(1 for m in merged if m["confirmed"]),
                "masked_integrations": scan_payload["masked_integrations"],
                "report": str(anomaly_dir / "anomaly_report.json"),
            }

            # -- pass 2: corrected joint white refit (+ GP), channels ----
            # Build every detector's corrections first: the refit is one
            # joint fit, so it needs all of them at once.
            p2: dict[str, dict[str, Any]] = {}
            for det, prep in preps.items():
                keep = masks[det]
                # A step confirmed by the Stage 5.5 scan but missed by
                # the Stage 4 trace search would otherwise be corrected
                # by NOBODY: the scan declines to mask steps because a
                # Heaviside is the right correction, and the Heaviside
                # regressors were already built. Add them here, so a
                # step found by EITHER stage reaches the fit.
                # Pass the residuals so each break time and transition
                # width is FITTED, not taken from the flagged span's
                # centre (see step_events_for_regressors).
                scan_steps = step_events_for_regressors(
                    merged, det, residual=series.get(det, {}).get("residual_full"))
                regressors = prep["regressors"]
                regressor_names = list(prep["regressor_names"])
                spot_cols, spot_meta = spot_crossing_regressors(
                    prep["lc"]["time"], merged, det)
                if spot_cols.shape[1]:
                    regressors = np.column_stack([regressors, spot_cols])
                    regressor_names += [f"spot_gauss_{j}"
                                        for j in range(spot_cols.shape[1])]
                    log(f"[{slug}] {vname} {det}: {spot_cols.shape[1]} occulted "
                        "crossing(s) MODELLED as Gaussian(s) (not masked): "
                        + "; ".join(f"{m['kind']} {m['detected_amplitude_ppm']:+.0f} ppm "
                                    f"FWHM {m['fwhm_min']:.0f} min"
                                    for m in spot_meta))
                if scan_steps:
                    n_int = prep["lc"]["time"].size
                    extra = ramp_regressors(n_int, scan_steps)
                    regressors = np.column_stack([regressors, extra])
                    regressor_names += [f"scan_step_{j}"
                                        for j in range(extra.shape[1])]
                    # Drop the blended integrations at each transition,
                    # same rule as a Stage 4 tilt.
                    keep = keep & tilt_transition_keep_mask(n_int, scan_steps)
                    log(f"[{slug}] {vname} {det}: {len(scan_steps)} step(s) "
                        "from the Stage 5.5 scan added as Heaviside "
                        "regressor(s): "
                        + "; ".join(f"index {e['index']} "
                                    f"({e['amplitude_ppm']:+.0f} ppm, "
                                    f"{e['peak_sigma']:.1f} sigma, "
                                    f"width {e.get('width_ints', 0):.0f} ints)"
                                    for e in scan_steps))
                fit_tilt_report = dict(prep["tilt_report"] or {})
                fit_tilt_report["events"] = (
                    list(fit_tilt_report.get("events") or []) + scan_steps
                )
                p2[det] = {"keep": keep, "regressors": regressors,
                           "regressor_names": regressor_names,
                           "scan_steps": scan_steps, "spot_meta": spot_meta,
                           "n_masked": int((~keep).sum()),
                           "tilt_report": fit_tilt_report}

            log(f"[{slug}] fitting {vname} white light, pass 2 "
                f"({'+'.join(preps)} joint, GP; masked "
                + ", ".join(f"{d} {p2[d]['n_masked']}" for d in p2) + ")...")
            whites = fit_white_lightcurve_joint(
                {det: p["lc"] for det, p in preps.items()},
                next(iter(preps.values()))["priors"],
                {det: root / "fits" / vname / det.lower() for det in preps},
                joint_dir=root / "fits" / vname / "joint",
                regressors={det: p2[det]["regressors"] for det in preps},
                regressor_names={det: p2[det]["regressor_names"] for det in preps},
                ld={det: p["ld"] for det, p in preps.items()},
                keep_masks={det: p2[det]["keep"] for det in preps},
                gp=True, program=program, visit=vname,
                anomaly_scan=scan_payload,
                tilt_reports={det: p2[det]["tilt_report"] for det in preps},
                n_spot_masked={det: 0 for det in preps},
                n_tilt_masked={det: p2[det]["n_masked"] for det in preps},
            )

            for det, prep in preps.items():
                fit_dir = root / "fits" / vname / det.lower()
                white = whites[det]
                keep, n_masked = p2[det]["keep"], p2[det]["n_masked"]
                log(f"[{slug}] fitting {vname} {det} spectroscopic channels...")
                spec = fit_transmission_spectrum(
                    prep["lc"], white, fit_dir / "spectro",
                    instrument=det,
                    program=program, visit=vname,
                    regressors=p2[det]["regressors"],
                    stellar=stellar,
                    keep_mask=keep,
                )
                summary["visits"].setdefault(vname, {}).setdefault("fits", {})[det] = {
                    "white_depth_ppm": white["depth_ppm"],
                    "white_depth_ppm_pass1": pass1[det]["depth_ppm"],
                    "white_rms_ppm": white["residual_rms_ppm"],
                    "fit_version": white["fit_version"],
                    "joint_fit": white.get("joint_fit"),
                    "noise_model": white.get("noise_model"),
                    "t0_p1": white.get("t0_p1"),
                    "b_p1": white.get("b_p1"),
                    "a_p1": white.get("a_p1"),
                    "tilt_events": len(prep["tilt_events"]),
                    "scan_steps_corrected": p2[det]["scan_steps"],
                    "spot_crossings_modelled": p2[det]["spot_meta"],
                    "tilt_handling": white.get("tilt_handling"),
                    "spot_handling": white.get("spot_handling"),
                    "ld_source": white["ld_source"],
                    "priors_source": prep["priors_source"],
                    "priors_override_used": prep.get("priors_override_used"),
                    "rednoise": white.get("rednoise"),
                    "rednoise_before_gp": white.get("rednoise_before_gp"),
                    "rednoise_pass1": pass1[det].get("rednoise"),
                    "depth_check": white.get("depth_check"),
                    "n_channels": spec["n_bins"],
                    "ld_fixed_per_channel": spec["ld_fixed_per_channel"],
                    "median_depth_err_ppm": spec["median_depth_err_ppm"],
                    "median_channel_rms_ppm": spec["median_rms_ppm"],
                    "spectrum_csv": spec["spectrum_csv"],
                }
                dc = white.get("depth_check") or {}
                if dc.get("status") == "suspect":
                    log(f"[{slug}] {vname} {det}: SUSPECT white-light depth "
                        f"({dc['measured_ppm']:.0f} ppm vs {dc['band']} "
                        f"reference {dc['expected_ppm']:.0f} ppm, "
                        f"{dc['difference_sigma']:.1f} sigma) — verify before using.")
                elif dc.get("status") == "indicative":
                    log(f"[{slug}] {vname} {det}: depth "
                        f"{dc['difference_frac'] * 100:+.1f}% from the optical "
                        "archive value (different band — indicative only).")
                # Corrections are only worth their integrations if they
                # actually improve the noise. Record the comparison so a
                # target where they did not can be spotted without a
                # rerun. Compare like with like: pass 1 has no GP, so its
                # beta is set against pass 2's beta BEFORE the GP term.
                b1 = (pass1[det].get("rednoise") or {}).get("beta_median")
                b2 = ((white.get("rednoise_before_gp") or white.get("rednoise")
                       or {}).get("beta_median"))
                bgp = (white.get("rednoise") or {}).get("beta_median")
                d1 = pass1[det]["depth_ppm"]["median"]
                d2 = white["depth_ppm"]["median"]
                log(f"[{slug}] {vname} {det}: pass 2 moved the depth "
                    f"{d2 - d1:+.0f} ppm"
                    + (f"; beta {b1:.2f} -> {b2:.2f} (after GP {bgp:.2f})"
                       if b1 and b2 and bgp else "") + ".")

    # -- combine ------------------------------------------------------
    if "combine" in steps:
        combined_dir = root / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        combined = {}
        for det in detectors:
            csvs = sorted(glob.glob(str(
                root / "fits" / "*" / det.lower() / "spectro"
                / "transmission_spectrum.csv"
            )))
            if not csvs:
                continue
            S = combine_visit_spectra(csvs)
            combined[det] = S
            rows = [
                {"wave": w, "wave_err": we, "depth": d / 1e6,
                 "depth_err": de / 1e6, "rms_ppm": r}
                for w, we, d, de, r in zip(
                    S["wave"], S["wave_err"], S["depth_ppm"],
                    S["depth_err_ppm"], S["rms_ppm"])
            ]
            csv_path = combined_dir / f"combined_{det.lower()}_transmission_spectrum.csv"
            write_spectrum_csv(
                csv_path, rows,
                header=f"{planet} — {det}, {S['n_visits']} visit(s), "
                       f"Patchwork v{PATCHWORK_FIT_VERSION}",
            )
            summary.setdefault("combined", {})[det] = {
                "n_visits": S["n_visits"],
                "csv": str(csv_path),
                "median_depth_err_ppm": float(np.nanmedian(S["depth_err_ppm"])),
                "median_channel_rms_ppm": float(np.nanmedian(S["rms_ppm"])),
            }
        if "NRS1" in combined and "NRS2" in combined:
            summary["nrs1_nrs2_offset_ppm"] = detector_offset_ppm(
                combined["NRS1"], combined["NRS2"]
            )

        # Per-visit NRS1-NRS2 white-light t0 offset. COMPASS reports
        # statistically significant transit-time offsets between the two
        # detectors on some targets; record it as a survey health metric.
        # A v1.6 joint fit has ONE t0 by construction, so the offset is not
        # measured there and is recorded as shared rather than as a fake 0.
        for vname in visit_dirs:
            t0s, joint = {}, False
            for det in ("NRS1", "NRS2"):
                ws = root / "fits" / vname / det.lower() / "white_fit_summary.json"
                if ws.is_file():
                    with ws.open() as fh:
                        w = json.load(fh)
                    if "t0_p1" in w:
                        t0s[det] = w["t0_p1"]["median"]
                    joint = joint or (w.get("joint_fit") or {}).get("n_detectors", 1) > 1
            if joint:
                visit_rec = summary["visits"].setdefault(vname, {})
                visit_rec.pop("nrs1_nrs2_t0_offset_s", None)
                visit_rec["t0_shared_across_detectors"] = True
            elif len(t0s) == 2:
                summary["visits"].setdefault(vname, {})[
                    "nrs1_nrs2_t0_offset_s"
                ] = (t0s["NRS1"] - t0s["NRS2"]) * 86400.0
        if combined:
            n_vis = max(S["n_visits"] for S in combined.values())
            _plot_combined(
                combined, combined_dir,
                figure_title(planet, program=program,
                             suffix=f"{n_vis} visit"
                                    f"{'s' if n_vis != 1 else ''} combined, "
                                    f"R = {resolution:g}"))

    # -- contamination (Stage 6.5) ------------------------------------
    # Unocculted spots and faculae never appear in the lightcurve, so
    # Stage 5.5 cannot see them; they multiply the whole spectrum by
    # epsilon(lambda). Runs on the combined spectrum, costs seconds, and
    # never modifies the measured spectrum — it writes a corrected copy
    # alongside it.
    if "contamination" in steps:
        contam_dir = root / "contamination"
        csvs = sorted(glob.glob(str(
            root / "combined" / "combined_*_transmission_spectrum.csv")))
        t_phot = (stellar or {}).get("st_teff") if stellar else None
        if not csvs:
            log(f"[{slug}] contamination: no combined spectrum — run the "
                "combine step first.")
        elif not t_phot:
            log(f"[{slug}] contamination: no stellar effective temperature "
                "available — skipping. Add 'stellar' to the manifest.")
        else:
            from .juliet import read_spectrum_csv

            waves, depths, errs, is_nrs2 = [], [], [], []
            for c in csvs:
                s = read_spectrum_csv(c)
                waves.append(np.asarray(s["wave"], dtype=float))
                # read_spectrum_csv already returns ppm (depth_ppm columns).
                depths.append(np.asarray(s["depth_ppm"], dtype=float))
                errs.append(np.asarray(s["depth_err_ppm"], dtype=float))
                # Which half of the combined spectrum each channel came
                # from, so Stage 6.5 can fit the inter-detector step as
                # its own parameter. Without it epsilon(lambda) absorbs
                # the step and reports it as stellar contamination: on
                # 2026-08-27 that produced GJ 357 b at delta_BIC +74.7
                # and TOI-836.01 at +31.0, while TOI-836 b -- the SAME
                # star -- came out at -1.3. With the offset fitted and
                # the null raised to a straight line, all three are
                # non-detections and the two TOI-836 planets agree.
                is_nrs2.append(np.full(len(s["wave"]),
                                       "nrs2" in os.path.basename(c).lower()))
            wave = np.concatenate(waves)
            offset_mask = np.concatenate(is_nrs2) if len(csvs) > 1 else None
            order = np.argsort(wave)
            try:
                result = retrieve_contamination(
                    wave[order],
                    np.concatenate(depths)[order],
                    np.concatenate(errs)[order],
                    t_phot=float(t_phot),
                    offset_mask=(offset_mask[order]
                                 if offset_mask is not None else None),
                )
                written = write_contamination_report(result, contam_dir)
                p = result["posterior"]
                summary["contamination"] = {
                    "contam_version": PATCHWORK_CONTAM_VERSION,
                    "stellar_model": result["stellar_model"],
                    "f_het": p["f_het"],
                    "f_het_95pct_upper": result["f_het_95pct_upper"],
                    "T_het": p["T_het"],
                    "delta_bic": result["delta_bic"],
                    "delta_bic_linear": result["delta_bic_linear"],
                    "railed": result["railed"],
                    "offset_ppm": result["offset_ppm"],
                    "detected": result["contamination_detected"],
                    "corrected_csv": written["corrected_csv"],
                }
                log(f"[{slug}] Stage 6.5: f_het = {p['f_het']['median']:.3f} "
                    f"(<{result['f_het_95pct_upper']:.3f} at 95%), "
                    f"T_het = {p['T_het']['median']:.0f} K, "
                    f"delta_BIC = {result['delta_bic']:+.1f} — "
                    + ("contamination favoured; confirm with stctm on a "
                       "PHOENIX grid before publishing."
                       if result["contamination_detected"]
                       else "no correction needed."))
            except Exception as exc:
                # Stage 6.5 is diagnostic. A failure here must not cost
                # the reduction and fits that produced the spectrum.
                log(f"[{slug}] contamination: FAILED ({type(exc).__name__}: "
                    f"{exc}). The measured spectrum is unaffected.")
                summary["contamination"] = {"error": f"{type(exc).__name__}: {exc}"}

    summary_path = root / "patchwork_summary.json"
    _write_json_atomic(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    log(f"[{slug}] done. Summary: {summary_path}")
    return summary


# -------------------- Fir cluster (SLURM) --------------------

FIR_SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --account={account}
#SBATCH --job-name=patchwork_{slug}
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
# Log to the SUBMISSION directory: an --output path under {output_root}/{slug}
# does not exist before the first run, and SLURM kills a job with no log
# at all when the --output directory is missing.
#SBATCH --output=%x-%j.out

# --- Patchwork on DRAC Fir ---------------------------------------
# Load the module aster-env was built from. Stages 1-3 run in a separate
# environment on a different python, so ASTER_EXOTEDRF_PYTHON should point
# at a wrapper that loads its own module (see docs) rather than straight
# at the interpreter.
module load StdEnv/2023 {python_module}
source {aster_env}/bin/activate           # env with juliet + aster deps

# aster_toolkit is used from the repo checkout, not installed into the env.
export PYTHONPATH={aster_repo}${{PYTHONPATH:+:$PYTHONPATH}}
# Stages 1-3 run as a subprocess in the pinned exoTEDRF environment.
export ASTER_EXOTEDRF_PYTHON={exotedrf_python}
# Run exoTEDRF from the source checkout, so reductions and optimizer sweeps
# execute the SAME stage code. Unset this to use the installed release.
export ASTER_EXOTEDRF_REPO={exotedrf_repo}
export CRDS_PATH={crds_path}
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
# Pin the reference context so targets reduced weeks apart stay uniform.
export CRDS_CONTEXT={crds_context}
export ASTER_EXOTIC_LD_DATA={exotic_ld_data}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# juliet reloads existing posteriors instead of refitting, so a rerun after any
# change silently returns the OLD answers. FORCE_REFIT=1 deletes them first.
REFIT_FLAG=""; [[ "${{FORCE_REFIT:-0}}" == "1" ]] && REFIT_FLAG="--force-refit"

# Fail in seconds rather than hours if the exoTEDRF environment regressed
# (pip installs silently revert the compatibility patches).
python -c "
from aster_toolkit.data_reduction.exotedrf import verify_exotedrf_environment, format_environment_report
r = verify_exotedrf_environment(); print(format_environment_report(r))
raise SystemExit(0 if r['ok'] else 1)" || exit 1

# STEPS can be overridden at submit time WITHOUT regenerating the script:
#   STEPS=inspect,reduce sbatch --export=ALL <script>
# (set it in the submitting shell — SLURM's --export=NAME=VALUE splits on
# the comma inside the value no matter how it is quoted).
python -m aster_toolkit.data_reduction.survey \\
    --manifest {manifest_path} \\
    --output-root {output_root} \\
    --steps "${{STEPS:-{steps}}}" $REFIT_FLAG
"""


def write_fir_slurm_script(
    manifest_path: str | os.PathLike[str],
    output_root: str,
    *,
    script_path: str | os.PathLike[str] | None = None,
    account: str = "def-ncowan",
    time: str = "24:00:00",
    cpus: int = 8,
    mem: str = "40G",
    aster_env: str = "~/aster/aster-env",
    aster_repo: str = "~/aster/maiea",
    python_module: str = "python/3.13.2",
    exotedrf_python: str = "~/bin/exotedrf-python",
    exotedrf_repo: str = "~/exoTEDRF",
    crds_path: str = "~/scratch/crds_cache",
    crds_context: str = CRDS_CONTEXT,
    exotic_ld_data: str = "~/scratch/exotic_ld_data",
    steps: str = "inspect,reduce,fit,combine,contamination",
) -> Path:
    """Write the sbatch file that runs one target's Patchwork chain on
    Fir. Paths default to the Fir layout (scratch for the big caches);
    adjust ``aster_env``/``exotedrf_python`` to where the two
    environments actually live. Does NOT submit — inspect, then sbatch it.
    """
    manifest = load_manifest(manifest_path)
    slug = _slug(manifest["planet_name"])
    script = FIR_SBATCH_TEMPLATE.format(
        account=account, slug=slug, time=time, cpus=cpus, mem=mem,
        output_root=output_root, aster_env=aster_env, aster_repo=aster_repo,
        python_module=python_module,
        exotedrf_python=exotedrf_python, exotedrf_repo=exotedrf_repo,
        crds_path=crds_path,
        crds_context=crds_context, exotic_ld_data=exotic_ld_data,
        manifest_path=os.path.abspath(manifest_path), steps=steps,
    )
    path = Path(script_path) if script_path else (
        Path(manifest_path).parent / f"run_{slug}.sbatch"
    )
    path.write_text(script)
    return path


# -------------------- CLI --------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aster_toolkit.data_reduction.survey",
        description="Run the Patchwork uncals->spectrum chain for one target.",
    )
    parser.add_argument("--manifest", required=True, help="Target manifest JSON.")
    parser.add_argument("--output-root", required=True,
                        help="Root directory for all per-target outputs.")
    parser.add_argument("--steps", default=",".join(ALL_STEPS),
                        help="Comma-separated subset of: inspect,reduce,fit,combine,contamination.")
    parser.add_argument("--detectors", default="NRS1,NRS2")
    parser.add_argument("--force-refit", action="store_true",
                        help="Delete previous posteriors before fitting. "
                             "Required when refitting an existing output_root, "
                             "because juliet reloads posteriors it finds.")
    args = parser.parse_args(argv)

    steps = tuple(s.strip() for s in args.steps.split(",") if s.strip())
    unknown = set(steps) - set(ALL_STEPS)
    if unknown:
        parser.error(f"Unknown steps: {sorted(unknown)}")

    run_patchwork_target(
        load_manifest(args.manifest),
        args.output_root,
        steps=steps,
        detectors=tuple(d.strip().upper() for d in args.detectors.split(",")),
        force_refit=args.force_refit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -------------------- orchestral tools --------------------


class RunPatchworkTarget(BaseTool):
    """
    Run the full Patchwork chain for ONE target from a manifest JSON:
    segment inspection -> exoTEDRF Stages 1-3 (per visit, NRS1+NRS2) ->
    Stage 4 lightcurves + tilt scan -> juliet white-light + spectroscopic
    fits (frozen v1.1: ExoTiC-LD priors, trace decorrelation, tilt steps)
    -> multi-visit combination with the NRS1-NRS2 offset and rms metrics.

    Compute warning: the reduce step is HOURS per visit and the fit step
    is hours more. On the laptop use this only on subset data
    (``MakeUncalTestSubset``); full targets belong on Fir via
    ``GenerateFirJobs``. Restartable — rerun with a later
    ``steps`` subset to continue from existing on-disk products.

    Manifest format (see module docstring): planet_name, planet_letter,
    visits {name: uncal_dir | {raw_root, visit_prefix}}, optional stellar
    overrides.

    Outputs under {output_root}/{target}/: uncals/ (staged links),
    reductions/, fits/, combined/, and patchwork_summary.json — the
    per-target Patchwork log (white rms, channel rms, tilt events,
    NRS1-NRS2 offset).

    Example
    -------
        RunPatchworkTarget(
            manifest_path="patchwork/GJ_9827_d.json",
            output_root="patchwork/results",
            steps="inspect,reduce,fit,combine,contamination",
        )
    """

    manifest_path: str = RuntimeField(description="Path to the target manifest JSON.")
    output_root: str = RuntimeField(description="Root directory for outputs.")
    steps: str = RuntimeField(
        default="inspect,reduce,fit,combine,contamination",
        description="Comma-separated subset of inspect,reduce,fit,combine,contamination.",
    )
    force_refit: bool = RuntimeField(
        default=False,
        description="Delete previous posteriors before fitting. Required when "
                    "refitting an existing output_root — juliet reloads any "
                    "posteriors it finds instead of refitting.",
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        manifest_path = self.manifest_path
        if not os.path.isabs(manifest_path):
            manifest_path = os.path.join(self.base_directory, manifest_path)
        output_root = self.output_root
        if not os.path.isabs(output_root):
            output_root = os.path.join(self.base_directory, output_root)

        lines: list[str] = []
        summary = run_patchwork_target(
            load_manifest(manifest_path),
            output_root,
            steps=tuple(s.strip() for s in self.steps.split(",") if s.strip()),
            force_refit=self.force_refit,
            log=lines.append,
        )
        lines.append(f"Patchwork summary: {summary['summary_path']}")
        if "nrs1_nrs2_offset_ppm" in summary:
            lines.append(
                f"NRS1-NRS2 offset: {summary['nrs1_nrs2_offset_ppm']:+.1f} ppm"
            )
        return "\n".join(lines)


# Per-job-type SLURM resource defaults (walltime, cpus, mem). The optimizer
# runs ~50 Stage 1-3 trials per visit x detector, hence the longer default.
FIR_JOB_DEFAULTS = {
    "patchwork": {"time": "24:00:00", "cpus": 8, "mem": "40G"},
    "optimizer": {"time": "3-00:00:00", "cpus": 16, "mem": "64G"},
}


class GenerateFirJobs(BaseTool):
    """
    The ONE tool that writes SLURM sbatch scripts for DRAC **Fir** (where
    the full uncal data and the pinned exoTEDRF/juliet environments live).
    Does NOT submit anything — copy the manifest + script(s) to Fir, check
    the environment paths, then ``sbatch`` them yourself.

    ``job`` selects what the scripts run:
      * ``"patchwork"`` (default) — ONE script running the full Patchwork
        chain for one target (``python -m aster_toolkit.data_reduction.survey``);
        ``steps`` picks the subset. Defaults: 24:00:00, 8 CPUs, 40G.
      * ``"optimizer"`` — ONE script PER visit x detector running the frozen
        Class-2 optimizer sweep (``python -m aster_toolkit.data_reduction.optimize``;
        ~50 reductions each, so never bundled). ``visits`` / ``detectors``
        pick the subset; ``exotedrf_repo`` must be the pulled
        'optimizer'-branch checkout. Defaults: 3-00:00:00, 16 CPUs, 64G.
        After completion, run ``SummarizeG395hOptimization`` on the cost tables.

    Every script loads the python module, activates the ASTER env, points
    ASTER_EXOTEDRF_PYTHON / ASTER_EXOTEDRF_REPO at the exoTEDRF interpreter
    and checkout, and puts the CRDS (and ExoTiC-LD) caches on scratch.
    Manifest paths (raw_root, etc.) must be Fir paths.

    Examples
    --------
        GenerateFirJobs(manifest_path="patchwork/manifests/GJ_9827_d.json",
                        output_root="/scratch/wasi/patchwork", time="36:00:00")
        GenerateFirJobs(job="optimizer",
                        manifest_path="patchwork/manifests/GJ_9827_d.json",
                        output_root="/scratch/wasi/patchwork", detectors="NRS1")
    """

    manifest_path: str = RuntimeField(description="Path to the target manifest JSON.")
    output_root: str = RuntimeField(
        description="Output root ON FIR (e.g. /scratch/<user>/patchwork)."
    )
    job: str = RuntimeField(
        default="patchwork",
        description="'patchwork' (full target chain, one script) or "
                    "'optimizer' (one script per visit x detector).",
    )
    steps: str = RuntimeField(
        default="inspect,reduce,fit,combine,contamination",
        description="job='patchwork' only: comma-separated steps to run.",
    )
    visits: str | None = OptionalRuntimeField(
        description="job='optimizer' only: comma-separated visit keys; default all.",
    )
    detectors: str = RuntimeField(
        default="NRS1,NRS2",
        description="job='optimizer' only: comma-separated detectors.",
    )
    account: str = RuntimeField(
        default="def-ncowan", description="DRAC allocation account."
    )
    time: str | None = OptionalRuntimeField(
        description="SLURM walltime; default 24:00:00 (patchwork) or 3-00:00:00 (optimizer).",
    )
    cpus: int | str | None = OptionalRuntimeField(
        description="CPUs per task; default 8 (patchwork) or 16 (optimizer).",
    )
    mem: str | None = OptionalRuntimeField(
        description="Memory request; default 40G (patchwork) or 64G (optimizer).",
    )
    aster_env: str = RuntimeField(
        default="~/aster/aster-env",
        description="Path on Fir to the venv with juliet and the aster deps.",
    )
    aster_repo: str = RuntimeField(
        default="~/aster/maiea",
        description="Path on Fir to the repo containing aster_toolkit/ "
                    "(exported as PYTHONPATH).",
    )
    python_module: str = RuntimeField(
        default="python/3.13.2",
        description="Module providing the python that aster_env was built from.",
    )
    exotedrf_python: str = RuntimeField(
        default="~/bin/exotedrf-python",
        description="Path on Fir to the exoTEDRF interpreter (prefer a wrapper "
                    "that module-loads its own python).",
    )
    exotedrf_repo: str = RuntimeField(
        default="~/exoTEDRF",
        description="Fir path to the exoTEDRF checkout (the 'optimizer' branch "
                    "for job='optimizer'; it also shadows the release for "
                    "reductions so both share one stage-code tree).",
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        job = (self.job or "patchwork").strip().lower()
        if job not in FIR_JOB_DEFAULTS:
            raise ValueError(f"job must be one of {sorted(FIR_JOB_DEFAULTS)}, got {job!r}.")
        defaults = FIR_JOB_DEFAULTS[job]
        time = (self.time or "").strip() or defaults["time"]
        cpus = int(self.cpus) if str(self.cpus or "").strip() else defaults["cpus"]
        mem = (self.mem or "").strip() or defaults["mem"]

        manifest_path = self.manifest_path
        if not os.path.isabs(manifest_path):
            manifest_path = os.path.join(self.base_directory, manifest_path)
        common = dict(
            account=self.account, time=time, cpus=cpus, mem=mem,
            aster_env=self.aster_env, aster_repo=self.aster_repo,
            python_module=self.python_module,
            exotedrf_python=self.exotedrf_python,
            exotedrf_repo=self.exotedrf_repo,
        )

        if job == "patchwork":
            path = write_fir_slurm_script(
                manifest_path, self.output_root, steps=self.steps, **common
            )
            return (
                f"Wrote {path} (job=patchwork, {time}, {cpus} CPUs, {mem}).\n"
                "Copy the manifest + this script to Fir, verify aster_env / "
                "exotedrf_python paths, then submit with: sbatch "
                f"{os.path.basename(str(path))}\n"
                "NOTE: manifest paths (raw_root, etc.) must be Fir paths."
            )

        from .optimize import write_fir_optimizer_slurm_scripts

        visits = ([v.strip() for v in self.visits.split(",") if v.strip()]
                  if self.visits else None)
        written = write_fir_optimizer_slurm_scripts(
            manifest_path,
            self.output_root,
            visits=visits,
            detectors=tuple(d.strip().upper() for d in self.detectors.split(",")
                            if d.strip()),
            **common,
        )
        lines = [f"Wrote {len(written)} optimizer sbatch script(s) "
                 f"(job=optimizer, {time}, {cpus} CPUs, {mem}):"]
        lines += [f"  {p}" for p in written]
        lines.append(
            "Copy to Fir with the manifest, verify aster_env / "
            "exotedrf_python / exotedrf_repo paths, then sbatch each. "
            "After completion: SummarizeG395hOptimization on the cost tables."
        )
        return "\n".join(lines)
