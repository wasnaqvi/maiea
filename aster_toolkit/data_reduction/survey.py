"""Patchwork survey driver — one command from uncals to transmission spectrum.

Ties the whole Patchwork chain together for one target:

    uncals (mast.py download tree)
      -> segment inventory check           (exotedrf.inspect_uncal_directory)
      -> exoTEDRF Stages 1-3 per visit     (exotedrf.run_reduction)
      -> Stage 4 lightcurves + tilt scan   (lightcurves.py)
      -> juliet white fit, pass 1          (juliet.py, frozen v1.3)
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
    from orchestral.tools.base.tool import BaseTool
    from orchestral.tools.base.field_utils import RuntimeField, StateField
except ModuleNotFoundError:
    class BaseTool:
        """Fallback that keeps the driver importable without Orchestral."""

    def RuntimeField(default=None, description=None):
        return default

    def StateField(default=None, description=None):
        return default

from .exotedrf import (
    CRDS_CONTEXT,
    inspect_uncal_directory,
    run_reduction,
)
from .lightcurves import DEFAULT_RESOLUTION
from .juliet import (
    PATCHWORK_FIT_VERSION,
    figure_title,
    combine_visit_spectra,
    detector_offset_ppm,
    fetch_transit_priors,
    fit_transmission_spectrum,
    fit_white_lightcurve,
    prepare_visit_fit_inputs,
    write_spectrum_csv,
    _plot_combined,
)
from .contamination import (
    PATCHWORK_CONTAM_VERSION,
    anomaly_keep_mask,
    detect_lightcurve_anomalies,
    match_detector_anomalies,
    plot_anomaly_diagnostic,
    remap_anomaly_report,
    retrieve_contamination,
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

    stellar = manifest.get("stellar")
    if stellar is None and {"reduce", "fit", "contamination"} & set(steps):
        try:
            priors = fetch_transit_priors(planet)
        except Exception:
            if not cached_priors:
                raise
            priors = cached_priors
        stellar = {k: priors[k] for k in ("st_teff", "st_logg", "st_met")}

    summary: dict[str, Any] = {
        "planet_name": planet,
        "program": program,
        "fit_version": PATCHWORK_FIT_VERSION,
        "steps": list(steps),
        "visits": {},
    }

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
            with summary_path.open("w") as fh:
                json.dump(summary, fh, indent=2)
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

        for vname in visit_dirs:
            # -- pass 1: unmasked white fits, one per detector ----------
            # The Stage 5.5 spot scan needs a clean transit model to
            # subtract, and it needs BOTH detectors before it can confirm
            # anything (a bump in one detector alone is instrumental).
            # So every detector is fitted once unmasked, then scanned
            # together, then refitted with the mask.
            preps: dict[str, dict[str, Any]] = {}
            pass1: dict[str, dict[str, Any]] = {}
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
                log(f"[{slug}] fitting {vname} {det} white light (pass 1)...")
                prep = prepare_visit_fit_inputs(
                    products[0], planet,
                    instrument=det, reduction_dir=str(det_dir),
                    fallback_priors=cached_priors,
                )
                preps[det] = prep
                pass1[det] = fit_white_lightcurve(
                    prep["lc"], prep["priors"],
                    root / "fits" / vname / det.lower() / "pass1",
                    instrument=det, program=program, visit=vname,
                    regressors=prep["regressors"],
                    regressor_names=prep["regressor_names"],
                    ld=prep["ld"],
                    keep_mask=prep["tilt_keep_mask"],
                    tilt_report=prep["tilt_report"],
                    n_tilt_masked=prep["n_tilt_masked"],
                )
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

            # -- Stage 5.5: scan pass-1 residuals, cross-confirm --------
            anomaly_dir = root / "fits" / vname / "anomalies"
            anomaly_dir.mkdir(parents=True, exist_ok=True)
            reports, series = {}, {}
            for det, prep in preps.items():
                npz = np.load(Path(pass1[det]["output_dir"])
                              / "white_lightcurve_residuals.npz")
                series[det] = {"time": npz["time"], "residual": npz["residual"]}
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
            masks = {det: (anomaly_keep_mask(preps[det]["lc"]["time"].size,
                                             merged, detector=det)
                           & preps[det]["tilt_keep_mask"])
                     for det in preps}
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

            # -- pass 2: masked white refit + spectroscopic channels ----
            for det, prep in preps.items():
                fit_dir = root / "fits" / vname / det.lower()
                keep = masks[det]
                n_masked = int((~keep).sum())
                log(f"[{slug}] fitting {vname} {det} white light "
                    f"(pass 2, {n_masked} integration(s) masked)...")
                white = fit_white_lightcurve(
                    prep["lc"], prep["priors"], fit_dir,
                    instrument=det,
                    program=program, visit=vname,
                    regressors=prep["regressors"],
                    regressor_names=prep["regressor_names"],
                    ld=prep["ld"],
                    keep_mask=keep,
                    anomaly_scan=scan_payload,
                    tilt_report=prep["tilt_report"],
                    n_spot_masked=int(n_masked - prep["n_tilt_masked"]),
                    n_tilt_masked=prep["n_tilt_masked"],
                )
                log(f"[{slug}] fitting {vname} {det} spectroscopic channels...")
                spec = fit_transmission_spectrum(
                    prep["lc"], white, fit_dir / "spectro",
                    instrument=det,
                    program=program, visit=vname,
                    regressors=prep["regressors"],
                    stellar=stellar,
                    keep_mask=keep,
                )
                summary["visits"].setdefault(vname, {}).setdefault("fits", {})[det] = {
                    "white_depth_ppm": white["depth_ppm"],
                    "white_depth_ppm_pass1": pass1[det]["depth_ppm"],
                    "white_rms_ppm": white["residual_rms_ppm"],
                    "fit_version": white["fit_version"],
                    "tilt_events": len(prep["tilt_events"]),
                    "tilt_handling": white.get("tilt_handling"),
                    "spot_handling": white.get("spot_handling"),
                    "ld_source": white["ld_source"],
                    "priors_source": prep["priors_source"],
                    "rednoise": white.get("rednoise"),
                    "rednoise_pass1": pass1[det].get("rednoise"),
                    "depth_check": white.get("depth_check"),
                    "n_channels": spec["n_bins"],
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
                # Masking is only worth its integrations if it actually
                # improves the noise. Record the comparison so a target
                # where it did not can be spotted without a rerun.
                if n_masked:
                    b1 = (pass1[det].get("rednoise") or {}).get("beta_median")
                    b2 = (white.get("rednoise") or {}).get("beta_median")
                    d1 = pass1[det]["depth_ppm"]["median"]
                    d2 = white["depth_ppm"]["median"]
                    log(f"[{slug}] {vname} {det}: masking moved the depth "
                        f"{d2 - d1:+.0f} ppm and beta {b1:.2f} -> {b2:.2f}."
                        if b1 and b2 else
                        f"[{slug}] {vname} {det}: masking moved the depth "
                        f"{d2 - d1:+.0f} ppm.")

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
        for vname in visit_dirs:
            t0s = {}
            for det in ("NRS1", "NRS2"):
                ws = root / "fits" / vname / det.lower() / "white_fit_summary.json"
                if ws.is_file():
                    with ws.open() as fh:
                        w = json.load(fh)
                    if "t0_p1" in w:
                        t0s[det] = w["t0_p1"]["median"]
            if len(t0s) == 2:
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
                                    f"R = {DEFAULT_RESOLUTION}"))

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

            waves, depths, errs = [], [], []
            for c in csvs:
                s = read_spectrum_csv(c)
                waves.append(np.asarray(s["wave"], dtype=float))
                # read_spectrum_csv already returns ppm (depth_ppm columns).
                depths.append(np.asarray(s["depth_ppm"], dtype=float))
                errs.append(np.asarray(s["depth_err_ppm"], dtype=float))
            wave = np.concatenate(waves)
            order = np.argsort(wave)
            try:
                result = retrieve_contamination(
                    wave[order],
                    np.concatenate(depths)[order],
                    np.concatenate(errs)[order],
                    t_phot=float(t_phot),
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
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
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
    ``GeneratePatchworkFirJob``. Restartable — rerun with a later
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


class GeneratePatchworkFirJob(BaseTool):
    """
    Write the SLURM sbatch script that runs one Patchwork target
    end-to-end on the Digital Research Alliance of Canada **Fir** cluster
    (where the full uncal data and the exoTEDRF/juliet environments
    live). Does NOT submit anything — copy the manifest + script to Fir,
    check the environment paths, then ``sbatch`` it yourself.

    The script loads StdEnv/2023 python, activates the ASTER environment
    (juliet + aster_toolkit), points ASTER_EXOTEDRF_PYTHON at the pinned
    exoTEDRF environment, sets CRDS and ExoTiC-LD caches on scratch, and
    invokes ``python -m aster_toolkit.data_reduction.survey``.

    Example
    -------
        GeneratePatchworkFirJob(
            manifest_path="patchwork/GJ_9827_d.json",
            output_root="/scratch/wasi/patchwork",
            account="def-ncowan",
            time="36:00:00",
        )
    """

    manifest_path: str = RuntimeField(description="Path to the target manifest JSON.")
    output_root: str = RuntimeField(
        description="Output root ON FIR (e.g. /scratch/<user>/patchwork)."
    )
    account: str = RuntimeField(
        default="def-ncowan", description="DRAC allocation account."
    )
    time: str = RuntimeField(default="24:00:00", description="SLURM walltime.")
    cpus: int = RuntimeField(default=8, description="CPUs per task.")
    mem: str = RuntimeField(default="40G", description="Memory request.")
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
    exotedrf_repo: str = RuntimeField(
        default="~/exoTEDRF",
        description="Fir path to the exoTEDRF checkout that shadows the "
                    "installed release (keeps reductions and optimizer sweeps "
                    "on one stage-code tree).",
    )
    exotedrf_python: str = RuntimeField(
        default="~/bin/exotedrf-python",
        description="Path on Fir to the exoTEDRF interpreter. Prefer a wrapper "
                    "that module-loads its own python, since that env runs a "
                    "different python version than aster_env.",
    )
    steps: str = RuntimeField(
        default="inspect,reduce,fit,combine,contamination",
        description="Steps the job should run.",
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        manifest_path = self.manifest_path
        if not os.path.isabs(manifest_path):
            manifest_path = os.path.join(self.base_directory, manifest_path)
        path = write_fir_slurm_script(
            manifest_path,
            self.output_root,
            account=self.account,
            time=self.time,
            cpus=self.cpus,
            mem=self.mem,
            aster_env=self.aster_env,
            aster_repo=self.aster_repo,
            python_module=self.python_module,
            exotedrf_python=self.exotedrf_python,
            exotedrf_repo=self.exotedrf_repo,
            steps=self.steps,
        )
        return (
            f"Wrote {path}.\n"
            "Copy the manifest + this script to Fir, verify aster_env / "
            "exotedrf_python paths, then submit with: sbatch "
            f"{os.path.basename(str(path))}\n"
            "NOTE: manifest paths (raw_root, etc.) must be Fir paths."
        )
