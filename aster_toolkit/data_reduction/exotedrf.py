"""Patchwork uniform JWST TSO reduction built on exoTEDRF.

This module turns raw JWST time-series uncal files (downloaded with the
mast.py tools) into time-resolved extracted stellar spectra using the
exoTEDRF pipeline (Radica 2024), which wraps the official STScI ``jwst``
DMS with TSO-specific custom steps (group-level 1/f destriping, custom
superbias handling, bad-pixel interpolation, box extraction).

Uniformity contract
-------------------
Every Patchwork target is reduced with the *same* frozen configuration
(``PATCHWORK_G395H_CONFIG``). Per-target inputs are limited to file paths
and stellar parameters (Teff/logg/[Fe/H], used only for wavelength
calibration). A copy of the exact YAML config used is archived by
exoTEDRF next to the outputs, so any reduction is reproducible.

Transit -> Phase curve. Scale up!!

Environment isolation
---------------------
exoTEDRF pins ``numpy==1.26`` / ``jwst==1.17.1`` and cannot be imported in
the main ASTER environment. The reduction therefore runs as a subprocess
in a dedicated conda environment (default ``exotedrf``), located via the
``ASTER_EXOTEDRF_PYTHON`` environment variable or the default path below.

Outputs
-------
``{output_dir}/{nrs1|nrs2}/pipeline_outputs_directory/Stage3/
{target}_box_spectra_fullres.fits`` with FITS extensions
Wave / Wave Err / Flux / Flux Err / Time — the input expected by the
lightcurve-fitting tools in ``juliet.py``.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
from pathlib import Path
from typing import Any

try:
    from orchestral.tools.base.tool import BaseTool
    from orchestral.tools.base.field_utils import RuntimeField, StateField
except ModuleNotFoundError:
    class BaseTool:
        """Fallback that keeps plain reduction wrappers importable without Orchestral."""

    def RuntimeField(default=None, description=None):
        return default

    def StateField(default=None, description=None):
        return default


DEFAULT_EXOTEDRF_PYTHON = "/opt/anaconda3/envs/exotedrf/bin/python"
DEFAULT_CRDS_CACHE = str(Path.home() / "crds_cache")
NIRSPEC_DETECTORS = ("NRS1", "NRS2")

# Frozen Patchwork reduction settings for NIRSpec/G395H BOTS (SUB2048,
# NRSRAPID). Changing any value here changes the survey definition — bump
# ``PATCHWORK_CONFIG_VERSION`` if you do, so old and new reductions are
# never silently mixed in one population analysis.
PATCHWORK_CONFIG_VERSION = "1.0"
PATCHWORK_G395H_CONFIG: dict[str, Any] = {
    "observing_mode": "NIRSpec/G395H",
    "input_filetag": "uncal",
    # --- Stage 1 ---
    "DQInitStep": "run",
    "EmiCorrStep": "skip",        # MIRI only
    "SaturationStep": "run",
    "ResetStep": "skip",          # MIRI only
    "SuperBiasStep": "run",
    "RefPixStep": "skip",         # SOSS only
    "DarkCurrentStep": "skip",
    "OneOverFStep_grp": "run",    # group-level 1/f destriping
    "LinearityStep": "run",
    "JumpStep": "run",
    "RampFitStep": "run",
    "GainScaleStep": "run",
    "hot_pixel_map": "None",
    "superbias_method": "crds",
    "soss_background_file": "None",
    "oof_method": "median",       # NIRSpec options: 'median' or 'slope'
    "soss_timeseries": "None",
    "soss_timeseries_o2": "None",
    "outlier_maps": "None",
    "soss_inner_mask_width": 40,
    "soss_outer_mask_width": 70,
    "nirspec_mask_width": 16,
    "miri_drop_groups": 12,
    "flag_up_ramp": False,
    "jump_threshold": 15,
    "flag_in_time": True,
    "time_jump_threshold": 10,
    "stage1_kwargs": {},
    # --- Stage 2 ---
    "AssignWCSStep": "run",
    "Extract2DStep": "run",       # NIRSpec only
    "SourceTypeStep": "run",
    "WaveCorrStep": "run",        # NIRSpec only
    "FlatFieldStep": "skip",      # SOSS/MIRI only
    "BackgroundStep": "skip",     # SOSS/MIRI only
    "OneOverFStep_int": "skip",
    "BadPixStep": "run",
    # PCA TSO reconstruction is diagnostic-only and crashes in exoTEDRF
    # 2.3.1 with sklearn >=1.3 (3D array into inverse_transform).
    "PCAReconstructStep": "skip",
    "TracingStep": "run",
    "miri_trace_width": 20,
    "miri_background_width": 14,
    "miri_background_method": "median",
    "space_outlier_threshold": 15,
    "time_outlier_threshold": 10,
    "pca_components": 10,
    "remove_components": "None",
    "generate_lc": False,         # SOSS only
    "smoothing_scale": "None",
    "generate_order0_mask": False,
    "f277w": "None",
    "stage2_kwargs": {},
    # --- Stage 3 ---
    "extract_method": "box",
    "extract_width": "optimize",  # scan widths, minimize white-LC scatter
    "soss_specprofile": "None",
    "stage3_kwargs": {},
    # --- General ---
    "save_results": True,
    "force_redo": False,
    "baseline_ints": [50, -50],   # out-of-transit ints for difference images
    "centroids": "None",
    "do_plots": True,
}


def _exotedrf_python() -> str:
    """Path to the python interpreter of the pinned exoTEDRF environment."""
    python = os.environ.get("ASTER_EXOTEDRF_PYTHON", DEFAULT_EXOTEDRF_PYTHON)
    if not Path(python).exists():
        raise FileNotFoundError(
            f"exoTEDRF python not found at {python}. Install the pinned "
            "environment (conda create -n exotedrf python=3.11; "
            "pip install 'exotedrf[stage4]') or set ASTER_EXOTEDRF_PYTHON."
        )
    return python


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return f"'{value}'"


def write_dms_config(
    config_path: str | os.PathLike[str],
    *,
    input_dir: str,
    filter_detector: str,
    crds_cache_path: str,
    run_stages: list[int],
    st_teff: float | None = None,
    st_logg: float | None = None,
    st_met: float | None = None,
    planet_letter: str = "b",
    output_tag: str = "",
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Write an exoTEDRF run_DMS YAML config from the frozen Patchwork settings.

    ``overrides`` exists for debugging single targets only; production
    survey reductions must not use it (it breaks uniformity).
    """
    config = dict(PATCHWORK_G395H_CONFIG)
    config.update(
        {
            "crds_cache_path": crds_cache_path,
            "input_dir": input_dir,
            "filter_detector": filter_detector,
            "st_teff": st_teff if st_teff is not None else "None",
            "st_logg": st_logg if st_logg is not None else "None",
            "st_met": st_met if st_met is not None else "None",
            "planet_letter": planet_letter,
            "output_tag": output_tag,
            "run_stages": run_stages,
        }
    )
    if overrides:
        config.update(overrides)

    lines = [
        "# Auto-generated by aster_toolkit.data_reduction.exotedrf",
        f"# Patchwork uniform config version {PATCHWORK_CONFIG_VERSION}",
    ]
    for key, value in config.items():
        lines.append(f"{key} : {_yaml_scalar(value)}")

    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def inspect_uncal_directory(input_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Report segment completeness per detector for a directory of uncal files.

    JWST TSOs are split into segments (EXSEGNUM of EXSEGTOT) per detector.
    A reduction on an incomplete segment set silently produces a lightcurve
    with chunks of the transit missing — check this before reducing.
    """
    from astropy.io import fits  # lazy: keep module import light

    report: dict[str, Any] = {"input_dir": str(input_dir), "detectors": {}}
    for f in sorted(glob.glob(str(Path(input_dir) / "*.fits"))):
        try:
            h = fits.getheader(f)
        except OSError:
            continue
        det = str(h.get("DETECTOR", "?"))
        entry = report["detectors"].setdefault(
            det,
            {
                "segments_found": [],
                "segments_expected": None,
                "nints_total": None,
                "ngroups": None,
                "subarray": None,
                "readpatt": None,
                "target": None,
                "date_obs": None,
                "files": [],
            },
        )
        entry["segments_found"].append(int(h.get("EXSEGNUM", 0)))
        entry["segments_expected"] = int(h.get("EXSEGTOT", 0)) or entry["segments_expected"]
        entry["nints_total"] = h.get("NINTS")
        entry["ngroups"] = h.get("NGROUPS")
        entry["subarray"] = h.get("SUBARRAY")
        entry["readpatt"] = h.get("READPATT")
        entry["target"] = h.get("TARGPROP")
        entry["date_obs"] = h.get("DATE-OBS")
        entry["files"].append(os.path.basename(f))

    for det, entry in report["detectors"].items():
        found = sorted(entry["segments_found"])
        expected = entry["segments_expected"] or 0
        entry["missing_segments"] = [s for s in range(1, expected + 1) if s not in found]
        entry["complete"] = expected > 0 and not entry["missing_segments"]
    return report


def make_uncal_subset(
    uncal_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    n_integrations: int = 100,
) -> Path:
    """Write a copy of an uncal file trimmed to its first ``n_integrations``.

    For smoke-testing the pipeline in minutes instead of hours. Never use
    subset outputs for science.
    """
    from astropy.io import fits

    src = Path(uncal_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # do_not_scale_image_data keeps the raw uint16 + BZERO representation
    # intact so the subset is byte-compatible with the original uncal.
    with fits.open(src, memmap=False, do_not_scale_image_data=True) as hdul:
        intstart = int(hdul[0].header.get("INTSTART", 1))
        n = min(n_integrations, hdul["SCI"].data.shape[0])
        out = fits.HDUList([hdul[0].copy()])
        for hdu in hdul[1:]:
            name = hdu.name.upper()
            if name in ("SCI", "GROUPDQ", "ZEROFRAME") and hdu.data is not None:
                new = type(hdu)(data=hdu.data[:n], header=hdu.header)
            elif name == "INT_TIMES" and hdu.data is not None:
                new = type(hdu)(data=hdu.data[:n], header=hdu.header)
            elif name == "GROUP" and hdu.data is not None:
                mask = hdu.data["integration_number"] < intstart + n
                new = type(hdu)(data=hdu.data[mask], header=hdu.header)
            else:
                new = hdu.copy()
            # astropy silently drops BZERO/BSCALE when an HDU is built from
            # data + header; without BZERO=32768 the uint16 ramps read
            # ~32768 counts too low and the pipeline produces garbage.
            for key in ("BZERO", "BSCALE", "BLANK"):
                if key in hdu.header and key not in new.header:
                    new.header[key] = hdu.header[key]
            out.append(new)
        out[0].header["NINTS"] = n
        out[0].header["INTSTART"] = 1
        out[0].header["INTEND"] = n
        # Present the subset as a complete single-segment exposure.
        out[0].header["EXSEGNUM"] = 1
        out[0].header["EXSEGTOT"] = 1
        out.writeto(dst, overwrite=True)
    return dst


def _env_python_version(python: str) -> str:
    result = subprocess.run(
        [python, "-c",
         "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def find_stage3_products(root: str | os.PathLike[str]) -> list[str]:
    """Locate extracted stellar-spectra products under a reduction directory."""
    pattern = str(Path(root) / "**" / "*_spectra_fullres.fits")
    return sorted(glob.glob(pattern, recursive=True))


def run_reduction(
    input_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    detectors: list[str] | tuple[str, ...] | str = NIRSPEC_DETECTORS,
    stages: list[int] | None = None,
    st_teff: float | None = None,
    st_logg: float | None = None,
    st_met: float | None = None,
    planet_letter: str = "b",
    crds_cache_path: str | None = None,
    overrides: dict[str, Any] | None = None,
    log_callback=None,
) -> dict[str, Any]:
    """Run the uniform exoTEDRF Stage 1-3 reduction for one observation.

    One subprocess per detector (NRS1/NRS2 are reduced independently, as in
    every G395H analysis). Blocks until finished — Stage 1 on a full TSO is
    hours of compute; drive long runs in a background shell.

    Returns a manifest with per-detector status, log paths, and the
    ``*_spectra_fullres.fits`` products for lightcurve fitting.
    """
    if isinstance(detectors, str):
        detectors = [detectors]
    stages = stages or [1, 2, 3]
    crds = crds_cache_path or os.environ.get("CRDS_PATH", DEFAULT_CRDS_CACHE)
    Path(crds).mkdir(parents=True, exist_ok=True)

    python = _exotedrf_python()
    # run_DMS.py executes at import time (reads sys.argv), so locate it by
    # path instead of importing it.
    script_path = (
        Path(python).parent.parent
        / "lib" / f"python{_env_python_version(python)}" / "site-packages"
        / "exotedrf" / "run_DMS.py"
    )
    if not script_path.exists():
        raise FileNotFoundError(f"exoTEDRF run_DMS.py not found at {script_path}")

    manifest: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "config_version": PATCHWORK_CONFIG_VERSION,
        "stages": stages,
        "detectors": {},
    }

    for detector in detectors:
        det_dir = Path(output_dir) / detector.lower()
        det_dir.mkdir(parents=True, exist_ok=True)
        config_path = write_dms_config(
            det_dir / "run_DMS.yaml",
            input_dir=str(Path(input_dir).resolve()),
            filter_detector=detector.upper(),
            crds_cache_path=crds,
            run_stages=stages,
            st_teff=st_teff,
            st_logg=st_logg,
            st_met=st_met,
            planet_letter=planet_letter,
            overrides=overrides,
        )

        log_path = det_dir / "reduction.log"
        env = dict(os.environ)
        env["CRDS_PATH"] = crds
        env["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

        with log_path.open("w") as log:
            process = subprocess.Popen(
                [python, str(script_path), config_path.name],
                cwd=det_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in process.stdout:
                log.write(line)
                log.flush()
                if log_callback is not None:
                    log_callback(line)
            returncode = process.wait()

        products = find_stage3_products(det_dir)
        manifest["detectors"][detector.upper()] = {
            "returncode": returncode,
            "success": returncode == 0,
            "config": str(config_path),
            "log": str(log_path),
            "spectra_fullres": products,
        }

    manifest_path = Path(output_dir) / "reduction_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _format_inspection(report: dict[str, Any]) -> str:
    lines = [f"Uncal inventory for {report['input_dir']}:"]
    if not report["detectors"]:
        return lines[0] + "\n  No uncal FITS files found."
    for det, entry in sorted(report["detectors"].items()):
        status = "COMPLETE" if entry["complete"] else (
            f"INCOMPLETE (missing segments {entry['missing_segments']})"
            if entry["segments_expected"] else "UNKNOWN segment count"
        )
        lines.append(
            f"  {det}: {len(entry['files'])} file(s), segments "
            f"{sorted(entry['segments_found'])} of {entry['segments_expected']} -> {status}"
        )
        lines.append(
            f"    target={entry['target']} date={entry['date_obs']} "
            f"subarray={entry['subarray']} readpatt={entry['readpatt']} "
            f"nints={entry['nints_total']} ngroups={entry['ngroups']}"
        )
        for f in entry["files"]:
            lines.append(f"    - {f}")
    return "\n".join(lines)


def _format_reduction_manifest(manifest: dict[str, Any]) -> str:
    lines = [
        f"exoTEDRF reduction finished (Patchwork config v{manifest['config_version']}).",
        f"Input:  {manifest['input_dir']}",
        f"Output: {manifest['output_dir']}",
    ]
    for det, entry in manifest["detectors"].items():
        state = "OK" if entry["success"] else f"FAILED (exit {entry['returncode']})"
        lines.append(f"  {det}: {state}")
        lines.append(f"    log: {entry['log']}")
        for product in entry["spectra_fullres"]:
            lines.append(f"    spectra: {product}")
        if not entry["spectra_fullres"]:
            lines.append("    spectra: (none found — check log)")
    lines.append(
        "Pass the *_spectra_fullres.fits paths to the lightcurve fitting "
        "tools (FitNirspecG395hWhiteLight / FitNirspecG395hTransmissionSpectrum)."
    )
    return "\n".join(lines)


class InspectNirspecG395hUncalData(BaseTool):
    """
    Inventory raw JWST NIRSpec/G395H uncal files for one observation
    before reduction. (Header logic is generic JWST, but the Patchwork
    workflow and downstream reduction are G395H-specific.)

    Reports, per detector (NRS1/NRS2), which exposure segments are present
    versus expected (EXSEGNUM / EXSEGTOT), plus NINTS, NGROUPS, subarray,
    and readout pattern.

    ALWAYS run this before ``ReduceNirspecG395hTso``: a missing segment
    means part of the transit is absent and the reduction, while it will
    run, is not usable for science.

    Example
    -------
        InspectNirspecG395hUncalData(
            input_dir="mast/jwst_raw/GJ_1214_b/jw01185-o018_t009_nirspec_f290lp-g395h-s1600a1-sub2048"
        )
    """

    input_dir: str = RuntimeField(
        description="Directory containing *_uncal.fits files for ONE observation."
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        path = self.input_dir
        if not os.path.isabs(path):
            path = os.path.join(self.base_directory, path)
        return _format_inspection(inspect_uncal_directory(path))


class ReduceNirspecG395hTso(BaseTool):
    """
    Uniform Patchwork reduction of one JWST **NIRSpec/G395H BOTS**
    time-series observation: raw uncal ramps -> extracted time-resolved
    stellar spectra. This tool is G395H-specific — the frozen config
    assumes the SUB2048 subarray, NRSRAPID readout, and NRS1/NRS2
    detectors. Do not point it at SOSS/MIRI/NIRCam data.

    Runs exoTEDRF Stages 1-3 (superbias, group-level 1/f destriping, jump
    detection, ramp fitting, WCS/wavelength calibration, bad-pixel cleanup,
    optimized-width box extraction) in the pinned ``exotedrf`` conda
    environment. All reduction settings are frozen survey-wide
    (PATCHWORK_G395H_CONFIG) — do not override them for individual targets.

    Compute warning
    ---------------
    Stage 1 processes every group of every integration: a full TSO takes
    HOURS and the first-ever run also downloads several GB of CRDS
    reference files. For a quick functional test, first create a trimmed
    file with ``MakeUncalTestSubset`` and reduce that instead.

    Outputs
    -------
    Per detector: ``{output_dir}/{nrs1|nrs2}/pipeline_outputs_directory/``
    with Stage1/Stage2/Stage3 subdirectories, a reduction.log, the archived
    YAML config, and the key product ``*_spectra_fullres.fits``
    (extensions: Wave, Wave Err, Flux, Flux Err, Time) for lightcurve
    fitting with the juliet tools.

    Example
    -------
        ReduceNirspecG395hTso(
            input_dir="mast/jwst_raw/GJ_1214_b/jw01185-o018_.../",
            output_dir="reductions/GJ_1214_b/visit1",
            detectors=["NRS1", "NRS2"],
            st_teff=3250, st_logg=5.03, st_met=0.29,
        )
    """

    input_dir: str = RuntimeField(
        description="Directory with the *_uncal.fits segments of ONE observation."
    )
    output_dir: str = RuntimeField(
        description="Directory for reduction outputs (created if missing)."
    )
    detectors: list | str = RuntimeField(
        default=["NRS1", "NRS2"],
        description="Detectors to reduce: ['NRS1'], ['NRS2'], or both.",
    )
    stages: list | None = RuntimeField(
        default=None,
        description="Pipeline stages to run, default [1, 2, 3]. Rerunning with "
                    "[3] reuses existing Stage 2 outputs.",
    )
    st_teff: float | None = RuntimeField(
        default=None,
        description="Stellar Teff in K (wavelength-calibration refinement; optional).",
    )
    st_logg: float | None = RuntimeField(
        default=None,
        description="Stellar log g (optional).",
    )
    st_met: float | None = RuntimeField(
        default=None,
        description="Stellar [Fe/H] (optional).",
    )
    planet_letter: str = RuntimeField(
        default="b",
        description="Planet letter designation, e.g. 'b' or 'd'.",
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        input_dir = self.input_dir
        if not os.path.isabs(input_dir):
            input_dir = os.path.join(self.base_directory, input_dir)
        output_dir = self.output_dir
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(self.base_directory, output_dir)

        detectors = self.detectors
        if isinstance(detectors, str):
            import ast
            try:
                parsed = ast.literal_eval(detectors)
                detectors = parsed if isinstance(parsed, (list, tuple)) else [detectors]
            except (SyntaxError, ValueError):
                detectors = [detectors]

        manifest = run_reduction(
            input_dir,
            output_dir,
            detectors=list(detectors),
            stages=self.stages,
            st_teff=self.st_teff,
            st_logg=self.st_logg,
            st_met=self.st_met,
            planet_letter=self.planet_letter,
        )
        return _format_reduction_manifest(manifest)


class MakeUncalTestSubset(BaseTool):
    """
    Create a small test copy of an uncal file containing only the first N
    integrations. Reduces pipeline smoke-test time from hours to minutes.

    NEVER use subset reductions for science — the trimmed lightcurve does
    not cover the transit.

    Example
    -------
        MakeUncalTestSubset(
            uncal_path="mast/jwst_raw/GJ_1214_b/.../jw01185018001_04102_00001-seg001_nrs1_uncal.fits",
            output_path="reductions/GJ_1214_b/test_subset/jw01185018001_04102_00001-seg001_nrs1_uncal.fits",
            n_integrations=100,
        )
    """

    uncal_path: str = RuntimeField(description="Path to the source *_uncal.fits file.")
    output_path: str = RuntimeField(
        description="Destination path. Keep the original filename so exoTEDRF "
                    "can parse the segment naming."
    )
    n_integrations: int = RuntimeField(
        default=100,
        description="Number of integrations to keep from the start of the exposure.",
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        src = self.uncal_path
        dst = self.output_path
        if not os.path.isabs(src):
            src = os.path.join(self.base_directory, src)
        if not os.path.isabs(dst):
            dst = os.path.join(self.base_directory, dst)
        path = make_uncal_subset(src, dst, n_integrations=self.n_integrations)
        return (
            f"Wrote test subset ({self.n_integrations} integrations) to {path}. "
            "For pipeline testing only — not science-usable."
        )
