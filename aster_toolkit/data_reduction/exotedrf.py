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

Transit -> Phase curve. Scale up LATER!!

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
# Pinned reference context. Part of the survey definition: an unpinned CRDS
# resolves to "latest", so two targets reduced weeks apart could use
# different reference files. The pip build of run_DMS.py has no
# crds_context config key, so this is set via the environment instead.
CRDS_CONTEXT = "jwst_1322.pmap"
NIRSPEC_DETECTORS = ("NRS1", "NRS2")

# Frozen Patchwork reduction settings for NIRSpec/G395H BOTS (SUB2048,
# NRSRAPID). Changing any value here changes the survey definition — bump
# ``PATCHWORK_CONFIG_VERSION`` if you do, so old and new reductions are
# never silently mixed in one population analysis.
#
# v1.1: settings aligned with the validated GJ 9827 d run
# (workspace/patchwork/GJ_9827d.ipynb): 1/f via 'scale-achromatic',
# fixed 16-px box extraction (was 'optimize' — a per-target optimized
# width silently breaks survey uniformity), per-visit baseline_ints
# computed from the data instead of a fixed [50, -50].
PATCHWORK_CONFIG_VERSION = "1.1"
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
    "oof_method": "scale-achromatic",  # group-level 1/f treatment
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
    "extract_width": 16,          # fixed survey-wide (= nirspec_mask_width)
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
            "pip install 'exotedrf[stage4]') or set ASTER_EXOTEDRF_PYTHON.\n"
            "On an HPC cluster whose environments are virtualenvs (e.g. DRAC), "
            "point ASTER_EXOTEDRF_PYTHON at a wrapper that ACTIVATES the venv "
            "and module-loads its python/opencv -- see "
            "scripts/patchwork/exotedrf-python. Calling <venv>/bin/python "
            "directly leaves site-packages off sys.path on a compute node."
        )
    return python


DEFAULT_EXOTEDRF_REPO = "/Users/wasi/Desktop/exoTEDRF"


def exotedrf_repo() -> str | None:
    """Path to an exoTEDRF source checkout that should SHADOW the installed
    package, or None to use whatever is installed in the environment.

    Set ``ASTER_EXOTEDRF_REPO`` to run the survey from a source checkout
    (e.g. the ``optimizer`` branch, which is the only place ``optimize.py``
    exists). When set, it is prepended to PYTHONPATH for every exoTEDRF
    subprocess, so the reduction and the optimizer sweep run the SAME stage
    code — parameters tuned against one tree are not valid for another.

    Returns None rather than raising when unset: running against the
    installed release is a legitimate configuration.
    """
    repo = os.environ.get("ASTER_EXOTEDRF_REPO")
    if not repo:
        return None
    repo = os.path.expanduser(repo)
    if not (Path(repo) / "exotedrf" / "__init__.py").exists():
        raise FileNotFoundError(
            f"ASTER_EXOTEDRF_REPO={repo} does not contain an exotedrf/ "
            "package. Point it at the root of an exoTEDRF checkout."
        )
    return repo


def _subprocess_env(crds: str, crds_context: str | None = None) -> dict[str, str]:
    """Environment for an exoTEDRF subprocess: CRDS settings plus the
    source-checkout shadow, if one is configured."""
    env = dict(os.environ)
    env["CRDS_PATH"] = crds
    env["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
    if crds_context:
        env["CRDS_CONTEXT"] = crds_context
    repo = exotedrf_repo()
    if repo:
        env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    return env


# Checks run inside the exoTEDRF environment. Each entry hard-won from a
# real failure on DRAC Fir; see scripts/patchwork/patch_exotedrf_env.py.
_ENV_CHECK_SCRIPT = r"""
import json, os, sys
report = {"python": sys.executable,
          "version": "%d.%d.%d" % sys.version_info[:3],
          "checks": [], "packages": {}}

def check(name, fatal, fn):
    try:
        detail = fn()
        report["checks"].append({"name": name, "ok": True, "fatal": fatal,
                                 "detail": detail or ""})
    except Exception as exc:
        report["checks"].append({"name": name, "ok": False, "fatal": fatal,
                                 "detail": "%s: %s" % (type(exc).__name__, exc)})

def _numpy():
    import numpy
    report["packages"]["numpy"] = numpy.__version__
    major = int(numpy.__version__.split(".")[0])
    if major >= 2:
        raise RuntimeError(
            "numpy %s is too new; jwst 1.17.1 requires <2.0. "
            "Fix: pip install --no-index 'numpy<2'" % numpy.__version__)
    return numpy.__version__

def _jwst():
    import jwst
    report["packages"]["jwst"] = getattr(jwst, "__version__", "?")
    # jwst/stpipe/core.py imports this at module level; the +computecanada
    # wheel omits it.
    from jwst import __version_commit__  # noqa: F401
    return report["packages"]["jwst"]

def _crds_locate():
    import crds
    report["packages"]["crds"] = getattr(crds, "__version__", "?")
    # Fails when CRDS is newer than the stdatamodels jwst permits.
    from crds.jwst import locate  # noqa: F401
    return report["packages"]["crds"]

def _cv2():
    # stcal.jump imports cv2 at module level, so JumpStep cannot run without
    # it. On DRAC it comes from the opencv MODULE, never from pip.
    import cv2
    report["packages"]["cv2"] = cv2.__version__
    return cv2.__file__

def _stages():
    from exotedrf.stage1 import run_stage1  # noqa: F401
    from exotedrf.stage2 import run_stage2  # noqa: F401
    from exotedrf.stage3 import run_stage3  # noqa: F401
    import exotedrf, os as _os
    report["packages"]["exotedrf"] = _os.path.dirname(exotedrf.__file__)
    return report["packages"]["exotedrf"]

def _run_dms():
    import exotedrf, os as _os
    p = _os.path.join(_os.path.dirname(exotedrf.__file__), "run_DMS.py")
    if not _os.path.exists(p):
        raise FileNotFoundError(p)
    return p

def _badpix_resume():
    # exoTEDRF 2.3.1 raises UnboundLocalError when BadPixStep is fully
    # resumed (every segment already on disk). Non-fatal for a fresh run,
    # fatal for any rerun -- which is the normal mode under a walltime.
    import exotedrf, os as _os, re
    src = open(_os.path.join(_os.path.dirname(exotedrf.__file__),
                             "stage2.py")).read()
    body = src[src.index("class BadPixStep"):]
    body = body[:body.index("class ", 10)] if "class " in body[10:] else body
    if "to_flag = None" not in body.split("for i, segment")[0]:
        raise RuntimeError(
            "BadPixStep cannot be resumed (to_flag unbound). "
            "Fix: python scripts/patchwork/patch_exotedrf_env.py")
    return "resume guard present"

def _pandas():
    import pandas
    report["packages"]["pandas"] = pandas.__version__
    return pandas.__version__

check("numpy < 2",            True,  _numpy)
check("pandas",               True,  _pandas)
check("jwst.__version_commit__", True, _jwst)
check("crds.jwst.locate",     True,  _crds_locate)
check("cv2 (JumpStep)",       True,  _cv2)
check("exotedrf stages 1-3",  True,  _stages)
check("run_DMS.py present",   True,  _run_dms)
check("BadPixStep resumable", False, _badpix_resume)

print("PATCHWORK_ENV_JSON " + json.dumps(report))
"""


def verify_exotedrf_environment(python: str | None = None,
                                timeout: float = 300.0) -> dict[str, Any]:
    """Preflight the exoTEDRF environment before committing hours of compute.

    Runs every check inside the target interpreter in ONE subprocess and
    returns a structured report. Each check corresponds to a failure mode
    that has actually cost a job on DRAC Fir: numpy 2.x against jwst
    1.17.1, a CRDS newer than the permitted stdatamodels, missing cv2
    (JumpStep imports it at module level), the missing
    ``jwst.__version_commit__`` in the +computecanada wheel, and the
    exoTEDRF BadPixStep resume bug.
    """
    python = python or _exotedrf_python()
    repo = exotedrf_repo()
    env = dict(os.environ)
    if repo:
        env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([python, "-c", _ENV_CHECK_SCRIPT], env=env,
                            capture_output=True, text=True, timeout=timeout)

    report: dict[str, Any] = {"python": python, "repo": repo,
                              "checks": [], "packages": {}}
    for line in result.stdout.splitlines():
        if line.startswith("PATCHWORK_ENV_JSON "):
            report.update(json.loads(line[len("PATCHWORK_ENV_JSON "):]))
            break
    else:
        report["checks"] = [{
            "name": "interpreter runs", "ok": False, "fatal": True,
            "detail": (result.stderr.strip()[-800:] or
                       "no output from the environment check"),
        }]

    report["fatal_failures"] = [c for c in report["checks"]
                                if not c["ok"] and c["fatal"]]
    report["warnings"] = [c for c in report["checks"]
                          if not c["ok"] and not c["fatal"]]
    report["ok"] = not report["fatal_failures"]
    return report


def format_environment_report(report: dict[str, Any]) -> str:
    lines = [f"exoTEDRF environment: {report['python']}"]
    if report.get("version"):
        lines.append(f"  python {report['version']}")
    lines.append(f"  source       {report.get('repo') or '(installed release)'}")
    for name, value in sorted(report.get("packages", {}).items()):
        lines.append(f"  {name:<12} {value}")
    lines.append("")
    for c in report["checks"]:
        mark = "OK   " if c["ok"] else ("FAIL " if c["fatal"] else "WARN ")
        lines.append(f"  {mark}{c['name']}")
        if not c["ok"] or (c["detail"] and not c["ok"]):
            for detail_line in str(c["detail"]).splitlines():
                lines.append(f"         {detail_line}")
    lines.append("")
    lines.append("READY — safe to launch a reduction." if report["ok"]
                 else "NOT READY — fix the FAIL items before launching.")
    return "\n".join(lines)


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
    baseline_ints: list[int] | None = None,
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
    if baseline_ints is not None:
        config["baseline_ints"] = list(baseline_ints)
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


def count_integrations(input_dir: str | os.PathLike[str],
                       detector: str = "NRS1") -> int:
    """Total integrations across the uncal segments of one detector."""
    from astropy.io import fits

    n_seg, n_tot = 0, 0
    for f in sorted(glob.glob(str(Path(input_dir) / "**" / "*.fits"),
                              recursive=True)):
        name = os.path.basename(f)
        if f"_{detector.lower()}_" not in name or "uncal" not in name:
            continue
        try:
            h = fits.getheader(f)
        except OSError:
            continue
        if h.get("INTEND") and h.get("INTSTART"):
            n_seg += int(h["INTEND"]) - int(h["INTSTART"]) + 1
        n_tot = max(n_tot, int(h.get("NINTS", 0)))
    return n_seg or n_tot


def compute_baseline_ints(input_dir: str | os.PathLike[str],
                          *, fraction: float = 0.20,
                          min_side: int = 5) -> list[int]:
    """Per-visit out-of-transit baseline window, ``[n_pre, -n_post]``.

    The survey-uniform *rule* is fractional (20% of the integrations each
    side, floor of 5), not a fixed count — visits differ in length, so a
    fixed count is what would actually break uniformity.
    """
    n = count_integrations(input_dir, "NRS1") or count_integrations(input_dir, "NRS2")
    if n == 0:
        raise FileNotFoundError(f"No uncal files with NINTS found in {input_dir}")
    side = max(min_side, int(fraction * n))
    return [side, -side]


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


def exotedrf_script_path(python: str, script: str = "run_DMS.py",
                         extra_pythonpath: str | None = None) -> Path:
    """Locate a script inside the exoTEDRF package of a given interpreter.

    Asks the interpreter where ``exotedrf`` actually lives rather than
    reconstructing a site-packages path — venv, conda, ``pip --user`` and
    editable installs all lay that out differently, and a PYTHONPATH
    shadow (the optimizer checkout) moves it again. ``run_DMS.py`` reads
    ``sys.argv`` at import time, so it is located by path and never
    imported here.
    """
    env = dict(os.environ)
    if extra_pythonpath:
        env["PYTHONPATH"] = extra_pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [python, "-c",
         "import exotedrf, os; print(os.path.dirname(exotedrf.__file__))"],
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        raise FileNotFoundError(
            f"Could not import exotedrf with {python}:\n{result.stderr.strip()[-500:]}"
        )
    package_dir = Path(result.stdout.strip())
    path = package_dir / script
    if not path.exists():
        raise FileNotFoundError(
            f"{script} not found in the exotedrf package at {package_dir}."
        )
    return path


def exotedrf_version(python: str, extra_pythonpath: str | None = None) -> dict[str, str]:
    """Report which exoTEDRF an interpreter resolves, and from where.

    The reduction and the optimizer must agree on this — parameters tuned
    against one copy of the stage code are not valid for another.
    """
    env = dict(os.environ)
    if extra_pythonpath:
        env["PYTHONPATH"] = extra_pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import exotedrf, os, json;"
        "print(json.dumps({'path': os.path.dirname(exotedrf.__file__),"
        " 'version': getattr(exotedrf, '__version__', 'unknown')}))"
    )
    result = subprocess.run([python, "-c", code], capture_output=True,
                            text=True, env=env)
    if result.returncode != 0:
        raise FileNotFoundError(result.stderr.strip()[-500:])
    return json.loads(result.stdout.strip().splitlines()[-1])


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
    crds_context: str = CRDS_CONTEXT,
    baseline_ints: list[int] | None = None,
    overrides: dict[str, Any] | None = None,
    preflight: bool = True,
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

    if baseline_ints is None:
        baseline_ints = compute_baseline_ints(input_dir)

    python = _exotedrf_python()
    repo = exotedrf_repo()

    # Preflight before committing hours of compute. Every check here maps to
    # a failure that has already killed a job mid-reduction; finding them in
    # seconds beats finding them after Stage 1.
    if preflight:
        env_report = verify_exotedrf_environment(python)
        if not env_report["ok"]:
            raise RuntimeError(
                "exoTEDRF environment is not usable:\n\n"
                + format_environment_report(env_report)
            )
        for warning in env_report["warnings"]:
            print(f"[patchwork] WARNING: {warning['name']} — {warning['detail']}")
    else:
        env_report = None

    script_path = exotedrf_script_path(python, "run_DMS.py",
                                       extra_pythonpath=repo)

    manifest: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "config_version": PATCHWORK_CONFIG_VERSION,
        "stages": stages,
        "baseline_ints": list(baseline_ints),
        # Which exoTEDRF actually ran. Must match whatever the optimizer
        # was swept against, or tuned parameters do not transfer.
        "exotedrf": exotedrf_version(python, extra_pythonpath=repo),
        # Reference-file context actually in force. Unpinned CRDS resolves
        # to "latest", so targets reduced weeks apart would silently use
        # different reference files and no longer form a uniform survey.
        "crds_context": crds_context,
        "environment": env_report,
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
            baseline_ints=baseline_ints,
            overrides=overrides,
        )

        log_path = det_dir / "reduction.log"
        env = _subprocess_env(crds, crds_context)

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


class VerifyPatchworkEnvironment(BaseTool):
    """
    Preflight the pinned exoTEDRF environment before committing hours of
    compute to a reduction. Runs in seconds.

    Checks, inside the exoTEDRF interpreter itself: numpy is <2 (jwst
    1.17.1 breaks on 2.x), pandas present, ``jwst.__version_commit__``
    importable (absent from some vendor wheels), ``crds.jwst.locate``
    importable (fails when CRDS is newer than the stdatamodels jwst
    permits), cv2 importable (stcal.jump imports it at module level, so
    JumpStep cannot run without it), exoTEDRF Stages 1-3 importable,
    run_DMS.py present, and the exoTEDRF BadPixStep resume guard applied.

    Every one of these corresponds to a failure that has actually killed
    a Patchwork job mid-reduction. Run it after building or changing an
    environment, and after any pip install — pip silently reverts patched
    files.

    Fixes for anything reported are in
    ``scripts/patchwork/patch_exotedrf_env.py``.

    Example
    -------
        VerifyPatchworkEnvironment()
    """

    base_directory: str = StateField()

    def _run(self) -> str:
        try:
            report = verify_exotedrf_environment()
        except FileNotFoundError as exc:
            return f"Cannot locate the exoTEDRF interpreter.\n{exc}"
        return format_environment_report(report)


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
