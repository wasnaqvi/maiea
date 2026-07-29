"""Patchwork wrapper for the exoTEDRF coordinate-descent optimizer.

Drives ``exotedrf/optimize.py`` (the ``optimizer`` branch of
radicamc/exoTEDRF) as Patchwork's **Class-2 rule generator**: a frozen,
deterministic procedure that turns reduction knobs which would otherwise
be hand-picked (trace mask width, jump/outlier thresholds, BadPix
box/window, extraction aperture) into audited, replayable choices.

The optimizer greedily sweeps each parameter (coordinate descent, one
1D sweep per knob) and scores every trial with a point-to-point noise
cost on the Stage 3 flux cube: the 2nd time-difference of each
normalized channel, medianed over out-of-transit integrations and over
channels — i.e. the effective white-noise level that sets the error
bars of the final transmission spectrum.

Uniformity contract (PATCHWORK_OPTIMIZER_VERSION)
-------------------------------------------------
The greedy descent is order-dependent and the cost is definition-
dependent, so for the optimizer to be a Class-2 rule (and not a per-run
judgment call) everything that changes its answer is frozen here:

- the sweep grids (``G395H_SWEEP``, integers only),
- the sweep order (insertion order of ``G395H_SWEEP``),
- the cost definition (w1=0, w2=1, per-detector wave range, fractional
  baseline_ints rule),
- the pipeline step configuration around the sweep, and
- the CRDS context (pinned, so reference files cannot drift mid-survey).

The SHA-256 of the generated YAML (``omega_hash``) is recorded in every
output so a reduction can state exactly which rule produced it. Change
anything -> bump the version; never mix versions in one analysis.

Branch note
-----------
On the ``optimizer`` branch, NIRSpec group-level 1/f silently maps
'scale-achromatic' -> 'median' (stage1.py). The template therefore pins
``oof_method: median`` explicitly — the Ω record should say what
actually ran, not what was requested.

Environment
-----------
The optimizer is NOT in the pip release; it lives in the pulled
checkout. ``ASTER_EXOTEDRF_REPO`` must point at the repo root (e.g.
``/Users/wasi/Desktop/exoTEDRF`` locally, wherever it is cloned on
Fir). The wrapper prepends it to PYTHONPATH of the pinned-env python so
the checkout — optimizer plus its exact stage code — shadows any
installed release. Compute: ~50 trials with stage caching; days per
visit x detector on a full TSO. Calibrator targets only.
"""

from __future__ import annotations

import hashlib
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
        """Fallback that keeps the wrapper importable without Orchestral."""

    def RuntimeField(default=None, description=None):
        return default

    def StateField(default=None, description=None):
        return default

from .exotedrf import _exotedrf_python, exotedrf_version, DEFAULT_CRDS_CACHE
from .lightcurves import G395H_WAVE_RANGES

PATCHWORK_OPTIMIZER_VERSION = "1.0"

DEFAULT_EXOTEDRF_REPO = "/Users/wasi/Desktop/exoTEDRF"

# Pinned CRDS context: reference-file drift mid-survey would make the
# same Ω produce different spectra. Part of the frozen rule.
CRDS_CONTEXT = "jwst_1322.pmap"

# Cost-function weights: pure spectral point-to-point stability (the
# quantity that sets transmission-spectrum error bars).
COST_W1 = 0.0
COST_W2 = 1.0

# Frozen sweep grids. Insertion order IS the coordinate-descent order
# (Stage 1 -> 2 -> 3); integers only (the optimizer initializes each
# knob to int(median(grid))). Survey definition — do not edit per
# target; bump PATCHWORK_OPTIMIZER_VERSION if changed at all.
G395H_SWEEP: dict[str, list[int]] = {
    # Stage 1
    "nirspec_mask_width": [10, 12, 14, 16, 18, 20, 22],
    "time_jump_threshold": [5, 6, 7, 8, 9, 10],
    "time_window": [3, 5, 7, 9, 11],
    # Stage 2
    "space_outlier_threshold": [5, 7, 9, 11, 13, 15],
    "time_outlier_threshold": [5, 7, 9, 11, 13, 15],
    "box_size": [3, 4, 5, 6, 7, 8],
    "window_size": [3, 5, 7, 9, 11],
    # Stage 3
    "extract_width": [8, 10, 12, 14, 16, 18, 20, 22, 24],
}

# Non-swept optimizer knobs (SOSS/MIRI-only), fixed to their defaults so
# the YAML stays schema-complete for optimize.py's parser.
G395H_FIXED_SWEEPABLES: dict[str, int] = {
    "soss_inner_mask_width": 40,
    "soss_outer_mask_width": 70,
    "miri_trace_width": 20,
    "miri_background_width": 14,
}


def _yaml_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(v) for v in value) + "]"
    if isinstance(value, (int, float)):
        return str(value)
    return f"'{value}'"


def write_optimize_config(
    config_path: str | os.PathLike[str],
    *,
    input_dir: str,
    detector: str,
    baseline_ints: list[int],
    name_tag: str,
    crds_cache_path: str,
    st_teff: float | None = None,
    st_logg: float | None = None,
    st_met: float | None = None,
    planet_letter: str = "b",
) -> Path:
    """Write the frozen Patchwork ``run_optimize.yaml`` for one
    NIRSpec/G395H visit x detector.

    Everything except the per-visit inputs (paths, detector, baseline
    window, stellar parameters) is fixed by the module constants. The
    cost is evaluated only inside the detector's usable wavelength range
    so filter-edge noise cannot steer the sweep.
    """
    detector = detector.upper()
    wave_range = list(G395H_WAVE_RANGES[detector])

    entries: list[tuple[str, Any]] = [
        # --- key parameters ---
        ("crds_cache_path", crds_cache_path),
        ("crds_context", CRDS_CONTEXT),
        ("input_dir", input_dir),
        ("input_filetag", "uncal"),
        ("observing_mode", "NIRSpec/G395H"),
        ("filter_detector", detector),
        ("name_tag", name_tag),
        # --- Stage 1 steps (G395H) ---
        ("DQInitStep", "run"),
        ("EmiCorrStep", "skip"),          # MIRI only
        ("SaturationStep", "run"),
        ("ResetStep", "skip"),            # MIRI only
        ("SuperBiasStep", "run"),
        ("RefPixStep", "skip"),           # SOSS only
        ("DarkCurrentStep", "skip"),
        ("OneOverFStep_grp", "run"),
        ("LinearityStep", "run"),
        ("JumpStep", "run"),
        ("RampFitStep", "run"),
        ("GainScaleStep", "run"),
        ("hot_pixel_map", None),
        ("superbias_method", "crds"),
        ("soss_background_file", None),
        # Branch behavior: NIRSpec accepts 'median'/'slope' only, and
        # silently coerces 'scale-achromatic' to 'median' — pin what runs.
        ("oof_method", "median"),
        ("soss_timeseries", None),
        ("soss_timeseries_o2", None),
        ("outlier_maps", None),
        ("miri_drop_groups", 12),
        ("flag_up_ramp", False),
        ("jump_threshold", 15),
        ("flag_in_time", True),
        ("stage1_kwargs", "{}"),
        # --- Stage 2 steps (G395H) ---
        ("AssignWCSStep", "run"),
        ("Extract2DStep", "run"),         # NIRSpec only
        ("SourceTypeStep", "run"),
        ("WaveCorrStep", "run"),          # NIRSpec only
        ("FlatFieldStep", "skip"),        # SOSS/MIRI only
        ("BackgroundStep", "skip"),       # SOSS/MIRI only
        ("OneOverFStep_int", "skip"),
        ("BadPixStep", "run"),
        ("PCAReconstructStep", "skip"),   # crashes with sklearn >= 1.3
        ("TracingStep", "run"),
        ("miri_background_method", "median"),
        ("pca_components", 10),
        ("remove_components", None),
        ("generate_lc", False),           # SOSS only
        ("smoothing_scale", None),
        ("generate_order0_mask", False),
        ("f277w", None),
        ("stage2_kwargs", "{}"),
        # --- Stage 3 ---
        ("extract_method", "box"),
        ("soss_specprofile", None),
        ("st_teff", st_teff),
        ("st_logg", st_logg),
        ("st_met", st_met),
        ("planet_letter", planet_letter),
        ("stage3_kwargs", "{}"),
        # --- general ---
        ("output_tag", ""),
        ("baseline_ints", list(baseline_ints)),
        ("centroids", None),
        ("do_plots", False),
        # --- optimize block (frozen cost definition) ---
        ("wave_range", wave_range),
        ("wave_range_plot", wave_range),
        ("ylim_plot", None),
        ("w1", COST_W1),
        ("w2", COST_W2),
    ]

    lines = [
        "# Auto-generated by aster_toolkit.data_reduction.optimize",
        f"# Patchwork optimizer rule version {PATCHWORK_OPTIMIZER_VERSION}",
        "# Frozen Class-2 procedure: grids, order, and cost are survey-wide.",
    ]
    for key, value in entries:
        lines.append(f"{key} : {_yaml_value(value)}")

    lines.append("# ===== sweep grids (order = coordinate-descent order) =====")
    for param, grid in G395H_SWEEP.items():
        lines.append(f"optimize_{param} : True")
        lines.append(f"{param} : {_yaml_value(grid)}")
    for param, value in G395H_FIXED_SWEEPABLES.items():
        lines.append(f"optimize_{param} : False")
        lines.append(f"{param} : {_yaml_value(value)}")

    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def omega_hash(config_path: str | os.PathLike[str]) -> str:
    """SHA-256 of the generated YAML — the identity of the frozen rule
    (grids + order + cost + step config + CRDS context) for provenance."""
    return hashlib.sha256(Path(config_path).read_bytes()).hexdigest()[:16]


def _exotedrf_repo() -> str:
    repo = os.environ.get("ASTER_EXOTEDRF_REPO", DEFAULT_EXOTEDRF_REPO)
    if not (Path(repo) / "exotedrf" / "optimize.py").exists():
        raise FileNotFoundError(
            f"exoTEDRF optimizer checkout not found at {repo} "
            "(needs exotedrf/optimize.py — the pip release does not ship it). "
            "Clone the 'optimizer' branch and set ASTER_EXOTEDRF_REPO."
        )
    return repo


def run_optimization(
    input_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    detector: str,
    name_tag: str,
    baseline_ints: list[int] | None = None,
    st_teff: float | None = None,
    st_logg: float | None = None,
    st_met: float | None = None,
    planet_letter: str = "b",
    crds_cache_path: str | None = None,
    log_callback=None,
) -> dict[str, Any]:
    """Run the frozen optimizer sweep for one visit x detector.

    Blocks until finished — DAYS of compute on a full TSO (about 50
    Stage 1-3 trials, softened by the optimizer's stage caching). Run
    per-detector; NRS1 and NRS2 are independent sweeps.

    Returns a manifest with the omega hash, cost-table path, best
    parameter vector, and per-knob sensitivity summary.
    """
    from .exotedrf import compute_baseline_ints

    detector = detector.upper()
    out = Path(output_dir) / detector.lower()
    out.mkdir(parents=True, exist_ok=True)
    # optimize.py opens its cost/scatter logs here before any stage runs.
    (out / "pipeline_outputs_directory" / "Files").mkdir(
        parents=True, exist_ok=True
    )

    if baseline_ints is None:
        baseline_ints = compute_baseline_ints(input_dir)
    crds = crds_cache_path or os.environ.get("CRDS_PATH", DEFAULT_CRDS_CACHE)
    Path(crds).mkdir(parents=True, exist_ok=True)

    config_path = write_optimize_config(
        out / "run_optimize.yaml",
        input_dir=str(Path(input_dir).resolve()),
        detector=detector,
        baseline_ints=baseline_ints,
        name_tag=name_tag,
        crds_cache_path=crds,
        st_teff=st_teff,
        st_logg=st_logg,
        st_met=st_met,
        planet_letter=planet_letter,
    )
    rule_hash = omega_hash(config_path)

    python = _exotedrf_python()
    repo = _exotedrf_repo()
    env = dict(os.environ)
    # The checkout must shadow any installed exotedrf: the optimizer and
    # the stage code it sweeps have to come from the same tree.
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    env["CRDS_PATH"] = crds
    env["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
    env["CRDS_CONTEXT"] = CRDS_CONTEXT

    log_path = out / "optimizer.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [python, "-m", "exotedrf.optimize", "--config", config_path.name],
            cwd=out,
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

    cost_path = out / "pipeline_outputs_directory" / "Files" / f"Cost_{name_tag}.txt"
    manifest: dict[str, Any] = {
        "optimizer_version": PATCHWORK_OPTIMIZER_VERSION,
        "omega_hash": rule_hash,
        "detector": detector,
        "input_dir": str(input_dir),
        "output_dir": str(out),
        "config": str(config_path),
        "log": str(log_path),
        "baseline_ints": list(baseline_ints),
        # The stage code the sweep actually ran against (the checkout,
        # via the PYTHONPATH shadow). Compare against the 'exotedrf'
        # entry of any reduction that adopts these parameters — if the
        # paths differ, the tuning was done on different code.
        "exotedrf": exotedrf_version(python, extra_pythonpath=repo),
        "returncode": returncode,
        "success": returncode == 0 and cost_path.exists(),
        "cost_table": str(cost_path) if cost_path.exists() else None,
    }
    if cost_path.exists():
        table = parse_cost_table(cost_path)
        manifest["n_trials"] = len(table["rows"])
        manifest["best_params"] = table["best_params"]
        manifest["best_cost"] = table["best_cost"]
        manifest["sensitivity"] = summarize_sweep(table)

    manifest_path = out / "optimizer_manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


# -------------------- cost-table analysis --------------------


def parse_cost_table(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Parse ``Cost_<tag>.txt``: tab-delimited, header of swept parameter
    names + duration_s + cost, one row per trial. The optimizer's own
    final choice is the global-minimum row; reproduce that here."""
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    header = lines[0].split("\t")
    params = [h for h in header if h not in ("duration_s", "cost")]
    rows = []
    for ln in lines[1:]:
        vals = ln.split("\t")
        if len(vals) != len(header):
            continue
        row = dict(zip(header, vals))
        try:
            rows.append({
                "params": {p: float(row[p]) for p in params},
                "duration_s": float(row["duration_s"]),
                "cost": float(row["cost"]),
            })
        except (ValueError, KeyError):
            continue
    if not rows:
        raise ValueError(f"No parseable trial rows in {path}")
    best = min(rows, key=lambda r: r["cost"])
    return {
        "params": params,
        "rows": rows,
        "best_params": {k: int(v) if float(v).is_integer() else v
                        for k, v in best["params"].items()},
        "best_cost": best["cost"],
        "total_hours": sum(r["duration_s"] for r in rows) / 3600.0,
    }


def summarize_sweep(table: dict[str, Any]) -> dict[str, Any]:
    """Per-knob sensitivity: cost spread over each parameter's trials.

    'flat' knobs (spread small vs the global best) are candidates for a
    plain Class-1 freeze; 'sharp' knobs are the ones actually earning
    their Class-2 status. Advisory only — the adoption decision (and the
    depth-shift ablation gate) stays with the human.
    """
    best_cost = table["best_cost"]
    out: dict[str, Any] = {}
    for param in table["params"]:
        by_value: dict[float, list[float]] = {}
        for r in table["rows"]:
            by_value.setdefault(r["params"][param], []).append(r["cost"])
        # Trials where this knob was swept: min cost per tried value.
        per_value = {v: min(costs) for v, costs in by_value.items()}
        if len(per_value) < 2:
            continue
        spread = max(per_value.values()) - min(per_value.values())
        out[param] = {
            "best_value": min(per_value, key=per_value.get),
            "cost_spread": spread,
            "spread_pct_of_best": 100.0 * spread / best_cost if best_cost else None,
            "classification": "sharp" if best_cost and spread / best_cost > 0.02
                              else "flat",
        }
    return out


def best_params_to_overrides(best_params: dict[str, Any]) -> dict[str, Any]:
    """Map an optimizer best vector onto ``run_reduction`` overrides
    (the keys run_DMS understands), for reducing WITH the optimized rule.

    Using these overrides is a survey-definition event: record the omega
    hash next to the reduction, and do not mix optimized and default
    reductions in one analysis.
    """
    overrides: dict[str, Any] = {}
    direct = ("nirspec_mask_width", "time_jump_threshold",
              "space_outlier_threshold", "time_outlier_threshold",
              "extract_width")
    for key in direct:
        if key in best_params:
            overrides[key] = int(best_params[key])
    if "time_window" in best_params:
        overrides["stage1_kwargs"] = {
            "JumpStep": {"time_window": int(best_params["time_window"])}
        }
    badpix = {k: int(best_params[k]) for k in ("box_size", "window_size")
              if k in best_params}
    if badpix:
        overrides["stage2_kwargs"] = {"BadPixStep": badpix}
    return overrides


def format_optimizer_result(manifest: dict[str, Any]) -> str:
    lines = [
        f"Optimizer run ({manifest['detector']}) — Patchwork rule "
        f"v{manifest['optimizer_version']}, omega {manifest['omega_hash']}.",
    ]
    if not manifest["success"]:
        lines.append(f"  FAILED (exit {manifest['returncode']}) — see {manifest['log']}")
        return "\n".join(lines)
    lines.append(
        f"  {manifest['n_trials']} trials, best cost {manifest['best_cost']:.6g}."
    )
    lines.append("  Best parameters:")
    for k, v in manifest["best_params"].items():
        lines.append(f"    {k} = {v}")
    sens = manifest.get("sensitivity", {})
    sharp = [k for k, s in sens.items() if s["classification"] == "sharp"]
    flat = [k for k, s in sens.items() if s["classification"] == "flat"]
    if sharp:
        lines.append(f"  Sharp knobs (earn Class-2 status): {', '.join(sharp)}")
    if flat:
        lines.append(f"  Flat knobs (freezable as Class-1): {', '.join(flat)}")
    lines.append(f"  Cost table: {manifest['cost_table']}")
    lines.append(f"  Manifest:   {manifest['manifest_path']}")
    lines.append(
        "Gate before adoption: reduce with these values AND the frozen "
        "defaults, fit both, compare depths (must agree well within 1 sigma)."
    )
    return "\n".join(lines)


# -------------------- Fir sbatch --------------------

FIR_OPTIMIZER_SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --account={account}
#SBATCH --job-name=pw_opt_{slug}_{visit}_{det}
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --output={output_root}/{slug}/optimizer/slurm-%x-%j.out

# --- Patchwork optimizer (Class-2 rule) on DRAC Fir -----------------
# One visit x one detector per job; NRS1/NRS2 are independent sweeps.
module load StdEnv/2023 {python_module}
source {aster_env}/bin/activate

export PYTHONPATH={aster_repo}${{PYTHONPATH:+:$PYTHONPATH}}
export ASTER_EXOTEDRF_PYTHON={exotedrf_python}
export ASTER_EXOTEDRF_REPO={exotedrf_repo}   # git pull of the optimizer branch
export CRDS_PATH={crds_path}
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -m aster_toolkit.data_reduction.optimize \\
    --manifest {manifest_path} \\
    --output-root {output_root} \\
    --visit {visit} \\
    --detector {det}
"""


def write_fir_optimizer_slurm_scripts(
    manifest_path: str | os.PathLike[str],
    output_root: str,
    *,
    visits: list[str] | None = None,
    detectors: tuple[str, ...] = ("NRS1", "NRS2"),
    account: str = "def-ncowan",
    time: str = "3-00:00:00",
    cpus: int = 16,
    mem: str = "64G",
    aster_env: str = "~/aster/aster-env",
    aster_repo: str = "~/aster/maiea",
    python_module: str = "python/3.13",
    exotedrf_python: str = "~/bin/exotedrf-python",
    exotedrf_repo: str = "~/exoTEDRF",
    crds_path: str = "~/scratch/crds_cache",
    script_dir: str | os.PathLike[str] | None = None,
) -> list[Path]:
    """One sbatch per visit x detector (a sweep is ~50 reductions — never
    bundle them into one job). Does NOT submit."""
    from .survey import _slug, load_manifest

    manifest = load_manifest(manifest_path)
    slug = _slug(manifest["planet_name"])
    visits = visits or list(manifest["visits"])
    out_dir = Path(script_dir) if script_dir else Path(manifest_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for visit in visits:
        if visit not in manifest["visits"]:
            raise KeyError(f"Visit '{visit}' not in manifest {manifest_path}")
        for det in detectors:
            script = FIR_OPTIMIZER_SBATCH_TEMPLATE.format(
                account=account, slug=slug, visit=visit, det=det.lower(),
                time=time, cpus=cpus, mem=mem, output_root=output_root,
                aster_env=aster_env, aster_repo=aster_repo,
                python_module=python_module,
                exotedrf_python=exotedrf_python,
                exotedrf_repo=exotedrf_repo, crds_path=crds_path,
                manifest_path=os.path.abspath(str(manifest_path)),
            )
            path = out_dir / f"opt_{slug}_{visit}_{det.lower()}.sbatch"
            path.write_text(script)
            written.append(path)
    return written


# -------------------- CLI (what the sbatch runs) --------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .survey import _slug, load_manifest, stage_visit_uncals

    parser = argparse.ArgumentParser(
        prog="python -m aster_toolkit.data_reduction.optimize",
        description="Run the frozen Patchwork optimizer sweep for one "
                    "visit x detector of a target manifest.",
    )
    parser.add_argument("--manifest", required=True, help="Target manifest JSON.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--visit", required=True,
                        help="Visit key from the manifest, e.g. o010.")
    parser.add_argument("--detector", required=True, choices=["NRS1", "NRS2"])
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    if args.visit not in manifest["visits"]:
        parser.error(f"Visit '{args.visit}' not in manifest.")
    slug = _slug(manifest["planet_name"])
    root = Path(args.output_root) / slug

    input_dir = stage_visit_uncals(
        manifest["visits"][args.visit], root / "uncals" / args.visit
    )
    stellar = manifest.get("stellar") or {}
    result = run_optimization(
        input_dir,
        root / "optimizer" / args.visit,
        detector=args.detector,
        name_tag=f"{slug}_{args.visit}_{args.detector.lower()}",
        st_teff=stellar.get("st_teff"),
        st_logg=stellar.get("st_logg"),
        st_met=stellar.get("st_met"),
        planet_letter=manifest.get("planet_letter", "b"),
        log_callback=lambda line: print(line, end=""),
    )
    print(format_optimizer_result(result))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


# -------------------- orchestral tools --------------------


class OptimizeNirspecG395hReduction(BaseTool):
    """
    Run Patchwork's frozen Class-2 optimizer rule (the exoTEDRF
    coordinate-descent optimizer) on one NIRSpec/G395H visit x detector:
    sweep trace-mask width, jump/outlier thresholds, BadPix box/window,
    and extraction aperture, scoring each trial by out-of-transit
    point-to-point scatter of the Stage 3 spectra.

    The sweep grids, order, cost weights, wavelength windows, and CRDS
    context are all frozen module-wide; the YAML's SHA-256 (omega hash)
    is recorded so any result is replayable. This is a CALIBRATION
    procedure — run it on calibrator targets (GJ 9827 d), not per survey
    target, and gate adoption on the depth-shift ablation.

    Compute warning: ~50 Stage 1-3 trials — DAYS on a full TSO even with
    the optimizer's stage caching. On the laptop, only ever on a
    MakeUncalTestSubset trim; real runs belong on Fir via
    ``GenerateFirOptimizerJobs``. Requires the exoTEDRF 'optimizer'
    branch checkout (ASTER_EXOTEDRF_REPO).

    Outputs under {output_dir}/{nrs1|nrs2}/: run_optimize.yaml (frozen
    rule), optimizer.log, Cost/Scatter tables + plots, and
    optimizer_manifest.json (omega hash, best vector, per-knob
    flat/sharp sensitivity).

    Example
    -------
        OptimizeNirspecG395hReduction(
            input_dir="patchwork/results/GJ_9827_d/uncals/o010",
            output_dir="patchwork/results/GJ_9827_d/optimizer/o010",
            detector="NRS1",
            name_tag="GJ_9827_d_o010_nrs1",
            st_teff=4340, st_logg=4.66, st_met=-0.26, planet_letter="d",
        )
    """

    input_dir: str = RuntimeField(
        description="Directory with ONE visit's *_uncal.fits segments."
    )
    output_dir: str = RuntimeField(description="Directory for optimizer outputs.")
    detector: str = RuntimeField(description="'NRS1' or 'NRS2' (one per run).")
    name_tag: str = RuntimeField(
        description="Tag for the cost/scatter files, e.g. 'GJ_9827_d_o010_nrs1'."
    )
    st_teff: float | None = RuntimeField(default=None, description="Stellar Teff [K].")
    st_logg: float | None = RuntimeField(default=None, description="Stellar log g.")
    st_met: float | None = RuntimeField(default=None, description="Stellar [Fe/H].")
    planet_letter: str = RuntimeField(default="b", description="Planet letter.")
    base_directory: str = StateField()

    def _run(self) -> str:
        input_dir = self.input_dir
        if not os.path.isabs(input_dir):
            input_dir = os.path.join(self.base_directory, input_dir)
        output_dir = self.output_dir
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(self.base_directory, output_dir)
        manifest = run_optimization(
            input_dir, output_dir,
            detector=self.detector,
            name_tag=self.name_tag,
            st_teff=self.st_teff,
            st_logg=self.st_logg,
            st_met=self.st_met,
            planet_letter=self.planet_letter,
        )
        return format_optimizer_result(manifest)


class SummarizeG395hOptimization(BaseTool):
    """
    Analyze a completed optimizer run from its ``Cost_<tag>.txt`` table:
    best parameter vector (global-minimum row, the optimizer's own
    selection rule), per-knob flat/sharp sensitivity classification, and
    the ready-to-use ``run_reduction`` overrides JSON for reducing with
    the optimized values.

    Use after a Fir optimizer job finishes, or on a partial table while
    one is still running (every completed trial is a row). 'Flat' knobs
    barely move the cost — freeze them as plain survey constants.
    'Sharp' knobs are the ones the optimizer is actually earning its
    keep on.

    Adoption gate (do NOT skip): reduce a calibrator with the overrides
    AND with the frozen defaults, fit both with the juliet tools, and
    compare transmission depths — scatter-minimizing cost is blind to
    depth bias, so agreement well within 1 sigma is required before the
    optimized values touch the survey config.

    Example
    -------
        SummarizeG395hOptimization(
            cost_table="patchwork/results/GJ_9827_d/optimizer/o010/nrs1/"
                       "pipeline_outputs_directory/Files/Cost_GJ_9827_d_o010_nrs1.txt"
        )
    """

    cost_table: str = RuntimeField(description="Path to a Cost_<tag>.txt file.")
    base_directory: str = StateField()

    def _run(self) -> str:
        path = self.cost_table
        if not os.path.isabs(path):
            path = os.path.join(self.base_directory, path)
        table = parse_cost_table(path)
        sens = summarize_sweep(table)
        overrides = best_params_to_overrides(table["best_params"])

        lines = [
            f"Cost table: {len(table['rows'])} trials, "
            f"{table['total_hours']:.1f} h of compute, "
            f"best cost {table['best_cost']:.6g}.",
            "Best parameters (global-minimum row):",
        ]
        for k, v in table["best_params"].items():
            s = sens.get(k, {})
            lines.append(
                f"  {k} = {v}"
                + (f"   [{s['classification']}, spread "
                   f"{s['spread_pct_of_best']:.1f}% of best cost]" if s else "")
            )
        lines.append("run_reduction overrides for these values:")
        lines.append(json.dumps(overrides, indent=2))
        lines.append(
            "Adoption gate: reduce with overrides AND defaults, fit both, "
            "compare depths before touching the survey config."
        )
        return "\n".join(lines)


class GenerateFirOptimizerJobs(BaseTool):
    """
    Write the SLURM sbatch scripts that run the frozen Patchwork
    optimizer on DRAC Fir — one job per visit x detector of a target
    manifest (a sweep is ~50 reductions; visit x detector jobs must
    never be bundled). Does NOT submit anything.

    Each script activates the ASTER env, points ASTER_EXOTEDRF_PYTHON at
    the pinned exoTEDRF env and ASTER_EXOTEDRF_REPO at the pulled
    'optimizer'-branch checkout (the pip release does not ship the
    optimizer), and invokes ``python -m
    aster_toolkit.data_reduction.optimize`` for that visit/detector.
    Default walltime is 3 days; deep targets (K2-18-length visits) need
    more.

    Example
    -------
        GenerateFirOptimizerJobs(
            manifest_path="patchwork/manifests/GJ_9827_d.json",
            output_root="/scratch/wasi/patchwork",
            exotedrf_repo="~/exoTEDRF",
        )
    """

    manifest_path: str = RuntimeField(description="Path to the target manifest JSON.")
    output_root: str = RuntimeField(
        description="Output root ON FIR (e.g. /scratch/<user>/patchwork)."
    )
    visits: str | None = RuntimeField(
        default=None,
        description="Comma-separated visit keys; default all in the manifest.",
    )
    detectors: str = RuntimeField(
        default="NRS1,NRS2", description="Comma-separated detectors."
    )
    account: str = RuntimeField(default="def-ncowan", description="DRAC allocation.")
    time: str = RuntimeField(default="3-00:00:00", description="SLURM walltime per job.")
    cpus: int = RuntimeField(default=16, description="CPUs per task.")
    mem: str = RuntimeField(default="64G", description="Memory request.")
    aster_env: str = RuntimeField(
        default="~/aster/aster-env", description="Fir path to the ASTER venv."
    )
    aster_repo: str = RuntimeField(
        default="~/aster/maiea",
        description="Fir path to the repo containing aster_toolkit/ "
                    "(exported as PYTHONPATH).",
    )
    python_module: str = RuntimeField(
        default="python/3.13",
        description="Module providing the python that aster_env was built from.",
    )
    exotedrf_python: str = RuntimeField(
        default="~/bin/exotedrf-python",
        description="Fir path to the exoTEDRF interpreter (prefer a wrapper "
                    "that module-loads its own python).",
    )
    exotedrf_repo: str = RuntimeField(
        default="~/exoTEDRF",
        description="Fir path to the pulled exoTEDRF optimizer-branch checkout.",
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        manifest_path = self.manifest_path
        if not os.path.isabs(manifest_path):
            manifest_path = os.path.join(self.base_directory, manifest_path)
        visits = ([v.strip() for v in self.visits.split(",") if v.strip()]
                  if self.visits else None)
        written = write_fir_optimizer_slurm_scripts(
            manifest_path,
            self.output_root,
            visits=visits,
            detectors=tuple(d.strip().upper() for d in self.detectors.split(",")),
            account=self.account,
            time=self.time,
            cpus=self.cpus,
            mem=self.mem,
            aster_env=self.aster_env,
            aster_repo=self.aster_repo,
            python_module=self.python_module,
            exotedrf_python=self.exotedrf_python,
            exotedrf_repo=self.exotedrf_repo,
        )
        lines = [f"Wrote {len(written)} optimizer sbatch script(s):"]
        lines += [f"  {p}" for p in written]
        lines.append(
            "Copy to Fir with the manifest, verify aster_env / "
            "exotedrf_python / exotedrf_repo paths, then sbatch each. "
            "After completion: SummarizeG395hOptimization on the cost tables."
        )
        return "\n".join(lines)
