"""Patchwork uniform transit lightcurve fitting built on juliet.

Stages 5-6 of the Patchwork pipeline. Consumes exoTEDRF
``*_spectra_fullres.fits`` products (via ``lightcurves.py`` Stage 4) and
produces, per visit x detector:

1. White-light transit fit -> orbital parameters (t0, a/Rs, b, Rp/Rs).
2. Spectroscopic per-channel fits with the orbit frozen to the
   white-light posterior -> transmission spectrum (ppm CSV + TauREx dat).

and per detector: an inverse-variance combination of all visits, plus the
survey health metrics (NRS1-NRS2 offset near the detector gap, white and
spectroscopic residual rms).

Uniformity contract (PATCHWORK_FIT_VERSION)
-------------------------------------------
Fitting choices are frozen module-wide. v1.1 (the "patched" fit of the
Patchwork plan, superseding the as-is v1.0):

- Out-of-transit baseline normalization (ephemeris + literature duration).
- ExoTiC-LD Gaussian priors on Kipping (q1, q2) from the stellar
  parameters (truncated normal, sigma 0.1, mode JWST_NIRSpec_G395H);
  uniform fallback when ExoTiC-LD or its grids are unavailable — the
  fallback is recorded in the fit summary.
- Linear decorrelation regressors: time slope + trace x/y + FWHM (from
  Stage 2 calints, standardized), via juliet ``linear_regressors_lc``.
- Tilt-event step terms: step discontinuities detected on the white
  lightcurve (out-of-transit only) get one fitted 0/1 step regressor
  each, shared by the spectroscopic fits.
- dynesty nested sampling; ecc fixed 0, omega 90.

Set ``decorrelate=False, tilt_correction=False, ld_priors='uniform'`` to
reproduce an as-is v1.0-style fit for comparison.

NOTE: this module is named ``juliet.py`` but the fitting engine is the
``juliet`` package. ``_import_juliet()`` strips this file's directory
from ``sys.path`` during import so site-packages always wins.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .lightcurves import (
    DEFAULT_RESOLUTION,
    G395H_WAVE_RANGES,
    build_lightcurves,
    build_regressor_matrix,
    detect_tilt_events,
    find_stage2_calints,
    load_stage3_spectra,
    propagate_t0,
    trace_diagnostics,
)

try:
    from orchestral.tools.base.tool import BaseTool
    from orchestral.tools.base.field_utils import RuntimeField, StateField
except ModuleNotFoundError:
    class BaseTool:
        """Fallback that keeps plain fitting wrappers importable without Orchestral."""

    def RuntimeField(default=None, description=None):
        return default

    def StateField(default=None, description=None):
        return default


def _import_juliet():
    """Import the installed ``juliet`` package, never this module."""
    import importlib
    import sys

    cached = sys.modules.get("juliet")
    if cached is not None:
        if str(getattr(cached, "__file__", "")) == str(__file__):
            del sys.modules["juliet"]
        else:
            return cached

    this_dir = os.path.abspath(os.path.dirname(__file__))
    saved_path = list(sys.path)
    sys.path = [
        p for p in sys.path if os.path.abspath(p or os.getcwd()) != this_dir
    ]
    try:
        return importlib.import_module("juliet")
    finally:
        sys.path = saved_path


# Frozen survey-wide fitting settings. Bump the version if any change.
PATCHWORK_FIT_VERSION = "1.1"
SAMPLER = "dynesty"
N_LIVE_WHITE = 500
N_LIVE_SPEC = 300
LD_LAW = "quadratic"          # Kipping q1/q2 parameterization in juliet
LD_PRIOR_SIGMA = 0.1          # Gaussian width on ExoTiC-LD (q1, q2)
LD_MODE = "JWST_NIRSpec_G395H"
LD_MODEL = "mps1"
THETA_BOUNDS = [-0.1, 0.1]    # uniform prior on every linear regressor coeff

# Back-compat alias (pre-Stage4 name).
load_extracted_spectra = load_stage3_spectra


# -------------------- archive priors --------------------


def fetch_transit_priors(planet_name: str) -> dict[str, Any]:
    """Literature ephemeris/orbit/star from the NASA Exoplanet Archive
    (pscomppars). Seeds the white-light priors and the OOT mask."""
    from aster_toolkit.data_acquisition.mast import archive_tap_query

    rows = archive_tap_query(
        [f"pl_name = '{planet_name}'"],
        columns=[
            "pl_name", "pl_orbper", "pl_orbpererr1", "pl_tranmid",
            "pl_tranmiderr1", "pl_ratdor", "pl_ratdorerr1", "pl_ratror",
            "pl_orbincl", "pl_imppar", "pl_impparerr1", "pl_orbeccen",
            "pl_trandur", "st_rad", "st_teff", "st_logg", "st_met",
        ],
    )
    if not rows:
        raise ValueError(
            f"No NASA Exoplanet Archive entry for '{planet_name}'. "
            "Check the name format (e.g. 'GJ 1214 b')."
        )
    row = rows[0]

    def _f(key: str) -> float | None:
        value = row.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return {
        "planet_name": planet_name,
        "period": _f("pl_orbper"),
        "period_err": _f("pl_orbpererr1"),
        "t0": _f("pl_tranmid"),
        "t0_err": _f("pl_tranmiderr1"),
        "a_rs": _f("pl_ratdor"),
        "a_rs_err": _f("pl_ratdorerr1"),
        "rp_rs": _f("pl_ratror"),
        "inclination": _f("pl_orbincl"),
        "impact_param": _f("pl_imppar"),
        "impact_param_err": _f("pl_impparerr1"),
        "eccentricity": _f("pl_orbeccen"),
        "duration_hr": _f("pl_trandur"),
        "st_rad": _f("st_rad"),
        "st_teff": _f("st_teff"),
        "st_logg": _f("st_logg"),
        "st_met": _f("st_met"),
    }


# Back-compat alias.
def predict_mid_transit(t0_ref: float, period: float, times: np.ndarray) -> float:
    return propagate_t0(t0_ref, period, times)


# -------------------- ExoTiC-LD priors --------------------


def _ld_data_path() -> str:
    return os.environ.get(
        "ASTER_EXOTIC_LD_DATA", str(Path.home() / "exotic_ld_data")
    )


def compute_ld_coeffs(
    st_teff: float, st_logg: float, st_met: float,
    wave_min_um: float, wave_max_um: float,
) -> dict[str, float] | None:
    """Quadratic limb-darkening (u1, u2) -> Kipping (q1, q2) from
    ExoTiC-LD stellar grids over one wavelength range.

    Tries an in-process import first, then falls back to running inside
    the pinned exoTEDRF environment (where exotic_ld is installed).
    Returns None (-> uniform LD priors) if both fail; callers must record
    that in the fit summary.
    """
    args = (st_teff, st_logg, st_met, wave_min_um, wave_max_um)
    if any(a is None for a in args):
        return None

    script = (
        "import json, sys\n"
        "from exotic_ld import StellarLimbDarkening\n"
        "teff, logg, met, wmin, wmax, path = (\n"
        "    float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]),\n"
        "    float(sys.argv[4]), float(sys.argv[5]), sys.argv[6])\n"
        f"sld = StellarLimbDarkening(M_H=met, Teff=teff, logg=logg,\n"
        f"                           ld_model='{LD_MODEL}', ld_data_path=path)\n"
        "u1, u2 = sld.compute_quadratic_ld_coeffs(\n"
        "    wavelength_range=[wmin * 1e4, wmax * 1e4],\n"
        f"    mode='{LD_MODE}')\n"
        "print(json.dumps([u1, u2]))\n"
    )

    def _in_process():
        from exotic_ld import StellarLimbDarkening

        sld = StellarLimbDarkening(
            M_H=st_met, Teff=st_teff, logg=st_logg,
            ld_model=LD_MODEL, ld_data_path=_ld_data_path(),
        )
        return sld.compute_quadratic_ld_coeffs(
            wavelength_range=[wave_min_um * 1e4, wave_max_um * 1e4],
            mode=LD_MODE,
        )

    def _subprocess():
        from .exotedrf import _exotedrf_python

        result = subprocess.run(
            [_exotedrf_python(), "-c", script,
             str(st_teff), str(st_logg), str(st_met),
             str(wave_min_um), str(wave_max_um), _ld_data_path()],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-500:])
        return json.loads(result.stdout.strip().splitlines()[-1])

    u1 = u2 = None
    for attempt in (_in_process, _subprocess):
        try:
            u1, u2 = attempt()
            break
        except Exception:
            continue
    if u1 is None:
        return None

    u1, u2 = float(u1), float(u2)
    s = u1 + u2
    q1 = s**2
    q2 = u1 / (2 * s) if s != 0 else 0.5
    return {"u1": u1, "u2": u2,
            "q1": float(np.clip(q1, 0, 1)), "q2": float(np.clip(q2, 0, 1))}


def _ld_prior_entries(instrument: str, ld: dict[str, float] | None) -> dict[str, dict]:
    """Truncated-normal (q1, q2) priors when ExoTiC-LD coefficients are
    available, uniform otherwise."""
    if ld is None:
        return {
            f"q1_{instrument}": {"distribution": "uniform", "hyperparameters": [0.0, 1.0]},
            f"q2_{instrument}": {"distribution": "uniform", "hyperparameters": [0.0, 1.0]},
        }
    return {
        f"q1_{instrument}": {
            "distribution": "truncatednormal",
            "hyperparameters": [ld["q1"], LD_PRIOR_SIGMA, 0.0, 1.0],
        },
        f"q2_{instrument}": {
            "distribution": "truncatednormal",
            "hyperparameters": [ld["q2"], LD_PRIOR_SIGMA, 0.0, 1.0],
        },
    }


# -------------------- priors --------------------


def _common_instrument_priors(instrument: str, n_regressors: int) -> dict[str, dict]:
    p = {
        f"mdilution_{instrument}": {"distribution": "fixed", "hyperparameters": 1.0},
        f"mflux_{instrument}": {"distribution": "normal", "hyperparameters": [0.0, 0.1]},
        f"sigma_w_{instrument}": {"distribution": "loguniform",
                                  "hyperparameters": [0.1, 10000.0]},
    }
    for i in range(n_regressors):
        p[f"theta{i}_{instrument}"] = {
            "distribution": "uniform", "hyperparameters": list(THETA_BOUNDS),
        }
    return p


def _white_priors(
    priors: dict[str, Any],
    t0_obs: float,
    instrument: str,
    n_regressors: int,
    ld: dict[str, float] | None,
) -> dict[str, dict[str, Any]]:
    period = priors["period"]
    if period is None:
        raise ValueError("No orbital period available from the archive.")

    p: dict[str, dict[str, Any]] = {
        "P_p1": {
            "distribution": "normal",
            "hyperparameters": [period, max(priors["period_err"] or 0, 1e-5) * 3],
        },
        # t0 propagated to this visit; 0.01 d Gaussian (validated run).
        "t0_p1": {"distribution": "normal", "hyperparameters": [t0_obs, 0.01]},
        "p_p1": {"distribution": "uniform", "hyperparameters": [0.0, 0.3]},
        "b_p1": {"distribution": "uniform", "hyperparameters": [0.0, 1.0]},
        "ecc_p1": {"distribution": "fixed", "hyperparameters": 0.0},
        "omega_p1": {"distribution": "fixed", "hyperparameters": 90.0},
    }
    a_rs = priors["a_rs"]
    if a_rs is not None:
        err = priors["a_rs_err"] or 0.1 * a_rs
        p["a_p1"] = {"distribution": "normal", "hyperparameters": [a_rs, 3 * err]}
    else:
        p["a_p1"] = {"distribution": "uniform", "hyperparameters": [1.0, 300.0]}
    p.update(_ld_prior_entries(instrument, ld))
    p.update(_common_instrument_priors(instrument, n_regressors))
    return p


# -------------------- fitting --------------------


def fit_white_lightcurve(
    lc: dict[str, Any],
    priors: dict[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    instrument: str = "NRS1",
    regressors: np.ndarray | None = None,
    regressor_names: list[str] | None = None,
    ld: dict[str, float] | None = None,
    model_type: str = "transit",
    n_live: int = N_LIVE_WHITE,
) -> dict[str, Any]:
    """Fit the white-light transit of one visit x detector with juliet.

    ``lc`` is a Stage 4 product from ``build_lightcurves``. Returns the
    posterior summary (medians + 1-sigma), residual rms, and metadata
    (regressors used, LD source, tilt events) — the orbit is frozen to
    this in the spectroscopic fits.
    """
    if model_type != "transit":
        raise NotImplementedError(
            "Only model_type='transit' is implemented. Phase-curve support "
            "is a planned extension."
        )
    juliet = _import_juliet()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    times = lc["time"]
    if regressors is None:
        regressors, regressor_names = build_regressor_matrix(times)
    n_reg = regressors.shape[1]

    prior_dict = _white_priors(priors, lc["t0_obs"], instrument, n_reg, ld)
    dataset = juliet.load(
        priors=prior_dict,
        t_lc={instrument: times},
        y_lc={instrument: lc["wl_flux"]},
        yerr_lc={instrument: lc["wl_err"]},
        linear_regressors_lc={instrument: regressors},
        out_folder=str(out),
    )
    results = dataset.fit(sampler=SAMPLER, n_live_points=n_live, verbose=False)

    posterior = results.posteriors["posterior_samples"]
    summary: dict[str, Any] = {
        "instrument": instrument,
        "fit_version": PATCHWORK_FIT_VERSION,
        "regressor_names": list(regressor_names or []),
        "ld_source": "exotic-ld" if ld is not None else "uniform",
        "ld_coeffs": ld,
        "t0_obs": lc["t0_obs"],
    }
    keys = ["P_p1", "t0_p1", "p_p1", "b_p1", "a_p1",
            f"q1_{instrument}", f"q2_{instrument}", f"sigma_w_{instrument}"]
    keys += [f"theta{i}_{instrument}" for i in range(n_reg)]
    for key in keys:
        if key in posterior:
            samples = posterior[key]
            med, lo, hi = np.percentile(samples, [50, 16, 84])
            summary[key] = {"median": float(med),
                            "minus": float(med - lo), "plus": float(hi - med)}
    d = np.percentile(posterior["p_p1"] ** 2, [50, 16, 84]) * 1e6
    summary["depth_ppm"] = {"median": float(d[0]),
                            "minus": float(d[0] - d[1]), "plus": float(d[2] - d[0])}

    model = results.lc.evaluate(instrument)
    summary["residual_rms_ppm"] = float(np.nanstd(lc["wl_flux"] - model) * 1e6)

    _plot_white_fit(results, lc, instrument, out)

    with (out / "white_fit_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    summary["output_dir"] = str(out)
    return summary


_PLOT_STYLE = {
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False,
}


def _savefig(fig, out: Path, stem: str) -> None:
    # Patchwork figure rule: PDF + SVG, nothing else.
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.svg")


def _plot_white_fit(results, lc, instrument: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(_PLOT_STYLE):
        times = lc["time"]
        t_hr = (times - lc["t0_obs"]) * 24
        flux = lc["wl_flux"]
        err = lc["wl_err"]
        model = results.lc.evaluate(instrument)
        residual = flux - model
        sig = np.std(residual)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(9, 6), sharex=True, height_ratios=[3, 1]
        )
        ax1.errorbar(t_hr, flux, yerr=err, fmt="o", ms=3, color="#1a2f6b",
                     elinewidth=0.7, alpha=0.8, label="white lightcurve")
        ax1.plot(t_hr, model, color="crimson", lw=1.5, label="juliet best fit")
        core = np.percentile(flux, [0.5, 99.5])
        pad = 5 * sig
        ax1.set_ylim(min(core[0], model.min()) - pad, max(core[1], model.max()) + pad)
        ax1.set_ylabel("Relative flux")
        ax1.set_title(
            f"{instrument} white lightcurve — residual rms {sig * 1e6:.0f} ppm"
        )
        ax1.legend(fontsize=9)
        ax2.plot(t_hr, residual * 1e6, "o", ms=2.5, color="#1a2f6b", alpha=0.7)
        ax2.axhline(0, color="crimson", lw=1)
        ax2.set_ylim(-5 * sig * 1e6, 5 * sig * 1e6)
        ax2.set_xlabel("Time from mid-transit (h)")
        ax2.set_ylabel("Residual (ppm)")
        fig.tight_layout()
        _savefig(fig, out, "white_lightcurve_fit")
        plt.close(fig)


def fit_transmission_spectrum(
    lc: dict[str, Any],
    white_summary: dict[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    instrument: str = "NRS1",
    regressors: np.ndarray | None = None,
    stellar: dict[str, float] | None = None,
    n_live: int = N_LIVE_SPEC,
) -> dict[str, Any]:
    """Fit every spectroscopic channel with the orbit (P, t0, b, a/Rs)
    frozen to the white-light posterior medians. Free per channel: Rp/Rs,
    (q1, q2) with per-channel ExoTiC-LD priors when ``stellar`` is given,
    mflux, jitter, and the linear regressor coefficients.

    Writes ``transmission_spectrum.csv`` (ppm, with per-channel residual
    rms) and ``transmission_spectrum.dat`` (fractional depth, TauREx
    input format).
    """
    juliet = _import_juliet()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    times = lc["time"]
    if regressors is None:
        regressors, _ = build_regressor_matrix(times)
    n_reg = regressors.shape[1]

    fixed = {
        "P_p1": white_summary["P_p1"]["median"],
        "t0_p1": white_summary["t0_p1"]["median"],
        "b_p1": white_summary["b_p1"]["median"],
        "a_p1": white_summary["a_p1"]["median"],
    }

    use_ld = stellar is not None and white_summary.get("ld_source") == "exotic-ld"

    rows = []
    for i in range(lc["wave"].size):
        f, e = lc["sp_flux"][:, i], lc["sp_err"][:, i]
        good = np.isfinite(f) & np.isfinite(e)
        if good.sum() < 50:
            continue

        ld = None
        if use_ld:
            w0 = lc["wave"][i] - lc["wave_err"][i]
            w1 = lc["wave"][i] + lc["wave_err"][i]
            ld = compute_ld_coeffs(stellar["st_teff"], stellar["st_logg"],
                                   stellar["st_met"], w0, w1)

        prior_dict: dict[str, dict[str, Any]] = {
            key: {"distribution": "fixed", "hyperparameters": value}
            for key, value in fixed.items()
        }
        prior_dict.update({
            "ecc_p1": {"distribution": "fixed", "hyperparameters": 0.0},
            "omega_p1": {"distribution": "fixed", "hyperparameters": 90.0},
            "p_p1": {"distribution": "uniform", "hyperparameters": [0.0, 0.3]},
        })
        prior_dict.update(_ld_prior_entries(instrument, ld))
        prior_dict.update(_common_instrument_priors(instrument, n_reg))

        dataset = juliet.load(
            priors=prior_dict,
            t_lc={instrument: times[good]},
            y_lc={instrument: f[good]},
            yerr_lc={instrument: e[good]},
            linear_regressors_lc={instrument: regressors[good]},
            out_folder=str(out / f"bin{i:03d}"),
        )
        results = dataset.fit(sampler=SAMPLER, n_live_points=n_live, verbose=False)
        p_samples = results.posteriors["posterior_samples"]["p_p1"]
        depth = np.percentile(p_samples**2, [50, 16, 84])
        model = results.lc.evaluate(instrument)
        rows.append({
            "wave": float(lc["wave"][i]),
            "wave_err": float(lc["wave_err"][i]),
            "depth": float(depth[0]),
            "depth_err": float(0.5 * (depth[2] - depth[1])),
            "rms_ppm": float(np.nanstd(f[good] - model) * 1e6),
        })

    csv_path = out / "transmission_spectrum.csv"
    write_spectrum_csv(csv_path, rows, header=(
        f"Patchwork juliet fit v{PATCHWORK_FIT_VERSION}, "
        f"instrument={instrument}, ld={white_summary.get('ld_source')}, "
        f"regressors={','.join(white_summary.get('regressor_names', []))}"
    ))

    dat_path = out / "transmission_spectrum.dat"
    with dat_path.open("w") as handle:
        handle.write(
            "# wavelength_um  transit_depth  depth_err  bin_half_width_um\n"
            f"# Patchwork juliet fit v{PATCHWORK_FIT_VERSION}, instrument={instrument}\n"
        )
        for r in rows:
            handle.write(
                f"{r['wave']:.6f}  {r['depth']:.8f}  "
                f"{r['depth_err']:.8f}  {r['wave_err']:.6f}\n"
            )

    _plot_spectrum(rows, out, instrument)
    return {"spectrum_csv": str(csv_path), "spectrum_path": str(dat_path),
            "n_bins": len(rows), "rows": rows,
            "median_depth_err_ppm": float(np.nanmedian(
                [r["depth_err"] for r in rows]) * 1e6) if rows else None,
            "median_rms_ppm": float(np.nanmedian(
                [r["rms_ppm"] for r in rows])) if rows else None}


def write_spectrum_csv(path: str | os.PathLike[str],
                       rows: list[dict[str, float]], header: str = "") -> None:
    """ppm-unit spectrum CSV: wave_um, wave_err_um, depth_ppm,
    depth_err_ppm, resid_rms_ppm."""
    with Path(path).open("w", newline="") as fh:
        w = csv.writer(fh)
        if header:
            w.writerow([f"# {header}"])
        w.writerow(["wave_um", "wave_err_um", "depth_ppm",
                    "depth_err_ppm", "resid_rms_ppm"])
        for r in rows:
            w.writerow([
                f"{r['wave']:.6f}", f"{r['wave_err']:.6f}",
                f"{r['depth'] * 1e6:.2f}", f"{r['depth_err'] * 1e6:.2f}",
                f"{r.get('rms_ppm', float('nan')):.1f}",
            ])


def read_spectrum_csv(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Read a Patchwork spectrum CSV back into arrays (ppm depths)."""
    wave, wave_err, depth, depth_err, rms = [], [], [], [], []
    with Path(path).open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("#") or row[0] == "wave_um":
                continue
            wave.append(float(row[0]))
            wave_err.append(float(row[1]))
            depth.append(float(row[2]))
            depth_err.append(float(row[3]))
            rms.append(float(row[4]) if len(row) > 4 and row[4] else np.nan)
    return {"wave": np.asarray(wave), "wave_err": np.asarray(wave_err),
            "depth_ppm": np.asarray(depth), "depth_err_ppm": np.asarray(depth_err),
            "rms_ppm": np.asarray(rms)}


def _plot_spectrum(rows: list[dict[str, float]], out: Path, instrument: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(_PLOT_STYLE):
        wave = np.array([r["wave"] for r in rows])
        depth = np.array([r["depth"] for r in rows]) * 1e6
        err = np.array([r["depth_err"] for r in rows]) * 1e6
        xerr = np.array([r["wave_err"] for r in rows])
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.errorbar(wave, depth, yerr=err, xerr=xerr, fmt="o", ms=4,
                    color="#1a2f6b", ecolor="#1a2f6b", elinewidth=0.9,
                    capsize=0, alpha=0.9)
        med = np.median(depth)
        ax.axhline(med, color="crimson", lw=1, alpha=0.6,
                   label=f"median depth {med:.0f} ppm")
        lo, hi = np.percentile(depth - err, 2), np.percentile(depth + err, 98)
        span = hi - lo
        ax.set_ylim(lo - 0.15 * span, hi + 0.15 * span)
        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel("Transit depth (ppm)")
        ax.set_title(f"Transmission spectrum — {instrument}, {len(rows)} channels")
        ax.legend(fontsize=9)
        fig.tight_layout()
        _savefig(fig, out, "transmission_spectrum")
        plt.close(fig)


# -------------------- Stage 6: visit combination + metrics --------------


def combine_visit_spectra(csv_paths: list[str]) -> dict[str, np.ndarray]:
    """Inverse-variance combine per-visit spectra (same binning scheme ->
    same wavelength grid required). rms column: median across visits."""
    specs = [read_spectrum_csv(p) for p in csv_paths]
    ref = specs[0]["wave"]
    for s in specs[1:]:
        if s["wave"].size != ref.size or not np.allclose(s["wave"], ref, rtol=1e-4):
            raise ValueError(
                "Visit spectra are on different wavelength grids — they must "
                "come from the same Patchwork binning scheme."
            )
    D = np.vstack([s["depth_ppm"] for s in specs])
    E = np.vstack([s["depth_err_ppm"] for s in specs])
    R = np.vstack([s["rms_ppm"] for s in specs])
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / E**2
        depth = np.nansum(D * w, axis=0) / np.nansum(w, axis=0)
        err = np.sqrt(1.0 / np.nansum(w, axis=0))
    return {"wave": ref, "wave_err": specs[0]["wave_err"],
            "depth_ppm": depth, "depth_err_ppm": err,
            "rms_ppm": np.nanmedian(R, axis=0), "n_visits": len(specs)}


def detector_offset_ppm(nrs1: dict[str, np.ndarray],
                        nrs2: dict[str, np.ndarray],
                        n_edge: int = 5) -> float:
    """NRS1-NRS2 depth offset near the detector gap: median of the last
    ``n_edge`` finite NRS1 channels minus the first ``n_edge`` finite NRS2
    channels. A large jump flags an inter-detector systematic."""
    d1 = nrs1["depth_ppm"][np.isfinite(nrs1["depth_ppm"])]
    d2 = nrs2["depth_ppm"][np.isfinite(nrs2["depth_ppm"])]
    return float(np.median(d1[-n_edge:]) - np.median(d2[:n_edge]))


def _plot_combined(combined: dict[str, dict[str, np.ndarray]],
                   out: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"NRS1": "#1a2f6b", "NRS2": "crimson"}
    with plt.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        for det, S in combined.items():
            m = np.isfinite(S["depth_ppm"])
            ax.errorbar(S["wave"][m], S["depth_ppm"][m],
                        xerr=S["wave_err"][m], yerr=S["depth_err_ppm"][m],
                        fmt="o", ms=4, lw=0.8, alpha=0.85,
                        color=colors.get(det, "gray"), label=det)
        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel("Transit depth (ppm)")
        ax.set_title(title)
        ax.legend(fontsize=9)
        fig.tight_layout()
        _savefig(fig, out, "combined_transmission_spectrum")
        plt.close(fig)


# -------------------- shared tool plumbing --------------------


def _resolve(path: str, base: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base, path)


def prepare_visit_fit_inputs(
    spectra_path: str,
    planet_name: str,
    *,
    instrument: str,
    reduction_dir: str | None = None,
    decorrelate: bool = True,
    tilt_correction: bool = True,
    ld_priors: str = "exotic-ld",
    resolution: float = DEFAULT_RESOLUTION,
    n_bins: int | None = None,
    wave_min: float | None = None,
    wave_max: float | None = None,
) -> dict[str, Any]:
    """Shared Stage 4 -> Stage 5 preparation: archive priors, lightcurves,
    trace diagnostics, tilt events, regressor matrix, white-band LD."""
    priors = fetch_transit_priors(planet_name)
    spectra = load_stage3_spectra(spectra_path)
    lc = build_lightcurves(
        spectra,
        detector=instrument,
        t0_ref=priors["t0"],
        period=priors["period"],
        duration_hr=priors["duration_hr"],
        resolution=resolution,
        n_bins=n_bins,
        wave_min=wave_min,
        wave_max=wave_max,
    )

    diagnostics = None
    if decorrelate and reduction_dir:
        calints = find_stage2_calints(reduction_dir, instrument)
        if calints:
            diagnostics = trace_diagnostics(calints)
            for key in list(diagnostics):
                if diagnostics[key].size != lc["time"].size:
                    diagnostics = None  # cube/spectra mismatch: skip, do not crash
                    break

    tilt_events = (
        detect_tilt_events(lc["wl_flux"], exclude_mask=lc["oot_mask"])
        if tilt_correction else []
    )
    regressors, regressor_names = build_regressor_matrix(
        lc["time"], diagnostics, tilt_events
    )

    ld = None
    if ld_priors == "exotic-ld":
        wlo, whi = G395H_WAVE_RANGES.get(instrument.upper(), (None, None))
        ld = compute_ld_coeffs(
            priors["st_teff"], priors["st_logg"], priors["st_met"],
            wave_min if wave_min is not None else wlo,
            wave_max if wave_max is not None else whi,
        )

    return {"priors": priors, "lc": lc, "regressors": regressors,
            "regressor_names": regressor_names, "tilt_events": tilt_events,
            "ld": ld, "diagnostics_used": diagnostics is not None}


# -------------------- orchestral tools --------------------


class FitNirspecG395hWhiteLight(BaseTool):
    """
    Fit the white-light transit of one reduced JWST NIRSpec/G395H visit
    with juliet. One detector (NRS1 or NRS2) per call.

    Input is an exoTEDRF ``*_spectra_fullres.fits`` product (from
    ``ReduceNirspecG395hTso``). Ephemeris/orbit/stellar priors come from
    the NASA Exoplanet Archive via ``planet_name``; t0 is propagated to
    the observation epoch; the lightcurve is normalized by the
    out-of-transit baseline.

    Survey-frozen fit (Patchwork v1.1): Gaussian P and a/Rs, uniform
    Rp/Rs and b, ExoTiC-LD truncated-normal Kipping (q1, q2) [uniform
    fallback recorded in the summary], linear decorrelation against time
    + trace x/y + FWHM (pass ``reduction_dir`` so Stage 2 calints can be
    read), one fitted step per detected tilt event, dynesty sampling.
    Set decorrelate/tilt_correction False and ld_priors='uniform' for an
    as-is comparison fit.

    Outputs in ``output_dir``: juliet posteriors,
    ``white_lightcurve_fit.pdf/.svg``, ``white_fit_summary.json``
    (includes residual rms ppm and tilt events).

    Run BEFORE ``FitNirspecG395hTransmissionSpectrum`` — the spectroscopic
    fits freeze the orbit to this posterior.

    Example
    -------
        FitNirspecG395hWhiteLight(
            spectra_path="reductions/GJ_9827_d/o010/nrs1/.../..._box_spectra_fullres.fits",
            planet_name="GJ 9827 d",
            output_dir="fits/GJ_9827_d/o010/nrs1",
            instrument="NRS1",
            reduction_dir="reductions/GJ_9827_d/o010/nrs1",
        )
    """

    spectra_path: str = RuntimeField(
        description="Path to an exoTEDRF *_spectra_fullres.fits product."
    )
    planet_name: str = RuntimeField(
        description="Planet name for archive priors, e.g. 'GJ 9827 d'."
    )
    output_dir: str = RuntimeField(description="Directory for fit outputs.")
    instrument: str = RuntimeField(
        default="NRS1",
        description="Detector label for this dataset ('NRS1' or 'NRS2').",
    )
    reduction_dir: str | None = RuntimeField(
        default=None,
        description="Reduction dir containing Stage 2 calints — enables "
                    "trace x/y/FWHM decorrelation regressors.",
    )
    decorrelate: bool = RuntimeField(
        default=True,
        description="Include trace x/y/FWHM linear regressors (needs reduction_dir).",
    )
    tilt_correction: bool = RuntimeField(
        default=True,
        description="Detect tilt events and fit one step term per event.",
    )
    ld_priors: str = RuntimeField(
        default="exotic-ld",
        description="'exotic-ld' for Gaussian (q1,q2) priors, 'uniform' to disable.",
    )
    wave_min: float | None = RuntimeField(
        default=None, description="Optional lower wavelength cut in microns."
    )
    wave_max: float | None = RuntimeField(
        default=None, description="Optional upper wavelength cut in microns."
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        prep = prepare_visit_fit_inputs(
            _resolve(self.spectra_path, self.base_directory),
            self.planet_name,
            instrument=self.instrument,
            reduction_dir=(_resolve(self.reduction_dir, self.base_directory)
                           if self.reduction_dir else None),
            decorrelate=self.decorrelate,
            tilt_correction=self.tilt_correction,
            ld_priors=self.ld_priors,
            wave_min=self.wave_min,
            wave_max=self.wave_max,
        )
        summary = fit_white_lightcurve(
            prep["lc"],
            prep["priors"],
            _resolve(self.output_dir, self.base_directory),
            instrument=self.instrument,
            regressors=prep["regressors"],
            regressor_names=prep["regressor_names"],
            ld=prep["ld"],
        )
        lines = [
            f"White-light fit complete for {self.planet_name} ({self.instrument}).",
            f"Outputs: {summary['output_dir']}",
            f"  regressors: {', '.join(summary['regressor_names'])}"
            + ("" if prep["diagnostics_used"] else " (trace diagnostics unavailable)"),
            f"  LD priors: {summary['ld_source']}",
            f"  tilt events: {len(prep['tilt_events'])}",
        ]
        for key in ["t0_p1", "p_p1", "b_p1", "a_p1"]:
            if key in summary:
                s = summary[key]
                lines.append(
                    f"  {key} = {s['median']:.6f} +{s['plus']:.6f} -{s['minus']:.6f}"
                )
        d = summary["depth_ppm"]
        lines.append(
            f"  white depth = {d['median']:.0f} +{d['plus']:.0f} -{d['minus']:.0f} ppm; "
            f"residual rms = {summary['residual_rms_ppm']:.0f} ppm"
        )
        lines.append(
            "Now run FitNirspecG395hTransmissionSpectrum with the same "
            "spectra_path and this output_dir as white_fit_dir."
        )
        return "\n".join(lines)


class FitNirspecG395hTransmissionSpectrum(BaseTool):
    """
    Fit the spectroscopic channels of one NIRSpec/G395H visit x detector
    and produce a transmission spectrum.

    Requires a completed ``FitNirspecG395hWhiteLight`` run: P, t0, b, a/Rs
    are frozen to the white-light posterior medians. Free per channel:
    Rp/Rs, (q1, q2) with per-channel ExoTiC-LD priors, normalization,
    jitter, and the same decorrelation + tilt-step regressors as the
    white fit. Channels are constant-resolution (default R=100).

    Outputs in ``output_dir``: per-channel posteriors,
    ``transmission_spectrum.csv`` (ppm + per-channel residual rms — the
    input to ``CombineNirspecG395hVisits``), ``transmission_spectrum.dat``
    (TauREx format, usable by SimulateTaurexRetrieval), and
    ``transmission_spectrum.pdf/.svg``.

    Compute: one nested-sampling fit per channel (~30-60), a few minutes
    each.

    Example
    -------
        FitNirspecG395hTransmissionSpectrum(
            spectra_path="reductions/GJ_9827_d/o010/nrs1/.../..._box_spectra_fullres.fits",
            planet_name="GJ 9827 d",
            white_fit_dir="fits/GJ_9827_d/o010/nrs1",
            output_dir="fits/GJ_9827_d/o010/nrs1/spectro",
            reduction_dir="reductions/GJ_9827_d/o010/nrs1",
        )
    """

    spectra_path: str = RuntimeField(
        description="Path to the same *_spectra_fullres.fits used for the white fit."
    )
    planet_name: str = RuntimeField(
        description="Planet name for archive priors, e.g. 'GJ 9827 d'."
    )
    white_fit_dir: str = RuntimeField(
        description="output_dir of the completed FitNirspecG395hWhiteLight run "
                    "(must contain white_fit_summary.json)."
    )
    output_dir: str = RuntimeField(description="Directory for spectroscopic outputs.")
    instrument: str = RuntimeField(
        default="NRS1",
        description="Detector label, must match the white-light fit.",
    )
    reduction_dir: str | None = RuntimeField(
        default=None,
        description="Reduction dir with Stage 2 calints for trace regressors.",
    )
    decorrelate: bool = RuntimeField(
        default=True, description="Include trace x/y/FWHM regressors."
    )
    tilt_correction: bool = RuntimeField(
        default=True, description="Include tilt-event step terms."
    )
    ld_priors: str = RuntimeField(
        default="exotic-ld",
        description="'exotic-ld' for per-channel Gaussian (q1,q2), 'uniform' to disable.",
    )
    resolution: float = RuntimeField(
        default=DEFAULT_RESOLUTION,
        description="Spectral binning resolution R.",
    )
    n_bins: int | None = RuntimeField(
        default=None,
        description="Optional fixed number of equal-width bins (overrides resolution).",
    )
    wave_min: float | None = RuntimeField(
        default=None, description="Optional lower wavelength cut in microns."
    )
    wave_max: float | None = RuntimeField(
        default=None, description="Optional upper wavelength cut in microns."
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        white_dir = _resolve(self.white_fit_dir, self.base_directory)
        summary_path = Path(white_dir) / "white_fit_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"{summary_path} not found — run FitNirspecG395hWhiteLight first."
            )
        with summary_path.open() as handle:
            white_summary = json.load(handle)

        prep = prepare_visit_fit_inputs(
            _resolve(self.spectra_path, self.base_directory),
            self.planet_name,
            instrument=self.instrument,
            reduction_dir=(_resolve(self.reduction_dir, self.base_directory)
                           if self.reduction_dir else None),
            decorrelate=self.decorrelate,
            tilt_correction=self.tilt_correction,
            ld_priors=self.ld_priors,
            resolution=self.resolution,
            n_bins=self.n_bins,
            wave_min=self.wave_min,
            wave_max=self.wave_max,
        )
        stellar = {k: prep["priors"][k] for k in ("st_teff", "st_logg", "st_met")}
        result = fit_transmission_spectrum(
            prep["lc"],
            white_summary,
            _resolve(self.output_dir, self.base_directory),
            instrument=self.instrument,
            regressors=prep["regressors"],
            stellar=stellar if self.ld_priors == "exotic-ld" else None,
        )
        return (
            f"Transmission spectrum complete: {result['n_bins']} channels.\n"
            f"  median depth error: {result['median_depth_err_ppm']:.0f} ppm; "
            f"median channel rms: {result['median_rms_ppm']:.0f} ppm\n"
            f"CSV (for visit combining): {result['spectrum_csv']}\n"
            f"TauREx .dat: {result['spectrum_path']}\n"
            f"Plot: transmission_spectrum.pdf/.svg in the same directory."
        )


class CombineNirspecG395hVisits(BaseTool):
    """
    Stage 6: inverse-variance combine the per-visit transmission spectra
    of one target into the final Patchwork spectrum, and report the
    survey health metrics.

    Input is a JSON mapping detector -> list of per-visit
    ``transmission_spectrum.csv`` paths (from
    ``FitNirspecG395hTransmissionSpectrum``). Visits of one detector must
    share the binning scheme (they do, if fit with defaults). A single
    visit per detector is fine — the "combination" is then a pass-through.

    Outputs in ``output_dir``: ``combined_{nrs1,nrs2}_transmission_spectrum.csv``,
    ``combined_transmission_spectrum.pdf/.svg`` (both detectors, ready to
    overlay a published spectrum on), and ``combined_summary.json`` with
    the NRS1-NRS2 offset near the detector gap and median rms values —
    the numbers the Patchwork log asks for.

    Example
    -------
        CombineNirspecG395hVisits(
            spectra_by_detector='{"NRS1": ["fits/GJ_9827_d/o010/nrs1/spectro/transmission_spectrum.csv", '
                                '"fits/GJ_9827_d/o091/nrs1/spectro/transmission_spectrum.csv"], '
                                '"NRS2": ["fits/GJ_9827_d/o010/nrs2/spectro/transmission_spectrum.csv", '
                                '"fits/GJ_9827_d/o091/nrs2/spectro/transmission_spectrum.csv"]}',
            output_dir="fits/GJ_9827_d/combined",
            title="GJ 9827 d — Patchwork G395H",
        )
    """

    spectra_by_detector: str = RuntimeField(
        description='JSON dict: {"NRS1": [csv, ...], "NRS2": [csv, ...]}. '
                    "One or both detectors."
    )
    output_dir: str = RuntimeField(description="Directory for combined outputs.")
    title: str = RuntimeField(
        default="Patchwork G395H transmission spectrum",
        description="Plot title, e.g. 'GJ 9827 d — Patchwork G395H'.",
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        mapping = self.spectra_by_detector
        if isinstance(mapping, str):
            mapping = json.loads(mapping)

        out = Path(_resolve(self.output_dir, self.base_directory))
        out.mkdir(parents=True, exist_ok=True)

        combined: dict[str, dict[str, np.ndarray]] = {}
        summary: dict[str, Any] = {"fit_version": PATCHWORK_FIT_VERSION}
        lines = []
        for det, paths in mapping.items():
            det = det.upper()
            paths = [_resolve(p, self.base_directory) for p in paths]
            S = combine_visit_spectra(paths)
            combined[det] = S
            rows = [
                {"wave": w, "wave_err": we, "depth": d / 1e6,
                 "depth_err": de / 1e6, "rms_ppm": r}
                for w, we, d, de, r in zip(
                    S["wave"], S["wave_err"], S["depth_ppm"],
                    S["depth_err_ppm"], S["rms_ppm"])
            ]
            csv_path = out / f"combined_{det.lower()}_transmission_spectrum.csv"
            write_spectrum_csv(
                csv_path, rows,
                header=f"{self.title} — {det}, {S['n_visits']} visit(s) combined",
            )
            summary[det] = {
                "n_visits": S["n_visits"],
                "n_channels": int(np.isfinite(S["depth_ppm"]).sum()),
                "median_depth_err_ppm": float(np.nanmedian(S["depth_err_ppm"])),
                "median_channel_rms_ppm": float(np.nanmedian(S["rms_ppm"])),
                "csv": str(csv_path),
            }
            lines.append(
                f"  {det}: {S['n_visits']} visit(s), "
                f"{summary[det]['n_channels']} channels, median depth err "
                f"{summary[det]['median_depth_err_ppm']:.0f} ppm, median rms "
                f"{summary[det]['median_channel_rms_ppm']:.0f} ppm"
            )

        if "NRS1" in combined and "NRS2" in combined:
            offset = detector_offset_ppm(combined["NRS1"], combined["NRS2"])
            summary["nrs1_nrs2_offset_ppm"] = offset
            lines.append(f"  NRS1-NRS2 offset near detector gap: {offset:+.1f} ppm")

        _plot_combined(combined, out, self.title)
        with (out / "combined_summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)

        return "\n".join(
            [f"Combined transmission spectrum written to {out}."]
            + lines
            + ["Plot: combined_transmission_spectrum.pdf/.svg. Overlay your "
               "literature spectrum on the CSVs for the reproduction check."]
        )
