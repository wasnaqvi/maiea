"""Patchwork uniform transit lightcurve fitting built on juliet.

Consumes the ``*_spectra_fullres.fits`` products written by the exoTEDRF
reduction (``exotedrf.py``; extensions Wave / Wave Err / Flux / Flux Err /
Time) and produces:

1. White-light transit fit -> orbital parameters (t0, a/Rs, b, Rp/Rs).
2. Spectroscopic per-bin transit fits with the orbit frozen to the
   white-light posterior -> transmission spectrum.

The transmission spectrum is written as a 4-column ASCII file
(wavelength [um], transit depth (Rp/Rs)^2, depth error, bin half-width)
directly compatible with ``SimulateTaurexRetrieval``.

Uniformity contract
-------------------
Fitting choices (priors shapes, sampler, binning scheme, ld law) are
frozen module-wide so every Patchwork target is fit identically. The only
per-target inputs are the planet's literature ephemeris/orbit (fetched
automatically from the NASA Exoplanet Archive) and the data files.

Scalability note
----------------
``model_type='transit'`` is the only mode implemented. The fitting entry
points accept the parameter so that phase-curve (eclipse + phase
variation) support can be added later without changing tool signatures.

NOTE: this module is named ``juliet.py`` but the fitting engine is the
``juliet`` package. ``_import_juliet()`` below strips this file's own
directory from ``sys.path`` during import so the site-packages package is
always the one loaded, even when scripts run with this directory as cwd.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

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
    """Import the installed ``juliet`` package, never this module.

    When scripts run with this file's directory as sys.path[0] (direct
    execution, cwd here), ``import juliet`` would resolve to this file.
    Drop that path entry (and any stale self-reference in sys.modules)
    before importing.
    """
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
PATCHWORK_FIT_VERSION = "1.0"
SAMPLER = "dynesty"
N_LIVE_WHITE = 500
N_LIVE_SPEC = 300
LD_LAW = "quadratic"          # Kipping q1/q2 parameterization in juliet
DEFAULT_RESOLUTION = 100      # spectral binning R for the transmission spectrum
MJD_TO_BJD_OFFSET = 2400000.5

# Usable wavelength ranges (um) for G395H per detector. Columns outside
# these limits sit on the filter cut-on/detector edges where flux ~ 0 and
# bins are pure noise; they are cut unless the caller overrides.
G395H_WAVE_RANGES = {"NRS1": (2.87, 3.72), "NRS2": (3.82, 5.18)}


def _default_wave_cuts(instrument: str,
                       wave_min: float | None,
                       wave_max: float | None) -> tuple[float | None, float | None]:
    lo, hi = G395H_WAVE_RANGES.get(str(instrument).upper(), (None, None))
    return (wave_min if wave_min is not None else lo,
            wave_max if wave_max is not None else hi)


# -------------------- data loading and binning --------------------


def load_extracted_spectra(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Load an exoTEDRF ``*_spectra_fullres.fits`` product.

    Returns arrays: wave (nwave), wave_err (nwave), flux (ntime, nwave),
    flux_err (ntime, nwave), time (ntime; BJD_TDB).
    """
    from astropy.io import fits

    with fits.open(path) as hdul:
        wave = np.asarray(hdul["Wave"].data, dtype=float)
        wave_err = np.asarray(hdul["Wave Err"].data, dtype=float)
        flux = np.asarray(hdul["Flux"].data, dtype=float)
        flux_err = np.asarray(hdul["Flux Err"].data, dtype=float)
        time = np.asarray(hdul["Time"].data, dtype=float)

    # exoTEDRF wavelength arrays can be 2D (per-integration); collapse.
    if wave.ndim == 2:
        wave = np.nanmedian(wave, axis=0)
    if wave_err.ndim == 2:
        wave_err = np.nanmedian(wave_err, axis=0)
    # Times are MJD_TDB; convert to BJD_TDB for ephemeris arithmetic.
    if np.nanmedian(time) < 1e6:
        time = time + MJD_TO_BJD_OFFSET
    return {
        "wave": wave,
        "wave_err": wave_err,
        "flux": flux,
        "flux_err": flux_err,
        "time": time,
    }


def bin_lightcurves(
    spectra: dict[str, np.ndarray],
    *,
    resolution: float | None = DEFAULT_RESOLUTION,
    n_bins: int | None = None,
    wave_min: float | None = None,
    wave_max: float | None = None,
) -> dict[str, Any]:
    """Bin the 2D flux time series into white + spectroscopic lightcurves.

    Binning is inverse-variance weighted over detector columns. Columns
    that are all-NaN (detector gaps, reference pixels) are dropped. Bins
    are constant-R (``resolution``) unless ``n_bins`` is given, in which
    case they are equal-width in wavelength.
    """
    wave = spectra["wave"]
    flux = spectra["flux"]
    flux_err = spectra["flux_err"]

    good = np.isfinite(wave) & np.any(np.isfinite(flux), axis=0)
    lo = wave_min if wave_min is not None else np.nanmin(wave[good])
    hi = wave_max if wave_max is not None else np.nanmax(wave[good])

    if n_bins is not None:
        edges = np.linspace(lo, hi, n_bins + 1)
    else:
        edges = [lo]
        while edges[-1] < hi:
            edges.append(edges[-1] * (1 + 1.0 / resolution))
        edges = np.asarray(edges)

    def _weighted_bin(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = flux[:, mask]
        e = flux_err[:, mask]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / e**2
            w[~np.isfinite(w) | ~np.isfinite(f)] = 0.0
            f = np.where(np.isfinite(f), f, 0.0)
            wsum = w.sum(axis=1)
            fbin = (f * w).sum(axis=1) / wsum
            ebin = 1.0 / np.sqrt(wsum)
        return fbin, ebin

    # White lightcurve over the full usable range.
    white_flux, white_err = _weighted_bin(good & (wave >= lo) & (wave <= hi))

    bins = []
    for i in range(len(edges) - 1):
        mask = good & (wave >= edges[i]) & (wave < edges[i + 1])
        if mask.sum() == 0:
            continue
        fbin, ebin = _weighted_bin(mask)
        if not np.all(np.isfinite(fbin)):
            continue
        bins.append(
            {
                "wave_center": 0.5 * (edges[i] + edges[i + 1]),
                "wave_half_width": 0.5 * (edges[i + 1] - edges[i]),
                "flux": fbin,
                "flux_err": ebin,
            }
        )

    norm = np.nanmedian(white_flux)
    return {
        "time": spectra["time"],
        "white_flux": white_flux / norm,
        "white_err": white_err / norm,
        "bins": [
            {
                **b,
                "flux": b["flux"] / np.nanmedian(b["flux"]),
                "flux_err": b["flux_err"] / np.nanmedian(b["flux"]),
            }
            for b in bins
        ],
        "edges": np.asarray(edges),
    }


# -------------------- archive priors --------------------


def fetch_transit_priors(planet_name: str) -> dict[str, Any]:
    """Fetch literature ephemeris and orbit for a planet from the NASA
    Exoplanet Archive (pscomppars). Used to seed the white-light priors.
    """
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


def predict_mid_transit(t0_ref: float, period: float, times: np.ndarray) -> float:
    """Propagate a literature ephemeris to the transit epoch nearest the
    middle of the observation. All times BJD_TDB."""
    t_mid = 0.5 * (times.min() + times.max())
    n_orbits = np.round((t_mid - t0_ref) / period)
    return t0_ref + n_orbits * period


# -------------------- juliet fitting --------------------


def _white_priors(
    priors: dict[str, Any],
    times: np.ndarray,
    instrument: str,
) -> dict[str, dict[str, Any]]:
    """Build the frozen Patchwork white-light prior set for juliet.

    Wide/uninformative on what JWST constrains (Rp/Rs, t0, a/Rs); Gaussian
    on what it does not (P); eccentricity fixed to zero (sub-Neptune
    default; COMPASS does the same for circular-assumed literature fits).
    """
    period = priors["period"]
    if period is None:
        raise ValueError("No orbital period available from the archive.")
    t0_guess = (
        predict_mid_transit(priors["t0"], period, times)
        if priors["t0"] is not None
        else 0.5 * (times.min() + times.max())
    )
    a_rs = priors["a_rs"]
    b = priors["impact_param"]

    p: dict[str, dict[str, Any]] = {
        "P_p1": {
            "distribution": "normal",
            "hyperparameters": [period, max(priors["period_err"] or 0, 1e-5) * 3],
        },
        "t0_p1": {
            "distribution": "uniform",
            "hyperparameters": [float(times.min()), float(times.max())],
        },
        "p_p1": {"distribution": "uniform", "hyperparameters": [0.0, 0.3]},
        "b_p1": {"distribution": "uniform", "hyperparameters": [0.0, 1.0]},
        "ecc_p1": {"distribution": "fixed", "hyperparameters": 0.0},
        "omega_p1": {"distribution": "fixed", "hyperparameters": 90.0},
        f"q1_{instrument}": {"distribution": "uniform", "hyperparameters": [0.0, 1.0]},
        f"q2_{instrument}": {"distribution": "uniform", "hyperparameters": [0.0, 1.0]},
        f"mdilution_{instrument}": {"distribution": "fixed", "hyperparameters": 1.0},
        f"mflux_{instrument}": {
            "distribution": "normal",
            "hyperparameters": [0.0, 0.1],
        },
        f"sigma_w_{instrument}": {
            "distribution": "loguniform",
            "hyperparameters": [0.1, 10000.0],
        },
        # Linear slope in time absorbs the dominant NIRSpec settling trend
        # uniformly for all targets.
        f"theta0_{instrument}": {
            "distribution": "uniform",
            "hyperparameters": [-0.1, 0.1],
        },
    }
    if a_rs is not None:
        err = priors["a_rs_err"] or 0.1 * a_rs
        p["a_p1"] = {"distribution": "normal", "hyperparameters": [a_rs, 3 * err]}
    else:
        p["a_p1"] = {"distribution": "uniform", "hyperparameters": [1.0, 300.0]}
    return p


def fit_white_lightcurve(
    lightcurves: dict[str, Any],
    priors: dict[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    instrument: str = "NRS1",
    model_type: str = "transit",
    n_live: int = N_LIVE_WHITE,
) -> dict[str, Any]:
    """Fit the white-light transit with juliet/dynesty.

    Returns posterior medians and 1-sigma intervals for the orbital
    parameters, to be frozen in the spectroscopic fits.
    """
    if model_type != "transit":
        raise NotImplementedError(
            "Only model_type='transit' is implemented. Phase-curve support "
            "is a planned extension."
        )
    juliet = _import_juliet()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    times = lightcurves["time"]
    prior_dict = _white_priors(priors, times, instrument)
    theta = {instrument: np.vstack([times - times.mean()]).T}

    dataset = juliet.load(
        priors=prior_dict,
        t_lc={instrument: times},
        y_lc={instrument: lightcurves["white_flux"]},
        yerr_lc={instrument: lightcurves["white_err"]},
        linear_regressors_lc=theta,
        out_folder=str(out),
    )
    results = dataset.fit(sampler=SAMPLER, n_live_points=n_live, verbose=False)

    posterior = results.posteriors["posterior_samples"]
    summary: dict[str, Any] = {"instrument": instrument, "fit_version": PATCHWORK_FIT_VERSION}
    for key in ["P_p1", "t0_p1", "p_p1", "b_p1", "a_p1",
                f"q1_{instrument}", f"q2_{instrument}", f"sigma_w_{instrument}"]:
        if key in posterior:
            samples = posterior[key]
            med, lo, hi = np.percentile(samples, [50, 16, 84])
            summary[key] = {"median": float(med), "minus": float(med - lo), "plus": float(hi - med)}
    summary["depth_ppm"] = {
        "median": float(np.percentile(posterior["p_p1"] ** 2, 50) * 1e6),
        "minus": float((np.percentile(posterior["p_p1"] ** 2, 50)
                        - np.percentile(posterior["p_p1"] ** 2, 16)) * 1e6),
        "plus": float((np.percentile(posterior["p_p1"] ** 2, 84)
                       - np.percentile(posterior["p_p1"] ** 2, 50)) * 1e6),
    }

    _plot_white_fit(results, lightcurves, instrument, out)

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


def _plot_white_fit(results, lightcurves, instrument: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(_PLOT_STYLE):
        times = lightcurves["time"]
        t_min = (times - times.min()) * 24 * 60
        flux = lightcurves["white_flux"]
        err = lightcurves["white_err"]
        model = results.lc.evaluate(instrument)
        residual = flux - model
        sig = np.std(residual)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(9, 6), sharex=True, height_ratios=[3, 1]
        )
        ax1.errorbar(t_min, flux, yerr=err, fmt="o", ms=3, color="#1a2f6b",
                     elinewidth=0.7, alpha=0.8, label="white lightcurve")
        ax1.plot(t_min, model, color="crimson", lw=1.5, label="juliet best fit")
        # Robust display range: don't let a couple of cosmic-ray outliers
        # flatten the transit visually.
        core = np.percentile(flux, [0.5, 99.5])
        pad = 5 * sig
        ax1.set_ylim(min(core[0], model.min()) - pad, max(core[1], model.max()) + pad)
        ax1.set_ylabel("Relative flux")
        ax1.set_title(
            f"{instrument} white lightcurve — residual rms "
            f"{sig * 1e6:.0f} ppm"
        )
        ax1.legend(fontsize=9)
        ax2.plot(t_min, residual * 1e6, "o", ms=2.5, color="#1a2f6b", alpha=0.7)
        ax2.axhline(0, color="crimson", lw=1)
        ax2.set_ylim(-5 * sig * 1e6, 5 * sig * 1e6)
        ax2.set_xlabel("Time from start (min)")
        ax2.set_ylabel("Residual (ppm)")
        fig.tight_layout()
        fig.savefig(out / "white_lightcurve_fit.pdf")
        plt.close(fig)


def fit_transmission_spectrum(
    lightcurves: dict[str, Any],
    white_summary: dict[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    instrument: str = "NRS1",
    n_live: int = N_LIVE_SPEC,
) -> dict[str, Any]:
    """Fit every spectroscopic bin with the orbit frozen to the white-light
    posterior medians; only Rp/Rs, limb darkening, mflux, jitter, and the
    linear slope are free per bin. Writes the transmission spectrum.
    """
    juliet = _import_juliet()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    times = lightcurves["time"]
    theta = {instrument: np.vstack([times - times.mean()]).T}

    fixed = {
        "P_p1": white_summary["P_p1"]["median"],
        "t0_p1": white_summary["t0_p1"]["median"],
        "b_p1": white_summary["b_p1"]["median"],
        "a_p1": white_summary["a_p1"]["median"],
    }

    rows = []
    for i, b in enumerate(lightcurves["bins"]):
        prior_dict: dict[str, dict[str, Any]] = {
            key: {"distribution": "fixed", "hyperparameters": value}
            for key, value in fixed.items()
        }
        prior_dict.update(
            {
                "ecc_p1": {"distribution": "fixed", "hyperparameters": 0.0},
                "omega_p1": {"distribution": "fixed", "hyperparameters": 90.0},
                "p_p1": {"distribution": "uniform", "hyperparameters": [0.0, 0.3]},
                f"q1_{instrument}": {"distribution": "uniform", "hyperparameters": [0.0, 1.0]},
                f"q2_{instrument}": {"distribution": "uniform", "hyperparameters": [0.0, 1.0]},
                f"mdilution_{instrument}": {"distribution": "fixed", "hyperparameters": 1.0},
                f"mflux_{instrument}": {"distribution": "normal", "hyperparameters": [0.0, 0.1]},
                f"sigma_w_{instrument}": {"distribution": "loguniform",
                                          "hyperparameters": [0.1, 10000.0]},
                f"theta0_{instrument}": {"distribution": "uniform",
                                         "hyperparameters": [-0.1, 0.1]},
            }
        )

        bin_out = out / f"bin{i:03d}"
        dataset = juliet.load(
            priors=prior_dict,
            t_lc={instrument: times},
            y_lc={instrument: b["flux"]},
            yerr_lc={instrument: b["flux_err"]},
            linear_regressors_lc=theta,
            out_folder=str(bin_out),
        )
        results = dataset.fit(sampler=SAMPLER, n_live_points=n_live, verbose=False)
        p_samples = results.posteriors["posterior_samples"]["p_p1"]
        depth = np.percentile(p_samples**2, [50, 16, 84])
        rows.append(
            {
                "wave": b["wave_center"],
                "half_width": b["wave_half_width"],
                "depth": float(depth[0]),
                "depth_err": float(0.5 * (depth[2] - depth[1])),
            }
        )

    spectrum_path = out / "transmission_spectrum.dat"
    with spectrum_path.open("w") as handle:
        handle.write(
            "# wavelength_um  transit_depth  depth_err  bin_half_width_um\n"
            f"# Patchwork juliet fit v{PATCHWORK_FIT_VERSION}, "
            f"instrument={instrument}\n"
        )
        for row in rows:
            handle.write(
                f"{row['wave']:.6f}  {row['depth']:.8f}  "
                f"{row['depth_err']:.8f}  {row['half_width']:.6f}\n"
            )

    _plot_spectrum(rows, out)
    return {"spectrum_path": str(spectrum_path), "n_bins": len(rows), "rows": rows}


def _plot_spectrum(rows: list[dict[str, float]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(_PLOT_STYLE):
        wave = np.array([r["wave"] for r in rows])
        depth = np.array([r["depth"] for r in rows]) * 1e6
        err = np.array([r["depth_err"] for r in rows]) * 1e6
        xerr = np.array([r["half_width"] for r in rows])
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.errorbar(wave, depth, yerr=err, xerr=xerr, fmt="o", ms=4,
                    color="#1a2f6b", ecolor="#1a2f6b", elinewidth=0.9,
                    capsize=0, alpha=0.9)
        med = np.median(depth)
        ax.axhline(med, color="crimson", lw=1, alpha=0.6,
                   label=f"median depth {med:.0f} ppm")
        # Robust y-range so one blown bin doesn't hide the spectrum shape.
        lo, hi = np.percentile(depth - err, 2), np.percentile(depth + err, 98)
        span = hi - lo
        ax.set_ylim(lo - 0.15 * span, hi + 0.15 * span)
        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel("Transit depth (ppm)")
        ax.set_title(f"Transmission spectrum — {len(rows)} bins")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "transmission_spectrum.pdf")
        plt.close(fig)


# -------------------- orchestral tools --------------------


def _resolve(path: str, base: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base, path)


class FitNirspecG395hWhiteLight(BaseTool):
    """
    Fit the white-light transit of a reduced JWST NIRSpec/G395H TSO with
    juliet. One detector (NRS1 or NRS2) per call — G395H detectors are
    always fit independently.

    Input is an exoTEDRF ``*_spectra_fullres.fits`` product (from
    ``ReduceNirspecG395hTso``). Ephemeris/orbit priors are fetched automatically
    from the NASA Exoplanet Archive using ``planet_name``; the transit
    epoch is propagated to the observation window.

    All prior shapes and sampler settings are frozen survey-wide: Gaussian
    on P and a/Rs (literature), uniform on Rp/Rs, t0, b, Kipping q1/q2;
    eccentricity fixed to 0; linear-in-time systematics; dynesty nested
    sampling.

    Outputs (in ``output_dir``): juliet posteriors,
    ``white_lightcurve_fit.pdf``, ``white_fit_summary.json``.

    Run this BEFORE ``FitNirspecG395hTransmissionSpectrum`` — the
    spectroscopic fits freeze the orbit to this fit's posterior.

    Example
    -------
        FitNirspecG395hWhiteLight(
            spectra_path="reductions/GJ_1214_b/visit1/nrs1/.../GJ-1214_box_spectra_fullres.fits",
            planet_name="GJ 1214 b",
            output_dir="fits/GJ_1214_b/nrs1",
        )
    """

    spectra_path: str = RuntimeField(
        description="Path to an exoTEDRF *_spectra_fullres.fits product."
    )
    planet_name: str = RuntimeField(
        description="Planet name for archive priors, e.g. 'GJ 1214 b'."
    )
    output_dir: str = RuntimeField(
        description="Directory for fit outputs."
    )
    instrument: str = RuntimeField(
        default="NRS1",
        description="Detector label for this dataset ('NRS1' or 'NRS2').",
    )
    wave_min: float | None = RuntimeField(
        default=None, description="Optional lower wavelength cut in microns."
    )
    wave_max: float | None = RuntimeField(
        default=None, description="Optional upper wavelength cut in microns."
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        spectra = load_extracted_spectra(_resolve(self.spectra_path, self.base_directory))
        wave_min, wave_max = _default_wave_cuts(self.instrument, self.wave_min, self.wave_max)
        lightcurves = bin_lightcurves(spectra, wave_min=wave_min, wave_max=wave_max)
        priors = fetch_transit_priors(self.planet_name)
        summary = fit_white_lightcurve(
            lightcurves,
            priors,
            _resolve(self.output_dir, self.base_directory),
            instrument=self.instrument,
        )
        lines = [
            f"White-light fit complete for {self.planet_name} ({self.instrument}).",
            f"Outputs: {summary['output_dir']}",
        ]
        for key in ["t0_p1", "p_p1", "b_p1", "a_p1"]:
            if key in summary:
                s = summary[key]
                lines.append(
                    f"  {key} = {s['median']:.6f} +{s['plus']:.6f} -{s['minus']:.6f}"
                )
        d = summary["depth_ppm"]
        lines.append(
            f"  white transit depth = {d['median']:.0f} +{d['plus']:.0f} "
            f"-{d['minus']:.0f} ppm"
        )
        lines.append(
            "Now run FitNirspecG395hTransmissionSpectrum with the same "
            "spectra_path and this output_dir as white_fit_dir."
        )
        return "\n".join(lines)


class FitNirspecG395hTransmissionSpectrum(BaseTool):
    """
    Fit spectroscopic lightcurve bins of one NIRSpec/G395H detector and
    produce a transmission spectrum.

    Requires a completed ``FitNirspecG395hWhiteLight`` run: orbital parameters
    (P, t0, a/Rs, b) are frozen to the white-light posterior medians and
    only Rp/Rs, limb darkening, normalization, jitter, and a linear slope
    are fit per bin. Bins are constant-resolution (default R=100).

    Outputs (in ``output_dir``): per-bin juliet posteriors,
    ``transmission_spectrum.dat`` (wavelength um, depth, error, bin
    half-width — directly usable by SimulateTaurexRetrieval), and
    ``transmission_spectrum.pdf``.

    Compute: roughly one nested-sampling fit per bin (~30-60 bins), a few
    minutes each.

    Example
    -------
        FitNirspecG395hTransmissionSpectrum(
            spectra_path="reductions/GJ_1214_b/visit1/nrs1/.../GJ-1214_box_spectra_fullres.fits",
            white_fit_dir="fits/GJ_1214_b/nrs1",
            output_dir="fits/GJ_1214_b/nrs1/spectro",
        )
    """

    spectra_path: str = RuntimeField(
        description="Path to the same *_spectra_fullres.fits used for the white fit."
    )
    white_fit_dir: str = RuntimeField(
        description="output_dir of the completed FitWhiteLightCurve run "
                    "(must contain white_fit_summary.json)."
    )
    output_dir: str = RuntimeField(description="Directory for spectroscopic outputs.")
    instrument: str = RuntimeField(
        default="NRS1",
        description="Detector label, must match the white-light fit.",
    )
    resolution: float = RuntimeField(
        default=DEFAULT_RESOLUTION,
        description="Spectral binning resolution R for the transmission spectrum.",
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

        spectra = load_extracted_spectra(_resolve(self.spectra_path, self.base_directory))
        wave_min, wave_max = _default_wave_cuts(self.instrument, self.wave_min, self.wave_max)
        lightcurves = bin_lightcurves(
            spectra,
            resolution=self.resolution,
            n_bins=self.n_bins,
            wave_min=wave_min,
            wave_max=wave_max,
        )
        result = fit_transmission_spectrum(
            lightcurves,
            white_summary,
            _resolve(self.output_dir, self.base_directory),
            instrument=self.instrument,
        )
        return (
            f"Transmission spectrum complete: {result['n_bins']} bins.\n"
            f"Spectrum file: {result['spectrum_path']}\n"
            f"Plot: {Path(result['spectrum_path']).parent / 'transmission_spectrum.pdf'}\n"
            "The .dat file is directly usable as observation_path in "
            "SimulateTaurexRetrieval."
        )
