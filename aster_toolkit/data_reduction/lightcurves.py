"""Patchwork Stage 4 — lightcurve construction and systematics diagnostics.

Sits between the exoTEDRF Stage 1-3 reduction (``exotedrf.py``) and the
juliet transit fitting (``juliet.py``). Ports the validated GJ 9827 d
notebook workflow (workspace/patchwork/GJ_9827d.ipynb) into reusable,
target-agnostic form:

1. Load a ``*_spectra_fullres.fits`` product, trim reference-pixel
   columns, apply the G395H per-detector wavelength cuts.
2. Build the white lightcurve and constant-R spectroscopic lightcurves,
   normalized by the **out-of-transit baseline** (not the global median —
   a global median is biased low by the transit itself).
3. Extract per-integration trace diagnostics (cross-dispersion position
   y, dispersion drift x, trace FWHM) from the Stage 2 calints cubes —
   the decorrelation regressors for the juliet fits.
4. Detect **tilt events** (sudden mirror-segment tilts) and build the
   corresponding Heaviside step regressors. The search runs over the
   trace diagnostics — FWHM, position, PCA components — because a tilt
   changes the PSF before it changes the flux, and those series carry no
   transit signal, so an event *during* transit is as visible as one
   outside it. See ``find_tilt_events``.

Everything here is plain numpy/astropy — no jwst/exotedrf import — so it
runs in the main ASTER environment.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from orchestral.tools.base.tool import BaseTool
    from orchestral.tools.base.field_utils import RuntimeField, StateField
except ModuleNotFoundError:
    class BaseTool:
        """Fallback that keeps plain functions importable without Orchestral."""

    def RuntimeField(default=None, description=None):
        return default

    def StateField(default=None, description=None):
        return default


# 1.1: final bin edge clipped to the wavelength cut (the last channel's
# centre used to be reported outside the detector range).
# 1.2: tilt events are searched in the trace diagnostics (FWHM, position,
# PCA) rather than the out-of-transit flux alone, so an event during
# transit is detectable. Different events found => different step
# regressors => different fits, so this is a survey-definition change.
PATCHWORK_STAGE4_VERSION = "1.3"

N_REFPIX_COLS = 5             # detector reference columns trimmed each edge
# Survey-frozen binning (decision Wasi 2026-07-30): one resolution for
# every Patchwork target. R=100 is the standard presentation for
# sub-Neptune G395H transmission spectra (e.g. the K2-18 b reanalysis,
# arXiv:2501.18477, primary R~100; CO2 4.3 um and CH4 3.3 um bands are
# broad against it) and gives ~29 NRS1 + ~24 NRS2 channels. COMPASS
# (arXiv:2511.18196) publishes at R~200 — rebin THEIR spectra to R=100
# for overlays, never mix binnings within Patchwork.
DEFAULT_RESOLUTION = 100      # constant-R spectroscopic binning
MJD_TO_BJD_OFFSET = 2400000.5
# A constant-R bin clipped by the detector wavelength cut contains only
# part of the columns its neighbours do. Such a bin is dropped rather
# than published: see bin_at_resolution.
MIN_BIN_COLUMNS = 3
MIN_BIN_FILL_FRAC = 0.5

# Usable wavelength ranges (um) for G395H per detector. Outside these the
# throughput is ~0 and channels are pure noise.
G395H_WAVE_RANGES = {"NRS1": (2.87, 3.72), "NRS2": (3.82, 5.18)}

# Tilt-event detection: frozen survey-wide.
TILT_WINDOW = 15              # integrations each side of candidate step
TILT_THRESHOLD = 6.0          # robust sigma of window-median differences
TILT_MIN_SEPARATION = 30      # integrations between distinct events
# A mirror tilt event must show in at least this many independent series
# before it is believed. One series stepping is a glitch in that series;
# the PSF changing shape shows up in several at once.
TILT_MIN_SOURCES = 2
# Integrations dropped at the step itself. The tilt completes in under
# 1.4 s (Schlawin et al. 2023, PASP 135 018001) but the integration that
# straddles it is a blend of both PSF states, which no Heaviside
# describes. Rigby et al. / the WASP-39 b G395H analysis (arXiv:2405.06737)
# "exclude three integrations around the tilt event" and step-correct the
# rest; three is enough at any NIRSpec BOTS cadence and costs ~0.1% of a
# visit, against splitting the lightcurve which costs the orbit constraint.
TILT_TRANSITION_MASK = 3
# Loic Albert (2026-08-26): tilt events are rare, "no more than ~1 per day".
# A 5 h visit therefore expects ~0.2. Finding several in one visit means
# the detection is firing on something else, so flag it rather than
# silently fitting a step per false positive.
TILT_EXPECTED_PER_DAY = 1.0
TILT_MATCH_TOL_MIN = 5.0      # cross-detector coincidence window


# -------------------- loading --------------------


def load_stage3_spectra(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Load an exoTEDRF ``*_spectra_fullres.fits`` product.

    Returns wave (nwave), wave_err (nwave), flux (ntime, nwave),
    flux_err (ntime, nwave), time (ntime, BJD_TDB).
    """
    from astropy.io import fits

    with fits.open(path) as hdul:
        wave = np.asarray(hdul["Wave"].data, dtype=float)
        wave_err = np.asarray(hdul["Wave Err"].data, dtype=float)
        flux = np.asarray(hdul["Flux"].data, dtype=float)
        flux_err = np.asarray(hdul["Flux Err"].data, dtype=float)
        time = np.asarray(hdul["Time"].data, dtype=float)

    if wave.ndim == 2:
        wave = np.nanmedian(wave, axis=0)
    if wave_err.ndim == 2:
        wave_err = np.nanmedian(wave_err, axis=0)
    # exoTEDRF writes MJD_TDB; ephemerides are BJD_TDB.
    if np.nanmedian(time) < 1e6:
        time = time + MJD_TO_BJD_OFFSET
    return {"wave": wave, "wave_err": wave_err, "flux": flux,
            "flux_err": flux_err, "time": time}


def propagate_t0(t0_ref: float, period: float, times: np.ndarray) -> float:
    """Propagate a literature ephemeris to the epoch nearest this visit."""
    t_mid = float(np.nanmedian(np.asarray(times, dtype=float)))
    n = np.round((t_mid - t0_ref) / period)
    return float(t0_ref + n * period)


def oot_mask_from_baseline(n: int, baseline_ints: list[int]) -> np.ndarray:
    """Boolean out-of-transit mask from ``[n_pre, -n_post]``."""
    mask = np.zeros(n, dtype=bool)
    mask[: baseline_ints[0]] = True
    mask[baseline_ints[1]:] = True
    return mask


def oot_mask_from_ephemeris(times: np.ndarray, t0_obs: float,
                            duration_hr: float, pad: float = 1.15) -> np.ndarray:
    """Boolean out-of-transit mask from the propagated mid-time and the
    literature transit duration (padded to keep ingress/egress out)."""
    half = 0.5 * duration_hr / 24.0 * pad
    return np.abs(times - t0_obs) > half


# -------------------- binning + lightcurves --------------------


def bin_at_resolution(wave: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
                      *, resolution: float = DEFAULT_RESOLUTION,
                      n_bins: int | None = None) -> dict[str, np.ndarray]:
    """Inverse-variance bin (ntime, nwave) spectra into constant-R channels
    (or ``n_bins`` equal-width channels). Mirrors exotedrf.stage4 binning
    without needing the pinned environment."""
    good_col = np.isfinite(wave) & np.any(np.isfinite(flux), axis=0)
    lo, hi = np.nanmin(wave[good_col]), np.nanmax(wave[good_col])

    if n_bins is not None:
        edges = np.linspace(lo, hi, n_bins + 1)
    else:
        e = [lo]
        while e[-1] < hi:
            e.append(e[-1] * (1 + 1.0 / resolution))
        # Clip the final edge to the wavelength cut: otherwise the last
        # channel's reported centre lies outside [lo, hi] even though its
        # flux only comes from columns inside the cut.
        e[-1] = hi
        edges = np.asarray(e)

    # How many detector columns each bin SHOULD contain. An edge bin
    # clipped by the wavelength cut holds only a fraction of that, so its
    # depth is both noisier than its error bar implies and sitting where
    # throughput is falling -- the two together make it the single most
    # likely channel in a spectrum to be wrong. On the 2026-08-27 survey
    # products three of the four >4 sigma outliers were exactly this:
    # GJ 3090 b NRS2 4.038 um at -7.0 sigma (1075 vs 1426 ppm), L 98-59 d
    # NRS1 2.913 um, GJ 357 b NRS2 5.077 um -- every one a first or last
    # channel. Requiring a bin to be reasonably filled drops them before
    # they reach a retrieval.
    counts = np.array([int((good_col & (wave >= edges[i]) & (wave < edges[i + 1])).sum())
                       for i in range(len(edges) - 1)])
    typical = float(np.median(counts[counts > 0])) if np.any(counts > 0) else 0.0
    min_cols = max(MIN_BIN_COLUMNS, MIN_BIN_FILL_FRAC * typical)

    centers, half_widths, fbins, ebins = [], [], [], []
    for i in range(len(edges) - 1):
        m = good_col & (wave >= edges[i]) & (wave < edges[i + 1])
        if not m.any() or int(m.sum()) < min_cols:
            continue
        f, e = flux[:, m], flux_err[:, m]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / e**2
            w[~np.isfinite(w) | ~np.isfinite(f)] = 0.0
            f = np.where(np.isfinite(f), f, 0.0)
            wsum = w.sum(axis=1)
            fb = (f * w).sum(axis=1) / wsum
            eb = 1.0 / np.sqrt(wsum)
        if not np.all(np.isfinite(fb)):
            continue
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        half_widths.append(0.5 * (edges[i + 1] - edges[i]))
        fbins.append(fb)
        ebins.append(eb)

    return {
        "wave": np.asarray(centers),
        "wave_err": np.asarray(half_widths),
        "flux": np.column_stack(fbins) if fbins else np.empty((flux.shape[0], 0)),
        "flux_err": np.column_stack(ebins) if ebins else np.empty((flux.shape[0], 0)),
    }


def build_lightcurves(
    spectra: dict[str, np.ndarray],
    *,
    detector: str,
    t0_ref: float,
    period: float,
    duration_hr: float | None = None,
    baseline_ints: list[int] | None = None,
    resolution: float = DEFAULT_RESOLUTION,
    n_bins: int | None = None,
    wave_min: float | None = None,
    wave_max: float | None = None,
) -> dict[str, Any]:
    """Stage 4: white + spectroscopic lightcurves for one visit x detector.

    Normalization is by the median of the out-of-transit baseline. The
    mask comes from the propagated ephemeris + literature duration when
    available, else from ``baseline_ints``; one of the two must be given.
    """
    wave = spectra["wave"][N_REFPIX_COLS:-N_REFPIX_COLS]
    wave_err = spectra["wave_err"][N_REFPIX_COLS:-N_REFPIX_COLS]
    flux = spectra["flux"][:, N_REFPIX_COLS:-N_REFPIX_COLS]
    flux_err = spectra["flux_err"][:, N_REFPIX_COLS:-N_REFPIX_COLS]
    time = spectra["time"]

    lo, hi = G395H_WAVE_RANGES.get(detector.upper(), (None, None))
    lo = wave_min if wave_min is not None else lo
    hi = wave_max if wave_max is not None else hi
    keep = np.isfinite(wave)
    if lo is not None:
        keep &= wave >= lo
    if hi is not None:
        keep &= wave <= hi
    wave, wave_err = wave[keep], wave_err[keep]
    flux, flux_err = flux[:, keep], flux_err[:, keep]

    t0_obs = propagate_t0(t0_ref, period, time)
    if duration_hr:
        oot = oot_mask_from_ephemeris(time, t0_obs, duration_hr)
    elif baseline_ints is not None:
        oot = oot_mask_from_baseline(time.size, baseline_ints)
    else:
        raise ValueError("Need duration_hr or baseline_ints for the OOT mask.")
    if oot.sum() < 10:
        raise ValueError(
            f"Only {oot.sum()} out-of-transit integrations — check the "
            "ephemeris/duration or baseline_ints."
        )

    # The transit must actually be in the data. If a stale ephemeris (or an
    # unconverted MJD time axis) puts the propagated mid-transit outside the
    # observation, every fit still "succeeds" -- but the Rp/Rs posterior is
    # just the prior, giving a plausible-looking spectrum with error bars
    # larger than the signal. Fail here, in milliseconds, rather than after
    # ~100 nested-sampling runs.
    half_window_hr = 0.5 * (float(np.nanmax(time)) - float(np.nanmin(time))) * 24
    t_centre = 0.5 * (float(np.nanmin(time)) + float(np.nanmax(time)))
    offset_hr = (t0_obs - t_centre) * 24
    reach_hr = half_window_hr + 0.5 * (duration_hr or 0.0)
    if abs(offset_hr) > reach_hr:
        raise ValueError(
            f"Propagated mid-transit is {offset_hr:+.2f} h from the centre of "
            f"a {2 * half_window_hr:.2f} h observation — the transit is NOT in "
            f"this data, so a fit would return the Rp/Rs prior rather than a "
            f"measurement.\n"
            f"  t0_obs = {t0_obs:.5f} BJD; data span {float(np.nanmin(time)):.5f} "
            f"to {float(np.nanmax(time)):.5f}.\n"
            f"  Check the ephemeris is current (a stale reference epoch "
            f"propagated over many periods drifts by hours) and that the time "
            f"axis is BJD, not MJD."
        )
    transit_coverage = (
        min(1.0, max(0.0,
            (min(offset_hr + 0.5 * duration_hr, half_window_hr)
             - max(offset_hr - 0.5 * duration_hr, -half_window_hr)) / duration_hr))
        if duration_hr else None
    )

    def _normalize(f: np.ndarray, e: np.ndarray):
        base = np.nanmedian(f[oot], axis=0)
        return f / base, e / base

    wl_flux = np.nansum(flux, axis=1)
    wl_err = np.sqrt(np.nansum(flux_err**2, axis=1))
    wl_flux, wl_err = _normalize(wl_flux, wl_err)

    binned = bin_at_resolution(wave, flux, flux_err,
                               resolution=resolution, n_bins=n_bins)
    sp_flux, sp_err = _normalize(binned["flux"], binned["flux_err"])

    return {
        "detector": detector.upper(),
        "time": time,
        "t0_obs": t0_obs,
        "transit_coverage": transit_coverage,
        "oot_mask": oot,
        "wl_flux": wl_flux,
        "wl_err": wl_err,
        "wave": binned["wave"],
        "wave_err": binned["wave_err"],
        "sp_flux": sp_flux,
        "sp_err": sp_err,
        "oot_scatter_ppm": float(np.nanstd(wl_flux[oot]) * 1e6),
        "stage4_version": PATCHWORK_STAGE4_VERSION,
    }


# -------------------- trace diagnostics (decorrelation regressors) ------


def estimate_transit_midpoint(time: np.ndarray, flux: np.ndarray, *,
                              smooth: int = 15) -> dict[str, Any]:
    """Measure mid-transit straight from a white lightcurve.

    For a target whose archive ephemeris misses a transit that is
    plainly in the data, this is how the corrected ``t0`` for a
    ``priors_override`` is obtained — measured, not guessed. It is a
    diagnostic, NOT a fit: the number it returns seeds the juliet prior,
    which then fits t0 properly.

    Method: normalize by the median, smooth, take the out-of-transit
    level from the 90th percentile, and find the longest run below the
    half-depth level. The midpoint of that run is the mid-transit time —
    the half-depth crossing is used rather than the minimum because it
    is insensitive to limb darkening curving the floor of the transit.

    ``partial`` is the flag that matters: a run touching either end of
    the series means ingress or egress fell outside the exposure, so the
    depth is degenerate with the baseline and the midpoint is a lower
    bound on the truth, not a measurement. Do not build an override from
    a partial detection.
    """
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    if t.size != f.size:
        raise ValueError(f"time ({t.size}) and flux ({f.size}) differ in length")
    f = f / np.nanmedian(f)

    w = max(1, int(smooth))
    if w % 2 == 0:
        w += 1
    good = np.isfinite(f)
    kern = np.ones(w)
    num = np.convolve(np.where(good, f, 0.0), kern, mode="same")
    den = np.convolve(good.astype(float), kern, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(den >= w / 2.0, num / den, np.nan)

    base = float(np.nanpercentile(s, 90))
    depth = base - float(np.nanmin(s))
    if not np.isfinite(depth) or depth <= 0:
        return {"found": False, "note": "No dip in this lightcurve."}

    below = np.isfinite(s) & (s < base - 0.5 * depth)
    idx = np.flatnonzero(below)
    if idx.size == 0:
        return {"found": False, "note": "No samples below the half-depth level."}
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    run = max(runs, key=len)
    i0, i1 = int(run[0]), int(run[-1])

    return {
        "found": True,
        "t0": 0.5 * (t[i0] + t[i1]),
        "depth_ppm": depth * 1e6,
        "duration_hr": float((t[i1] - t[i0]) * 24),
        "index_start": i0,
        "index_end": i1,
        "n_in_transit": int(i1 - i0 + 1),
        # Ingress or egress outside the exposure: the baseline on that
        # side is missing, so depth and midpoint are both unreliable.
        "partial": bool(i0 == 0 or i1 == t.size - 1),
    }


def find_stage2_calints(reduction_dir: str | os.PathLike[str],
                        detector: str) -> list[str]:
    """Locate Stage 2 calints cubes for one detector under a reduction dir."""
    pats = [
        str(Path(reduction_dir) / "**" / f"*{detector.lower()}*calints.fits"),
        str(Path(reduction_dir) / "**" / f"*{detector.lower()}*badpixstep.fits"),
    ]
    for pat in pats:
        files = sorted(glob.glob(pat, recursive=True))
        if files:
            return files
    return []


def trace_diagnostics(calints_files: list[str]) -> dict[str, np.ndarray]:
    """Per-integration trace diagnostics from Stage 2 cubes.

    - ``y``: flux-weighted cross-dispersion centroid of the collapsed
      spatial profile (first moment).
    - ``fwhm``: 2.355 x sqrt(second moment) of the same profile.
    - ``x``: dispersion-direction drift from cross-correlating each
      integration's 1D spectrum against the median spectrum (parabolic
      sub-pixel refinement, +/-2 px search).

    All series are median-subtracted, ready to use as linear regressors.
    """
    from astropy.io import fits

    ys, fwhms, specs = [], [], []
    for f in calints_files:
        with fits.open(f) as hdul:
            cube = np.asarray(hdul["SCI"].data, dtype=float)  # (nints, ny, nx)
        cube = np.where(np.isfinite(cube), cube, 0.0)
        cube = np.clip(cube, 0, None)
        profile = cube.sum(axis=2)                            # (nints, ny)
        rows = np.arange(profile.shape[1], dtype=float)
        norm = profile.sum(axis=1)
        norm[norm == 0] = np.nan
        y = (profile * rows).sum(axis=1) / norm
        var = (profile * (rows[None, :] - y[:, None]) ** 2).sum(axis=1) / norm
        ys.append(y)
        fwhms.append(2.355 * np.sqrt(np.clip(var, 0, None)))
        specs.append(cube.sum(axis=1))                        # (nints, nx)

    y = np.concatenate(ys)
    fwhm = np.concatenate(fwhms)
    spec = np.concatenate(specs, axis=0)

    ref = np.nanmedian(spec, axis=0)
    ref = ref - ref.mean()
    lags = np.arange(-2, 3)
    x = np.zeros(spec.shape[0])
    for i in range(spec.shape[0]):
        s = spec[i] - spec[i].mean()
        cc = np.array([np.nansum(s * np.roll(ref, lag)) for lag in lags])
        k = int(np.argmax(cc))
        if 0 < k < len(lags) - 1:
            denom = cc[k - 1] - 2 * cc[k] + cc[k + 1]
            frac = 0.5 * (cc[k - 1] - cc[k + 1]) / denom if denom != 0 else 0.0
        else:
            frac = 0.0
        x[i] = lags[k] + frac

    return {
        "x": x - np.nanmedian(x),
        "y": y - np.nanmedian(y),
        "fwhm": fwhm - np.nanmedian(fwhm),
    }


def pca_regressors(calints_files: list[str], *, n_components: int = 6,
                   max_pixels: int = 1000) -> np.ndarray:
    """COMPASS-style PCA systematics regressors from Stage 2 calints.

    Following the COMPASS uniform reanalysis (Ahrer et al. 2025,
    arXiv:2511.18196): the principal components of the *relative pixel
    flux* timeseries RPF_ij(t) = F_ij(t) / sum_ij F_ij(t) trace changes
    in the shape and position of the spectral trace that x/y-shift
    regressors miss. Used as linear regressors they substantially reduce
    red noise, most strongly on NRS1 and for few-group observations.

    Deviation from the paper, for memory: the PCA runs on the
    ``max_pixels`` brightest pixels (median flux) rather than every
    pixel — a 21k-integration TSO over a full subarray would need >10 GB
    otherwise, and the trace morphology signal lives in the bright
    pixels. Each pixel series is mean-subtracted and variance-normalized
    before the SVD; the returned temporal components (nints,
    n_components) are ready for ``build_regressor_matrix``.
    """
    from astropy.io import fits

    chunks = []
    for f in calints_files:
        with fits.open(f) as hdul:
            cube = np.asarray(hdul["SCI"].data, dtype=float)
        cube = np.where(np.isfinite(cube), cube, 0.0)
        chunks.append(cube.reshape(cube.shape[0], -1))
    X = np.concatenate(chunks, axis=0)                    # (nints, npix)

    total = X.sum(axis=1, keepdims=True)
    total[total == 0] = np.nan
    X = X / total                                         # relative pixel flux

    bright = np.argsort(np.nanmedian(X, axis=0))[::-1][:max_pixels]
    X = X[:, bright]
    X = X - np.nanmean(X, axis=0, keepdims=True)
    std = np.nanstd(X, axis=0, keepdims=True)
    std[std == 0] = 1.0
    X = np.where(np.isfinite(X / std), X / std, 0.0)

    # Temporal principal components: left singular vectors of (nints, npix).
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    k = min(n_components, U.shape[1])
    comps = U[:, :k] * s[:k]                              # scaled scores
    # Standardize each component so the frozen theta priors apply.
    med = np.nanmedian(comps, axis=0, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(comps - med), axis=0, keepdims=True)
    mad[mad == 0] = 1.0
    return (comps - med) / mad


def rednoise_beta(residuals: np.ndarray, time: np.ndarray,
                  *, min_minutes: float = 5.0,
                  max_minutes: float = 30.0) -> dict[str, float]:
    """Time-correlated (red) noise diagnostic for a fit residual series.

    Classic Pont, Zucker & Queloz (2006) beta: bin the residuals at a
    range of timescales and compare the binned rms to the white-noise
    expectation rms_1 / sqrt(n). beta ~ 1 means the errors are honest;
    beta > 1 means depth uncertainties are underestimated by that
    factor. The COMPASS G395H reanalysis finds real errors run ~5-12%
    above the photon prediction, so recording this per fit is the
    cheapest way to know when a target needs its error bars inflated.

    Returns {"beta_median", "beta_max", "rms_unbinned_ppm"} with beta
    evaluated over bins spanning ``min_minutes`` to ``max_minutes``.
    """
    r = np.asarray(residuals, dtype=float)
    good = np.isfinite(r)
    r = r[good]
    t = np.asarray(time, dtype=float)[good]
    if r.size < 50:
        return {"beta_median": float("nan"), "beta_max": float("nan"),
                "rms_unbinned_ppm": float("nan")}
    rms1 = float(np.std(r))
    cadence_min = float(np.median(np.diff(t))) * 24 * 60
    betas = []
    for minutes in np.linspace(min_minutes, max_minutes, 8):
        n = max(2, int(round(minutes / cadence_min)))
        if n >= r.size // 3:
            break
        m = r.size // n
        binned = r[: m * n].reshape(m, n).mean(axis=1)
        expected = rms1 / np.sqrt(n) * np.sqrt(m / max(1, m - 1))
        if expected > 0:
            betas.append(float(np.std(binned) / expected))
    if not betas:
        return {"beta_median": float("nan"), "beta_max": float("nan"),
                "rms_unbinned_ppm": rms1 * 1e6}
    return {"beta_median": float(np.median(betas)),
            "beta_max": float(np.max(betas)),
            "rms_unbinned_ppm": rms1 * 1e6}


# -------------------- tilt events --------------------


def detect_tilt_events(
    flux: np.ndarray,
    *,
    window: int = TILT_WINDOW,
    threshold: float = TILT_THRESHOLD,
    min_separation: int = TILT_MIN_SEPARATION,
    exclude_mask: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Detect step discontinuities in a single series (legacy, flux-only).

    At each interior integration i, compare the median of ``window``
    points after i to the median of ``window`` points before; a step
    exceeding ``threshold`` robust sigma (MAD of the difference series)
    is an event. ``exclude_mask`` is a KEEP mask (True = use this point;
    False = ignore it, e.g. in-transit points).

    **Prefer ``find_tilt_events``.** This function can only search one
    series, so on the flux it must mask the transit to stop ingress and
    egress registering as steps — and an event *inside* transit is then
    structurally undetectable. That is the TOI-270 c failure. It remains
    here because it is still the right tool for a single diagnostic
    series, and ``find_tilt_events`` uses the same statistic.

    Returns a list of ``{"index", "amplitude"}`` (amplitude in relative
    flux, positive = brightening), ordered by index, merged so events are
    at least ``min_separation`` apart.
    """
    z = step_statistic(flux, window=window, exclude_mask=exclude_mask)
    if not np.any(np.isfinite(z)):
        return []

    f = np.asarray(flux, dtype=float).copy()
    if exclude_mask is not None:
        f[~np.asarray(exclude_mask, dtype=bool)] = np.nan

    candidates = np.flatnonzero(np.abs(z) > threshold)
    events: list[dict[str, float]] = []
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for idx in candidates[np.argsort(-np.abs(z[candidates]))]:
            if any(abs(int(idx) - e["index"]) < min_separation for e in events):
                continue
            i = int(idx)
            amplitude = float(np.nanmedian(f[i: i + window])
                              - np.nanmedian(f[i - window: i]))
            events.append({"index": i, "amplitude": amplitude})
    return sorted(events, key=lambda e: e["index"])


def step_statistic(series: np.ndarray, *, window: int = TILT_WINDOW,
                   exclude_mask: np.ndarray | None = None) -> np.ndarray:
    """Robust step-detection statistic for one per-integration series.

    At each interior point, the median of ``window`` samples after it
    minus the median of ``window`` before; that difference series is
    then divided by its own MAD. A step of any size produces one large
    excursion, and because a single step contributes only ~``window``
    points out of thousands, it barely moves the MAD that normalizes it.

    Returns a z-array (NaN where the windows do not fit). ``exclude_mask``
    is a KEEP mask: False entries are ignored, which is how the transit
    is kept out of a search run on the *flux*.
    """
    s = np.asarray(series, dtype=float).copy()
    if exclude_mask is not None:
        s[~np.asarray(exclude_mask, dtype=bool)] = np.nan

    n = s.size
    diff = np.full(n, np.nan)
    import warnings

    with warnings.catch_warnings():
        # Windows fully inside an excluded span are all-NaN by
        # construction; their diff stays NaN and is ignored.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in range(window, n - window):
            diff[i] = np.nanmedian(s[i: i + window]) - np.nanmedian(s[i - window: i])

    med = np.nanmedian(diff)
    mad = np.nanmedian(np.abs(diff - med))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma == 0:
        return np.full(n, np.nan)
    return (diff - med) / sigma


def find_tilt_events(
    flux: np.ndarray,
    *,
    diagnostics: dict[str, np.ndarray] | None = None,
    pca_components: np.ndarray | None = None,
    oot_mask: np.ndarray | None = None,
    time: np.ndarray | None = None,
    t0_obs: float | None = None,
    window: int = TILT_WINDOW,
    threshold: float = TILT_THRESHOLD,
    min_separation: int = TILT_MIN_SEPARATION,
    min_sources: int = TILT_MIN_SOURCES,
    detector: str = "",
) -> dict[str, Any]:
    """Detect mirror-segment tilt events from the trace diagnostics.

    A tilt event is a primary-mirror segment moving, so what it changes
    first is the **point spread function**, and only consequently the
    flux in the extraction aperture. Loic Albert (2026-08-26): *"The most
    direct effect that a tilt event has on the PSF is a change in its
    FWHM, so PCA are good to catch that."* The KELT-7 b analysis
    (arXiv:2509.12479) confirmed its event through the guide star,
    finding "a definite jump in the guide star **width**", and SOSSISSE
    detects tilts through the spatial-derivative term precisely because a
    tilt changes the trace's *shape* while barely moving its position.

    This is the reason to search the diagnostics rather than the flux.
    Searching the flux alone forces a choice between two failures: leave
    the transit in and every ingress is a step, or mask the transit — as
    Patchwork did until now — and an event *during* transit becomes
    structurally invisible. It is the second that cost us TOI-270 c.
    The FWHM, trace position and PCA series contain no transit at all, so
    they can be searched over the whole visit, in-transit included.

    Series searched: ``fwhm``, ``y``, ``x`` from ``diagnostics``, each
    column of ``pca_components``, and the white ``flux`` — the last
    restricted to ``oot_mask`` where one is given, since it is the only
    series the transit lives in. An event must trigger in at least
    ``min_sources`` of them: one series stepping on its own is a glitch
    in that series, while a PSF genuinely changing shape moves several
    together.

    Returns ``{"events": [...], "sources": [...], "rate_warning": ...}``.
    Each event carries ``index``, ``amplitude`` (flux step, relative —
    positive means brighter after), ``n_sources``, ``z`` per triggering
    series, and, when ``time`` is given, ``time``/``hours_from_mid``.
    """
    flux = np.asarray(flux, dtype=float)
    n = flux.size

    panel: dict[str, np.ndarray] = {}
    if diagnostics:
        for key in ("fwhm", "y", "x"):
            v = diagnostics.get(key)
            if v is not None and np.asarray(v).size == n:
                panel[f"trace_{key}"] = np.asarray(v, dtype=float)
    if pca_components is not None:
        pca = np.asarray(pca_components, dtype=float)
        if pca.ndim == 2 and pca.shape[0] == n:
            for j in range(pca.shape[1]):
                panel[f"pca_{j}"] = pca[:, j]

    # The flux is searched too, but only outside transit: it is the one
    # series where ingress and egress are themselves steps.
    z_by_source = {name: step_statistic(v, window=window)
                   for name, v in panel.items()}
    z_by_source["flux"] = step_statistic(flux, window=window,
                                         exclude_mask=oot_mask)

    triggers = {name: np.abs(z) > threshold
                for name, z in z_by_source.items()}
    stacked = np.vstack([np.where(np.isfinite(z_by_source[name]), t, False)
                         for name, t in triggers.items()])
    n_trig = stacked.sum(axis=0)

    # Total significance, used only to pick the representative index
    # inside a group of adjacent candidate points.
    total_z = np.nansum(
        np.vstack([np.abs(np.nan_to_num(z, nan=0.0))
                   for z in z_by_source.values()]), axis=0)

    candidates = np.flatnonzero(n_trig >= int(min_sources))
    events: list[dict[str, Any]] = []
    for idx in candidates[np.argsort(-total_z[candidates])]:
        if any(abs(int(idx) - e["index"]) < min_separation for e in events):
            continue
        i = int(idx)
        pre = flux[max(0, i - window): i]
        post = flux[i: i + window]
        amplitude = (float(np.nanmedian(post) - np.nanmedian(pre))
                     if pre.size and post.size else float("nan"))
        sources = sorted(name for name, t in triggers.items()
                         if t[i] and np.isfinite(z_by_source[name][i]))
        event: dict[str, Any] = {
            "index": i,
            "amplitude": amplitude,
            "amplitude_ppm": amplitude * 1e6,
            "n_sources": len(sources),
            "sources": sources,
            "z": {name: float(z_by_source[name][i]) for name in sources},
            "detector": detector.upper(),
        }
        if time is not None:
            event["time"] = float(np.asarray(time, dtype=float)[i])
            if t0_obs is not None:
                event["hours_from_mid"] = float((event["time"] - t0_obs) * 24)
            if oot_mask is not None:
                event["in_transit"] = bool(~np.asarray(oot_mask, dtype=bool)[i])
        events.append(event)

    events.sort(key=lambda e: e["index"])

    # Rate sanity check (see TILT_EXPECTED_PER_DAY).
    rate_warning = None
    if time is not None and len(events):
        t_arr = np.asarray(time, dtype=float)
        span_days = float(np.nanmax(t_arr) - np.nanmin(t_arr))
        expected = TILT_EXPECTED_PER_DAY * span_days
        if len(events) > max(2, 5 * expected):
            rate_warning = (
                f"{len(events)} tilt events in a {span_days * 24:.1f} h visit, "
                f"against ~{expected:.2f} expected at the observed rate of "
                f"~{TILT_EXPECTED_PER_DAY:.0f}/day. Tilt events are rare; this "
                "many means the search is firing on something else (a "
                "systematic ramp, a noisy diagnostic). Inspect before fitting "
                "a step per detection."
            )

    return {
        "detector": detector.upper(),
        "events": events,
        "sources_searched": sorted(z_by_source),
        "diagnostics_available": bool(panel),
        "threshold": float(threshold),
        "min_sources": int(min_sources),
        "window": int(window),
        "rate_warning": rate_warning,
    }


def match_tilt_events(reports: dict[str, dict[str, Any]], *,
                      tol_min: float = TILT_MATCH_TOL_MIN) -> list[dict[str, Any]]:
    """Cross-confirm tilt events between detectors.

    A segment tilt is a *telescope* event: the same mirror moves for
    every detector at the same instant. NRS1 and NRS2 read through
    independent chains, so a step in only one of them is detector
    electronics, not the observatory. Coincidence is therefore a cheap,
    strong discriminator — the same argument as for spot crossings in
    ``contamination.py``, but with the opposite chromatic expectation:
    a tilt changes the PSF, so its flux step can differ in size *and
    sign* between detectors and even between pixels (arXiv:2405.06737:
    "the change in count can be positive or negative and varies between
    pixels"). Do NOT reject a tilt for being chromatic.

    Returns one entry per matched or unmatched event, earliest first.
    """
    dets = [d for d in ("NRS1", "NRS2") if d in reports]
    dets += sorted(d for d in reports if d not in ("NRS1", "NRS2"))
    tol_days = tol_min / (24 * 60)
    merged: list[dict[str, Any]] = []
    used: set[tuple[str, int]] = set()

    if len(dets) >= 2:
        a_det, b_det = dets[0], dets[1]
        for i, a in enumerate(reports[a_det].get("events", [])):
            if "time" not in a:
                continue
            best, best_dt = None, None
            for j, b in enumerate(reports[b_det].get("events", [])):
                if (b_det, j) in used or "time" not in b:
                    continue
                dt = abs(a["time"] - b["time"])
                if dt <= tol_days and (best_dt is None or dt < best_dt):
                    best, best_dt = j, dt
            if best is None:
                continue
            b = reports[b_det]["events"][best]
            used.add((a_det, i))
            used.add((b_det, best))
            merged.append({
                "confirmed": True,
                "detectors": {a_det: a, b_det: b},
                "time": 0.5 * (a["time"] + b["time"]),
                "delta_t_min": float(best_dt * 24 * 60),
                "hours_from_mid": a.get("hours_from_mid"),
                "in_transit": bool(a.get("in_transit")
                                   or b.get("in_transit")),
                "amplitude_ppm": {a_det: a["amplitude_ppm"],
                                  b_det: b["amplitude_ppm"]},
                "n_sources": max(a["n_sources"], b["n_sources"]),
            })

    for det in dets:
        for i, a in enumerate(reports[det].get("events", [])):
            if (det, i) in used:
                continue
            merged.append({
                "confirmed": False,
                "detectors": {det: a},
                "time": a.get("time"),
                "delta_t_min": None,
                "hours_from_mid": a.get("hours_from_mid"),
                "in_transit": bool(a.get("in_transit")),
                "amplitude_ppm": {det: a["amplitude_ppm"]},
                "n_sources": a["n_sources"],
                "single_detector": True,
            })

    # None times sort last; the 0.0 placeholder keeps two timeless
    # events comparable (None < None raises in Python 3).
    merged.sort(key=lambda m: (m["time"] is None,
                               m["time"] if m["time"] is not None else 0.0))
    return merged


def tilt_transition_keep_mask(n: int, events: list[dict[str, Any]], *,
                              pad: int = TILT_TRANSITION_MASK) -> np.ndarray:
    """KEEP mask (True = use) excluding the integrations at each step.

    A Heaviside cannot describe the integration that straddles the tilt:
    for part of that exposure the PSF was in one state and for the rest
    in another, so its flux lies somewhere between the two levels and
    would drag the fitted step amplitude. Dropping ``pad`` integrations
    at the transition is what the WASP-39 b G395H analysis does, and it
    costs ~0.1% of a visit — as against splitting the lightcurve in two,
    which throws away the joint constraint on the orbit.
    """
    keep = np.ones(int(n), dtype=bool)
    for e in events:
        idx = int(e["index"] if "index" in e
                  else next(iter(e["detectors"].values()))["index"])
        lo = max(0, idx - int(pad))
        hi = min(int(n) - 1, idx + int(pad))
        keep[lo: hi + 1] = False
    return keep


def step_regressors(n: int, events: list[dict[str, float]]) -> np.ndarray:
    """(n, n_events) matrix of 0/1 step functions, one column per tilt
    event (0 before the event index, 1 from it on). Empty (n, 0) if none."""
    if not events:
        return np.empty((n, 0))
    cols = []
    for e in events:
        c = np.zeros(n)
        c[int(e["index"]):] = 1.0
        cols.append(c)
    return np.column_stack(cols)


def ramp_regressors(n: int, events: list[dict[str, Any]]) -> np.ndarray:
    """(n, n_events) matrix of SMOOTHED step functions, one per event.

    A hard Heaviside assumes the flux changes between two consecutive
    integrations. Real steps settle over a short but finite time, and
    forcing a vertical edge through a finite transition does two things
    wrong: the blended integrations pull the fitted amplitude towards
    the middle, so the step comes out too shallow, and the residual
    keeps a spike at the transition.

    Measured on TOI-270 c o016 (2026-08-27), where both detectors chose
    the same optimum independently: an erf ramp of width 8 integrations
    (1.4 min) at the fitted break beat a hard step at the flagged-span
    centre by 337 -> 230 ppm rms, and recovered a late-step amplitude of
    +2784 ppm against the Heaviside's +2275 -- 22% deeper, which is the
    difference between a model that reaches the post-event flux level
    and one that stops short of it.

    ``width_ints`` of 0 (or absent) gives the hard Heaviside, so an
    genuinely instantaneous event is unchanged.
    """
    if not events:
        return np.empty((n, 0))
    from math import erf as _erf

    x = np.arange(int(n), dtype=float)
    cols = []
    for e in events:
        b = float(e["index"])
        w = float(e.get("width_ints") or 0.0)
        if w <= 0:
            cols.append((x >= b).astype(float))
        else:
            z = (x - b) / (np.sqrt(2.0) * w)
            cols.append(0.5 * (1.0 + np.vectorize(_erf)(z)))
    return np.column_stack(cols)


def refine_step_shape(
    residual: np.ndarray,
    index_guess: int,
    *,
    search: int = 45,
    bounds: tuple[int, int] | None = None,
    widths: tuple[float, ...] = (0.0, 2.0, 4.0, 8.0, 14.0, 20.0),
    others: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fit a step's break time and transition width to the residuals.

    The Stage 5.5 scan localizes an event to a flagged span, but the
    span's centre is not the break: the span comes from a detrended
    statistic whose two lobes need not be symmetric about the
    transition. On TOI-270 c the centre sat 23 integrations late, which
    alone cost 22% of the fitted amplitude.

    Grid-searches break index and ramp width, fitting all amplitudes
    (plus a constant) by least squares at each trial, and returns the
    combination minimising the residual scatter. ``others`` are the
    other events in the same series, held at their current shape so
    each break is fitted in the presence of the rest.
    """
    r = np.asarray(residual, dtype=float)
    n = r.size
    if bounds is not None:
        # Search the FLAGGED SPAN, which brackets the transition by
        # construction. Searching a fixed window around the span's
        # centre fails whenever the two detrended lobes are lopsided:
        # on TOI-270 c the centre sat 240 integrations from the real
        # break, far outside any sane +/- window, and the fit settled on
        # a wrong break with a third of the true amplitude.
        lo, hi = max(0, int(bounds[0])), min(n - 1, int(bounds[1]))
    else:
        lo = max(0, int(index_guess) - int(search))
        hi = min(n - 1, int(index_guess) + int(search))
    if hi <= lo:
        lo, hi = max(0, int(index_guess) - 5), min(n - 1, int(index_guess) + 5)
    fixed = [e for e in (others or [])]
    best: dict[str, Any] | None = None
    good = np.isfinite(r)
    for b in range(lo, hi + 1, 2):
        for w in widths:
            trial = fixed + [{"index": b, "width_ints": w}]
            design = np.column_stack([np.ones(n), ramp_regressors(n, trial)])
            try:
                coef, *_ = np.linalg.lstsq(design[good], r[good], rcond=None)
            except np.linalg.LinAlgError:
                continue
            rms = float(np.nanstd(r - design @ coef))
            if best is None or rms < best["rms"]:
                best = {"index": int(b), "width_ints": float(w),
                        "amplitude": float(coef[-1]), "rms": rms}
    if best is None:
        return {"index": int(index_guess), "width_ints": 0.0,
                "amplitude": float("nan"), "rms": float("nan")}
    best["amplitude_ppm"] = best["amplitude"] * 1e6
    best["rms_ppm"] = best["rms"] * 1e6
    return best


def build_regressor_matrix(
    time: np.ndarray,
    diagnostics: dict[str, np.ndarray] | None = None,
    tilt_events: list[dict[str, float]] | None = None,
    pca_components: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Assemble the juliet ``linear_regressors_lc`` matrix.

    Columns (each standardized to zero median / unit robust scale, so the
    frozen theta priors are comparable across targets):
    time slope, [trace x, trace y, FWHM], [one step per tilt event],
    [optional PCA components from ``pca_regressors`` — enabling these
    changes the fit definition, see PATCHWORK_FIT_VERSION in juliet.py].
    Returns (matrix, column_names).
    """
    def _standardize(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        med = np.nanmedian(v)
        mad = 1.4826 * np.nanmedian(np.abs(v - med))
        scale = mad if mad > 0 else (np.nanstd(v) or 1.0)
        out = (v - med) / scale
        return np.where(np.isfinite(out), out, 0.0)

    cols = [_standardize(time)]
    names = ["time"]
    if diagnostics:
        for key in ("x", "y", "fwhm"):
            if key in diagnostics and np.asarray(diagnostics[key]).size == time.size:
                cols.append(_standardize(diagnostics[key]))
                names.append(f"trace_{key}")
    steps = step_regressors(time.size, tilt_events or [])
    for j in range(steps.shape[1]):
        cols.append(steps[:, j])  # keep 0/1 — amplitude is the fit coefficient
        names.append(f"tilt_step_{j}")
    if pca_components is not None and np.asarray(pca_components).size:
        pca = np.asarray(pca_components, dtype=float)
        if pca.shape[0] == time.size:
            for j in range(pca.shape[1]):
                cols.append(np.where(np.isfinite(pca[:, j]), pca[:, j], 0.0))
                names.append(f"pca_{j}")
    return np.column_stack(cols), names


# -------------------- orchestral tool --------------------


class DetectTiltEvents(BaseTool):
    """
    Scan a reduced NIRSpec/G395H visit for mirror-segment tilt events.

    A tilt event is a primary-mirror segment shifting, which changes the
    PSF and steps the flux in the extraction aperture. Undetected, it
    biases the transit depth; the KELT-7 b event sat a third of the way
    through transit and every pipeline that reduced it had to model it
    (arXiv:2509.12479).

    **Detection searches the trace diagnostics, not just the flux.** A
    tilt changes the PSF *width* first — Loic Albert: "the most direct
    effect that a tilt event has on the PSF is a change in its FWHM, so
    PCA are good to catch that"; the KELT-7 b team confirmed theirs as "a
    definite jump in the guide star width". So this tool steps-searches
    the trace FWHM, the trace x/y position and the PCA components, and
    requires an event to appear in at least two of them. Those series
    carry no transit signal, so an event *during* transit is as visible
    as one outside it — the flux-only search this replaces had to mask
    the transit to avoid flagging ingress, which made an in-transit tilt
    structurally undetectable. The white flux is searched too, but only
    out of transit.

    Pass ``reduction_dir`` — without the Stage 2 calints there are no
    diagnostics, the search falls back to flux-only, and the tool says so.

    **Correction is a fitted Heaviside step, not a mask.** The juliet fit
    tools add one step regressor per event with a free amplitude, fixed
    in time, and re-fit it per wavelength channel (the flux change is
    pixel-dependent and can be positive or negative, arXiv:2405.06737).
    Only ~3 integrations at the transition are dropped, because the
    integration straddling the tilt is a blend of both PSF states. Nothing
    else is discarded and the lightcurve is never split.

    Tilt events are rare — roughly one per day, so ~0.2 per visit. Several
    in one visit means the search is firing on something else, and the
    tool warns rather than fitting a step per false positive.

    Example
    -------
        DetectTiltEvents(
            spectra_path="reductions/TOI_270_c/o016/nrs1/.../..._box_spectra_fullres.fits",
            reduction_dir="reductions/TOI_270_c/o016/nrs1",
            detector="NRS1",
            period=11.38014, t0_ref=2458444.4677, duration_hr=2.17,
        )
    """

    spectra_path: str = RuntimeField(
        description="Path to an exoTEDRF *_spectra_fullres.fits product."
    )
    detector: str = RuntimeField(
        default="NRS1", description="'NRS1' or 'NRS2'."
    )
    period: float = RuntimeField(description="Orbital period in days.")
    t0_ref: float = RuntimeField(description="Reference mid-transit epoch, BJD_TDB.")
    duration_hr: float = RuntimeField(description="Transit duration in hours.")
    reduction_dir: str | None = RuntimeField(
        default=None,
        description="Reduction dir with Stage 2 calints. Strongly "
                    "recommended: without it there are no PSF diagnostics "
                    "and an in-transit tilt cannot be found.",
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        path = self.spectra_path
        if not os.path.isabs(path):
            path = os.path.join(self.base_directory, path)
        spectra = load_stage3_spectra(path)
        lc = build_lightcurves(
            spectra, detector=self.detector, t0_ref=self.t0_ref,
            period=self.period, duration_hr=self.duration_hr,
        )

        diagnostics = pca = None
        if self.reduction_dir:
            red = self.reduction_dir
            if not os.path.isabs(red):
                red = os.path.join(self.base_directory, red)
            calints = find_stage2_calints(red, self.detector)
            if calints:
                diagnostics = trace_diagnostics(calints)
                if any(np.asarray(v).size != lc["time"].size
                       for v in diagnostics.values()):
                    diagnostics = None
                else:
                    try:
                        pca = pca_regressors(calints)
                        if pca.shape[0] != lc["time"].size:
                            pca = None
                    except Exception:
                        pca = None

        report = find_tilt_events(
            lc["wl_flux"], diagnostics=diagnostics, pca_components=pca,
            oot_mask=lc["oot_mask"], time=lc["time"], t0_obs=lc["t0_obs"],
            detector=self.detector,
        )
        events = report["events"]

        lines = [
            f"Tilt-event scan ({self.detector}): {len(events)} event(s); "
            f"OOT scatter {lc['oot_scatter_ppm']:.0f} ppm.",
            f"  searched: {', '.join(report['sources_searched'])}",
        ]
        if not report["diagnostics_available"]:
            lines.append(
                "  WARNING: no trace diagnostics (pass reduction_dir). The "
                "search fell back to the out-of-transit flux only, so a tilt "
                "DURING transit would not be found — the TOI-270 c failure."
            )
        for e in events:
            lines.append(
                f"  index {e['index']} (BJD {e['time']:.5f}, "
                f"{e.get('hours_from_mid', float('nan')):+.2f} h from "
                f"mid-transit"
                + (", IN TRANSIT" if e.get("in_transit") else "")
                + f"): step {e['amplitude_ppm']:+.0f} ppm, "
                f"{e['n_sources']} source(s): "
                + ", ".join(f"{s} ({e['z'][s]:+.1f} sigma)" for s in e["sources"])
            )
        if not events:
            lines.append("  No step discontinuities above threshold.")
        else:
            lines.append(
                f"The juliet fit tools will fit one Heaviside step per event "
                f"and drop {TILT_TRANSITION_MASK} integration(s) either side "
                "of each transition (tilt_correction=True, the default). "
                "Nothing else is masked."
            )
        if report["rate_warning"]:
            lines.append(f"  WARNING: {report['rate_warning']}")
        return "\n".join(lines)


# Pre-v1.3 name; the tool now searches the PSF diagnostics rather than
# the out-of-transit flux alone.
DetectNirspecTiltEvents = DetectTiltEvents
