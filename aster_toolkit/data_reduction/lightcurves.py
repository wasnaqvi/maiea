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
4. Detect **tilt events** (sudden mirror-segment tilts, seen in NIRSpec
   TSOs as step discontinuities in flux / trace position) and build the
   corresponding step regressors.

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


PATCHWORK_STAGE4_VERSION = "1.0"

N_REFPIX_COLS = 5             # detector reference columns trimmed each edge
DEFAULT_RESOLUTION = 100      # constant-R spectroscopic binning
MJD_TO_BJD_OFFSET = 2400000.5

# Usable wavelength ranges (um) for G395H per detector. Outside these the
# throughput is ~0 and channels are pure noise.
G395H_WAVE_RANGES = {"NRS1": (2.87, 3.72), "NRS2": (3.82, 5.18)}

# Tilt-event detection: frozen survey-wide.
TILT_WINDOW = 15              # integrations each side of candidate step
TILT_THRESHOLD = 6.0          # robust sigma of window-median differences
TILT_MIN_SEPARATION = 30      # integrations between distinct events


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
        edges = np.asarray(e)

    centers, half_widths, fbins, ebins = [], [], [], []
    for i in range(len(edges) - 1):
        m = good_col & (wave >= edges[i]) & (wave < edges[i + 1])
        if not m.any():
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


# -------------------- tilt events --------------------


def detect_tilt_events(
    flux: np.ndarray,
    *,
    window: int = TILT_WINDOW,
    threshold: float = TILT_THRESHOLD,
    min_separation: int = TILT_MIN_SEPARATION,
    exclude_mask: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Detect step discontinuities (mirror tilt events) in a lightcurve.

    At each interior integration i, compare the median of ``window``
    points after i to the median of ``window`` points before; a step
    exceeding ``threshold`` robust sigma (MAD of the difference series)
    is an event. ``exclude_mask`` (True = ignore, e.g. in-transit points)
    keeps ingress/egress from being flagged as steps.

    Returns a list of ``{"index", "amplitude"}`` (amplitude in relative
    flux, positive = brightening), strongest first, merged so events are
    at least ``min_separation`` apart.
    """
    f = np.asarray(flux, dtype=float).copy()
    if exclude_mask is not None:
        f[~np.asarray(exclude_mask, dtype=bool)] = np.nan

    n = f.size
    diff = np.full(n, np.nan)
    import warnings

    with warnings.catch_warnings():
        # Windows fully inside the excluded (in-transit) span are all-NaN
        # by construction; their diff stays NaN and is ignored.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in range(window, n - window):
            pre = np.nanmedian(f[i - window: i])
            post = np.nanmedian(f[i: i + window])
            diff[i] = post - pre

    med = np.nanmedian(diff)
    mad = np.nanmedian(np.abs(diff - med))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma == 0:
        return []

    candidates = np.where(np.abs(diff - med) > threshold * sigma)[0]
    events: list[dict[str, float]] = []
    for idx in candidates[np.argsort(-np.abs(diff[candidates] - med))]:
        if any(abs(idx - e["index"]) < min_separation for e in events):
            continue
        events.append({"index": int(idx), "amplitude": float(diff[idx] - med)})
    return sorted(events, key=lambda e: e["index"])


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


def build_regressor_matrix(
    time: np.ndarray,
    diagnostics: dict[str, np.ndarray] | None = None,
    tilt_events: list[dict[str, float]] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Assemble the juliet ``linear_regressors_lc`` matrix.

    Columns (each standardized to zero median / unit robust scale, so the
    frozen theta priors are comparable across targets):
    time slope, [trace x, trace y, FWHM], [one step per tilt event].
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
    return np.column_stack(cols), names


# -------------------- orchestral tool --------------------


class DetectNirspecTiltEvents(BaseTool):
    """
    Scan a reduced NIRSpec/G395H visit for mirror tilt events — sudden
    step discontinuities in the white lightcurve caused by primary-mirror
    segment tilts. Undetected tilt events bias the transit depth; the
    juliet fitting tools correct them with a fitted step term.

    Input is an exoTEDRF ``*_spectra_fullres.fits`` product. In-transit
    points are excluded from the step search using the planet ephemeris
    (period, t0, duration), so real ingress/egress is never flagged.

    Reports each event's integration index, time, and amplitude in ppm.
    The same detection runs automatically inside
    ``FitNirspecG395hWhiteLight`` — use this tool to inspect a visit
    before fitting, or to diagnose a suspicious lightcurve.

    Example
    -------
        DetectNirspecTiltEvents(
            spectra_path="reductions/GJ_9827_d/o010/nrs1/.../..._box_spectra_fullres.fits",
            detector="NRS1",
            period=6.20146980, t0_ref=2457740.96115, duration_hr=1.28,
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
        events = detect_tilt_events(lc["wl_flux"], exclude_mask=lc["oot_mask"])
        lines = [
            f"Tilt-event scan ({self.detector}): {len(events)} event(s); "
            f"OOT scatter {lc['oot_scatter_ppm']:.0f} ppm.",
        ]
        for e in events:
            t = lc["time"][e["index"]]
            lines.append(
                f"  index {e['index']} (BJD {t:.5f}, "
                f"{(t - lc['t0_obs']) * 24:+.2f} h from mid-transit): "
                f"step {e['amplitude'] * 1e6:+.0f} ppm"
            )
        if not events:
            lines.append("  No step discontinuities above threshold.")
        else:
            lines.append(
                "The juliet fit tools will include one step regressor per "
                "event (tilt_correction=True, the default)."
            )
        return "\n".join(lines)
