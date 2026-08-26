"""Patchwork stellar contamination — Stages 5.5 and 6.5.

Two physically distinct problems share the name "stellar contamination",
and Patchwork handles them at two different points in the chain:

**Stage 5.5 — occulted heterogeneities (spot / facula crossings).**
The planet crosses a starspot, the blocked flux is fainter than the mean
photosphere, and the lightcurve shows a positive bump *inside* transit.
This is a per-visit, per-integration problem: it distorts the transit
shape, inflates the red-noise beta, and biases the depth. It is only
visible once a clean transit model exists, so the scan runs on the
Stage 5 white-light residuals — after the white fit, before the
spectroscopic fits. Detected crossings are masked (the survey default;
see the module note below), and the masked lightcurve is refit.

**Stage 6.5 — unocculted heterogeneities (the transit light source
effect).** Spots and faculae *outside* the transit chord never appear in
the lightcurve at all, but they make the disk-integrated spectrum differ
from the spectrum of the transit chord, multiplying every measured depth
by a wavelength-dependent factor. This is a per-spectrum problem, so it
runs after visit combination on the final transmission spectrum.

Why masking rather than modelling, for the survey
-------------------------------------------------
Referee feedback on Patchwork 1 (2026-08-03): *"for a wholesale analysis
like yours, masking starspots seems a reasonable compromise between
expediency and accuracy."* Spot modelling (SPOTROD) adds >= 4 free
parameters per crossing per visit, is degenerate with limb darkening at
G395H precision, and cannot be made uniform across a heterogeneous
survey. Masking is deterministic, uniform, and costs only integrations.
The modelling path stays available (``fit_spot_crossing_spotrod``) for
individual targets where a crossing is too large to mask away.

Everything in the detection/masking half is plain numpy — no juliet, no
exoTEDRF — so it runs in the main ASTER environment. The Stage 6.5
retrieval additionally needs ``emcee``. SPOTROD, stctm and SAGE are
optional external backends, probed at call time and never imported at
module scope.
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
        """Fallback that keeps plain functions importable without Orchestral."""

    def RuntimeField(default=None, description=None):
        return default

    def StateField(default=None, description=None):
        return default


# Frozen survey-wide contamination settings. Bump the version if any
# change: the Stage 5.5 mask changes which integrations enter the fit,
# so it is part of the survey definition exactly like the Stage 4 binning.
PATCHWORK_CONTAM_VERSION = "1.0"

# --- Stage 5.5 anomaly detection ---------------------------------------
# The running-mean window is the timescale a spot crossing occupies. A
# spot of radius r_spot/R* takes roughly (2 r_spot / v_chord) to cross;
# for a sub-Neptune on an M dwarf that is tens of minutes, i.e. tens of
# integrations at the few-second cadence of a NIRSpec BOTS TSO. 15 is
# the same window used for tilt steps, and keeps a single timescale
# across both diagnostics.
ANOMALY_WINDOW = 15           # integrations in the centred running mean
ANOMALY_THRESHOLD = 3.0       # robust sigma of the SMOOTHED residual
ANOMALY_MIN_RUN = 5           # consecutive flagged integrations required
ANOMALY_MASK_PAD = 5          # integrations masked either side of a run
# Flagged runs whose peaks fall within this many integrations are one
# event. Removing the long-window trend splits a single feature into
# lobes: a step becomes an antisymmetric -/+ pair (the trend ramps
# across the transition, so each side reads ~0.49x the step height), and
# a strong bump grows shallow negative wings. Treated as separate events
# they are reported at the wrong amplitude and the wrong sign, and a
# wing next to a real crossing would be masked as a crossing of its own.
# The lobes can only live inside the detrend window, so merging within
# that window collects them; two genuinely distinct crossings are much
# further apart (TOI-1231 b's pair sit ~1 h, i.e. ~180 integrations).
ANOMALY_MERGE_RADIUS = 6 * ANOMALY_WINDOW
# Two detectors read the same photons at the same time, so a stellar or
# telescope event lands at the same clock time in both. Matching is on
# the OVERLAP of the two flagged spans; this tolerance is only the slack
# allowed between spans that fail to touch (each span is systematically
# narrower than the event by the run threshold, so two detections of one
# event can leave a small gap). It is NOT applied to t_peak: the peak of
# the detrended statistic can land on either lobe of a broad feature and
# wander tens of minutes between detectors, which is why peak-time
# matching missed two thirds of injected coincident crossings.
ANOMALY_MATCH_TOL_MIN = 5.0
# A spot is cooler than the photosphere, so its contrast — and therefore
# the crossing amplitude — falls towards the infrared. NRS2 (3.8-5.2 um)
# should show a *smaller* bump than NRS1 (2.9-3.7 um). A ratio at or
# above this is achromatic and points at the instrument, not the star.
ACHROMATIC_RATIO = 0.9

_KIND_SPOT = "spot_crossing"
_KIND_FACULA = "facula_crossing"
_KIND_STEP = "step"
_KIND_UNCONFIRMED = "unconfirmed"


# -------------------- residual smoothing --------------------


def running_mean(x: np.ndarray, window: int = ANOMALY_WINDOW) -> np.ndarray:
    """Centred, NaN-aware running mean.

    The window is forced odd so the output stays aligned with the input
    (an even window biases every feature half an integration early).
    Positions with fewer than half a window of finite points — the two
    ends of the series — return NaN rather than a noisy partial average,
    which would otherwise be the most likely place to raise a false
    anomaly.
    """
    x = np.asarray(x, dtype=float)
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    if w == 1:
        return x.copy()

    good = np.isfinite(x)
    kern = np.ones(w)
    num = np.convolve(np.where(good, x, 0.0), kern, mode="same")
    den = np.convolve(good.astype(float), kern, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den >= w / 2.0, out, np.nan)


def _robust_sigma(values: np.ndarray) -> float:
    """MAD-based sigma, immune to the very outliers being searched for."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return float("nan")
    mad = np.median(np.abs(v - np.median(v)))
    return float(1.4826 * mad)


def _consecutive_runs(flagged: np.ndarray) -> list[tuple[int, int]]:
    """[(start, end)] inclusive index ranges of consecutive True values."""
    idx = np.flatnonzero(np.asarray(flagged, dtype=bool))
    if idx.size == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1) + 1
    return [(int(g[0]), int(g[-1])) for g in np.split(idx, splits) if g.size]


# -------------------- Stage 5.5 detection --------------------


def detect_lightcurve_anomalies(
    time: np.ndarray,
    residual: np.ndarray,
    *,
    oot_mask: np.ndarray | None = None,
    window: int = ANOMALY_WINDOW,
    threshold: float = ANOMALY_THRESHOLD,
    min_run: int = ANOMALY_MIN_RUN,
    min_separation: int = ANOMALY_MERGE_RADIUS,
    detector: str = "",
) -> dict[str, Any]:
    """Find correlated excursions in a white-light fit residual series.

    ``residual`` is (flux - full transit+systematics model) from the
    Stage 5 white fit — so a *clean* transit, limb darkening, the linear
    regressors and any fitted tilt steps have already been removed, and
    what is left is either noise or something the model does not
    describe.

    Method, following the referee's prescription:

    1. Smooth the residuals with a centred ``window``-integration running
       mean. A spot crossing is coherent over tens of integrations, so
       smoothing raises it above the white noise; uncorrelated noise
       averages down as sqrt(window).
    2. Set the significance scale from the MAD of the *smoothed*
       residuals in the out-of-transit baseline. Using the smoothed
       series rather than sigma/sqrt(window) means any residual red
       noise is already folded into the threshold, so a target with
       correlated systematics does not flag its whole transit.
    3. Flag runs of at least ``min_run`` consecutive integrations beyond
       ``threshold`` sigma.

    ``oot_mask`` is a KEEP-style out-of-transit mask (True = outside
    transit), the same convention as ``lightcurves.build_lightcurves``.
    It sets the noise scale and labels each run in/out of transit; the
    search itself runs over the whole series, because an event can
    straddle ingress.

    Returns a report dict with ``anomalies`` (a list, strongest first)
    and the detection metadata. Each anomaly carries::

        index_start, index_end, n_integrations, sign,
        amplitude_ppm      mean raw residual over the run
        peak_sigma         largest |smoothed residual| / sigma in the run
        duration_min       wall-clock length of the run
        t_peak             BJD of the peak integration
        hours_from_mid     t_peak relative to mid-transit (needs oot_mask)
        in_transit_frac    fraction of the run inside transit
        persistent         True if the flux level does not return after
                           the run (step-like, i.e. tilt rather than spot)
        kind               single-detector provisional classification
    """
    t = np.asarray(time, dtype=float)
    r = np.asarray(residual, dtype=float)
    if t.size != r.size:
        raise ValueError(f"time ({t.size}) and residual ({r.size}) differ in length")

    n = r.size
    in_transit = (
        ~np.asarray(oot_mask, dtype=bool) if oot_mask is not None
        else np.zeros(n, dtype=bool)
    )

    smoothed = running_mean(r, window)
    oot_sel = (np.asarray(oot_mask, dtype=bool) if oot_mask is not None
               else np.ones(n, dtype=bool))

    # The search runs on the smoothed residuals with a LONG-WINDOW trend
    # removed, and both halves of that matter.
    #
    # Smoothing raises a coherent excursion above the white noise. But
    # searching the smoothed series against its out-of-transit median
    # breaks on the case that matters most, a tilt step part-way through
    # a visit: the out-of-transit points then sit at two different
    # levels, the median lands between them, and BOTH halves of the
    # visit read as large excursions while the step itself — the actual
    # event — is never localized. Removing a 6x-window running mean
    # makes any such level shift show up where it belongs, as a compact
    # feature at the transition, and leaves the two flat stretches at
    # zero. Correlated noise on the timescale of a crossing survives the
    # detrending, so it still enters the significance scale, which is
    # the reason for working with the smoothed series rather than
    # dividing white noise by sqrt(window).
    #
    # The cost is deliberate: an excursion LONGER than ~6 windows is
    # suppressed along with the trend. At the survey defaults that is
    # ~30 minutes of a NIRSpec TSO; a stellar feature broader than that
    # is not a spot crossing but a change in the transit itself.
    trend = running_mean(smoothed, 6 * window)
    detrended = smoothed - trend
    baseline = 0.0
    # Detrending also removes ~1/6 of the variance (var(s - <s>_6w) =
    # var(s)(1 - 1/6) for noise correlated on the window scale). Undo it
    # so "3 sigma" still means 3 sigma and the threshold is not quietly 2.7.
    sigma = _robust_sigma(detrended[oot_sel]) / np.sqrt(5.0 / 6.0)
    if not np.isfinite(sigma) or sigma <= 0:
        return {
            "detector": detector.upper(),
            "contam_version": PATCHWORK_CONTAM_VERSION,
            "n_integrations": int(n),
            "sigma_smoothed_ppm": float("nan"),
            "anomalies": [],
            "note": "Could not establish a noise scale (too few finite "
                    "out-of-transit residuals) — no scan performed.",
        }

    z = (detrended - baseline) / sigma
    runs = [rr for rr in _consecutive_runs(np.abs(z) > threshold)
            if rr[1] - rr[0] + 1 >= min_run]

    # Collect the lobes of one feature into one event (see
    # ANOMALY_MERGE_RADIUS). Strongest run first, so the merged span
    # grows outwards from the real peak rather than from a wing.
    def _peak_of(i0: int, i1: int) -> int:
        return int(i0 + int(np.nanargmax(np.abs(z[i0: i1 + 1]))))

    runs.sort(key=lambda rr: -abs(z[_peak_of(*rr)]))
    merged_runs: list[list[int]] = []
    for i0, i1 in runs:
        p = _peak_of(i0, i1)
        for span in merged_runs:
            if abs(p - span[2]) < int(min_separation):
                span[0] = min(span[0], i0)
                span[1] = max(span[1], i1)
                break
        else:
            merged_runs.append([i0, i1, p])
    runs = [(s[0], s[1]) for s in sorted(merged_runs)]

    # High-frequency scatter, for the persistence test below. Same
    # reasoning as sigma: the MAD of the raw out-of-transit residuals is
    # contaminated by any level shift, so measure it about the local mean.
    sigma_raw = _robust_sigma((r - smoothed)[oot_sel])
    t0_obs = float(np.nanmedian(t[in_transit])) if in_transit.any() else float("nan")

    anomalies: list[dict[str, Any]] = []
    for i0, i1 in runs:
        peak = int(i0 + int(np.nanargmax(np.abs(z[i0: i1 + 1]))))
        # Step-like vs transient: does the flux level differ after the
        # event from before it? A spot crossing returns to baseline; a
        # mirror tilt does not. This is what separates a Stage 5.5
        # stellar event from a leftover tilt event that the
        # out-of-transit-only tilt search could not see.
        #
        # The reference windows stand off by 2x the smoothing window
        # rather than butting against the flagged span. The span comes
        # from the detrended series and does not track the feature's
        # true extent — for a negative bump the positive wing can be the
        # strongest lobe and drag the span off-centre — so an adjacent
        # window lands *inside* the event, and the measured "shift" is
        # then the event's own depth with the sign flipped.
        standoff, ref_len = 2 * window, 2 * window
        pre = r[max(0, i0 - standoff - ref_len): max(0, i0 - standoff)]
        post = r[i1 + standoff: i1 + standoff + ref_len]
        have_refs = pre.size >= 3 and post.size >= 3
        shift = (float(np.nanmedian(post) - np.nanmedian(pre))
                 if have_refs else 0.0)
        local_base = (0.5 * float(np.nanmedian(pre) + np.nanmedian(post))
                      if have_refs else 0.0)
        transient_amp = float(np.nanmean(r[i0: i1 + 1]) - local_base)
        # Purely a significance test on the level change: a transient
        # gives shift ~ 0 whatever its amplitude. Comparing the shift to
        # the event amplitude as well would misfire, because the merged
        # span dilutes the amplitude by including the wings.
        sigma_shift = sigma_raw * np.sqrt(2.0 / max(1, min(pre.size, post.size)))
        persistent = bool(have_refs and abs(shift) > 4 * sigma_shift)
        # A step's amplitude is its level shift, not the mean over the
        # flagged span: that span covers both lobes of the detrended
        # feature, so its mean sits halfway up the step.
        amp = shift if persistent else transient_amp
        frac_in = float(np.mean(in_transit[i0: i1 + 1])) if in_transit.any() else 0.0

        if persistent:
            kind = _KIND_STEP
        elif frac_in > 0.5 and amp > 0:
            kind = _KIND_SPOT
        elif frac_in > 0.5 and amp < 0:
            kind = _KIND_FACULA
        else:
            kind = _KIND_UNCONFIRMED

        anomalies.append({
            "detector": detector.upper(),
            "index_start": int(i0),
            "index_end": int(i1),
            "n_integrations": int(i1 - i0 + 1),
            "sign": int(np.sign(amp)) or 1,
            "amplitude_ppm": amp * 1e6,
            # Height of the smoothed excursion at its peak. The mean over
            # the run is NOT comparable between detectors: a weaker bump
            # clears the threshold over fewer integrations, so its run
            # average sits closer to its own peak and the NRS2/NRS1 ratio
            # comes out biased towards 1 — i.e. towards calling a real
            # chromatic spot crossing achromatic. The peak height has no
            # such run-length dependence, so the ratio uses it.
            "peak_amplitude_ppm": float(detrended[peak] - baseline) * 1e6,
            "peak_sigma": float(abs(z[peak])),
            "duration_min": float((t[i1] - t[i0]) * 24 * 60),
            "t_peak": float(t[peak]),
            # Wall-clock extent of the flagged run. Cross-detector
            # confirmation matches on the overlap of these intervals,
            # not on t_peak: the peak of the detrended statistic can
            # land on either lobe of a broad feature and wander tens of
            # minutes between detectors, while the flagged span tracks
            # the feature itself.
            "t_start": float(t[i0]),
            "t_end": float(t[i1]),
            "hours_from_mid": (float((t[peak] - t0_obs) * 24)
                               if np.isfinite(t0_obs) else None),
            "in_transit_frac": frac_in,
            "persistent": persistent,
            "kind": kind,
            "confirmed": False,
        })

    anomalies.sort(key=lambda a: -a["peak_sigma"])
    return {
        "detector": detector.upper(),
        "contam_version": PATCHWORK_CONTAM_VERSION,
        "n_integrations": int(n),
        "window": int(window),
        "threshold_sigma": float(threshold),
        "min_run": int(min_run),
        "min_separation": int(min_separation),
        "sigma_smoothed_ppm": sigma * 1e6,
        "sigma_raw_ppm": sigma_raw * 1e6,
        "anomalies": anomalies,
    }


def remap_anomaly_report(
    report: dict[str, Any],
    index: np.ndarray,
    n_total: int,
) -> dict[str, Any]:
    """Map a scan's indices from a masked series back to original coords.

    The white fit that writes ``white_lightcurve_residuals.npz`` may have
    been masked (the tilt transition mask in pass 1), in which case its
    ``time``/``residual`` arrays are COMPRESSED and the scan's
    ``index_start``/``index_end`` count integrations of the compressed
    series. A mask built from those indices and applied to the full
    Stage 4 lightcurve lands early by however many integrations were
    dropped before the event — up to 7 per prior tilt event, which the
    mask padding cannot be relied on to absorb.

    ``index`` is the npz's ``index`` array (original integration number
    of each kept point) and ``n_total`` the full lightcurve length.
    Returns the report with every anomaly's indices remapped in place
    and ``n_integrations`` set to ``n_total``, so ``anomaly_keep_mask``
    lines up with the unmasked lightcurve. Times are untouched — they
    were always correct.
    """
    idx = np.asarray(index, dtype=int)
    for a in report.get("anomalies", []):
        a["index_start"] = int(idx[int(a["index_start"])])
        a["index_end"] = int(idx[int(a["index_end"])])
    report["n_integrations"] = int(n_total)
    return report


def match_detector_anomalies(
    reports: dict[str, dict[str, Any]],
    *,
    tol_min: float = ANOMALY_MATCH_TOL_MIN,
    achromatic_ratio: float = ACHROMATIC_RATIO,
) -> list[dict[str, Any]]:
    """Cross-confirm anomalies between NRS1 and NRS2.

    ``reports`` maps detector name -> the dict from
    ``detect_lightcurve_anomalies``. The two detectors record the same
    photons at the same times through independent readout chains, which
    makes coincidence the single most useful discriminator available.
    Coincidence is judged on the overlap of the flagged spans
    (``t_start``..``t_end``), with ``tol_min`` of slack for spans that
    narrowly fail to touch — never on ``t_peak``, whose position on a
    broad detrended feature is noise-driven (see ANOMALY_MATCH_TOL_MIN).

    - **Both detectors, transient, in transit, positive** -> a real
      occulted spot crossing. Its amplitude should also *drop* towards
      the infrared, because spot contrast does; the returned
      ``amplitude_ratio`` (NRS2/NRS1) records this and a ratio above
      ``achromatic_ratio`` sets ``achromatic: True``, which argues for an
      instrumental origin despite the coincidence.
    - **Both detectors, persistent** -> a tilt event, not a stellar one.
      Reported as ``kind='step'`` so it is routed to the tilt handling.
    - **One detector only** -> detector-level systematics. Never masked
      as a stellar event; masking it would silently remove real
      integrations from one half of the spectrum and not the other,
      breaking the NRS1/NRS2 offset.

    Returns one entry per matched or unmatched event, strongest first.
    """
    # Order matters: ``amplitude_ratio`` is defined as the redder
    # detector over the bluer one, so NRS1 must come first whatever
    # order the caller's dict happens to be in.
    known = [d for d in ("NRS1", "NRS2")
             if d in reports and reports[d].get("anomalies") is not None]
    other = sorted(d for d in reports
                   if d not in ("NRS1", "NRS2")
                   and reports[d].get("anomalies") is not None)
    dets = known + other
    tol_days = tol_min / (24 * 60)
    merged: list[dict[str, Any]] = []
    used: set[tuple[str, int]] = set()

    def _overlap_days(a: dict[str, Any], b: dict[str, Any]) -> float:
        """Overlap of the two flagged spans (negative = gap between them).

        Falls back to peak separation for anomaly dicts predating
        ``t_start``/``t_end`` (an old anomaly_report.json re-read)."""
        if all(k in a and k in b for k in ("t_start", "t_end")):
            return (min(a["t_end"], b["t_end"])
                    - max(a["t_start"], b["t_start"]))
        return -abs(a["t_peak"] - b["t_peak"])

    if len(dets) >= 2:
        a_det, b_det = dets[0], dets[1]
        for i, a in enumerate(reports[a_det]["anomalies"]):
            best, best_ov = None, None
            for j, b in enumerate(reports[b_det]["anomalies"]):
                if (b_det, j) in used:
                    continue
                ov = _overlap_days(a, b)
                if ov >= -tol_days and (best_ov is None or ov > best_ov):
                    best, best_ov = j, ov
            if best is None:
                continue
            b = reports[b_det]["anomalies"][best]
            best_dt = abs(a["t_peak"] - b["t_peak"])
            used.add((a_det, i))
            used.add((b_det, best))

            amp_a, amp_b = a["amplitude_ppm"], b["amplitude_ppm"]
            peak_a, peak_b = a["peak_amplitude_ppm"], b["peak_amplitude_ppm"]
            ratio = (abs(peak_b) / abs(peak_a)) if peak_a else float("nan")
            persistent = a["persistent"] or b["persistent"]
            same_sign = np.sign(amp_a) == np.sign(amp_b)
            if persistent:
                kind = _KIND_STEP
            elif not same_sign:
                kind = _KIND_UNCONFIRMED
            elif max(a["in_transit_frac"], b["in_transit_frac"]) > 0.5:
                kind = _KIND_SPOT if amp_a > 0 else _KIND_FACULA
            else:
                kind = _KIND_UNCONFIRMED

            merged.append({
                "kind": kind,
                "confirmed": kind in (_KIND_SPOT, _KIND_FACULA, _KIND_STEP),
                "detectors": {a_det: a, b_det: b},
                "t_peak": 0.5 * (a["t_peak"] + b["t_peak"]),
                "delta_t_min": float(best_dt * 24 * 60),
                "span_overlap_min": float(best_ov * 24 * 60),
                "hours_from_mid": a["hours_from_mid"],
                "amplitude_ppm": 0.5 * (amp_a + amp_b),
                "peak_amplitude_ppm": 0.5 * (peak_a + peak_b),
                "amplitude_ratio": float(ratio),
                "achromatic": bool(np.isfinite(ratio) and ratio >= achromatic_ratio),
                "peak_sigma": max(a["peak_sigma"], b["peak_sigma"]),
                "in_transit_frac": max(a["in_transit_frac"], b["in_transit_frac"]),
                "persistent": persistent,
            })

    for det in dets:
        for i, a in enumerate(reports[det]["anomalies"]):
            if (det, i) in used:
                continue
            merged.append({
                # Seen in one detector only: whatever the shape, the star
                # cannot produce it in one half of the spectrograph.
                "kind": _KIND_UNCONFIRMED,
                "confirmed": False,
                "detectors": {det: a},
                "t_peak": a["t_peak"],
                "delta_t_min": None,
                "hours_from_mid": a["hours_from_mid"],
                "amplitude_ppm": a["amplitude_ppm"],
                "peak_amplitude_ppm": a["peak_amplitude_ppm"],
                "amplitude_ratio": None,
                "achromatic": None,
                "peak_sigma": a["peak_sigma"],
                "in_transit_frac": a["in_transit_frac"],
                "persistent": a["persistent"],
                "single_detector": True,
            })

    merged.sort(key=lambda m: -m["peak_sigma"])
    return merged


def anomaly_keep_mask(
    n: int,
    anomalies: list[dict[str, Any]],
    *,
    detector: str,
    pad: int = ANOMALY_MASK_PAD,
    kinds: tuple[str, ...] = (_KIND_SPOT, _KIND_FACULA),
) -> np.ndarray:
    """KEEP mask (True = use this integration) for one detector.

    Only events whose ``kind`` is in ``kinds`` are masked — by default
    confirmed spot and facula crossings. Tilt steps are *not* masked:
    they are fitted with a step regressor, which preserves the
    integrations. Unconfirmed single-detector excursions are not masked
    either, for the reason in ``match_detector_anomalies``.

    ``pad`` integrations are removed either side of each run. The
    running mean smears a crossing's edges inwards by half a window, so
    the flagged run is systematically *narrower* than the event; padding
    covers the wings that would otherwise stay in the fit and pull the
    depth in exactly the direction the mask is meant to prevent.

    The masked span is the UNION of the event's flagged runs across
    detectors, so both detectors lose the same integrations. A spot
    crossing is a property of the star, so it contaminates NRS2 even
    where NRS2's own lower-amplitude detection clears threshold over a
    narrower run; masking each detector by its own run alone would leave
    contaminated integrations in the redder half and put the two
    detectors' depths on different data, which is exactly what the
    NRS1-NRS2 offset is supposed to be measuring against.
    """
    keep = np.ones(int(n), dtype=bool)
    for event in anomalies:
        if event.get("kind") not in kinds or not event.get("confirmed", False):
            continue
        entries = (event.get("detectors") or {})
        if detector.upper() not in entries:
            continue
        i0 = min(int(e["index_start"]) for e in entries.values())
        i1 = max(int(e["index_end"]) for e in entries.values())
        i0 = max(0, i0 - int(pad))
        i1 = min(int(n) - 1, i1 + int(pad))
        keep[i0: i1 + 1] = False
    return keep


def summarize_anomalies(merged: list[dict[str, Any]]) -> str:
    """Human-readable one-block summary of a cross-matched anomaly list."""
    if not merged:
        return "No anomalies above threshold in either detector."
    lines = []
    for m in merged:
        dets = "+".join(sorted(m["detectors"]))
        where = (f"{m['hours_from_mid']:+.2f} h from mid-transit"
                 if m.get("hours_from_mid") is not None else "unknown phase")
        line = (f"  [{m['kind']}] {dets}: {m['amplitude_ppm']:+.0f} ppm, "
                f"{m['peak_sigma']:.1f} sigma, {where}")
        if m.get("amplitude_ratio") is not None and np.isfinite(m["amplitude_ratio"]):
            line += f", NRS2/NRS1 amplitude {m['amplitude_ratio']:.2f}"
            if m.get("achromatic"):
                line += " (achromatic — suspect instrumental)"
        if m.get("single_detector"):
            line += " (single detector — NOT masked)"
        lines.append(line)
    return "\n".join(lines)


# -------------------- Stage 5.5 diagnostic figure --------------------


def plot_anomaly_diagnostic(
    reports: dict[str, dict[str, Any]],
    series: dict[str, dict[str, np.ndarray]],
    merged: list[dict[str, Any]],
    out_dir: str | os.PathLike[str],
    *,
    title: str = "",
    stem: str = "anomaly_scan",
) -> str:
    """Residuals per detector with the search statistic overlaid, flagged
    spans shaded, and the detection threshold drawn. Written as PDF and SVG.

    The orange curve is the *detrended* smoothed residual — the series
    the threshold is actually applied to — not a plain running mean. A
    figure showing the plain running mean would not explain its own
    shading: the whole point of removing the long-window trend is that a
    step reads as a compact feature rather than as two offset levels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .juliet import _PLOT_STYLE, _savefig, DATA_COLOR, MODEL_COLOR

    dets = [d for d in ("NRS1", "NRS2") if d in series] or sorted(series)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(_PLOT_STYLE):
        fig, axes = plt.subplots(len(dets), 1, figsize=(9, 3.1 * len(dets)),
                                 sharex=True, squeeze=False)
        for ax, det in zip(axes[:, 0], dets):
            s = series[det]
            t_hr = (s["time"] - np.nanmedian(s["time"])) * 24
            ax.plot(t_hr, s["residual"] * 1e6, ".", ms=1.6, alpha=0.35,
                    color=DATA_COLOR, label="residual")
            window = int(reports.get(det, {}).get("window") or ANOMALY_WINDOW)
            smoothed = running_mean(s["residual"], window)
            detrended = smoothed - running_mean(smoothed, 6 * window)
            ax.plot(t_hr, detrended * 1e6, "-", lw=1.6, color=MODEL_COLOR,
                    label=f"detrended running mean ({window} ints)")
            sig = reports.get(det, {}).get("sigma_smoothed_ppm")
            thr = float(reports.get(det, {}).get("threshold_sigma")
                        or ANOMALY_THRESHOLD)
            if sig and np.isfinite(sig):
                for k in (-thr, thr):
                    ax.axhline(k * sig, ls=":", lw=0.9, color="#8A94A6")
            # Shading works in TIME, not index: event indices are in
            # original integration numbers (see remap_anomaly_report)
            # while this series may be the compressed pass-1 arrays, so
            # indexing t_hr with them would land the shade off the event.
            cad_hr = float(np.nanmedian(np.diff(s["time"]))) * 24
            t_med = np.nanmedian(s["time"])
            for m in merged:
                entries = m.get("detectors") or {}
                if det not in entries:
                    continue
                # Shade what is actually removed: the union span plus the
                # padding, for a masked event. Shading this detector's own
                # run would show a narrower region than the mask covers.
                masked = m["confirmed"] and m["kind"] in (_KIND_SPOT, _KIND_FACULA)
                if masked:
                    lo = (min(e.get("t_start", e["t_peak"])
                              for e in entries.values())
                          - ANOMALY_MASK_PAD * cad_hr / 24)
                    hi = (max(e.get("t_end", e["t_peak"])
                              for e in entries.values())
                          + ANOMALY_MASK_PAD * cad_hr / 24)
                    shade = "#D9534F"
                else:
                    e = entries[det]
                    lo = e.get("t_start", e["t_peak"])
                    hi = e.get("t_end", e["t_peak"])
                    shade = "#B9C2CC"
                ax.axvspan((lo - t_med) * 24, (hi - t_med) * 24,
                           color=shade, alpha=0.18, lw=0)
            ax.axhline(0, lw=0.8, color="#8A94A6")
            ax.set_ylabel(f"{det} residual [ppm]")
            ax.legend(loc="upper right", frameon=False, fontsize=8)
        axes[-1, 0].set_xlabel("Time from visit centre  [h]")
        if title:
            axes[0, 0].set_title(title)
        fig.tight_layout()
        return _savefig(fig, out, stem)


# -------------------- optional external backends --------------------


def _probe(module: str) -> tuple[bool, str]:
    import importlib

    try:
        m = importlib.import_module(module)
    except Exception as exc:  # ImportError, or a broken install
        return False, f"{type(exc).__name__}: {exc}"
    return True, str(getattr(m, "__version__", "installed"))


def contamination_backends() -> dict[str, dict[str, str | bool]]:
    """Availability of the three optional contamination backends.

    None of these is required: Stage 5.5 masking and the Stage 6.5
    blackbody retrieval are self-contained. They are cross-checks and
    escape hatches for individual targets.

    - **spotrod** (Béky, Kocsis & Holman 2014) — semi-analytic occulted
      spot model. Used by the published TOI-1231 b analysis (Sarkar et
      al. 2026) to model, rather than mask, the crossings Patchwork
      masks; the comparison is the reason Patchwork's TOI-1231 b depth
      can be checked against theirs.
    - **stctm** (Piaulet-Ghorayeb) — unocculted spot/facula
      contamination retrieval on a transmission spectrum. This is the
      reference implementation of what Stage 6.5 does; run it against
      the built-in retrieval on any target where contamination is
      claimed.
    - **sage** — spot/facula forward modelling of contamination spectra,
      the independent third opinion.

    Set ``ASTER_STCTM_CMD`` / ``ASTER_SAGE_CMD`` to run those two as
    external commands (they are config-file driven) when they are not
    importable in the ASTER environment.
    """
    report: dict[str, dict[str, str | bool]] = {}
    for name in ("spotrod", "stctm", "sage"):
        ok, detail = _probe(name)
        report[name] = {"available": ok, "detail": detail}
    for name, var in (("stctm", "ASTER_STCTM_CMD"), ("sage", "ASTER_SAGE_CMD")):
        cmd = os.environ.get(var)
        if cmd:
            report[name]["external_command"] = cmd
            report[name]["available"] = True
    return report


def fit_spot_crossing_spotrod(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    period: float,
    t0: float,
    a_rs: float,
    rp_rs: float,
    inclination_deg: float,
    ld_u: tuple[float, float],
    spot_guesses: list[dict[str, float]],
    n_steps: int = 4000,
    n_walkers: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    """Model (rather than mask) occulted spot crossings with SPOTROD.

    This is the per-target escape hatch, NOT part of the frozen survey
    reduction — a fit that used it is not uniform with the rest of
    Patchwork and must be reported separately. Use it when a crossing is
    too large a fraction of the transit to mask, or to reproduce a
    published spot-modelled analysis.

    ``spot_guesses`` is one dict per spot with keys ``x``, ``y``
    (position in units of R*, in the frame where the transit chord runs
    along +x), ``radius`` (R*), ``contrast`` (spot intensity / photosphere
    intensity, < 1 for a dark spot). Each spot contributes 4 free
    parameters, so keep the count to what the data actually shows.

    Returns posterior medians per spot plus the best-fit model.
    Raises ``ModuleNotFoundError`` with an install hint if SPOTROD is
    absent — there is deliberately no silent fallback, because a
    "spot-modelled" result produced without a spot model would be a
    fabricated number.
    """
    try:
        import spotrod
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SPOTROD is not installed in this environment. It is optional: "
            "the survey path masks crossings instead (Stage 5.5). Install "
            "with `pip install spotrod` (needs a C compiler) to use the "
            "modelling path, or run this on Fir where it is available."
        ) from exc
    import emcee

    t = np.asarray(time, float)
    y = np.asarray(flux, float)
    e = np.asarray(flux_err, float)

    # Quadratic limb darkening on the standard annulus grid SPOTROD
    # integrates over. 1000 annuli is the package's own recommendation
    # and is far below the noise floor here.
    n_ann = 1000
    r = np.linspace(1.0 / (2 * n_ann), 1.0 - 1.0 / (2 * n_ann), n_ann)
    u1, u2 = ld_u
    mu = np.sqrt(1.0 - r**2)
    f_ld = 1.0 - u1 * (1 - mu) - u2 * (1 - mu) ** 2
    # Annulus areas, so the weighted profile integrates to the disk flux.
    edges = np.linspace(0.0, 1.0, n_ann + 1)
    f_w = f_ld * np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)

    incl = np.radians(inclination_deg)
    phase = 2 * np.pi * (t - t0) / period
    # Sky-plane track of the planet centre for a circular orbit.
    px = a_rs * np.sin(phase)
    py = a_rs * np.cos(phase) * np.cos(incl)
    z = np.sqrt(px**2 + py**2)

    keys = ("x", "y", "radius", "contrast")
    p0_spots = np.array([[g[k] for k in keys] for g in spot_guesses], float)
    n_spot = p0_spots.shape[0]

    def _model(theta: np.ndarray) -> np.ndarray:
        depth_scale = theta[0]
        spots = theta[1:].reshape(n_spot, 4)
        p = rp_rs * depth_scale
        planetangle = np.array([spotrod.circleangle(r, p, zi) for zi in z])
        return spotrod.integratetransit(
            px, py, z, p, r, f_w,
            spots[:, 0].copy(), spots[:, 1].copy(),
            spots[:, 2].copy(), spots[:, 3].copy(),
            planetangle,
        )

    def _log_prob(theta: np.ndarray) -> float:
        scale = theta[0]
        spots = theta[1:].reshape(n_spot, 4)
        if not 0.5 < scale < 1.5:
            return -np.inf
        if np.any(spots[:, 2] <= 0) or np.any(spots[:, 2] > 0.5):
            return -np.inf
        if np.any(spots[:, 3] < 0) or np.any(spots[:, 3] > 1.5):
            return -np.inf
        if np.any(np.hypot(spots[:, 0], spots[:, 1]) > 1.0):
            return -np.inf
        resid = y - _model(theta)
        return float(-0.5 * np.sum((resid / e) ** 2))

    p0 = np.concatenate([[1.0], p0_spots.ravel()])
    rng = np.random.default_rng(seed)
    start = p0 + 1e-4 * rng.standard_normal((n_walkers, p0.size))
    sampler = emcee.EnsembleSampler(n_walkers, p0.size, _log_prob)
    sampler.run_mcmc(start, n_steps, progress=False)
    chain = sampler.get_chain(discard=n_steps // 2, flat=True)

    med = np.median(chain, axis=0)
    lo, hi = np.percentile(chain, [16, 84], axis=0)
    spots_out = []
    for i in range(n_spot):
        s = slice(1 + 4 * i, 5 + 4 * i)
        spots_out.append({k: {"median": float(m), "minus": float(m - l),
                              "plus": float(h - m)}
                          for k, m, l, h in zip(keys, med[s], lo[s], hi[s])})
    return {
        "backend": "spotrod",
        "uniform_with_survey": False,
        "rp_rs": float(rp_rs * med[0]),
        "depth_ppm": float((rp_rs * med[0]) ** 2 * 1e6),
        "spots": spots_out,
        "model": _model(med),
        "residual_rms_ppm": float(np.std(y - _model(med)) * 1e6),
        "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
    }


# -------------------- Stage 6.5: unocculted heterogeneities ------------

# Planck constants in SI; the contamination factor only needs ratios of
# surface brightnesses, so the normalization cancels.
_H = 6.62607015e-34
_C = 2.99792458e8
_KB = 1.380649e-23


def planck(wave_um: np.ndarray, temperature: float) -> np.ndarray:
    """Planck surface brightness B_lambda (arbitrary units) at wave_um."""
    lam = np.asarray(wave_um, dtype=float) * 1e-6
    x = _H * _C / (lam * _KB * float(temperature))
    # expm1 keeps the Wien tail (large x, the G395H regime for cool
    # spots) from cancelling catastrophically.
    return 1.0 / (lam**5 * np.expm1(x))


def contamination_factor(
    wave_um: np.ndarray,
    *,
    t_phot: float,
    t_het: float,
    f_het: float,
    t_fac: float | None = None,
    f_fac: float = 0.0,
    stellar_spectra: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Transit light source contamination factor epsilon(lambda).

    The measured depth relates to the true one by

        D_obs(lambda) = D_true(lambda) * epsilon(lambda),
        epsilon = 1 / [1 - f_het (1 - S_het/S_phot)
                          - f_fac (1 - S_fac/S_phot)]

    (Rackham, Apai & Giampapa 2018, the "transit light source effect").
    ``f_het`` is the fraction of the *unocculted* stellar disk covered by
    spots at ``t_het``; ``f_fac`` the same for faculae at ``t_fac``.
    A cool spot (S_het < S_phot) gives epsilon > 1, deepening the
    measured transit at short wavelengths where the contrast is largest —
    a slope that mimics a hazy or H2-rich atmosphere.

    Surface brightnesses default to blackbodies, which is a deliberate
    first-order choice: it captures the continuum contrast that drives
    the slope but not the molecular features (notably water) that a
    PHOENIX or SPHINX grid puts into an M-dwarf spot spectrum. Pass
    ``stellar_spectra={'phot': ..., 'het': ..., 'fac': ...}`` — arrays
    already interpolated onto ``wave_um`` — to use a real grid, and
    record which was used. ``stctm`` with a PHOENIX grid is the
    cross-check.
    """
    wave = np.asarray(wave_um, dtype=float)
    if stellar_spectra:
        s_phot = np.asarray(stellar_spectra["phot"], dtype=float)
        s_het = np.asarray(stellar_spectra["het"], dtype=float)
        s_fac = (np.asarray(stellar_spectra["fac"], dtype=float)
                 if "fac" in stellar_spectra else None)
    else:
        s_phot = planck(wave, t_phot)
        s_het = planck(wave, t_het)
        s_fac = planck(wave, t_fac) if t_fac else None

    denom = 1.0 - f_het * (1.0 - s_het / s_phot)
    if s_fac is not None and f_fac:
        denom = denom - f_fac * (1.0 - s_fac / s_phot)
    # f_het -> 1 with a very dark spot drives the denominator to zero.
    # Clip rather than return infinities so a sampler walking into the
    # corner gets a large-but-finite likelihood penalty, not a NaN.
    denom = np.clip(denom, 1e-3, None)
    return 1.0 / denom


def retrieve_contamination(
    wave_um: np.ndarray,
    depth_ppm: np.ndarray,
    depth_err_ppm: np.ndarray,
    *,
    t_phot: float,
    t_phot_err: float = 100.0,
    f_het_max: float = 0.5,
    n_steps: int = 6000,
    n_walkers: int = 32,
    seed: int = 0,
    fit_faculae: bool = False,
) -> dict[str, Any]:
    """Fit an unocculted-heterogeneity model to a transmission spectrum.

    Model: a *flat* intrinsic spectrum times the contamination factor,

        D_obs(lambda) = D0 * epsilon(lambda; f_het, T_het, T_phot).

    A flat intrinsic spectrum is the null hypothesis on purpose. The
    question this answers is not "what is the atmosphere" but "can the
    structure in this spectrum be produced by the star alone" — so any
    preference the fit shows for contamination is an upper bound on how
    much of the signal is safe to attribute to the planet. Feeding the
    contamination-corrected spectrum into a retrieval that also fits
    composition is the next step, not this one.

    Free parameters: ``D0`` (ppm), ``f_het``, ``T_het``, and ``T_phot``
    with a Gaussian prior at the catalogue effective temperature.
    ``T_het`` is bounded to [0.5, 1.2] T_phot, covering spots down to the
    coldest observed on M dwarfs and mild faculae above; set
    ``fit_faculae`` to add a second component.

    Returns posteriors, the best-fit contamination factor, the
    corrected spectrum, and a BIC comparison against a flat line. Treat
    ``delta_bic > 10`` as the threshold for reporting contamination —
    anything less is not distinguishable from a constant depth at these
    error bars.

    **Do not quote f_het on its own.** ``f_het`` and ``T_het`` are
    strongly degenerate: a large area of mild spots and a small area of
    very cold ones produce nearly the same epsilon(lambda) over a single
    G395H octave. On injection tests the recovered pair can be far from
    the input (f=0.15, T=2900 K recovers as f=0.09, T=2350 K) while
    epsilon(lambda) and the corrected spectrum are right. What this fit
    constrains is the *shape*, so report epsilon(lambda), the corrected
    spectrum, and — when nothing is detected — the f_het upper limit at
    the fitted T_het. Breaking the degeneracy needs a bluer baseline
    than G395H provides, which is one concrete argument for the NIRISS
    SOSS overlap on the ten Patchwork targets that have it.

    A further limit of the 3-5 um window: blackbody spot contrast is
    weak there. A f_het = 0.15, T_het = 2900 K spot on a 3500 K M dwarf
    moves the depth by only ~20 ppm across the whole band, which
    Patchwork's ~25 ppm channel errors cannot see. Expect
    non-detections, and read them as "G395H alone cannot constrain
    this", not as "the star is quiet".
    """
    import emcee

    w = np.asarray(wave_um, dtype=float)
    d = np.asarray(depth_ppm, dtype=float)
    e = np.asarray(depth_err_ppm, dtype=float)
    good = np.isfinite(w) & np.isfinite(d) & np.isfinite(e) & (e > 0)
    w, d, e = w[good], d[good], e[good]
    if w.size < 6:
        raise ValueError(
            f"Only {w.size} usable channels — a contamination fit needs a "
            "spectrum, not a handful of points."
        )

    labels = ["D0_ppm", "f_het", "T_het", "T_phot"]
    p0 = [float(np.median(d)), 0.05, 0.8 * t_phot, float(t_phot)]
    if fit_faculae:
        labels += ["f_fac", "T_fac"]
        p0 += [0.02, 1.1 * t_phot]

    def _unpack(theta):
        d0, f_het, t_het, tp = theta[:4]
        f_fac, t_fac = (theta[4], theta[5]) if fit_faculae else (0.0, None)
        return d0, f_het, t_het, tp, f_fac, t_fac

    def _model(theta):
        d0, f_het, t_het, tp, f_fac, t_fac = _unpack(theta)
        return d0 * contamination_factor(
            w, t_phot=tp, t_het=t_het, f_het=f_het,
            t_fac=t_fac, f_fac=f_fac,
        )

    def _log_prob(theta):
        d0, f_het, t_het, tp, f_fac, t_fac = _unpack(theta)
        if not (0 < d0 < 1e6) or not (0.0 <= f_het <= f_het_max):
            return -np.inf
        if not (2000.0 < tp < 8000.0):
            return -np.inf
        if not (0.5 * tp <= t_het <= 1.2 * tp):
            return -np.inf
        if fit_faculae and not (0.0 <= f_fac <= f_het_max
                                and tp <= t_fac <= 1.5 * tp):
            return -np.inf
        if f_het + f_fac > f_het_max:
            return -np.inf
        lp = -0.5 * ((tp - t_phot) / max(t_phot_err, 1.0)) ** 2
        resid = d - _model(theta)
        return float(lp - 0.5 * np.sum((resid / e) ** 2))

    rng = np.random.default_rng(seed)
    p0 = np.asarray(p0, dtype=float)
    scatter = np.abs(p0) * 1e-3 + 1e-3
    start = p0 + scatter * rng.standard_normal((n_walkers, p0.size))
    start[:, 1] = np.abs(start[:, 1])
    sampler = emcee.EnsembleSampler(n_walkers, p0.size, _log_prob)
    sampler.run_mcmc(start, n_steps, progress=False)
    chain = sampler.get_chain(discard=n_steps // 2, flat=True)

    med = np.median(chain, axis=0)
    lo, hi = np.percentile(chain, [16, 84], axis=0)
    posterior = {
        label: {"median": float(m), "minus": float(m - l), "plus": float(h - m)}
        for label, m, l, h in zip(labels, med, lo, hi)
    }

    model = _model(med)
    chi2 = float(np.sum(((d - model) / e) ** 2))
    k = len(labels)
    n = w.size
    bic = chi2 + k * np.log(n)

    flat = float(np.sum(d / e**2) / np.sum(1.0 / e**2))
    chi2_flat = float(np.sum(((d - flat) / e) ** 2))
    bic_flat = chi2_flat + np.log(n)

    eps = contamination_factor(
        w, t_phot=med[3], t_het=med[2], f_het=med[1],
        t_fac=(med[5] if fit_faculae else None),
        f_fac=(med[4] if fit_faculae else 0.0),
    )
    # 95th percentile of f_het: the number to quote when the fit does not
    # detect contamination, which is the expected outcome for most
    # targets and is still a useful published limit.
    f_het_95 = float(np.percentile(chain[:, 1], 95))

    return {
        "backend": "patchwork-blackbody",
        "contam_version": PATCHWORK_CONTAM_VERSION,
        "stellar_model": "blackbody",
        "n_channels": int(n),
        "posterior": posterior,
        "f_het_95pct_upper": f_het_95,
        "chi2": chi2,
        "chi2_flat": chi2_flat,
        "bic": bic,
        "bic_flat": bic_flat,
        "delta_bic": float(bic_flat - bic),
        "contamination_detected": bool(bic_flat - bic > 10.0),
        "wave_um": w,
        "epsilon": eps,
        "model_ppm": model,
        "corrected_depth_ppm": d / eps,
        "corrected_depth_err_ppm": e / eps,
        "flat_depth_ppm": flat,
    }


def write_contamination_report(result: dict[str, Any],
                               out_dir: str | os.PathLike[str]) -> dict[str, str]:
    """Write the Stage 6.5 JSON summary, corrected spectrum CSV, and plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .juliet import _PLOT_STYLE, _savefig, DATA_COLOR, MODEL_COLOR, MEAN_COLOR

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    arrays = {k: result[k] for k in
              ("wave_um", "epsilon", "model_ppm",
               "corrected_depth_ppm", "corrected_depth_err_ppm")}
    scalar = {k: v for k, v in result.items() if k not in arrays}

    json_path = out / "contamination_summary.json"
    with json_path.open("w") as handle:
        json.dump(scalar, handle, indent=2)

    csv_path = out / "contamination_corrected_spectrum.csv"
    with csv_path.open("w") as handle:
        handle.write("wave_um,epsilon,corrected_depth_ppm,corrected_depth_err_ppm\n")
        for i in range(len(arrays["wave_um"])):
            handle.write(
                f"{arrays['wave_um'][i]:.6f},{arrays['epsilon'][i]:.6f},"
                f"{arrays['corrected_depth_ppm'][i]:.2f},"
                f"{arrays['corrected_depth_err_ppm'][i]:.2f}\n"
            )

    with plt.rc_context(_PLOT_STYLE):
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(9, 6), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1]})
        obs = arrays["corrected_depth_ppm"] * arrays["epsilon"]
        obs_err = arrays["corrected_depth_err_ppm"] * arrays["epsilon"]
        ax.errorbar(arrays["wave_um"], obs, yerr=obs_err, fmt="+",
                    color=DATA_COLOR, lw=1.1, ms=5, label="measured")
        ax.plot(arrays["wave_um"], arrays["model_ppm"], "-", lw=1.8,
                color=MODEL_COLOR, label="contamination model (flat intrinsic)")
        ax.axhline(result["flat_depth_ppm"], ls="--", lw=1.0, color=MEAN_COLOR,
                   label=f"flat {result['flat_depth_ppm']:.0f} ppm")
        ax.set_ylabel(r"Transit depth $(R_{\rm p}/R_\star)^2$  [ppm]")
        ax.legend(frameon=False, fontsize=8)
        p = result["posterior"]
        ax.set_title(
            f"Stage 6.5 unocculted-spot check — "
            f"$f_{{\\rm het}}$ = {p['f_het']['median']:.3f}"
            f"$^{{+{p['f_het']['plus']:.3f}}}_{{-{p['f_het']['minus']:.3f}}}$, "
            f"$T_{{\\rm het}}$ = {p['T_het']['median']:.0f} K, "
            f"$\\Delta$BIC = {result['delta_bic']:+.1f}"
        )
        ax2.plot(arrays["wave_um"], arrays["epsilon"], "-", lw=1.8,
                 color=MODEL_COLOR)
        ax2.axhline(1.0, ls="--", lw=1.0, color=MEAN_COLOR)
        ax2.set_ylabel(r"$\epsilon(\lambda)$")
        ax2.set_xlabel(r"Wavelength  [$\mu$m]")
        fig.tight_layout()
        fig_path = _savefig(fig, out, "contamination_fit")

    return {"summary_json": str(json_path), "corrected_csv": str(csv_path),
            "figure": fig_path}


# -------------------- orchestral tools --------------------


class DetectLightCurveAnomalies(BaseTool):
    """
    Stage 5.5 — scan white-light fit residuals for stellar anomalies
    (spot and facula crossings) and decide which integrations to mask.

    Runs AFTER ``FitNirspecG395hWhiteLight`` and BEFORE
    ``FitNirspecG395hTransmissionSpectrum``: it needs a clean transit
    model to subtract, and its output changes which integrations the
    spectroscopic fits see.

    Method: subtract the fitted transit + systematics model, smooth the
    residuals with a centred running mean, set the significance scale
    from the out-of-transit MAD of the smoothed series, and flag runs of
    at least 5 consecutive integrations beyond 3 sigma. Runs seen at the
    same time in BOTH detectors are confirmed as stellar; the NRS2/NRS1
    amplitude ratio is reported because a real spot's contrast falls
    towards the infrared, and an achromatic bump is more likely
    instrumental. Persistent (step-like) events are labelled as tilt
    events, not stellar, and are corrected with a step regressor instead
    of being masked. Single-detector excursions are reported but never
    masked.

    Point ``white_fit_dirs`` at the per-detector white-fit output
    directories (one or two). Each must contain the
    ``white_lightcurve_residuals.npz`` written by the white fit.

    Outputs in ``output_dir``: ``anomaly_report.json`` (the mask is
    stored there and read by the fit tools) and ``anomaly_scan.pdf/.svg``.

    Example
    -------
        DetectLightCurveAnomalies(
            white_fit_dirs=["fits/TOI_1231_b/o002/nrs1",
                            "fits/TOI_1231_b/o002/nrs2"],
            output_dir="fits/TOI_1231_b/o002/anomalies",
            planet_name="TOI-1231 b",
        )
    """

    white_fit_dirs: list[str] = RuntimeField(
        description="One or two white-fit output directories (NRS1, NRS2). "
                    "Each must contain white_lightcurve_residuals.npz."
    )
    output_dir: str = RuntimeField(description="Directory for the scan outputs.")
    planet_name: str = RuntimeField(
        default="", description="Planet name, for the figure title only."
    )
    threshold: float = RuntimeField(
        default=ANOMALY_THRESHOLD,
        description="Detection threshold in robust sigma of the smoothed "
                    "residual (survey default 3.0).",
    )
    min_run: int = RuntimeField(
        default=ANOMALY_MIN_RUN,
        description="Consecutive integrations required above threshold "
                    "(survey default 5).",
    )
    window: int = RuntimeField(
        default=ANOMALY_WINDOW,
        description="Running-mean window in integrations (survey default 15).",
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        out_dir = self.output_dir
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(self.base_directory, out_dir)

        reports: dict[str, dict[str, Any]] = {}
        series: dict[str, dict[str, np.ndarray]] = {}
        missing: list[str] = []
        for d in self.white_fit_dirs:
            path = d if os.path.isabs(d) else os.path.join(self.base_directory, d)
            npz = Path(path) / "white_lightcurve_residuals.npz"
            if not npz.exists():
                missing.append(str(npz))
                continue
            data = np.load(npz)
            det = str(data["detector"]) if "detector" in data else Path(path).name.upper()
            det = det.upper()
            series[det] = {"time": data["time"], "residual": data["residual"]}
            reports[det] = detect_lightcurve_anomalies(
                data["time"], data["residual"],
                oot_mask=data["oot_mask"], detector=det,
                window=self.window, threshold=self.threshold,
                min_run=self.min_run,
            )
            # The white fit that wrote this npz may have been masked
            # (tilt transitions), leaving its arrays compressed. Map the
            # scan's indices back to original integration numbers so the
            # mask lines up with the full Stage 4 lightcurve.
            if "index" in data:
                n_total = (int(data["n_total"]) if "n_total" in data
                           else int(data["index"][-1]) + 1)
                reports[det] = remap_anomaly_report(
                    reports[det], data["index"], n_total)
        if missing:
            return (
                "No residual file in:\n  " + "\n  ".join(missing)
                + "\nRun FitNirspecG395hWhiteLight first — it writes "
                  "white_lightcurve_residuals.npz alongside the posteriors."
            )

        merged = match_detector_anomalies(reports)
        masks = {det: anomaly_keep_mask(reports[det]["n_integrations"], merged,
                                        detector=det)
                 for det in reports}

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        payload = {
            "contam_version": PATCHWORK_CONTAM_VERSION,
            "planet_name": self.planet_name,
            "detectors": {det: reports[det] for det in reports},
            "events": [
                {k: v for k, v in m.items() if k != "detectors"}
                | {"detectors": {d: {"index_start": e["index_start"],
                                     "index_end": e["index_end"],
                                     "amplitude_ppm": e["amplitude_ppm"],
                                     "peak_sigma": e["peak_sigma"]}
                                 for d, e in m["detectors"].items()}}
                for m in merged
            ],
            "masked_integrations": {det: int((~m).sum()) for det, m in masks.items()},
            "mask_indices": {det: np.flatnonzero(~m).tolist()
                             for det, m in masks.items()},
        }
        with (Path(out_dir) / "anomaly_report.json").open("w") as handle:
            json.dump(payload, handle, indent=2)

        fig = plot_anomaly_diagnostic(
            reports, series, merged, out_dir,
            title=f"{self.planet_name} — Stage 5.5 anomaly scan".strip(" —"),
        )

        lines = [
            f"Stage 5.5 anomaly scan ({', '.join(sorted(reports))}): "
            f"{len(merged)} event(s).",
            summarize_anomalies(merged),
        ]
        for det, m in masks.items():
            n_bad = int((~m).sum())
            frac = 100.0 * n_bad / max(1, m.size)
            lines.append(f"  {det}: {n_bad} integration(s) masked ({frac:.1f}%).")
        confirmed = [m for m in merged if m["confirmed"]
                     and m["kind"] in (_KIND_SPOT, _KIND_FACULA)]
        steps = [m for m in merged if m["kind"] == _KIND_STEP]
        if steps:
            lines.append(
                f"  {len(steps)} persistent (step-like) event(s) — these are "
                "tilt events, not stellar, and must be corrected with a step "
                "regressor, never masked. NOTE: the fit only builds step "
                "regressors for events found by the Stage 4 tilt search "
                "(DetectTiltEvents). If that search did not report an event "
                "at this time, the step is currently UNCORRECTED — rerun the "
                "tilt scan with the reduction_dir diagnostics before fitting."
            )
        if confirmed:
            lines.append(
                "Refit the white lightcurve and the spectroscopic channels "
                "with anomaly_report=<this output_dir>/anomaly_report.json "
                "(force_refit=True) so the masked integrations are excluded."
            )
        else:
            lines.append("No confirmed stellar crossings — no refit needed.")
        lines.append(f"Figure: {fig}")
        return "\n".join(lines)


class ModelStellarContamination(BaseTool):
    """
    Stage 6.5 — test whether unocculted spots and faculae can account for
    the structure in a combined transmission spectrum (the transit light
    source effect).

    Unocculted heterogeneities never appear in the lightcurve, so Stage
    5.5 cannot see them. They multiply every measured depth by a
    wavelength-dependent factor epsilon(lambda) that rises towards short
    wavelengths for cool spots, mimicking a haze slope or an inflated
    scale height. This tool fits a flat intrinsic spectrum times
    epsilon(lambda) — the null hypothesis — so the result bounds how much
    of the observed structure the star alone could produce.

    Free: D0, spot covering fraction f_het, spot temperature T_het, and
    the photosphere temperature T_phot (Gaussian prior at the catalogue
    value). Optionally a facula component. Surface brightnesses are
    blackbodies by default; that captures the continuum contrast but not
    the molecular features a PHOENIX grid puts in an M-dwarf spot, so
    treat a detection as motivation to run stctm, not as a final answer.

    Report ``delta_bic > 10`` as contamination; otherwise quote
    ``f_het_95pct_upper`` as the limit.

    Outputs in ``output_dir``: ``contamination_summary.json``,
    ``contamination_corrected_spectrum.csv``,
    ``contamination_fit.pdf/.svg``.

    Example
    -------
        ModelStellarContamination(
            spectrum_csv="combined/combined_nrs1_transmission_spectrum.csv",
            output_dir="contamination/TOI_1231_b",
            planet_name="TOI-1231 b",
        )
    """

    spectrum_csv: str = RuntimeField(
        description="Combined transmission spectrum CSV (wave, depth, "
                    "depth_err in the Patchwork format). Pass a "
                    "comma-separated pair to fit NRS1 and NRS2 together."
    )
    output_dir: str = RuntimeField(description="Directory for Stage 6.5 outputs.")
    planet_name: str = RuntimeField(
        default="", description="Planet name — used to fetch T_eff from the archive."
    )
    t_phot: float | None = RuntimeField(
        default=None,
        description="Stellar effective temperature in K. Fetched from the "
                    "archive via planet_name when omitted.",
    )
    fit_faculae: bool = RuntimeField(
        default=False,
        description="Add an unocculted facula component (2 extra parameters). "
                    "Only worth it when the spectrum has many channels.",
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        from .juliet import read_spectrum_csv, fetch_transit_priors

        paths = [p.strip() for p in str(self.spectrum_csv).split(",") if p.strip()]
        waves, depths, errs = [], [], []
        for p in paths:
            full = p if os.path.isabs(p) else os.path.join(self.base_directory, p)
            s = read_spectrum_csv(full)
            waves.append(np.asarray(s["wave"], dtype=float))
            # read_spectrum_csv already returns ppm (depth_ppm columns).
            depths.append(np.asarray(s["depth_ppm"], dtype=float))
            errs.append(np.asarray(s["depth_err_ppm"], dtype=float))
        wave = np.concatenate(waves)
        depth = np.concatenate(depths)
        err = np.concatenate(errs)
        order = np.argsort(wave)
        wave, depth, err = wave[order], depth[order], err[order]

        t_phot = self.t_phot
        t_phot_err = 100.0
        if t_phot is None:
            if not self.planet_name:
                return ("Need t_phot or planet_name — the contamination factor "
                        "is defined relative to the photosphere temperature.")
            priors = fetch_transit_priors(self.planet_name)
            t_phot = priors.get("st_teff")
            if not t_phot:
                return (f"No st_teff in the archive for '{self.planet_name}'. "
                        "Pass t_phot explicitly.")

        result = retrieve_contamination(
            wave, depth, err, t_phot=float(t_phot), t_phot_err=t_phot_err,
            fit_faculae=self.fit_faculae,
        )
        out_dir = (self.output_dir if os.path.isabs(self.output_dir)
                   else os.path.join(self.base_directory, self.output_dir))
        written = write_contamination_report(result, out_dir)

        p = result["posterior"]
        lines = [
            f"Stage 6.5 unocculted-spot check for "
            f"{self.planet_name or 'this spectrum'} "
            f"({result['n_channels']} channels, blackbody heterogeneity).",
            f"  f_het   = {p['f_het']['median']:.4f} "
            f"+{p['f_het']['plus']:.4f} -{p['f_het']['minus']:.4f} "
            f"(95% upper limit {result['f_het_95pct_upper']:.4f})",
            f"  T_het   = {p['T_het']['median']:.0f} "
            f"+{p['T_het']['plus']:.0f} -{p['T_het']['minus']:.0f} K "
            f"(T_phot = {p['T_phot']['median']:.0f} K)",
            f"  D0      = {p['D0_ppm']['median']:.0f} ppm",
            f"  chi2 = {result['chi2']:.1f} vs flat {result['chi2_flat']:.1f}; "
            f"delta_BIC = {result['delta_bic']:+.1f}",
        ]
        if result["contamination_detected"]:
            lines.append(
                "  CONTAMINATION FAVOURED (delta_BIC > 10). The corrected "
                "spectrum is in contamination_corrected_spectrum.csv, but "
                "confirm with stctm on a PHOENIX grid before publishing — a "
                "blackbody spot has no molecular features."
            )
        else:
            lines.append(
                "  No contamination preferred over a flat spectrum. Quote "
                f"f_het < {result['f_het_95pct_upper']:.3f} (95%) at "
                f"T_het = {p['T_het']['median']:.0f} K as the limit; the "
                "measured spectrum needs no correction."
            )
        lines.append(
            "  Note: f_het and T_het are degenerate over a single G395H "
            "octave — report epsilon(lambda) and the corrected spectrum, not "
            "f_het alone. Blackbody spot contrast at 3-5 um is weak, so a "
            "non-detection means G395H cannot constrain this, not that the "
            "star is quiet."
        )
        lines += [f"  {k}: {v}" for k, v in written.items()]
        return "\n".join(lines)


class VerifyContaminationBackends(BaseTool):
    """
    Report which optional stellar-contamination backends are importable:
    SPOTROD (occulted spot modelling), stctm (unocculted contamination
    retrieval), and SAGE (contamination forward models).

    None is required. Stage 5.5 masking and the Stage 6.5 blackbody
    retrieval are self-contained pure-numpy/emcee code. These backends
    are cross-checks and per-target escape hatches — run this before
    promising a referee a SPOTROD or stctm comparison.
    """

    base_directory: str = StateField()

    def _run(self) -> str:
        report = contamination_backends()
        lines = ["Optional stellar-contamination backends:"]
        for name, info in report.items():
            state = "available" if info["available"] else "NOT INSTALLED"
            lines.append(f"  {name:8s} {state} — {info['detail']}")
            if "external_command" in info:
                lines.append(f"           external command: {info['external_command']}")
        lines.append(
            "Absent backends do not block the survey: Stage 5.5 masks "
            "crossings and Stage 6.5 runs its own blackbody retrieval."
        )
        return "\n".join(lines)
