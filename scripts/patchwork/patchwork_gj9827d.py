#!/usr/bin/env python
"""Patchwork — GJ 9827 d, full NIRSpec/BOTS G395H reduction and baseline fit.

Script port of ``GJ_9827d.ipynb``, for batch submission on DRAC Fir.

This is the **baseline** fit of the Patchwork plan: free quadratic limb
darkening, no decorrelation regressors, no tilt-event term. It is the
reference the patched fit is compared against — not a replacement for
``aster_toolkit.data_reduction.survey``, which runs the frozen v1.1
patched pipeline (ExoTiC-LD priors, trace decorrelation, tilt steps) and
is target-agnostic. Run both; the difference is the point.

Two phases, because exoTEDRF and juliet live in separate environments:

    --phase reduce   exoTEDRF Stages 1-3        (~/envs/exotedrf python)
    --phase fit      juliet Stages 5-6 + plots  (aster-env python)
    --phase all      both, if one env has both

The handoff is on disk: the fit phase reads the Stage 3
``*_box_spectra_fullres.fits`` products, so the phases need not share an
interpreter or even a job.

Outputs (in --workdir):
    pipeline_outputs_directory_gj9827d_<visit>_<det>/   exoTEDRF products
    juliet_<visit>_<det>_whitelight/                    white-light posteriors
    juliet_<visit>_<det>_bin###/                        per-channel posteriors
    gj9827d_<det>_transmission_spectrum.csv             combined over visits
    gj9827d_transmission_spectrum.pdf / .svg
    gj9827d_reproduction_check.pdf / .svg               if a published CSV exists
    gj9827d_summary.json                                the Patchwork log

Usage
-----
    python patchwork_gj9827d.py --phase reduce \
        --raw-root /project/def-ncowan/wasi/jwst_raw \
        --workdir  ~/scratch/patchwork/GJ_9827_d_asis
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

# ---------------------------------------------------------------- config

SCI_TAG = "04102"                       # science TSO exposure (not 02101 acq)
ALL_VISITS = {"o091": "jw04098091001", "o010": "jw04098010001"}
ALL_DETECTORS = ["NRS1", "NRS2"]

MODE = "NIRSpec/G395H"
MASK_W = 16          # trace mask / extraction half-width [pix]
NRS1_BLUE = 2.8      # micron; drop the NRS1 blue edge
RES = 100            # spectroscopic resolving power
NLIVE_WL = 400       # live points, white-light
NLIVE_SP = 250       # live points, per spectroscopic channel
BASELINE_FRAC = 0.20  # out-of-transit fraction per side

# GJ 9827 d system parameters (NASA Exoplanet Archive, pscomppars,
# retrieved 2026-07-29).
#
# Use a RECENT reference epoch. The previous values here (t0 =
# 2457740.96115, per = 6.20146980) were a 2016 epoch with a period 3.6e-4 d
# from the current best fit; propagating that 440 epochs forward to 2024
# accumulated ~3.8 h of error and put the model transit outside the
# observation entirely. The archive epoch below is only ~33 epochs away.
SYS = dict(
    per=6.20183000,          # days
    t0=2460265.10196,        # BJD_TDB reference epoch (propagated per visit)
    dur=1.2264,              # transit duration, hours
    inc=87.443,              # deg
    rprs=0.03093,            # -> depth ~957 ppm
    ars=19.739,
    ecc=0.0,
    omega=90.0,
    teff=4340.0,
    logg=4.66,
    feh=-0.26,
    rstar=0.602,             # Rsun
)
SYS["b"] = SYS["ars"] * np.cos(np.radians(SYS["inc"]))

# Patchwork house style for circulated figures: no grid (it competes with
# error bars at these amplitudes), muted data, one strong accent for the model.
PLOT_STYLE = {
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "legend.frameon": False, "legend.fontsize": 10.5,
    "figure.dpi": 150,
}
DATA_COLOR = "#1A2F6B"     # deep navy
BINNED_COLOR = "#0B1B45"   # darker navy for the binned overlay
MODEL_COLOR = "#E8A33D"    # amber -- reads strongly over dense navy points
PROGRAM = "GO 4098"


def savefig(fig, stem: str) -> None:
    """Patchwork figure rule: PDF and SVG, nothing else."""
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.svg")


# ------------------------------------------------------------- locating

def segment_files(raw_root: str, visit_root: str, detector: str) -> list[str]:
    """Sorted science uncal segments for one (visit, detector).

    Searched recursively: on Fir the visits sit inside obsid-numbered
    subfolders, and some folder names carry several targets at once, so
    the visit prefix in the filename is the only reliable key.
    """
    pattern = os.path.join(
        raw_root, "**",
        f"{visit_root}_{SCI_TAG}*_{detector.lower()}_uncal.fits",
    )
    return sorted(glob.glob(pattern, recursive=True))


def count_integrations(files: list[str]) -> int:
    n = 0
    for f in files:
        with fits.open(f) as h:
            head = h[0].header
            if head.get("INTSTART") and head.get("INTEND"):
                n += int(head["INTEND"]) - int(head["INTSTART"]) + 1
            else:
                n += int(head.get("NINTS", 0))
    return n


def baseline_for_visit(raw_root: str, visit_root: str) -> list[int]:
    """``[n_pre, -n_post]`` out-of-transit window, 20% of the visit each
    side (floor 5). Fractional rather than a fixed count, so visits of
    different length are treated uniformly."""
    n = count_integrations(segment_files(raw_root, visit_root, "NRS1"))
    if n == 0:
        raise FileNotFoundError(f"No NRS1 segments found for {visit_root}.")
    side = max(5, int(BASELINE_FRAC * n))
    return [side, -side]


def inventory(raw_root: str, visits: dict, detectors: list) -> dict:
    """Report segments per visit x detector and verify completeness."""
    report = {}
    for vname, vroot in visits.items():
        for det in detectors:
            files = segment_files(raw_root, vroot, det)
            segs, expected = [], 0
            for f in files:
                h = fits.getheader(f)
                segs.append(int(h.get("EXSEGNUM", 0)))
                expected = int(h.get("EXSEGTOT", 0)) or expected
            missing = [s for s in range(1, expected + 1) if s not in segs]
            report[f"{vname}_{det}"] = {
                "n_files": len(files), "segments": sorted(segs),
                "expected": expected, "missing": missing,
                "complete": bool(expected) and not missing,
                "nints": count_integrations(files) if files else 0,
            }
            status = "complete" if report[f"{vname}_{det}"]["complete"] else \
                     f"INCOMPLETE missing={missing}"
            print(f"  {vname} {det}: {len(files)} file(s), segments "
                  f"{sorted(segs)}/{expected} -> {status}")
    return report


# -------------------------------------------------------------- reduce

def phase_reduce(raw_root: str, visits: dict, detectors: list) -> dict:
    """exoTEDRF Stages 1-3 for every visit x detector. Runs in the
    exoTEDRF environment. Re-running skips completed stages
    (force_redo=False), so a timed-out job can simply be resubmitted."""
    from exotedrf.stage1 import run_stage1
    from exotedrf.stage2 import run_stage2
    from exotedrf.stage3 import run_stage3

    products = {}
    for vname, vroot in visits.items():
        bl = baseline_for_visit(raw_root, vroot)
        for det in detectors:
            files = segment_files(raw_root, vroot, det)
            if not files:
                print(f"!! no segments for {vname} {det} — skipping")
                continue
            tag = f"_gj9827d_{vname}_{det.lower()}"
            print(f"\n===== {vname} {det} (tag{tag}) "
                  f"{len(files)} seg(s) baseline {bl} =====", flush=True)

            s1 = run_stage1(files, mode=MODE, baseline_ints=bl,
                            oof_method="scale-achromatic",
                            superbias_method="crds",
                            nirspec_mask_width=MASK_W, save_results=True,
                            force_redo=False, output_tag=tag,
                            do_plot=True, show_plot=False)

            # PCAReconstructStep is a diagnostic-only TSO-stability step and
            # it crashes in exoTEDRF 2.3.1 against sklearn >= 1.3: it feeds a
            # 3D array to PCA.inverse_transform, which accepts <= 2D. Nothing
            # downstream needs it -- TracingStep computes its own deep frame
            # when deepframe is None -- so it is skipped, matching
            # PATCHWORK_G395H_CONFIG in aster_toolkit.
            # run_stage2 returns (results, centroids) -- it must be unpacked,
            # and the TracingStep centroids handed to Stage 3, exactly as
            # exoTEDRF's own run_DMS.py does (run_DMS.py:173, 192). Passing
            # the raw tuple makes Stage 3 fail in sort_datamodels with
            # "inhomogeneous shape ... (2,)".
            s2, centroids = run_stage2(
                s1, mode=MODE, baseline_ints=bl,
                nirspec_mask_width=MASK_W, generate_lc=True,
                skip_steps=['PCAReconstructStep'],
                save_results=True, force_redo=False,
                output_tag=tag, do_plot=True, show_plot=False)

            run_stage3(s2, extract_method="box", extract_width=MASK_W,
                       centroids=centroids,
                       planet_letter="d", st_teff=SYS["teff"],
                       st_logg=SYS["logg"], st_met=SYS["feh"],
                       save_results=True, force_redo=False, output_tag=tag,
                       do_plot=True, show_plot=False)

            found = find_spectra(tag, det)
            products[f"{vname}_{det}"] = found
            print(f"  -> Stage3 spectra: {found}")
    return products


def pipeline_dir(tag: str) -> str:
    """Reproduce exoTEDRF's output-directory naming exactly.

    exoTEDRF prepends its own separator to a non-empty output_tag
    (stage1.py: ``if output_tag != '': output_tag = '_' + output_tag``),
    so a tag that already starts with '_' produces a DOUBLED underscore.
    Constructing this by hand instead of mirroring that rule is how the
    fit phase ended up looking in a directory that never existed.
    """
    return "pipeline_outputs_directory" + ("_" + tag if tag else "")


def find_spectra(tag: str, det: str) -> str | None:
    """Locate the Stage 3 product for one visit x detector in the workdir."""
    # exoTEDRF writes the detector lowercase (GJ9827_nrs1_box_...), but the
    # case has varied across versions, so try both explicitly before falling
    # back to a detector-agnostic glob.
    patterns = [
        os.path.join(pipeline_dir(tag), "Stage3",
                     f"*_{det.lower()}_box_spectra_fullres.fits"),
        os.path.join(pipeline_dir(tag), "Stage3",
                     f"*_{det.upper()}_box_spectra_fullres.fits"),
        # Safe because each tag directory holds exactly one detector.
        os.path.join(pipeline_dir(tag), "Stage3",
                     "*box_spectra_fullres.fits"),
        # Last resort: tolerate any separator convention.
        os.path.join(f"pipeline_outputs_directory*{tag.lstrip('_')}",
                     "Stage3", "*box_spectra_fullres.fits"),
    ]
    for pattern in patterns:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None


# ------------------------------------------------------ lightcurves

MJD_TO_BJD_OFFSET = 2400000.5


def to_bjd(time: np.ndarray) -> np.ndarray:
    """exoTEDRF writes the Time extension in MJD; ephemerides are BJD.

    Without this the propagated mid-transit lands up to half a period from
    the data and the fit sees no transit at all -- it returns the Rp/Rs
    prior, which looks like a plausible spectrum with enormous error bars.
    """
    time = np.asarray(time, dtype=float)
    return time + MJD_TO_BJD_OFFSET if np.nanmedian(time) < 1e6 else time


def propagate_t0(time: np.ndarray) -> float:
    """Propagate the reference epoch to the transit nearest this visit.
    ``time`` must already be BJD (see ``to_bjd``)."""
    tmid = float(np.nanmedian(np.asarray(time, dtype=float)))
    n = round((tmid - SYS["t0"]) / SYS["per"])
    return SYS["t0"] + n * SYS["per"]


def check_transit_in_window(time: np.ndarray, t0_obs: float,
                            label: str) -> float:
    """Fail loudly if the propagated transit does not overlap the data.

    A transit outside the observation is the single most expensive silent
    failure in this pipeline: every fit still 'succeeds', but the Rp/Rs
    posterior is just the prior (median depth ~1800 ppm, errors ~1500 ppm)
    and the transmission spectrum is meaningless. Catch it in milliseconds
    instead of after ~110 nested-sampling runs.
    """
    t = np.asarray(time, dtype=float)
    tmid = 0.5 * (np.nanmin(t) + np.nanmax(t))
    half_window_hr = 0.5 * (np.nanmax(t) - np.nanmin(t)) * 24
    offset_hr = (t0_obs - tmid) * 24
    reach_hr = half_window_hr + 0.5 * SYS["dur"]
    if abs(offset_hr) > reach_hr:
        raise SystemExit(
            f"\n!! {label}: propagated mid-transit is {offset_hr:+.2f} h from "
            f"the centre of a {2 * half_window_hr:.2f} h observation — the "
            f"transit is NOT in this data.\n"
            f"   t0_obs = {t0_obs:.5f}, data span "
            f"{np.nanmin(t):.5f} to {np.nanmax(t):.5f} (BJD).\n"
            f"   Fitting would return the Rp/Rs prior, not a measurement.\n"
            f"   Check: (a) the SYS ephemeris is current — a stale epoch "
            f"propagated over many periods drifts hours; (b) times were "
            f"converted MJD -> BJD.\n"
        )
    coverage = min(1.0, max(0.0,
                   (min(offset_hr + 0.5 * SYS["dur"], half_window_hr)
                    - max(offset_hr - 0.5 * SYS["dur"], -half_window_hr))
                   / SYS["dur"]))
    print(f"   {label}: mid-transit {offset_hr:+.2f} h from window centre, "
          f"{coverage * 100:.0f}% of the transit covered")
    return coverage


def bin_at_resolution(wave, flux, err, res=RES):
    """Constant-R inverse-variance binning of (nints, nwave) spectra.

    Local reimplementation so the fit phase needs no exoTEDRF import
    (the two phases run in different environments).
    """
    good = np.isfinite(wave) & np.any(np.isfinite(flux), axis=0)
    lo, hi = np.nanmin(wave[good]), np.nanmax(wave[good])
    edges = [lo]
    while edges[-1] < hi:
        edges.append(edges[-1] * (1 + 1.0 / res))
    edges = np.asarray(edges)

    centers, halves, fbins, ebins = [], [], [], []
    for i in range(len(edges) - 1):
        m = good & (wave >= edges[i]) & (wave < edges[i + 1])
        if not m.any():
            continue
        f, e = flux[:, m], err[:, m]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / e ** 2
            w[~np.isfinite(w) | ~np.isfinite(f)] = 0.0
            f = np.where(np.isfinite(f), f, 0.0)
            wsum = w.sum(axis=1)
            fb = (f * w).sum(axis=1) / wsum
            eb = 1.0 / np.sqrt(wsum)
        if not np.all(np.isfinite(fb)):
            continue
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        halves.append(0.5 * (edges[i + 1] - edges[i]))
        fbins.append(fb)
        ebins.append(eb)
    return (np.asarray(centers), np.asarray(halves),
            np.column_stack(fbins), np.column_stack(ebins))


def build_lightcurves(spectra_file: str, det: str, bl: list[int],
                      label: str = "") -> dict:
    """White + spectroscopic lightcurves, normalized by the
    out-of-transit baseline (a global median would be biased low by the
    transit itself)."""
    wave = np.asarray(fits.getdata(spectra_file, 1), dtype=float)
    flux = np.asarray(fits.getdata(spectra_file, 3), dtype=float)
    err = np.asarray(fits.getdata(spectra_file, 4), dtype=float)
    time = to_bjd(fits.getdata(spectra_file, 5))
    if wave.ndim == 2:
        wave = np.nanmedian(wave, axis=0)

    wave, flux, err = wave[5:-5], flux[:, 5:-5], err[:, 5:-5]  # ref-pix cols
    if det == "NRS1":
        keep = wave >= NRS1_BLUE
        wave, flux, err = wave[keep], flux[:, keep], err[:, keep]

    pre, post = slice(0, bl[0]), slice(bl[1], None)

    def normalize(f, e):
        base = np.nanmedian(np.concatenate([f[pre], f[post]], axis=0), axis=0)
        return f / base, e / base

    wl_flux = np.nansum(flux, axis=1)
    wl_err = np.sqrt(np.nansum(err ** 2, axis=1))
    wl_flux, wl_err = normalize(wl_flux, wl_err)

    bw, bwe, bf, be = bin_at_resolution(wave, flux, err, res=RES)
    sp_flux, sp_err = normalize(bf, be)

    oot = np.concatenate([wl_flux[pre], wl_flux[post]])
    t0_obs = propagate_t0(time)
    coverage = check_transit_in_window(time, t0_obs, label)
    return dict(time=time, t0_obs=t0_obs, transit_coverage=coverage,
                wl_flux=wl_flux, wl_err=wl_err,
                wave=bw, wave_err=bwe, sp_flux=sp_flux, sp_err=sp_err,
                oot_scatter_ppm=float(np.nanstd(oot) * 1e6))


# -------------------------------------------------------------- fitting

def priors(names, dists, vals) -> dict:
    return {n: {"distribution": d, "hyperparameters": v}
            for n, d, v in zip(names, dists, vals)}


def fit_white_light(juliet, lc: dict, vname: str, det: str) -> dict:
    """As-is white-light fit: free t0, Rp/Rs, b, a/Rs and free quadratic
    LD (Kipping q1,q2). P, ecc, omega fixed."""
    inst = det
    t, f, e = lc["time"], lc["wl_flux"], lc["wl_err"]
    names = ["P_p1", "t0_p1", "p_p1", "b_p1", "a_p1", "ecc_p1", "omega_p1",
             "q1_" + inst, "q2_" + inst, "mdilution_" + inst,
             "mflux_" + inst, "sigma_w_" + inst]
    dists = ["fixed", "normal", "uniform", "uniform", "uniform", "fixed",
             "fixed", "uniform", "uniform", "fixed", "normal", "loguniform"]
    vals = [SYS["per"], [lc["t0_obs"], 0.01], [0.005, 0.08], [0.0, 1.0],
            [10.0, 30.0], SYS["ecc"], SYS["omega"], [0.0, 1.0], [0.0, 1.0],
            1.0, [0.0, 0.1], [1.0, 1e4]]

    out = f"juliet_{vname}_{det.lower()}_whitelight"
    ds = juliet.load(priors=priors(names, dists, vals), t_lc={inst: t},
                     y_lc={inst: f}, yerr_lc={inst: e}, out_folder=out)
    res = ds.fit(sampler="dynesty", n_live_points=NLIVE_WL)
    post = res.posteriors["posterior_samples"]
    orbit = {k: float(np.median(post[k])) for k in ["t0_p1", "b_p1", "a_p1"]}
    depth = float(np.median(post["p_p1"]) ** 2 * 1e6)
    rms = float(np.nanstd(f - res.lc.evaluate(inst)) * 1e6)
    print(f"{vname} {det} white light: depth={depth:.0f} ppm, "
          f"t0={orbit['t0_p1']:.5f}, b={orbit['b_p1']:.3f}, "
          f"a/Rs={orbit['a_p1']:.2f}, rms={rms:.0f} ppm", flush=True)
    plot_white(res, lc, vname, det, rms)
    return dict(orbit=orbit, depth=depth, rms=rms)


def bin_series(x, y, n_bins=90):
    """Equal-count binning for a readable overlay on dense photometry."""
    idx = np.argsort(x)
    x, y = np.asarray(x)[idx], np.asarray(y)[idx]
    edges = np.linspace(0, x.size, n_bins + 1).astype(int)
    bx, by, be = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 3:
            continue
        bx.append(np.mean(x[a:b])); by.append(np.mean(y[a:b]))
        be.append(np.std(y[a:b], ddof=1) / np.sqrt(b - a))
    return map(np.asarray, (bx, by, be))


def plot_white(res, lc, vname, det, rms) -> None:
    """White-light fit figure, formatted for circulation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    with plt.rc_context(PLOT_STYLE):
        t_hr = (lc["time"] - lc["t0_obs"]) * 24
        model = res.lc.evaluate(det)
        resid = lc["wl_flux"] - model
        depth_ppm = (1.0 - model.min()) * 1e6

        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.0, 6.4), sharex=True,
                                     height_ratios=[3, 1])

        # Unbinned photometry, deliberately faint: it shows the true scatter
        # without burying the model.
        a1.plot(t_hr, lc["wl_flux"], "o", ms=2.4, mew=0, color=DATA_COLOR,
                alpha=0.16, zorder=1, label="integrations")
        bx, by, be = bin_series(t_hr, lc["wl_flux"])
        a1.errorbar(bx, by, yerr=be, fmt="o", ms=5.0, mew=0,
                    color=BINNED_COLOR, ecolor=BINNED_COLOR, elinewidth=1.2,
                    capsize=0, alpha=0.95, zorder=3, label="binned")
        a1.plot(t_hr, model, color=MODEL_COLOR, lw=2.6, zorder=4,
                solid_capstyle="round", label="juliet transit model",
                path_effects=[pe.Stroke(linewidth=4.6, foreground="white"),
                              pe.Normal()])
        a1.set_ylabel("Relative flux")
        a1.set_title(f"GJ 9827 d   ·   {PROGRAM}   ·   JWST NIRSpec/G395H "
                     f"{det}   ·   visit {vname}", pad=12)
        a1.legend(loc="lower left", ncol=3, columnspacing=1.4,
                  handletextpad=0.5)
        a1.annotate(f"depth {depth_ppm:.0f} ppm\nresidual rms {rms:.0f} ppm",
                    xy=(0.985, 0.06), xycoords="axes fraction", ha="right",
                    va="bottom", fontsize=10.5, color="#4E5866")

        a2.plot(t_hr, resid * 1e6, "o", ms=2.4, mew=0, color=DATA_COLOR,
                alpha=0.16, zorder=1)
        rbx, rby, rbe = bin_series(t_hr, resid * 1e6)
        a2.errorbar(rbx, rby, yerr=rbe, fmt="o", ms=4.5, mew=0,
                    color=BINNED_COLOR, ecolor=BINNED_COLOR, elinewidth=1.1,
                    capsize=0, alpha=0.95, zorder=3)
        a2.axhline(0, color=MODEL_COLOR, lw=1.8, zorder=2)
        a2.set_xlabel("Time from mid-transit  [h]")
        a2.set_ylabel("Residual  [ppm]")
        a2.set_ylim(-4 * rms, 4 * rms)

        fig.tight_layout()
        savefig(fig, f"gj9827d_whitelight_{vname}_{det.lower()}")
        plt.close(fig)

    # Save the plotted series so the figure can be restyled later without
    # refitting anything.
    np.savetxt(f"gj9827d_whitelight_{vname}_{det.lower()}.csv",
               np.column_stack([lc["time"], lc["wl_flux"], lc["wl_err"], model]),
               delimiter=",", header="time_bjd,flux,flux_err,model",
               comments="", fmt="%.10g")


def fit_spectroscopic(juliet, lc: dict, orbit: dict,
                      vname: str, det: str) -> dict:
    """Per-channel fits with the orbit frozen to the white-light medians.
    Free: Rp/Rs, q1, q2, mflux, sigma_w."""
    inst = det
    t, F, E, W = lc["time"], lc["sp_flux"], lc["sp_err"], lc["wave"]
    n = W.size
    depth = np.full(n, np.nan)
    depth_err = np.full(n, np.nan)
    rms = np.full(n, np.nan)

    names = ["P_p1", "t0_p1", "p_p1", "b_p1", "a_p1", "ecc_p1", "omega_p1",
             "q1_" + inst, "q2_" + inst, "mdilution_" + inst,
             "mflux_" + inst, "sigma_w_" + inst]
    dists = ["fixed", "fixed", "uniform", "fixed", "fixed", "fixed", "fixed",
             "uniform", "uniform", "fixed", "normal", "loguniform"]

    for i in range(n):
        f, e = F[:, i], E[:, i]
        good = np.isfinite(f) & np.isfinite(e)
        if good.sum() < 50:
            continue
        vals = [SYS["per"], orbit["t0_p1"], [0.005, 0.08], orbit["b_p1"],
                orbit["a_p1"], SYS["ecc"], SYS["omega"], [0.0, 1.0],
                [0.0, 1.0], 1.0, [0.0, 0.1], [1.0, 1e4]]
        out = f"juliet_{vname}_{det.lower()}_bin{i:03d}"
        ds = juliet.load(priors=priors(names, dists, vals),
                         t_lc={inst: t[good]}, y_lc={inst: f[good]},
                         yerr_lc={inst: e[good]}, out_folder=out)
        res = ds.fit(sampler="dynesty", n_live_points=NLIVE_SP)
        d = res.posteriors["posterior_samples"]["p_p1"] ** 2 * 1e6
        depth[i] = np.median(d)
        depth_err[i] = 0.5 * (np.percentile(d, 84) - np.percentile(d, 16))
        rms[i] = np.nanstd(f[good] - res.lc.evaluate(inst)) * 1e6

    print(f"  {vname} {det}: fit {int(np.isfinite(depth).sum())} channels, "
          f"median depth err {np.nanmedian(depth_err):.0f} ppm", flush=True)
    return dict(wave=W, wave_err=lc["wave_err"], depth=depth,
                depth_err=depth_err, rms=rms)


# ---------------------------------------------------- combine + outputs

def ivw_combine(specs: list[dict]) -> dict:
    """Inverse-variance combine visits of one detector."""
    W, WE = specs[0]["wave"], specs[0]["wave_err"]
    D = np.vstack([s["depth"] for s in specs])
    E = np.vstack([s["depth_err"] for s in specs])
    R = np.vstack([s["rms"] for s in specs])
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / E ** 2
        depth = np.nansum(D * w, axis=0) / np.nansum(w, axis=0)
        derr = np.sqrt(1.0 / np.nansum(w, axis=0))
    return dict(wave=W, wave_err=WE, depth=depth, depth_err=derr,
                rms=np.nanmedian(R, axis=0), n_visits=len(specs))


def save_transpec(det: str, S: dict) -> str:
    path = f"gj9827d_{det.lower()}_transmission_spectrum.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"# GJ 9827 d {det} G395H baseline (juliet, free LD, "
                    f"R={RES}, {S['n_visits']} visit(s) combined)"])
        w.writerow(["wave_um", "wave_err_um", "depth_ppm", "depth_err_ppm",
                    "resid_rms_ppm"])
        for row in zip(S["wave"], S["wave_err"], S["depth"],
                       S["depth_err"], S["rms"]):
            w.writerow([f"{x:.6f}" if np.isfinite(x) else "" for x in row])
    return path


def plot_spectrum(combined: dict, published_csv: str | None) -> float | None:
    """Transmission-spectrum figure, formatted for circulation.

    The NIRSpec/G395H inter-detector gap is shaded and labelled so the eye
    does not read across a wavelength range that was never observed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"NRS1": DATA_COLOR, "NRS2": "#1B7F79"}
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        edge, all_d, all_e = {}, [], []

        if {"NRS1", "NRS2"} <= set(combined):
            m1 = np.isfinite(combined["NRS1"]["depth"])
            m2 = np.isfinite(combined["NRS2"]["depth"])
            gap_lo = combined["NRS1"]["wave"][m1].max()
            gap_hi = combined["NRS2"]["wave"][m2].min()
            ax.axvspan(gap_lo, gap_hi, color="#B9C2CC", alpha=0.30,
                       zorder=0, lw=0)
            ax.text(0.5 * (gap_lo + gap_hi), 0.035, "detector\ngap",
                    transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=9.5, color="#4E5866",
                    linespacing=1.15)

        for det in ("NRS1", "NRS2"):
            if det not in combined:
                continue
            S = combined[det]
            m = np.isfinite(S["depth"])
            all_d.append(S["depth"][m]); all_e.append(S["depth_err"][m])
            ax.errorbar(S["wave"][m], S["depth"][m], xerr=S["wave_err"][m],
                        yerr=S["depth_err"][m], fmt="o", ms=4.5, mew=0,
                        color=colors[det], ecolor=colors[det],
                        elinewidth=1.1, capsize=0, alpha=0.85, zorder=3,
                        label=f"{det}  ({int(m.sum())} channels)")
            d = S["depth"][m]
            edge[det] = np.median(d[-5:] if det == "NRS1" else d[:5])

        if all_d:
            d = np.concatenate(all_d); e = np.concatenate(all_e)
            wmean = np.sum(d / e ** 2) / np.sum(1 / e ** 2)
            ax.axhline(wmean, color="#8A94A6", lw=1.2, ls="--", zorder=1,
                       label=f"weighted mean  {wmean:.0f} ppm")
            lo, hi = (d - e).min(), (d + e).max()
            pad = 0.18 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)

        n_vis = combined[next(iter(combined))].get("n_visits", len(ALL_VISITS))
        ax.set_xlabel("Wavelength  [μm]")
        ax.set_ylabel(r"Transit depth  $(R_{\rm p}/R_{\star})^{2}$  [ppm]")
        ax.set_title(f"GJ 9827 d   ·   {PROGRAM}   ·   JWST NIRSpec/G395H   ·   "
                     f"{n_vis} visit{'s' if n_vis != 1 else ''} combined, "
                     f"R = {RES}", pad=12)
        ax.legend(loc="upper left", ncol=3, columnspacing=1.4,
                  handletextpad=0.5)
        fig.tight_layout()
        savefig(fig, "gj9827d_transmission_spectrum")
        plt.close(fig)

        offset = (edge["NRS1"] - edge["NRS2"]
                  if {"NRS1", "NRS2"} <= set(edge) else None)

        if published_csv and os.path.exists(published_csv):
            fig, ax = plt.subplots(figsize=(10.5, 5.2))
            for det in ("NRS1", "NRS2"):
                if det not in combined:
                    continue
                S = combined[det]; m = np.isfinite(S["depth"])
                ax.errorbar(S["wave"][m], S["depth"][m],
                            yerr=S["depth_err"][m], fmt="o", ms=4.5, mew=0,
                            color=colors[det], ecolor=colors[det],
                            elinewidth=1.1, capsize=0, alpha=0.85, zorder=3,
                            label=f"Patchwork {det}")
            pub = np.genfromtxt(published_csv, delimiter=",", names=True,
                                comments="#")
            cols = list(pub.dtype.names)
            wcol = cols[0]
            dcol = next(c for c in cols
                        if "depth" in c.lower() and "err" not in c.lower())
            ecol = [c for c in cols
                    if "depth" in c.lower() and "err" in c.lower()]
            ax.errorbar(pub[wcol], pub[dcol],
                        yerr=pub[ecol[0]] if ecol else None,
                        fmt="s", ms=4.5, mew=0, color="#4E5866",
                        ecolor="#4E5866", elinewidth=1.1, capsize=0,
                        alpha=0.75, zorder=2, label="published")
            ax.set_xlabel("Wavelength  [μm]")
            ax.set_ylabel(r"Transit depth  $(R_{\rm p}/R_{\star})^{2}$  [ppm]")
            ax.set_title(f"GJ 9827 d   ·   {PROGRAM}   ·   "
                         "Patchwork versus published", pad=12)
            ax.legend(loc="upper left", ncol=3, columnspacing=1.4,
                      handletextpad=0.5)
            fig.tight_layout()
            savefig(fig, "gj9827d_reproduction_check")
            plt.close(fig)
            print(f"overlaid published spectrum: {published_csv}")
        elif published_csv:
            print(f"No published CSV at {published_csv} — skipped overlay.")
    return offset


# ------------------------------------------------------------------ fit

def existing_juliet_runs() -> list[str]:
    """Directories of previous juliet fits in the current workdir."""
    return sorted(d for d in glob.glob("juliet_*") if os.path.isdir(d))


def clear_juliet_runs(dirs: list[str]) -> None:
    import shutil
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


def phase_fit(raw_root: str, visits: dict, detectors: list,
              published_csv: str | None, force_refit: bool = False) -> dict:
    """juliet Stages 5-6. Runs in the ASTER environment; reads the Stage 3
    products the reduce phase left on disk."""
    # juliet LOADS existing posteriors from out_folder instead of refitting.
    # After any change to the data, ephemeris, or priors that silently
    # returns the previous run's answers -- a rerun that "completes" in
    # minutes and reproduces the old numbers exactly.
    stale = existing_juliet_runs()
    if stale and force_refit:
        print(f"Clearing {len(stale)} previous juliet run(s) before refitting.")
        clear_juliet_runs(stale)
    elif stale:
        raise SystemExit(
            f"\n!! {len(stale)} previous juliet run(s) found in this workdir "
            f"(e.g. {stale[0]}).\n"
            "   juliet will RELOAD those posteriors instead of refitting, so "
            "any fix to the\n"
            "   ephemeris, priors, or lightcurves would be silently ignored "
            "and you would get\n"
            "   the previous answers back.\n"
            "   Re-run with --force-refit to delete them and fit properly, or "
            "use a fresh --workdir.\n"
        )

    try:
        import juliet
    except ImportError as exc:
        raise SystemExit(
            f"{exc}. The fit phase needs juliet — activate the ASTER "
            "environment (not the exoTEDRF one) and rerun with --phase fit. "
            "The reduce phase's Stage 3 products stay on disk, so nothing "
            "is lost."
        ) from exc

    wl_fits, sp_fits = {}, {}
    for vname, vroot in visits.items():
        bl = baseline_for_visit(raw_root, vroot)
        for det in detectors:
            tag = f"_gj9827d_{vname}_{det.lower()}"
            spectra = find_spectra(tag, det)
            if spectra is None:
                print(f"!! no Stage 3 spectra for {vname} {det} "
                      f"(expected under {pipeline_dir(tag)}/Stage3/) "
                      "— run --phase reduce first. Skipping.")
                continue
            lc = build_lightcurves(spectra, det, bl, f"{vname} {det}")
            print(f"{vname} {det}: {lc['time'].size} ints, {lc['wave'].size} "
                  f"channels, wl oot scatter={lc['oot_scatter_ppm']:.0f} ppm, "
                  f"t0_obs={lc['t0_obs']:.5f}", flush=True)
            wl = wl_fits[(vname, det)] = fit_white_light(juliet, lc, vname, det)
            print(f"===== spectroscopic fits: {vname} {det} =====", flush=True)
            sp_fits[(vname, det)] = fit_spectroscopic(
                juliet, lc, wl["orbit"], vname, det)

    if not sp_fits:
        raise SystemExit("No fits produced — nothing to combine.")

    combined, summary = {}, {"planet": "GJ 9827 d", "mode": "baseline",
                             "resolution": RES, "detectors": {}}
    for det in detectors:
        specs = [sp_fits[(v, det)] for v in visits if (v, det) in sp_fits]
        if not specs:
            continue
        S = combined[det] = ivw_combine(specs)
        path = save_transpec(det, S)
        wlrms = float(np.nanmedian([wl_fits[(v, det)]["rms"] for v in visits
                                    if (v, det) in wl_fits]))
        summary["detectors"][det] = {
            "n_visits": S["n_visits"],
            "n_channels": int(np.isfinite(S["depth"]).sum()),
            "median_depth_err_ppm": float(np.nanmedian(S["depth_err"])),
            "median_white_rms_ppm": wlrms,
            "median_channel_rms_ppm": float(np.nanmedian(S["rms"])),
            "csv": path,
        }
        print(f"wrote {path}")
        print(f"{det}: median white-light rms = {wlrms:.0f} ppm | "
              f"median spec-channel rms = "
              f"{np.nanmedian(S['rms']):.0f} ppm")

    offset = plot_spectrum(combined, published_csv)
    if offset is not None:
        summary["nrs1_nrs2_offset_ppm"] = float(offset)
        print(f"\nNRS1-NRS2 offset near detector gap: {offset:.1f} ppm")

    with open("gj9827d_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("wrote gj9827d_summary.json")
    return summary


# ----------------------------------------------------------------- main

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-root", default="/project/def-ncowan/wasi/jwst_raw",
                   help="Root of the uncal tree (searched recursively).")
    p.add_argument("--workdir", required=True,
                   help="All outputs are written here (created if missing).")
    p.add_argument("--phase", choices=["reduce", "fit", "all"], default="all")
    p.add_argument("--force-refit", action="store_true",
                   help="Delete previous juliet_* runs before fitting. Required "
                        "when refitting the same workdir, because juliet "
                        "reloads existing posteriors instead of refitting.")
    p.add_argument("--published-csv", default="published_gj9827d_g395h.csv",
                   help="Published GO 4098 spectrum for the overlay "
                        "(wave_um, depth_ppm, depth_err_ppm). Skipped if absent.")
    p.add_argument("--visits", default=",".join(ALL_VISITS),
                   help="Comma-separated visit names to process.")
    p.add_argument("--detectors", default=",".join(ALL_DETECTORS))
    args = p.parse_args(argv)

    keep = [v.strip() for v in args.visits.split(",") if v.strip()]
    unknown = set(keep) - set(ALL_VISITS)
    if unknown:
        p.error(f"Unknown visit(s): {sorted(unknown)}. "
                f"Known: {list(ALL_VISITS)}")
    visits = {k: ALL_VISITS[k] for k in keep}
    detectors = [d.strip().upper() for d in args.detectors.split(",")
                 if d.strip()]

    raw_root = os.path.abspath(os.path.expanduser(args.raw_root))
    workdir = os.path.abspath(os.path.expanduser(args.workdir))
    published = (os.path.abspath(os.path.expanduser(args.published_csv))
                 if args.published_csv else None)
    os.makedirs(workdir, exist_ok=True)
    # exoTEDRF writes pipeline_outputs_directory* relative to cwd.
    os.chdir(workdir)

    print(f"raw_root = {raw_root}")
    print(f"workdir  = {workdir}")
    print(f"visits   = {list(visits)}   detectors = {detectors}")
    print(f"python   = {sys.executable}\n")

    print("Uncal inventory:")
    report = inventory(raw_root, visits, detectors)
    if not any(v["n_files"] for v in report.values()):
        raise SystemExit(f"No science uncals found under {raw_root}.")
    if not all(v["complete"] for v in report.values() if v["n_files"]):
        print("!! WARNING: incomplete segment set — part of the transit is "
              "missing. The reduction will run but is not science-usable.")

    if args.phase in ("reduce", "all"):
        phase_reduce(raw_root, visits, detectors)
    if args.phase in ("fit", "all"):
        phase_fit(raw_root, visits, detectors, published,
                  force_refit=args.force_refit)

    print(f"\nDone. Outputs in {workdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
