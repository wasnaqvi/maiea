#!/usr/bin/env python
"""Patchwork — GJ 9827 d, full NIRSpec/BOTS G395H reduction and as-is fit.

Script port of ``GJ_9827d.ipynb``, for batch submission on DRAC Fir.

This is the **as-is baseline** of the Patchwork plan: free quadratic limb
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

# GJ 9827 d system parameters (NASA Exoplanet Archive).
SYS = dict(
    per=6.20146980,          # days
    t0=2457740.96115,        # BJD_TDB reference epoch (propagated per visit)
    inc=87.443,              # deg
    rprs=0.03073,            # -> depth ~945 ppm
    ars=20.003,
    ecc=0.0,
    omega=90.0,
    teff=4340.0,
    logg=4.66,
    feh=-0.26,
    rstar=0.602,             # Rsun
)
SYS["b"] = SYS["ars"] * np.cos(np.radians(SYS["inc"]))

PLOT_STYLE = {
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False,
}


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


def find_spectra(tag: str, det: str) -> str | None:
    """Locate the Stage 3 product for one visit x detector in the workdir."""
    hits = glob.glob(os.path.join(
        f"pipeline_outputs_directory{tag}", "Stage3",
        f"*_{det}_box_spectra_fullres.fits"))
    if not hits:  # exoTEDRF has varied the detector case across versions
        hits = glob.glob(os.path.join(
            f"pipeline_outputs_directory{tag}", "Stage3",
            "*box_spectra_fullres.fits"))
    return hits[0] if hits else None


# ------------------------------------------------------ lightcurves

def propagate_t0(time: np.ndarray) -> float:
    tmid = float(np.nanmedian(np.asarray(time, dtype=float)))
    n = round((tmid - SYS["t0"]) / SYS["per"])
    return SYS["t0"] + n * SYS["per"]


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


def build_lightcurves(spectra_file: str, det: str, bl: list[int]) -> dict:
    """White + spectroscopic lightcurves, normalized by the
    out-of-transit baseline (a global median would be biased low by the
    transit itself)."""
    wave = np.asarray(fits.getdata(spectra_file, 1), dtype=float)
    flux = np.asarray(fits.getdata(spectra_file, 3), dtype=float)
    err = np.asarray(fits.getdata(spectra_file, 4), dtype=float)
    time = np.asarray(fits.getdata(spectra_file, 5), dtype=float)
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
    return dict(time=time, t0_obs=propagate_t0(time),
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


def plot_white(res, lc, vname, det, rms) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(PLOT_STYLE):
        t_hr = (lc["time"] - lc["t0_obs"]) * 24
        model = res.lc.evaluate(det)
        resid = lc["wl_flux"] - model
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                     height_ratios=[3, 1])
        a1.errorbar(t_hr, lc["wl_flux"], yerr=lc["wl_err"], fmt="o", ms=3,
                    color="#1a2f6b", elinewidth=0.7, alpha=0.8, label="data")
        a1.plot(t_hr, model, color="crimson", lw=1.5, label="juliet best fit")
        a1.set_ylabel("Relative flux")
        a1.set_title(f"GJ 9827 d — {vname} {det} white light, "
                     f"residual rms {rms:.0f} ppm")
        a1.legend(fontsize=9)
        a2.plot(t_hr, resid * 1e6, "o", ms=2.5, color="#1a2f6b", alpha=0.7)
        a2.axhline(0, color="crimson", lw=1)
        a2.set_xlabel("Time from mid-transit (h)")
        a2.set_ylabel("Residual (ppm)")
        fig.tight_layout()
        savefig(fig, f"gj9827d_whitelight_{vname}_{det.lower()}")
        plt.close(fig)


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
        w.writerow([f"# GJ 9827 d {det} G395H as-is (juliet, free LD, "
                    f"R={RES}, {S['n_visits']} visit(s) combined)"])
        w.writerow(["wave_um", "wave_err_um", "depth_ppm", "depth_err_ppm",
                    "resid_rms_ppm"])
        for row in zip(S["wave"], S["wave_err"], S["depth"],
                       S["depth_err"], S["rms"]):
            w.writerow([f"{x:.6f}" if np.isfinite(x) else "" for x in row])
    return path


def plot_spectrum(combined: dict, published_csv: str | None) -> float | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"NRS1": "#1a2f6b", "NRS2": "crimson"}
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        edge = {}
        for det, S in combined.items():
            m = np.isfinite(S["depth"])
            ax.errorbar(S["wave"][m], S["depth"][m], xerr=S["wave_err"][m],
                        yerr=S["depth_err"][m], fmt="o", ms=4, lw=0.8,
                        color=colors.get(det, "gray"), label=det, alpha=0.85)
            d = S["depth"][m]
            edge[det] = np.median(d[-5:] if det == "NRS1" else d[:5])
        ax.set(xlabel="Wavelength (μm)",
               ylabel=r"Transit depth $(R_p/R_s)^2$ (ppm)",
               title="GJ 9827 d — Patchwork G395H, both visits combined (as-is)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        savefig(fig, "gj9827d_transmission_spectrum")
        plt.close(fig)

        offset = (edge["NRS1"] - edge["NRS2"]
                  if {"NRS1", "NRS2"} <= set(edge) else None)

        if published_csv and os.path.exists(published_csv):
            fig, ax = plt.subplots(figsize=(10, 5))
            for det, S in combined.items():
                m = np.isfinite(S["depth"])
                ax.errorbar(S["wave"][m], S["depth"][m],
                            yerr=S["depth_err"][m], fmt="o", ms=4,
                            color=colors.get(det, "gray"), alpha=0.85,
                            label="Patchwork " + det)
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
                        fmt="s", ms=4, color="k", alpha=0.6,
                        label="Published (GO 4098)")
            ax.set(xlabel="Wavelength (μm)", ylabel="Transit depth (ppm)",
                   title="GJ 9827 d — Patchwork vs published (GO 4098)")
            ax.legend(fontsize=9)
            fig.tight_layout()
            savefig(fig, "gj9827d_reproduction_check")
            plt.close(fig)
            print(f"overlaid published spectrum: {published_csv}")
        elif published_csv:
            print(f"No published CSV at {published_csv} — skipped overlay.")
    return offset


# ------------------------------------------------------------------ fit

def phase_fit(raw_root: str, visits: dict, detectors: list,
              published_csv: str | None) -> dict:
    """juliet Stages 5-6. Runs in the ASTER environment; reads the Stage 3
    products the reduce phase left on disk."""
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
                      f"(expected under pipeline_outputs_directory{tag}) "
                      "— run --phase reduce first. Skipping.")
                continue
            lc = build_lightcurves(spectra, det, bl)
            print(f"{vname} {det}: {lc['time'].size} ints, {lc['wave'].size} "
                  f"channels, wl oot scatter={lc['oot_scatter_ppm']:.0f} ppm, "
                  f"t0_obs={lc['t0_obs']:.5f}", flush=True)
            wl = wl_fits[(vname, det)] = fit_white_light(juliet, lc, vname, det)
            print(f"===== spectroscopic fits: {vname} {det} =====", flush=True)
            sp_fits[(vname, det)] = fit_spectroscopic(
                juliet, lc, wl["orbit"], vname, det)

    if not sp_fits:
        raise SystemExit("No fits produced — nothing to combine.")

    combined, summary = {}, {"planet": "GJ 9827 d", "mode": "as-is",
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
        phase_fit(raw_root, visits, detectors, published)

    print(f"\nDone. Outputs in {workdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
