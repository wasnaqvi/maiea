#!/usr/bin/env python
"""Sing-style survey figures for Patchwork: stacked spectra in scale heights,
and the spectral-modulation / mean-molecular-weight trend.

Modelled on the HST Spectral Library figure (Sing et al. 2016, Nature):
every planet's transmission spectrum as a relative altitude z/H, one row per
planet, coloured from hot to cool. Data only -- no model curves (the library
decision of 2026-10-20: models come with the paper).

    python scripts/patchwork/plot_sing_library.py ~/Patchwork_v16 -o figs \\
        --hold TOI_125_b TOI_270_c

Writes ``patchwork_library_stack.{pdf,svg,png}``,
``patchwork_mmw_trend.{pdf,svg,png}`` and ``patchwork_library_table.csv``.

Conventions
-----------
* Altitude: z(lambda) = R* (sqrt(D) - sqrt(<D>)) / H, with H = k Teq / (mu m_u g)
  at mu = 2.3 (solar H2/He) -- the normalisation Sing et al. used, so a
  feature's height reads directly against a clear hydrogen atmosphere.
  <D> is each planet's inverse-variance mean over both detectors.
* Spectra are shown as measured: no NRS1-NRS2 offset is applied. The
  fitted offset is listed in the table and printed by each row.
* Modulation: the wavelength-independent intrinsic scatter beyond the
  quoted errors, profiled with a free mean and a free NRS2 offset
  (``_profile_intrinsic_scatter`` in plot_survey_scale_heights.py), in
  units of H. Reported as a 2-sigma upper limit unless it is detected.
* mu floor: a cloud-free atmosphere's modulation scales as 1/mu, so
  mu >= 2.3 * A_clear / A_2sigma. A_clear -- the rms modulation a clear
  solar-composition G395H spectrum would show -- is NOT measured here; the
  figure draws the floor for the bracket ``A_CLEAR_RMS_H`` and says so.
  Aerosols mute features exactly as high mu does, so a floor is a floor on
  "mu or clouds", never a composition measurement. Replace the bracket with
  a forward-model grid before quoting any number.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from plot_survey_scale_heights import (  # noqa: E402
    MU_H2HE,
    _profile_intrinsic_scatter,
    planet_params,
    pretty,
)
from rebin_spectra import rebin  # noqa: E402

# Bracket on the rms modulation (in H at mu = 2.3) of a CLEAR solar-composition
# sub-Neptune across G395H (CH4 3.3 um, CO2 4.3 um, H2O, CO). An assumption,
# drawn as a band so it cannot be mistaken for a measurement.
A_CLEAR_RMS_H = (0.5, 1.0)
GAP_UM = (3.72, 3.82)
# Display only: a rebinned point whose error is this many times the planet's
# median is an under-filled edge bin; it is drawn as nothing rather than as an
# error bar crossing three rows. It stays in the modulation statistic (which
# weights it by 1/sigma^2 anyway) and in the table.
DISPLAY_ERR_CLIP = 4.0
FEATURES = {r"CH$_4$": 3.32, r"SO$_2$": 4.05, r"CO$_2$": 4.30, "CO": 4.65}
# Hot (top) -> cool (bottom), after Sing et al. 2016.
SING_COLORS = ["#F08A3C", "#D4425A", "#B2387E", "#7A3F99", "#4B53A8", "#3C7CC0"]


def planet_name(root: Path, slug: str) -> str:
    """The archive name the fit used. Slugs lose spaces and component letters
    (HD_15337_b, TOI_4336_A_b), so never rebuild a name from one."""
    import json
    path = root / slug / "patchwork_summary.json"
    if path.is_file():
        with path.open() as handle:
            name = json.load(handle).get("planet_name")
        if name:
            return str(name)
    return pretty(slug)


def load_combined(root: Path, slug: str) -> dict[str, dict[str, np.ndarray]]:
    from aster_toolkit.data_reduction.juliet import read_spectrum_csv
    out = {}
    for det in ("NRS1", "NRS2"):
        f = root / slug / "combined" / f"combined_{det.lower()}_transmission_spectrum.csv"
        if f.is_file():
            out[det] = read_spectrum_csv(f)
    return out


def to_altitude(spec, p, resolution):
    """Rebin each detector and convert to z/H about the planet's mean depth."""
    rows = {}
    for det, s in spec.items():
        rb = rebin(s["wave"], s["wave_err"], s["depth_ppm"], s["depth_err_ppm"],
                   np.full(len(s["wave"]), np.nan), resolution=resolution)
        rows[det] = {k: np.array([r[k] for r in rb]) for k in ("wave", "wave_err", "depth", "depth_err")}
        # rebin works in ppm in and returns fraction/1e6; back to ppm.
        rows[det]["depth"] *= 1e6
        rows[det]["depth_err"] *= 1e6
    d = np.concatenate([rows[k]["depth"] for k in rows])
    e = np.concatenate([rows[k]["depth_err"] for k in rows])
    mean_ppm = float(np.sum(d / e**2) / np.sum(1 / e**2))
    h_m = p["H_km"] * 1e3
    for r in rows.values():
        depth = np.clip(r["depth"], 1e-3, None) * 1e-6
        r["z"] = p["rs_m"] * (np.sqrt(depth) - np.sqrt(mean_ppm * 1e-6)) / h_m
        r["z_err"] = p["rs_m"] * (r["depth_err"] * 1e-6) / (2 * np.sqrt(depth)) / h_m
    med = float(np.median(np.concatenate([r["z_err"] for r in rows.values()])))
    for r in rows.values():
        r["show"] = r["z_err"] <= DISPLAY_ERR_CLIP * med
    return rows, mean_ppm


def modulation(spec, p):
    """Profiled intrinsic modulation (R=100 channels) in scale heights."""
    w = np.concatenate([spec[k]["wave"] for k in spec])
    d = np.concatenate([spec[k]["depth_ppm"] for k in spec])
    e = np.concatenate([spec[k]["depth_err_ppm"] for k in spec])
    n2 = np.concatenate([np.full(len(spec[k]["wave"]), k == "NRS2") for k in spec])
    ok = np.isfinite(w) & np.isfinite(d) & np.isfinite(e) & (e > 0)
    prof = _profile_intrinsic_scatter(d[ok], e[ok], n2[ok])
    delta = p["delta_ppm"]
    detected = prof["delta_loglike"] > 2.0     # ~2 sigma for one parameter
    return {
        "amp_H": prof["scatter_ppm"] / delta,
        "amp_H_2sig": prof["scatter_2sig_ppm"] / delta,
        "detected": bool(detected),
        "chi2_red": prof["chi2_red"],
        "nrs2_offset_ppm": prof["detector_offset_ppm"],
        "median_err_H": float(np.median(e[ok])) / delta,
    }


def mu_floor(amp_2sig, a_clear):
    if not np.isfinite(amp_2sig) or amp_2sig <= 0:
        return np.nan
    return max(MU_H2HE, MU_H2HE * a_clear / amp_2sig)


def _style():
    return {
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.linewidth": 1.0, "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    }


def _save(fig, out: Path, stem: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "svg", "png"):
        path = out / f"{stem}.{ext}"
        fig.savefig(path, dpi=200 if ext == "png" else None)
        paths.append(str(path))
    return paths


def plot_stack(items, out: Path, resolution: float, spacing: float) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("sing", SING_COLORS[::-1])
    teqs = np.array([it["p"]["teq"] for it in items])
    norm = plt.Normalize(teqs.min(), teqs.max())
    n = len(items)
    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(8.6, 1.05 * n + 1.6))
        for i, it in enumerate(items):           # items sorted hot -> cool
            base = (n - 1 - i) * spacing
            col = cmap(norm(it["p"]["teq"]))
            for det, r in it["alt"].items():
                k = r["show"]
                ax.errorbar(r["wave"][k], base + r["z"][k], yerr=r["z_err"][k], xerr=r["wave_err"][k],
                            fmt="o", ms=3.6, mfc="#3A3A3A", mec="#1E1E1E", mew=0.5,
                            ecolor=col, elinewidth=1.0, capsize=0, zorder=3)
                ax.plot(r["wave"][k], base + r["z"][k], "-", color=col, lw=1.6, alpha=0.75, zorder=2)
            ax.axhline(base, color=col, lw=0.6, ls=":", alpha=0.6, zorder=1)
            m = it["mod"]
            ax.text(5.22, base + 0.25 * spacing, it["name"], color=col, fontsize=10.5,
                    fontweight="bold", va="center")
            ax.text(5.22, base - 0.12 * spacing,
                    f"{it['p']['teq']:.0f} K · {it['p']['delta_ppm']:.0f} ppm/H",
                    color=col, fontsize=8, va="center")
        ax.axvspan(*GAP_UM, color="#C9CED6", alpha=0.45, lw=0, zorder=0)
        top = (n - 1) * spacing + 0.7 * spacing
        for label, w in FEATURES.items():
            ax.axvline(w, color="#8A8F99", lw=0.8, ls=(0, (3, 3)), zorder=0)
            ax.text(w, 1.005, label, ha="center", va="bottom", fontsize=10, color="#555B66",
                    transform=ax.get_xaxis_transform())
        ax.set_xlim(2.8, 5.2)
        ax.set_ylim(-0.8 * spacing, top)
        ax.set_xlabel(r"Wavelength  ($\mu$m)")
        ax.set_ylabel(r"Relative altitude in scale heights  $z(\lambda)/H_{\rm eq}$"
                      "\n" r"($\mu$ = 2.3, rows offset by " f"{spacing:g})")
        ax.set_yticks(np.arange(0, n) * spacing)
        ax.set_yticklabels([f"{k * spacing:g}" for k in range(n)])
        ax.set_title(f"Patchwork — JWST NIRSpec/G395H transmission spectra (R≈{resolution:g})",
                     fontsize=12, pad=26)
        fig.text(0.01, 0.005,
                 "Uniform Patchwork fit v1.6. Spectra as measured (no NRS1–NRS2 offset applied); "
                 f"shaded band: detector gap; bins with errors >{DISPLAY_ERR_CLIP:g}× the planet's median "
                 "not drawn.", fontsize=7.5, color="#6B7280")
        fig.subplots_adjust(right=0.80)
        paths = _save(fig, out, "patchwork_library_stack")
        plt.close(fig)
    return paths


def _repel_labels(ax, xs, ys, labels, color="#4B5563", fontsize=7.5):
    """Place labels beside points, nudged apart vertically so none overlap.
    A few lines of greedy repulsion in display space -- enough for ~20 points
    without pulling in adjustText."""
    if not labels:
        return
    trans = ax.transData
    pts = trans.transform(np.column_stack([xs, ys]))
    order = np.argsort(pts[:, 1])
    pos = pts.copy()
    pos[:, 0] += 6
    min_gap = fontsize * 1.35
    for _ in range(60):
        moved = False
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                a, b = order[i], order[j]
                if abs(pos[a, 0] - pos[b, 0]) < 70 and abs(pos[a, 1] - pos[b, 1]) < min_gap:
                    shift = (min_gap - abs(pos[a, 1] - pos[b, 1])) / 2 + 0.2
                    lo, hi = (a, b) if pos[a, 1] <= pos[b, 1] else (b, a)
                    pos[lo, 1] -= shift
                    pos[hi, 1] += shift
                    moved = True
        if not moved:
            break
    inv = trans.inverted()
    for k, lab in enumerate(labels):
        tx, ty = inv.transform(pos[k])
        ax.annotate(lab, (xs[k], ys[k]), xytext=(tx, ty), textcoords="data",
                    fontsize=fontsize, color=color, va="center",
                    arrowprops=dict(arrowstyle="-", color="#C4C9D1", lw=0.5,
                                    shrinkA=0, shrinkB=3))


def plot_trend(items, held, out: Path) -> list[str]:
    """(a) modulation limit vs precision, (b) mu floor vs mass, (c) vs Teq.

    Panel (a) is there because the limits track precision (log-log
    correlation ~0.8 on the v1.6 set): a planet sitting on the precision
    line has been measured, not constrained. Read (b) and (c) through it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FixedLocator, NullFormatter

    cmap = LinearSegmentedColormap.from_list("sing", SING_COLORS[::-1])
    teqs = np.array([it["p"]["teq"] for it in items])
    norm = plt.Normalize(teqs.min(), teqs.max())
    names = [it["name"] for it in items]
    prec = np.array([it["mod"]["median_err_H"] for it in items])
    a2 = np.array([it["mod"]["amp_H_2sig"] for it in items])
    det = np.array([it["mod"]["detected"] for it in items])
    mass = np.array([it["p"]["mp_me"] for it in items])
    flo = np.array([mu_floor(a, A_CLEAR_RMS_H[0]) for a in a2])
    fhi = np.array([mu_floor(a, A_CLEAR_RMS_H[1]) for a in a2])
    cols = [cmap(norm(t)) for t in teqs]

    with plt.rc_context(_style()):
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4),
                                 gridspec_kw={"width_ratios": [1, 1.15, 1.15]})
        ax = axes[0]
        for x, y, c, d in zip(prec, a2, cols, det):
            ax.plot(x, y, "o" if d else "v", ms=8, color=c, mec="#1E1E1E", mew=0.6, zorder=3)
        lim = np.array([0.1, 6])
        ax.plot(lim, lim, color="#9CA3AF", lw=1, ls="--")
        ax.text(1.5, 1.25, "limit = channel precision", rotation=40, fontsize=8, color="#6B7280")
        ax.axhspan(*A_CLEAR_RMS_H, color="#F08A3C", alpha=0.15, lw=0)
        ax.text(0.12, 0.72, "clear H$_2$/He bracket (assumed)", fontsize=8, color="#B45309")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(0.15, 5); ax.set_ylim(0.15, 5)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator([0.2, 0.3, 0.5, 1, 2, 3, 5]))
            axis.set_minor_formatter(NullFormatter())
            axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.set_xlabel("Channel precision  (median σ, scale heights)")
        ax.set_ylabel("Spectral modulation, 2σ limit  (rms, scale heights)")
        ax.set_title("(a) what each spectrum can rule out", fontsize=10.5, loc="left")
        _repel_labels(ax, prec, a2, names)

        for ax, x, xlabel, title, logx in (
                (axes[1], mass, r"Planet mass  ($M_\oplus$)", "(b) μ floor vs mass", True),
                (axes[2], teqs, r"Equilibrium temperature  (K)", "(c) μ floor vs temperature", False)):
            for xi, lo, hi, c in zip(x, flo, fhi, cols):
                if np.isfinite(lo) and hi > MU_H2HE * 1.02:
                    ax.vlines(xi, lo, hi, color=c, lw=5, alpha=0.85, zorder=3)
                    ax.plot(xi, hi, "^", color=c, mec="#1E1E1E", mew=0.5, ms=8, zorder=4)
                else:
                    ax.plot(xi, MU_H2HE, "o", mfc="white", mec=c, mew=1.6, ms=7, zorder=4)
            for mu, ls, lab in ((MU_H2HE, "--", r"H$_2$/He  μ = 2.3"),
                                (18.0, ":", "H$_2$O steam  μ = 18"),
                                (44.0, "-.", r"CO$_2$  μ = 44")):
                ax.axhline(mu, color="#6B7280", lw=1.0, ls=ls)
                ax.text(0.98, mu * 1.07, lab, transform=ax.get_yaxis_transform(),
                        ha="right", fontsize=8, color="#6B7280")
            ax.set_yscale("log"); ax.set_ylim(1.8, 70)
            ax.yaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20, 50]))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_yticklabels(["2", "3", "5", "10", "20", "50"])
            if logx:
                ax.set_xscale("log")
                ax.xaxis.set_major_locator(FixedLocator([1, 2, 3, 5, 10, 20]))
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.set_xticklabels(["1", "2", "3", "5", "10", "20"])
                ax.set_xlim(1.2, 22)
            ax.set_xlabel(xlabel)
            ax.set_title(title, fontsize=10.5, loc="left")
            _repel_labels(ax, x, np.where(np.isfinite(fhi), np.maximum(fhi, MU_H2HE), MU_H2HE), names)
        axes[1].set_ylabel("Cloud-free mean molecular weight floor  μ  (amu)")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = fig.colorbar(sm, ax=axes[2], pad=0.02, fraction=0.05)
        cb.set_label(r"$T_{\rm eq}$ (K)")
        fig.suptitle("Patchwork — muted spectra and mean molecular weight", fontsize=13)
        note = ("Bars: μ floor for a clear-atmosphere rms of "
                f"{A_CLEAR_RMS_H[0]}–{A_CLEAR_RMS_H[1]} H (an assumption, not a model fit); open circle: "
                "no constraint beyond H$_2$/He. A floor is on 'μ or aerosols' — clouds mute features "
                "exactly as a heavier atmosphere does. Limits track precision (panel a).")
        if held:
            note += "  Held (GP absorbs transit depth): " + ", ".join(held) + "."
        fig.text(0.01, -0.02, note, fontsize=7.5, color="#6B7280", wrap=True)
        fig.tight_layout(rect=(0, 0.02, 1, 0.95))
        paths = _save(fig, out, "patchwork_mmw_trend")
        plt.close(fig)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="Directory of per-target Patchwork outputs.")
    ap.add_argument("-o", "--out-dir", default=".")
    ap.add_argument("--hold", nargs="*", default=[],
                    help="Target slugs to leave out (listed on the figure).")
    ap.add_argument("--resolution", type=float, default=40.0,
                    help="Display resolution for the stacked figure (R=100 input).")
    ap.add_argument("--spacing", type=float, default=0.0,
                    help="Row offset in scale heights (default: from the data).")
    args = ap.parse_args(argv)

    root = Path(os.path.expanduser(args.root))
    out = Path(os.path.expanduser(args.out_dir))
    slugs = sorted(p.name for p in root.iterdir()
                   if (p / "combined").is_dir() and p.name not in args.hold)
    names = {s: planet_name(root, s) for s in slugs}
    params = planet_params(list(names.values()))
    # planet_params keys some planets by a display alias (TOI-836.01 ->
    # "TOI-836 c"); look them up by the archive name as well.
    params.update({v["archive_name"]: v for v in list(params.values())})
    items, missing = [], []
    for slug in slugs:
        name = names[slug]
        if name not in params:
            missing.append(slug)
            continue
        spec = load_combined(root, slug)
        if not spec:
            continue
        p = params[name]
        alt, mean_ppm = to_altitude(spec, p, args.resolution)
        items.append({"slug": slug, "name": name, "p": p, "alt": alt,
                      "mean_ppm": mean_ppm, "mod": modulation(spec, p)})
    items.sort(key=lambda it: -it["p"]["teq"])
    if not items:
        print("No spectra found.")
        return 1
    # Row gap from the TYPICAL planet, not the noisiest: one low-precision
    # row should overlap its neighbours a little (as in Sing et al.) rather
    # than flatten every other spectrum.
    spacing = args.spacing or float(max(5.0, np.ceil(6 * np.nanmedian([
        np.nanmedian(np.concatenate([r["z_err"] for r in it["alt"].values()]))
        for it in items]))))

    print(f"{'planet':13s} {'Teq':>5s} {'Mp':>5s} {'ppm/H':>6s} {'err/H':>6s} {'chi2':>5s} "
          f"{'NRS2off':>7s} {'A_H':>6s} {'A_H<2s':>7s} {'mu floor':>11s}")
    with (out.mkdir(parents=True, exist_ok=True) or (out / "patchwork_library_table.csv").open("w", newline="")) as fh:
        w = csv.writer(fh)
        w.writerow(["planet", "teq_K", "mass_Mearth", "radius_Rearth", "ppm_per_H", "mean_depth_ppm",
                    "median_err_H", "chi2_red_flat", "nrs2_offset_ppm", "modulation_H",
                    "modulation_H_2sig", "modulation_detected",
                    f"mu_floor_clear{A_CLEAR_RMS_H[0]}", f"mu_floor_clear{A_CLEAR_RMS_H[1]}"])
        for it in sorted(items, key=lambda it: it["p"]["teq"]):
            p, m = it["p"], it["mod"]
            lo, hi = (mu_floor(m["amp_H_2sig"], a) for a in A_CLEAR_RMS_H)
            print(f"{it['name']:13s} {p['teq']:5.0f} {p['mp_me']:5.1f} {p['delta_ppm']:6.0f} "
                  f"{m['median_err_H']:6.2f} {m['chi2_red']:5.2f} {m['nrs2_offset_ppm']:+7.0f} "
                  f"{m['amp_H']:6.2f} {m['amp_H_2sig']:7.2f} {lo:5.1f}-{hi:5.1f}")
            w.writerow([it["name"], p["teq"], p["mp_me"], p["rp_re"], p["delta_ppm"], it["mean_ppm"],
                        m["median_err_H"], m["chi2_red"], m["nrs2_offset_ppm"], m["amp_H"],
                        m["amp_H_2sig"], m["detected"], lo, hi])
    held = [planet_name(root, s) for s in args.hold]
    for path in plot_stack(items, out, args.resolution, spacing) + plot_trend(items, held, out):
        print(path)
    if missing:
        print("No archive parameters (left out): " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
