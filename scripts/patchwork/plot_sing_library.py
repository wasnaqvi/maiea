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
DISPLAY_ERR_CLIP = 3.0
DISPLAY_ERR_MAX_H = 3.0     # and never draw a bin whose error exceeds this many H
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
        r["show"] = (r["z_err"] <= DISPLAY_ERR_CLIP * med) & (r["z_err"] <= DISPLAY_ERR_MAX_H)
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


def white_depths(root: Path, slug: str) -> dict[str, tuple[float, float]]:
    """Patchwork white-light depth per detector, inverse-variance over visits."""
    import json
    path = root / slug / "patchwork_summary.json"
    if not path.is_file():
        return {}
    with path.open() as handle:
        visits = json.load(handle).get("visits") or {}
    acc: dict[str, list[tuple[float, float]]] = {}
    for v in visits.values():
        for det, f in (v.get("fits") or {}).items():
            w = f.get("white_depth_ppm") or {}
            if w.get("median") is None:
                continue
            acc.setdefault(det, []).append(
                (float(w["median"]), max(float(w.get("plus", 0)), float(w.get("minus", 0)), 1e-9)))
    out = {}
    for det, vals in acc.items():
        d = np.array([x[0] for x in vals]); e = np.array([x[1] for x in vals])
        wts = 1 / e**2
        out[det] = (float(np.sum(d * wts) / np.sum(wts)), float(1 / np.sqrt(np.sum(wts))))
    return out


def load_literature() -> dict:
    import json
    path = HERE / "literature_g395h.json"
    if not path.is_file():
        return {}
    with path.open() as handle:
        return {k: v for k, v in json.load(handle).items() if not k.startswith("_")}


def _colors(items):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("sing", SING_COLORS[::-1])
    teqs = np.array([it["p"]["teq"] for it in items])
    norm = plt.Normalize(teqs.min(), teqs.max())
    return {it["slug"]: cmap(norm(it["p"]["teq"])) for it in items}, cmap, norm


def plot_stack(items, out: Path, resolution: float, spacing: float) -> list[str]:
    """Sing et al. (2016) style: one row per planet, hot at the top.

    Points only. Each row is its own zero (dotted). Rows are at least
    ``spacing`` scale heights apart and further where a row's own points
    and error bars need the room (a low-precision planet would otherwise
    run through its neighbours). No cumulative tick numbers -- a scale bar
    gives the vertical size instead.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors, _, _ = _colors(items)
    n = len(items)

    def extent(it):
        """(below, above) the row's zero that its drawn points occupy."""
        lo, hi = [0.0], [0.0]
        for r in it["alt"].values():
            k = r["show"]
            lo += list(r["z"][k] - r["z_err"][k]); hi += list(r["z"][k] + r["z_err"][k])
        return -np.percentile(lo, 10), np.percentile(hi, 90)

    ext = [extent(it) for it in items]
    bases = np.zeros(n)
    for i in range(n - 2, -1, -1):            # bottom row (index n-1) at 0
        gap = max(spacing, ext[i + 1][1] + ext[i][0] + 0.8)
        bases[i] = bases[i + 1] + gap
    height = bases[0] + ext[0][1] + ext[-1][0] + 2
    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(9.0, 0.19 * height + 1.2))
        for i, it in enumerate(items):           # items sorted hot -> cool
            base = bases[i]
            col = colors[it["slug"]]
            ax.axhline(base, color=col, lw=0.8, ls=":", alpha=0.7, zorder=1)
            for det, r in it["alt"].items():
                k = r["show"]
                ax.errorbar(r["wave"][k], base + r["z"][k], yerr=r["z_err"][k],
                            fmt="o", ms=5.2, mfc=col, mec="#1F2937", mew=0.6,
                            ecolor=col, elinewidth=1.1, capsize=0, zorder=3)
            ax.text(5.18, base, it["name"], color=col, fontsize=11, fontweight="bold",
                    va="bottom")
            ax.text(5.18, base, f"{it['p']['teq']:.0f} K", color=col, fontsize=8.5, va="top")
        ax.axvspan(*GAP_UM, color="#D5D9E0", alpha=0.6, lw=0, zorder=0)
        for label, w in FEATURES.items():
            ax.axvline(w, color="#9CA3AF", lw=0.8, ls=(0, (4, 4)), zorder=0)
            ax.text(w, 1.005, label, ha="center", va="bottom", fontsize=10.5, color="#4B5563",
                    transform=ax.get_xaxis_transform())
        # vertical scale bar: 2 scale heights
        x0, y0 = 2.86, -ext[-1][0] - 1.6
        ax.plot([x0, x0], [y0, y0 + 2], color="#111827", lw=2.2, solid_capstyle="butt")
        ax.text(x0 + 0.03, y0 + 1, "2 H", fontsize=9.5, va="center", color="#111827")
        ax.set_xlim(2.8, 5.15)
        ax.set_ylim(-ext[-1][0] - 2.0, bases[0] + ext[0][1] + 0.6)
        ax.set_yticks([])
        ax.set_xlabel(r"Wavelength  ($\mu$m)", fontsize=11.5)
        ax.set_ylabel(r"Relative altitude  $z(\lambda)/H_{\rm eq}$   ($\mu$ = 2.3)", fontsize=11.5)
        ax.set_title(f"Patchwork — JWST NIRSpec/G395H transmission spectra  (R ≈ {resolution:g})",
                     fontsize=12.5, pad=24)
        ax.text(0.0, -0.045,
                f"Uniform Patchwork fit v1.6. Rows ≥{spacing:g} H apart; dotted line = each planet's mean. "
                "As measured (no NRS1–NRS2 offset applied). Grey band: detector gap. "
                f"Bins with errors >{DISPLAY_ERR_CLIP:g}× the planet's median or >{DISPLAY_ERR_MAX_H:g} H not drawn.",
                transform=ax.transAxes, fontsize=7.8, color="#6B7280", va="top")
        fig.subplots_adjust(right=0.82)
        paths = _save(fig, out, "patchwork_library_stack")
        plt.close(fig)
    return paths


def plot_comparison(items, lit, out: Path) -> list[str]:
    """Patchwork against the published JWST results, one row per planet.

    Left: Patchwork white-light depth / published depth, per detector.
    Right: Patchwork's cloud-free mu floor (bar, for the assumed clear-
    atmosphere bracket) against the published mu constraint (black).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors, cmap, norm = _colors(items)
    rows = list(items)                         # hot at the top
    n = len(rows)
    with plt.rc_context(_style()):
        fig, (axd, axm) = plt.subplots(1, 2, figsize=(13.5, 0.55 * n + 2.0), sharey=True,
                                       gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.05})
        ys = np.arange(n)[::-1]
        for y, it in zip(ys, rows):
            col = colors[it["slug"]]
            ref = lit.get(it["name"], {})
            held = it.get("held", False)
            # -- depth ratio ---------------------------------------------
            pub = ref.get("depth_ppm")
            wd = it.get("white", {})
            if pub:
                for k, det in enumerate(("NRS1", "NRS2")):
                    if det not in wd:
                        continue
                    pd = pub[k] if len(pub) > 1 else pub[0]
                    ratio = wd[det][0] / pd
                    err = wd[det][1] / pd
                    axd.errorbar(ratio, y + (0.12 if k == 0 else -0.12), xerr=err,
                                 fmt="o" if det == "NRS1" else "s", ms=6.5,
                                 mfc="white" if held else col, mec="#1F2937", mew=0.8,
                                 ecolor="#6B7280", elinewidth=1.0, capsize=0, zorder=3)
            else:
                axd.text(1.0, y, "no published G395H depth", ha="center", va="center",
                         fontsize=8, color="#9CA3AF")
            # -- mu ------------------------------------------------------
            m = it["mod"]
            lo, hi = (mu_floor(m["amp_H_2sig"], a) for a in A_CLEAR_RMS_H)
            if np.isfinite(lo) and hi > MU_H2HE * 1.02 and not held:
                axm.plot([lo, hi], [y + 0.1, y + 0.1], color=col, lw=7, alpha=0.9,
                         solid_capstyle="butt", zorder=3)
                axm.annotate("", xy=(hi * 1.35, y + 0.1), xytext=(hi, y + 0.1),
                             arrowprops=dict(arrowstyle="-|>", color=col, lw=2), zorder=3)
            elif not held:
                axm.plot(MU_H2HE, y + 0.1, "o", mfc="white", mec=col, mew=2, ms=8, zorder=3)
            pm = ref.get("mu") or {}
            if "lower" in pm:
                axm.annotate("", xy=(pm["lower"] * 1.6, y - 0.18), xytext=(pm["lower"], y - 0.18),
                             arrowprops=dict(arrowstyle="-|>", color="#111827", lw=1.6), zorder=4)
                axm.plot(pm["lower"], y - 0.18, "|", color="#111827", ms=11, mew=2, zorder=4)
            elif "low" in pm:
                axm.plot([pm["low"], pm["high"]], [y - 0.18, y - 0.18], color="#111827", lw=2.4,
                         solid_capstyle="butt", zorder=4)
                axm.plot([pm["low"], pm["high"]], [y - 0.18, y - 0.18], "|", color="#111827",
                         ms=11, mew=2, zorder=4)
            label = ref.get("verdict", "no published JWST spectrum")
            if held:
                label = "HELD in Patchwork (GP absorbs depth) · " + label
            axm.text(75, y, label, fontsize=7.8, va="center", ha="left",
                     color="#374151" if ref else "#9CA3AF")
        # reference lines
        axd.axvline(1.0, color="#111827", lw=1)
        axd.axvspan(0.95, 1.05, color="#E5E7EB", lw=0, zorder=0)
        axd.set_xlim(0.6, 1.25)
        axd.set_xlabel("Patchwork ÷ published white-light depth", fontsize=10.5)
        from matplotlib.lines import Line2D
        axd.legend(handles=[
            Line2D([], [], marker="o", ls="", mfc="#9CA3AF", mec="#1F2937", ms=7, label="NRS1"),
            Line2D([], [], marker="s", ls="", mfc="#9CA3AF", mec="#1F2937", ms=7, label="NRS2"),
            Line2D([], [], marker="o", ls="", mfc="white", mec="#1F2937", ms=7, label="held")],
            loc="lower left", fontsize=8.5, frameon=False, handletextpad=0.3)
        axd.set_yticks(ys)
        axd.set_yticklabels([it["name"] for it in rows], fontsize=10.5)
        for t, it in zip(axd.get_yticklabels(), rows):
            t.set_color(colors[it["slug"]]); t.set_fontweight("bold")
        for mu, ls, lab in ((MU_H2HE, "--", "H$_2$/He\n2.3"), (18.0, ":", "H$_2$O\n18"),
                            (44.0, "-.", "CO$_2$\n44")):
            axm.axvline(mu, color="#6B7280", lw=1.0, ls=ls, zorder=1)
            axm.text(mu, n - 0.35, lab, ha="center", va="bottom", fontsize=8.5, color="#4B5563")
        axm.set_xscale("log")
        axm.set_xlim(1.9, 70)
        axm.set_xticks([2, 3, 5, 10, 20, 50])
        axm.set_xticklabels(["2", "3", "5", "10", "20", "50"])
        axm.set_xlabel("Mean molecular weight  μ  (amu)", fontsize=10.5)
        axm.legend(handles=[
            Line2D([], [], color="#7A3F99", lw=6, label="Patchwork cloud-free floor"),
            Line2D([], [], marker="o", ls="", mfc="white", mec="#7A3F99", mew=2, ms=7,
                   label="Patchwork: no constraint beyond H$_2$/He"),
            Line2D([], [], color="#111827", lw=2, marker="|", ms=10, mew=2,
                   label="published constraint")],
            loc="upper right", bbox_to_anchor=(1.0, 0.94), fontsize=8.5, frameon=True,
            framealpha=0.95, edgecolor="#E5E7EB")
        axm.set_ylim(-0.7, n + 0.4)
        fig.suptitle("Patchwork v1.6 against the published JWST results", fontsize=13.5, y=0.995)
        fig.text(0.01, -0.01,
                 "Depth ratio: same detector where the paper gives both. Patchwork floor: 2σ modulation limit "
                 f"for an assumed clear-atmosphere rms of {A_CLEAR_RMS_H[0]}–{A_CLEAR_RMS_H[1]} H (bar spans the bracket; "
                 "arrow = floor). Published μ: quoted, or converted from metallicity for a solar-scaled gas "
                 "(scripts/patchwork/literature_g395h.json lists which). Both are floors on 'μ or aerosols'.",
                 fontsize=7.6, color="#6B7280", wrap=True)
        fig.subplots_adjust(right=0.62, top=0.93)
        paths = _save(fig, out, "patchwork_literature_comparison")
        plt.close(fig)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="Directory of per-target Patchwork outputs.")
    ap.add_argument("-o", "--out-dir", default=".")
    ap.add_argument("--hold", nargs="*", default=[],
                    help="Target slugs kept out of the stack (shown as held in the comparison).")
    ap.add_argument("--resolution", type=float, default=30.0,
                    help="Display resolution for the stacked figure (R=100 input).")
    ap.add_argument("--spacing", type=float, default=4.0,
                    help="Row offset in scale heights for the stacked figure.")
    args = ap.parse_args(argv)

    root = Path(os.path.expanduser(args.root))
    out = Path(os.path.expanduser(args.out_dir))
    out.mkdir(parents=True, exist_ok=True)
    slugs = sorted(p.name for p in root.iterdir() if (p / "combined").is_dir())
    names = {s: planet_name(root, s) for s in slugs}
    params = planet_params(list(names.values()))
    # planet_params keys some planets by a display alias (TOI-836.01 ->
    # "TOI-836 c"); look them up by the archive name as well.
    params.update({v["archive_name"]: v for v in list(params.values())})
    lit = load_literature()
    items, missing = [], []
    for slug in slugs:
        name = names[slug]
        spec = load_combined(root, slug)
        if name not in params or not spec:
            missing.append(slug)
            continue
        p = params[name]
        alt, mean_ppm = to_altitude(spec, p, args.resolution)
        items.append({"slug": slug, "name": name, "p": p, "alt": alt, "mean_ppm": mean_ppm,
                      "mod": modulation(spec, p), "white": white_depths(root, slug),
                      "held": slug in args.hold})
    items.sort(key=lambda it: -it["p"]["teq"])
    if not items:
        print("No spectra found.")
        return 1

    print(f"{'planet':13s} {'Teq':>5s} {'ppm/H':>6s} {'err/H':>6s} {'A_H<2s':>7s} {'mu floor':>11s}  "
          f"{'white NRS1/NRS2':>17s}  {'published':>15s}  ratio")
    with (out / "patchwork_library_table.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["planet", "held", "teq_K", "mass_Mearth", "radius_Rearth", "ppm_per_H",
                    "white_nrs1_ppm", "white_nrs1_err", "white_nrs2_ppm", "white_nrs2_err",
                    "published_depth_ppm", "median_err_H", "chi2_red_flat", "nrs2_offset_ppm",
                    "modulation_H_2sig", f"mu_floor_clear{A_CLEAR_RMS_H[0]}",
                    f"mu_floor_clear{A_CLEAR_RMS_H[1]}", "published_mu", "published_ref"])
        for it in sorted(items, key=lambda it: it["p"]["teq"]):
            p, m, wd = it["p"], it["mod"], it["white"]
            ref = lit.get(it["name"], {})
            lo, hi = (mu_floor(m["amp_H_2sig"], a) for a in A_CLEAR_RMS_H)
            n1 = wd.get("NRS1", (np.nan, np.nan)); n2 = wd.get("NRS2", (np.nan, np.nan))
            pub = ref.get("depth_ppm")
            ratio = ""
            if pub:
                p1, p2 = pub[0], pub[-1]
                ratio = f"{n1[0] / p1:.3f}/{n2[0] / p2:.3f}"
            print(f"{it['name']:13s} {p['teq']:5.0f} {p['delta_ppm']:6.0f} {m['median_err_H']:6.2f} "
                  f"{m['amp_H_2sig']:7.2f} {lo:5.1f}-{hi:5.1f}  {n1[0]:7.0f}/{n2[0]:<7.0f}  "
                  f"{'/'.join(f'{x:.0f}' for x in pub) if pub else '-':>15s}  {ratio}"
                  + ("  HELD" if it["held"] else ""))
            w.writerow([it["name"], it["held"], p["teq"], p["mp_me"], p["rp_re"], p["delta_ppm"],
                        n1[0], n1[1], n2[0], n2[1], "/".join(f"{x:g}" for x in pub) if pub else "",
                        m["median_err_H"], m["chi2_red"], m["nrs2_offset_ppm"], m["amp_H_2sig"], lo, hi,
                        json_dumps(ref.get("mu")), ref.get("ref", "")])
    shown = [it for it in items if not it["held"]]
    for path in plot_stack(shown, out, args.resolution, args.spacing) + plot_comparison(items, lit, out):
        print(path)
    if missing:
        print("No archive parameters or spectra (left out): " + ", ".join(missing))
    return 0


def json_dumps(x) -> str:
    import json
    return json.dumps(x) if x else ""


if __name__ == "__main__":
    raise SystemExit(main())
