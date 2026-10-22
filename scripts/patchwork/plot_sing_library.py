#!/usr/bin/env python
"""Sing-style survey figures for Patchwork: stacked spectra in scale heights,
and the spectral-modulation / mean-molecular-weight trend.

Modelled on the HST Spectral Library figure (Sing et al. 2016, Nature):
every planet's transmission spectrum as a relative altitude z/H, one row per
planet, coloured from hot to cool. Data only -- no model curves (the library
decision of 2026-10-20: models come with the paper).

    python scripts/patchwork/plot_sing_library.py ~/Patchwork_v16 -o figs \\
        --hold TOI_125_b TOI_270_c

Writes ``patchwork_library_stack{,_km}.{pdf,svg,png}``,
``patchwork_literature_comparison.{pdf,svg,png}`` and
``patchwork_library_table.csv``.

Conventions
-----------
* The measured quantity is the change in apparent radius,
  dR_p(lambda) = R* (sqrt(D(lambda)) - sqrt(<D>)), with <D> each planet's
  inverse-variance mean over both detectors. ``_km`` plots it as is: it
  needs only the depths and R*.
* The main stack divides it by a REFERENCE scale height,
  H_ref = k T_eq / (mu m_u g) with mu = 2.3 (solar H2/He). That is not the
  planet's scale height -- several of these planets are known not to be
  H2/He -- but the unit in which a clear hydrogen atmosphere's features
  span ~2-5, so a muted row reads directly as "not a clear H2/He
  atmosphere". It is labelled as a reference unit, not an altitude.
* T_eq is computed uniformly, T_eq = T_eff sqrt(R*/2a) (zero albedo, full
  redistribution), not taken from each discovery paper's own convention.
  g = G M_p / R_p^2 from the archive. sigma(H)/H propagates the archive
  errors on M_p, R_p, T_eff and a/R*; the idealisation in T_eq itself
  (albedo, redistribution) is not included. Rows with sigma(g)/g >
  ``FADE_SIGMA_G`` are drawn faded: their vertical scale is uncertain by
  more than the features they show.
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
    AMU,
    G_SI,
    K_B,
    M_EARTH,
    MU_H2HE,
    R_EARTH,
    R_SUN,
    _profile_intrinsic_scatter,
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
# A row whose surface gravity is uncertain by more than this has an uncertain
# vertical scale (in H units); it is drawn faded.
FADE_SIGMA_G = 0.30
AU_M = 1.495978707e11
# Hot (top) -> cool (bottom), after Sing et al. 2016.
SING_COLORS = ["#F08A3C", "#D4425A", "#B2387E", "#7A3F99", "#4B53A8", "#3C7CC0"]
# Row colours for the stack, hot (first) -> cool (last), after Sing et al.
# 2016 but saturated enough that neighbouring rows stay distinguishable.
# Sampled by RANK, not by T_eq, so two planets 10 K apart still differ.
STACK_COLORS = ["#F59E0B", "#F97316", "#EF4444", "#E11D48", "#DB2777", "#C026D3",
                "#9333EA", "#7C3AED", "#4F46E5", "#2563EB", "#0284C7", "#0891B2"]


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


def fetch_params(names: list[str]) -> dict[str, dict]:
    """Archive parameters with errors, T_eq computed uniformly.

    Every planet gets the same T_eq definition (zero albedo, full
    redistribution) from its host's T_eff and a/R*, so the scale-height
    unit differs between planets only through physics, not through each
    discovery paper's convention.
    """
    from aster_toolkit.data_acquisition.mast import archive_tap_query

    rows = archive_tap_query(
        ["pl_name in (" + ",".join("'" + n.replace("'", "''") + "'" for n in names) + ")"],
        columns=["pl_name", "pl_rade", "pl_radeerr1", "pl_radeerr2", "pl_bmasse",
                 "pl_bmasseerr1", "pl_bmasseerr2", "st_rad", "st_teff", "st_tefferr1",
                 "pl_ratdor", "pl_ratdorerr1", "pl_orbsmax", "pl_eqt"],
    )

    def f(r, k):
        try:
            v = float(r.get(k))
            return v if np.isfinite(v) else np.nan
        except (TypeError, ValueError):
            return np.nan

    def half(r, k):
        a, b = f(r, k + "err1"), f(r, k + "err2")
        vals = [abs(x) for x in (a, b) if np.isfinite(x)]
        return float(np.mean(vals)) if vals else np.nan

    out = {}
    for r in rows:
        rp_re, mp_me, rs = f(r, "pl_rade"), f(r, "pl_bmasse"), f(r, "st_rad")
        teff = f(r, "st_teff")
        ars = f(r, "pl_ratdor")
        ars_err = half(r, "pl_ratdor")
        if not np.isfinite(ars):
            ars = f(r, "pl_orbsmax") * AU_M / (rs * R_SUN)
            ars_err = np.nan
        if not all(np.isfinite(x) and x > 0 for x in (rp_re, mp_me, rs, teff, ars)):
            continue
        rp, mp = rp_re * R_EARTH, mp_me * M_EARTH
        teq = teff * np.sqrt(1.0 / (2.0 * ars))
        g = G_SI * mp / rp**2
        h = K_B * teq / (MU_H2HE * AMU * g)
        sg = np.hypot(half(r, "pl_bmasse") / mp_me, 2 * half(r, "pl_rade") / rp_re)
        st = np.hypot(f(r, "st_tefferr1") / teff if np.isfinite(f(r, "st_tefferr1")) else 0.0,
                      0.5 * ars_err / ars if np.isfinite(ars_err) else 0.0)
        out[r["pl_name"]] = {
            "archive_name": r["pl_name"], "rp_m": rp, "mp_kg": mp, "rs_m": rs * R_SUN,
            "rp_re": rp_re, "mp_me": mp_me, "teq": float(teq),
            "teq_archive": f(r, "pl_eqt"), "g": float(g), "H_km": float(h / 1e3),
            "delta_ppm": float(2 * rp * h / (rs * R_SUN) ** 2 * 1e6),
            "sigma_g_frac": float(sg) if np.isfinite(sg) else np.nan,
            "sigma_H_frac": float(np.hypot(sg, st)) if np.isfinite(sg) else np.nan,
        }
    return out


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
        r["km"] = p["rs_m"] * (np.sqrt(depth) - np.sqrt(mean_ppm * 1e-6)) / 1e3
        r["km_err"] = p["rs_m"] * (r["depth_err"] * 1e-6) / (2 * np.sqrt(depth)) / 1e3
        r["z"] = r["km"] * 1e3 / h_m
        r["z_err"] = r["km_err"] * 1e3 / h_m
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


def _rank_colors(n):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("stack", STACK_COLORS)
    return [cmap(x) for x in (np.linspace(0, 1, n) if n > 1 else [0.0])]


def plot_stack(items, out: Path, resolution: float, spacing: float,
               unit: str = "H") -> list[str]:
    """Sing et al. (2016) Figure 1 layout: one row per planet, hot at the top.

    Every row is offset by the same ``spacing`` and the y axis is numbered,
    as in the original, so a feature's size reads off the axis directly.
    Points are filled in the planet's colour with dark outlines and dark
    error bars (vertical: 1 sigma; horizontal: bin width); names sit inside
    the frame at the right end of each row.

    ``unit="H"``: dR_p in units of the reference H2/He scale height; each
    row also states H and its 1-sigma, and rows with sigma(g)/g >
    FADE_SIGMA_G are faded. ``unit="km"``: dR_p in km, no atmospheric
    assumption.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FixedLocator, NullFormatter

    key, ekey = ("z", "z_err") if unit == "H" else ("km", "km_err")
    n = len(items)
    cols = _rank_colors(n)
    top = n * spacing                       # rows at 0, spacing, ... (n-1)*spacing
    LABEL_X, XMAX = 5.15, 5.80
    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(13.0, 0.95 * n + 2.2))
        for i, it in enumerate(items):           # items sorted hot -> cool
            base = (n - 1 - i) * spacing
            col = cols[i]
            sg = it["p"].get("sigma_g_frac", np.nan)
            faded = unit == "H" and np.isfinite(sg) and sg > FADE_SIGMA_G
            a = 0.4 if faded else 1.0
            ax.axhline(base, color=col, lw=0.7, ls=(0, (1, 2)), alpha=0.6 * a, zorder=1)
            for det, r in it["alt"].items():
                k = r["show"]
                ax.errorbar(r["wave"][k], base + r[key][k], yerr=r[ekey][k], xerr=r["wave_err"][k],
                            fmt="none", ecolor="#374151", elinewidth=1.2, capsize=0,
                            alpha=0.85 * a, zorder=2)
                ax.plot(r["wave"][k], base + r[key][k], "o", ms=8.0, mfc=col, mec="#111827",
                        mew=0.8, alpha=a, zorder=3)
            # name inside the frame, right end, just above the row (as Sing)
            # name in a gutter inside the frame, past the last data (~5.1 um)
            ax.text(LABEL_X, base + 0.16 * spacing, it["name"], color=col, fontsize=15,
                    fontweight="bold", ha="left", va="center",
                    alpha=1.0 if not faded else 0.6, zorder=4)
            if unit == "H":
                sh = it["p"].get("sigma_H_frac", np.nan)
                sub = f"{it['p']['teq']:.0f} K · H {it['p']['H_km']:.0f} km ±{100 * sh:.0f}%"
            else:
                sub = f"{it['p']['teq']:.0f} K · {it['p']['rp_re']:.2f} R⊕"
            ax.text(LABEL_X, base - 0.12 * spacing, sub, color=col, fontsize=11, ha="left",
                    va="center", alpha=0.95 if not faded else 0.6, zorder=4)
        ax.axvspan(*GAP_UM, color="#E5E7EB", lw=0, zorder=0)
        for label, w in FEATURES.items():
            ax.axvline(w, color="#6B7280", lw=0.9, ls=(0, (5, 4)), zorder=0)
            ax.text(w, top - 0.12 * spacing, label, ha="center", va="top", fontsize=15,
                    color="#374151", zorder=4,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
        ax.set_xlim(2.80, XMAX)
        ax.set_ylim(-0.75 * spacing, top + 0.15 * spacing)
        ax.yaxis.set_major_locator(MultipleLocator(spacing))
        minor = spacing / 5 if unit == "H" else spacing / 2
        ax.yaxis.set_minor_locator(MultipleLocator(minor))
        ax.xaxis.set_major_locator(FixedLocator([3.0, 3.5, 4.0, 4.5, 5.0]))
        ax.axvline(LABEL_X - 0.04, color="#E5E7EB", lw=0.8, zorder=0)
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.tick_params(which="major", length=8, width=1.2, labelsize=14)
        ax.tick_params(which="minor", length=4, width=1.0)
        ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=16)
        if unit == "H":
            ax.set_ylabel(r"Relative altitude in H$_2$/He scale heights,  $\Delta R_p / H$"
                          "\n" r"($H = k_B T_{\rm eq}/\mu m_u g$,  $\mu$ = 2.3 reference; rows offset)",
                          fontsize=15)
        else:
            ax.set_ylabel(r"Relative altitude,  $\Delta R_p$  (km;  rows offset)", fontsize=15)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_linewidth(1.4)
        fig.subplots_adjust(left=0.085, right=0.985, top=0.99, bottom=0.045)
        paths = _save(fig, out, "patchwork_library_stack" + ("" if unit == "H" else "_km"))
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
    ap.add_argument("--spacing", type=float, default=5.0,
                    help="Constant row offset in scale heights for the stacked figure.")
    args = ap.parse_args(argv)

    root = Path(os.path.expanduser(args.root))
    out = Path(os.path.expanduser(args.out_dir))
    out.mkdir(parents=True, exist_ok=True)
    slugs = sorted(p.name for p in root.iterdir() if (p / "combined").is_dir())
    names = {s: planet_name(root, s) for s in slugs}
    params = fetch_params(list(names.values()))
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
        w.writerow(["planet", "held", "teq_K_uniform", "teq_K_archive", "mass_Mearth", "radius_Rearth",
                    "g_ms2", "sigma_g_frac", "H_km", "sigma_H_frac", "ppm_per_H",
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
            w.writerow([it["name"], it["held"], p["teq"], p["teq_archive"], p["mp_me"], p["rp_re"],
                        p["g"], p["sigma_g_frac"], p["H_km"], p["sigma_H_frac"], p["delta_ppm"],
                        n1[0], n1[1], n2[0], n2[1], "/".join(f"{x:g}" for x in pub) if pub else "",
                        m["median_err_H"], m["chi2_red"], m["nrs2_offset_ppm"], m["amp_H_2sig"], lo, hi,
                        json_dumps(ref.get("mu")), ref.get("ref", "")])
    shown = [it for it in items if not it["held"]]
    km_spacing = args.spacing * float(np.median([it["p"]["H_km"] for it in shown]))
    # one significant figure, so the numbered axis reads 0, 1000, 2000 km
    mag = 10 ** np.floor(np.log10(km_spacing))
    km_spacing = float(np.ceil(km_spacing / mag) * mag)
    for path in (plot_stack(shown, out, args.resolution, args.spacing, unit="H")
                 + plot_stack(shown, out, args.resolution, km_spacing, unit="km")
                 + plot_comparison(items, lit, out)):
        print(path)
    if missing:
        print("No archive parameters or spectra (left out): " + ", ".join(missing))
    return 0


def json_dumps(x) -> str:
    import json
    return json.dumps(x) if x else ""


if __name__ == "__main__":
    raise SystemExit(main())
