#!/usr/bin/env python
"""Sing-style survey figures for Patchwork: stacked spectra in scale heights,
and the spectral-modulation / mean-molecular-weight trend.

Modelled on the HST Spectral Library figure (Sing et al. 2016, Nature):
every planet's transmission spectrum as a relative altitude z/H, one row per
planet, coloured from hot to cool. Data only -- no model curves (the library
decision of 2026-10-20: models come with the paper).

    python scripts/patchwork/plot_sing_library.py ~/Patchwork_v16 -o figs \\
        --hold TOI_125_b TOI_270_c

Writes ``patchwork_library_stack{,_km}``, ``patchwork_band_indices``
(CH4 / CO2 band contrasts against T_eq) and ``patchwork_mu_landscape``
(published JWST mu constraints against T_eq and mass; data in
literature_subneptunes.json), each as pdf/svg/png, plus
``patchwork_library_table.csv``. See docs/patchwork_science_case.md.

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


def load_literature(filename: str = "literature_g395h.json") -> dict:
    import json
    path = HERE / filename
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


def _repel_labels(ax, xs, ys, labels, color="#374151", fontsize=9.0):
    """Label points, nudged apart vertically in display space so none overlap.
    Call after the axis limits and scales are final."""
    if not len(labels):
        return
    fig = ax.figure
    fig.canvas.draw()
    trans = ax.transData
    pts = trans.transform(np.column_stack([xs, ys]))
    pos = pts.copy()
    pos[:, 0] += 10
    gap = fontsize * 1.6
    order = np.argsort(pts[:, 1])
    for _ in range(80):
        moved = False
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                a, b = order[i], order[j]
                if abs(pos[a, 0] - pos[b, 0]) < 95 and abs(pos[a, 1] - pos[b, 1]) < gap:
                    shift = (gap - abs(pos[a, 1] - pos[b, 1])) / 2 + 0.3
                    lo, hi = (a, b) if pos[a, 1] <= pos[b, 1] else (b, a)
                    pos[lo, 1] -= shift
                    pos[hi, 1] += shift
                    moved = True
        if not moved:
            break
    inv = trans.inverted()
    for k, lab in enumerate(labels):
        tx, ty = inv.transform(pos[k])
        ax.annotate(lab, (xs[k], ys[k]), xytext=(tx, ty), textcoords="data", fontsize=fontsize,
                    color=color, va="center", ha="left", zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#9CA3AF", lw=0.6, shrinkA=0, shrinkB=5))


# Single-detector band indices: band minus neighbouring windows on the SAME
# detector, so the NRS1-NRS2 offset cannot enter. Windows avoid the band
# wings (CH4 3.15-3.55 um) and the SO2 4.0 / CO 4.6 um bands as far as
# G395H allows.
BAND_INDICES = {
    "CH4": {"band": (3.20, 3.45), "windows": [(2.87, 3.05), (3.58, 3.72)], "det": "NRS1",
            "label": r"CH$_4$ 3.3 $\mu$m"},
    "CO2": {"band": (4.20, 4.40), "windows": [(3.85, 4.10), (4.45, 4.55)], "det": "NRS2",
            "label": r"CO$_2$ 4.3 $\mu$m"},
}
# Regimes named in the literature, drawn on the T_eq axes.
COLD_TRAP_K = 350.0          # H2O cold trap, T_crit >~ 350 K (sub-Neptune review, arXiv:2509.19247)
AEROSOL_WINDOW_K = (500.0, 800.0)   # strongest muting (Brande et al. 2024, ApJL 961, L23)
NO_HAZE_K = 850.0            # no CH4 to photolyse into haze (Davenport et al. 2025)


def band_index(spec, p, name):
    """Band contrast in reference scale heights, with its 1-sigma."""
    cfg = BAND_INDICES[name]
    sp = spec.get(cfg["det"])
    if sp is None:
        return np.nan, np.nan
    w = np.asarray(sp["wave"], float)
    d = np.clip(np.asarray(sp["depth_ppm"], float) * 1e-6, 1e-9, None)
    e = np.asarray(sp["depth_err_ppm"], float) * 1e-6
    h = p["H_km"] * 1e3
    z = p["rs_m"] * np.sqrt(d) / h
    ze = p["rs_m"] * e / (2 * np.sqrt(d)) / h
    mb = (w >= cfg["band"][0]) & (w <= cfg["band"][1])
    mc = np.zeros_like(mb)
    for lo, hi in cfg["windows"]:
        mc |= (w >= lo) & (w <= hi)
    ok = np.isfinite(z) & np.isfinite(ze) & (ze > 0)
    mb, mc = mb & ok, mc & ok
    if mb.sum() < 2 or mc.sum() < 2:
        return np.nan, np.nan

    def wmean(m):
        wt = 1 / ze[m] ** 2
        return np.sum(z[m] * wt) / np.sum(wt), 1 / np.sqrt(np.sum(wt))

    (zb, eb), (zc, ec) = wmean(mb), wmean(mc)
    return float(zb - zc), float(np.hypot(eb, ec))


def _regimes(ax, ymax_text, fontsize=9.5):
    ax.axvspan(*AEROSOL_WINDOW_K, color="#FDE68A", alpha=0.35, lw=0, zorder=0)
    ax.axvspan(NO_HAZE_K, 2000, color="#E0E7FF", alpha=0.45, lw=0, zorder=0)
    ax.axvline(COLD_TRAP_K, color="#2563EB", lw=1.2, ls=(0, (6, 3)), zorder=1)
    kw = dict(transform=ax.get_xaxis_transform(), fontsize=fontsize, va="top")
    ax.text(COLD_TRAP_K - 8, ymax_text, "H$_2$O cold trap\n$T_{\\rm crit}\\gtrsim$350 K", ha="right",
            color="#1D4ED8", **kw)
    ax.text(np.mean(AEROSOL_WINDOW_K), ymax_text, "aerosol muting\nstrongest (Brande+24)",
            ha="center", color="#92400E", **kw)
    ax.text(NO_HAZE_K + 10, ymax_text, "no CH$_4$ haze\n(Davenport+25)", ha="left",
            color="#3730A3", **kw)


def plot_band_indices(items, lit_mu, out: Path) -> list[str]:
    """Patchwork's measured CH4 and CO2 band contrasts against T_eq.

    Each point is a measurement from the uniform v1.6 spectra, not a model.
    Planets whose published spectra detect the molecule are ringed, which
    is the check that the index measures what it says.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with plt.rc_context(_style()):
        fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6), sharex=True)
        for ax, name in zip(axes, ("CH4", "CO2")):
            xs, ys, labels = [], [], []
            for it in items:
                v, e = it["index"][name]
                if not np.isfinite(v):
                    continue
                x = it["p"]["teq"]
                snr = v / e if e > 0 else 0
                filled = snr >= 3
                col = "#B91C1C" if snr >= 3 else ("#1E3A8A" if snr >= 2 else "#6B7280")
                ax.errorbar(x, v, yerr=e, fmt="none", ecolor=col, elinewidth=1.6, capsize=3,
                            capthick=1.2, zorder=3)
                ax.plot(x, v, "o", ms=10, mfc=col if filled else "white", mec=col, mew=2.0,
                        zorder=4)
                if (lit_mu.get(it["name"]) or {}).get(name.lower()):
                    ax.plot(x, v, "o", ms=19, mfc="none", mec="#D97706", mew=2.2, zorder=5)
                xs.append(x); ys.append(v); labels.append(it["name"])
            ax.axhline(0, color="#111827", lw=1.0, zorder=2)
            _regimes(ax, 0.985)
            ax.set_xlim(280, 1080)
            ax.set_xlabel(r"Equilibrium temperature  $T_{\rm eq}$  (K)", fontsize=13.5)
            ax.set_ylabel(f"{BAND_INDICES[name]['label']} band contrast\n"
                          r"(H$_2$/He scale heights, $\mu$ = 2.3)", fontsize=13.5)
            ax.tick_params(labelsize=12, which="both", direction="in", top=True, right=True)
            lo = min(-1.8, min(ys) - 1.2) if ys else -2
            hi = max(5.0, max(ys) + 2.6) if ys else 5
            ax.set_ylim(lo, hi)
            _repel_labels(ax, np.array(xs), np.array(ys), labels, fontsize=9.5)
        axes[0].legend(handles=[
            Line2D([], [], marker="o", ls="", ms=10, mfc="#B91C1C", mec="#B91C1C", label="≥ 3σ"),
            Line2D([], [], marker="o", ls="", ms=10, mfc="white", mec="#1E3A8A", mew=2, label="2–3σ"),
            Line2D([], [], marker="o", ls="", ms=10, mfc="white", mec="#6B7280", mew=2, label="< 2σ"),
            Line2D([], [], marker="o", ls="", ms=17, mfc="none", mec="#D97706", mew=2.2,
                   label="published detection")],
            loc="center right", fontsize=10.5, frameon=True, framealpha=0.95, edgecolor="#E5E7EB")
        fig.tight_layout(w_pad=3)
        paths = _save(fig, out, "patchwork_band_indices")
        plt.close(fig)
    return paths


def plot_mu_landscape(lit_mu, params, patchwork_names, new_targets, out: Path) -> list[str]:
    """Published JWST mean-molecular-weight constraints for sub-Neptunes.

    Left against T_eq with the regimes the literature proposes, right
    against planet mass (the mass-metallicity question). Patchwork targets
    are dark; Patchwork targets with no published spectrum yet are marked
    along the bottom of the T_eq panel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullFormatter

    in_pw, out_pw = "#1E3A8A", "#9CA3AF"
    with plt.rc_context(_style()):
        fig, axes = plt.subplots(1, 2, figsize=(17.5, 8.4), sharey=True,
                                 gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.06})
        for ax, key in zip(axes, ("teq", "mp_me")):
            xs, ys, labels = [], [], []
            for name, ref in lit_mu.items():
                p = params.get(name)
                if not p:
                    continue
                x = p[key]
                col = in_pw if name in patchwork_names else out_pw
                mu, kind = ref["mu"], ref["kind"]
                if "lower" in mu:
                    y = mu["lower"]
                    open_ = kind == "lower_cloudfree"
                    ax.annotate("", xy=(x, y * 1.9), xytext=(x, y),
                                arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0,
                                                ls="--" if open_ else "-"), zorder=3)
                    ax.plot(x, y, "^", ms=11, mfc="white" if open_ else col, mec=col, mew=2, zorder=4)
                elif "low" in mu:
                    y = float(np.sqrt(mu["low"] * mu["high"]))
                    ax.plot([x, x], [mu["low"], mu["high"]], color=col, lw=3.0, solid_capstyle="butt",
                            zorder=3)
                    ax.plot(x, y, "o", ms=10, mfc=col, mec="#111827", mew=0.8, zorder=4)
                else:
                    y = mu["value"]
                    ax.errorbar(x, y, yerr=[[mu["minus"]], [mu["plus"]]], fmt="o", ms=10, mfc=col,
                                mec="#111827", mew=0.8, ecolor=col, elinewidth=2.2, capsize=0, zorder=4)
                if ref.get("ch4"):
                    ax.plot(x, y, "o", ms=18, mfc="none", mec="#D97706", mew=2.0, zorder=5)
                xs.append(x); ys.append(y); labels.append(name)
            for mu, ls, lab in ((MU_H2HE, "--", "H$_2$/He  μ = 2.3"), (18.0, ":", "H$_2$O  μ = 18"),
                                (44.0, "-.", "CO$_2$  μ = 44")):
                ax.axhline(mu, color="#4B5563", lw=1.1, ls=ls, zorder=1)
                ax.text(0.47 if key == "teq" else 0.40, mu * 1.05, lab,
                        transform=ax.get_yaxis_transform(), ha="center", va="bottom",
                        fontsize=10, color="#4B5563")
            ax.set_yscale("log")
            ax.set_ylim(1.7, 90)
            ax.yaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20, 50]))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_yticklabels(["2", "3", "5", "10", "20", "50"])
            ax.tick_params(labelsize=12.5, which="both", direction="in", top=True, right=True)
            if key == "teq":
                _regimes(ax, 0.99)
                ax.set_xlim(220, 1080)
                ax.set_xlabel(r"Equilibrium temperature  $T_{\rm eq}$  (K)", fontsize=14)
                # Patchwork targets with no published spectrum: first G395H spectra
                for name in new_targets:
                    p = params.get(name)
                    if p:
                        ax.plot(p["teq"], 1.85, "D", ms=9, mfc="#F59E0B", mec="#111827", mew=0.8,
                                zorder=4, clip_on=False)
                        xs.append(p["teq"]); ys.append(1.85); labels.append(name)
            else:
                ax.set_xscale("log")
                ax.set_xlim(1.2, 25)
                ax.xaxis.set_major_locator(FixedLocator([1.5, 2, 3, 5, 10, 20]))
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.set_xticklabels(["1.5", "2", "3", "5", "10", "20"])
                ax.set_xlabel(r"Planet mass  ($M_\oplus$)", fontsize=14)
                for nm, m in (("Uranus", 14.5), ("Neptune", 17.1)):
                    ax.plot(m, 2.5, "s", ms=9, mfc="#D1D5DB", mec="#4B5563", zorder=3)
                    xs.append(m); ys.append(2.5); labels.append(nm)
            _repel_labels(ax, np.array(xs), np.array(ys), labels, fontsize=9.5)
        axes[0].set_ylabel("Mean molecular weight  μ  (amu)", fontsize=14)
        fig.legend(handles=[
            Line2D([], [], marker="o", ls="", ms=10, mfc=in_pw, mec="#111827", label="Patchwork target"),
            Line2D([], [], marker="o", ls="", ms=10, mfc=out_pw, mec="#111827", label="other JWST sub-Neptune"),
            Line2D([], [], color="#4B5563", lw=3, label="measured / H$_2$-rich range"),
            Line2D([], [], marker="^", ls="", ms=10, mfc="#4B5563", mec="#4B5563", label="lower limit, aerosols modelled"),
            Line2D([], [], marker="^", ls="", ms=10, mfc="white", mec="#4B5563", mew=2,
                   label="lower limit only if cloud-free"),
            Line2D([], [], marker="o", ls="", ms=16, mfc="none", mec="#D97706", mew=2, label="CH$_4$ detected"),
            Line2D([], [], marker="D", ls="", ms=9, mfc="#F59E0B", mec="#111827",
                   label="Patchwork: no published spectrum")],
            loc="lower center", ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.005))
        fig.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.2)
        paths = _save(fig, out, "patchwork_mu_landscape")
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
                      "index": {k: band_index(spec, p, k) for k in BAND_INDICES},
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
    lit_mu = load_literature("literature_subneptunes.json")
    lit_params = fetch_params(list(lit_mu))
    lit_params.update({it["name"]: it["p"] for it in items})
    patchwork_names = {it["name"] for it in items}
    new_targets = [it["name"] for it in items if it["name"] not in lit_mu and it["name"] not in lit]
    for path in (plot_stack(shown, out, args.resolution, args.spacing, unit="H")
                 + plot_stack(shown, out, args.resolution, km_spacing, unit="km")
                 + plot_band_indices(shown, lit_mu, out)
                 + plot_mu_landscape(lit_mu, lit_params, patchwork_names, new_targets, out)):
        print(path)
    if missing:
        print("No archive parameters or spectra (left out): " + ", ".join(missing))
    return 0


def json_dumps(x) -> str:
    import json
    return json.dumps(x) if x else ""


if __name__ == "__main__":
    raise SystemExit(main())
