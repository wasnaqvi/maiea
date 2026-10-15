#!/usr/bin/env python
"""Compare spectra in reference scale heights with visit-level quality flags.

The reference assumes mu=2.3; it is not a measured composition. Different
atmospheres, aerosols, and pressure boundaries can suppress different numbers
of scale heights. Panels use adaptive limits so large features are not clipped.

Worst-visit beta and fit-version flags remain diagnostics, not calibrated
acceptance rules. Optional beta scaling is a sensitivity test, not the default.
The historical RMS-to-molecular-weight calculation is uncalibrated and is
exported only with --legacy-heuristic-mu; do not use it for atmospheric claims.

    python scripts/patchwork/plot_survey_scale_heights.py ~/Patchwork_Data \\
        -o ~/Desktop/patchwork_plots
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

G_SI = 6.674e-11
K_B = 1.380649e-23
AMU = 1.66053907e-27
R_EARTH = 6.371e6
M_EARTH = 5.9722e24
R_SUN = 6.957e8
MU_H2HE = 2.3          # solar-composition H2/He
# Amplitude a clear solar-composition atmosphere would show across
# G395H. The CO2 band at 4.3 um reaches ~2-3 scale heights above the
# continuum in cloud-free sub-Neptune models; 2 is the conservative end
# and is what the mu limit below is quoted against.
A_REF_SCALE_HEIGHTS = 2.0
# Hold population claims when time-correlated noise or unexplained spectral
# structure is large enough that the modulation limit is systematics-dominated.
QUALITY_BETA_MAX = 3.0
QUALITY_CHI2_MAX = 3.0
STEM_TAG = ""   # set from --tag; keeps R=50 and R=100 outputs distinct


def planet_params(names: list[str]) -> dict[str, dict]:
    from aster_toolkit.data_acquisition.mast import archive_tap_query

    aliases = {"TOI-836 c": "TOI-836.01"}
    query_names = sorted(set(names) | {aliases[n] for n in names if n in aliases})
    canonical_names = {value: key for key, value in aliases.items()}
    rows = archive_tap_query(
        ["pl_name in (" + ",".join("'" + n.replace("'", "''") + "'" for n in query_names) + ")"],
        columns=["pl_name", "pl_rade", "pl_bmasse", "pl_eqt", "st_rad",
                 "pl_dens", "st_teff"],
    )
    out = {}
    for r in rows:
        try:
            rp = float(r["pl_rade"]) * R_EARTH
            mp = float(r["pl_bmasse"]) * M_EARTH
            teq = float(r["pl_eqt"])
            rs = float(r["st_rad"]) * R_SUN
        except (TypeError, ValueError, KeyError):
            continue
        if not all(np.isfinite(v) and v > 0 for v in (rp, mp, teq, rs)):
            continue
        g = G_SI * mp / rp**2
        h = K_B * teq / (MU_H2HE * AMU * g)
        name = canonical_names.get(r["pl_name"], r["pl_name"])
        if name in out and r["pl_name"] != name:
            continue
        out[name] = {
            "archive_name": r["pl_name"],
            "rp_m": rp, "mp_kg": mp, "teq": teq, "rs_m": rs,
            "g": g, "H_km": h / 1e3,
            # ppm of transit depth per scale height
            "delta_ppm": 2 * rp * h / rs**2 * 1e6,
            "rp_re": float(r["pl_rade"]), "mp_me": float(r["pl_bmasse"]),
            "density": (float(r["pl_dens"]) if r.get("pl_dens") else np.nan),
        }
    return out


def _detector_diagnostics(root: Path, slug: str, detector: str) -> dict:
    import json

    betas, impact_parameters, versions = [], [], set()
    pattern = root / slug / "fits" / "*" / detector.lower() / "white_fit_summary.json"
    for path in glob.glob(str(pattern)):
        try:
            with Path(path).open() as handle:
                summary = json.load(handle)
                beta = (summary.get("rednoise") or {}).get("beta_median")
            if beta is not None and np.isfinite(float(beta)):
                betas.append(max(1.0, float(beta)))
            impact = (summary.get("b_p1") or {}).get("median")
            if impact is not None and np.isfinite(float(impact)):
                impact_parameters.append(float(impact))
            if summary.get("fit_version"):
                versions.add(str(summary["fit_version"]))
        except (OSError, ValueError, TypeError):
            continue
    return {
        "rednoise_beta": float(np.median(betas)) if betas else 1.0,
        "rednoise_beta_worst_visit": max(betas) if betas else np.nan,
        "diagnostics_missing": not bool(betas),
        "impact_parameter": (float(np.median(impact_parameters))
                             if impact_parameters else np.nan),
        "fit_versions": sorted(versions),
    }


def load_survey(root: Path, suffix: str = "", *,
                inflate_rednoise: bool = False) -> dict[str, dict[str, np.ndarray]]:
    from aster_toolkit.data_reduction.juliet import read_spectrum_csv

    out: dict[str, dict[str, np.ndarray]] = {}
    pattern = f"combined_*_transmission_spectrum{suffix}.csv"
    for f in sorted(glob.glob(str(root / "*" / "combined" / pattern))):
        slug = Path(f).parts[-3]
        det = "NRS2" if "nrs2" in os.path.basename(f).lower() else "NRS1"
        s = read_spectrum_csv(f)
        diagnostics = _detector_diagnostics(root, slug, det)
        if inflate_rednoise:
            s["depth_err_ppm"] = (
                np.asarray(s["depth_err_ppm"], dtype=float)
                * diagnostics["rednoise_beta"]
            )
        s.update(diagnostics)
        out.setdefault(slug, {})[det] = s
    return out


def pretty(slug: str) -> str:
    """Directory slug -> archive planet name."""
    special = {"TOI_836_01": "TOI-836 c", "L_98_59_d": "L 98-59 d",
               "K2_18_b": "K2-18 b"}
    if slug in special:
        return special[slug]
    p = slug.split("_")
    if p[0] in ("GJ", "LTT"):
        return f"{p[0]} {p[1]} {p[-1]}"
    return f"{'-'.join(p[:-1])} {p[-1]}"


def _profile_intrinsic_scatter(depth: np.ndarray, error: np.ndarray,
                               is_nrs2: np.ndarray) -> dict:
    """Profile a wavelength-independent intrinsic RMS above quoted errors."""
    design = np.ones((depth.size, 1 + int(is_nrs2.any() and (~is_nrs2).any())))
    if design.shape[1] == 2:
        design[:, 1] = is_nrs2.astype(float)

    def evaluate(scatter: float):
        variance = error**2 + float(scatter)**2
        weighted_design = design / np.sqrt(variance)[:, None]
        weighted_depth = depth / np.sqrt(variance)
        coeff, *_ = np.linalg.lstsq(weighted_design, weighted_depth, rcond=None)
        residual = depth - design @ coeff
        loglike = -0.5 * np.sum(np.log(2.0 * np.pi * variance)
                                + residual**2 / variance)
        return float(loglike), coeff, residual

    scale = max(float(np.nanstd(depth)), float(np.nanmedian(error)), 1.0)
    grid = np.concatenate(([0.0], np.geomspace(scale * 1e-4, scale * 20.0, 3000)))
    loglikes = np.asarray([evaluate(value)[0] for value in grid])
    best_index = int(np.nanargmax(loglikes))
    best = float(grid[best_index])
    threshold = float(loglikes[best_index] - 2.0)
    beyond = np.flatnonzero(loglikes[best_index:] <= threshold)
    if beyond.size:
        upper_index = best_index + int(beyond[0])
        if upper_index == 0:
            upper = float(grid[0])
        else:
            x0, x1 = grid[upper_index - 1:upper_index + 1]
            y0, y1 = loglikes[upper_index - 1:upper_index + 1]
            fraction = (threshold - y0) / (y1 - y0) if y1 != y0 else 1.0
            upper = float(x0 + np.clip(fraction, 0.0, 1.0) * (x1 - x0))
    else:
        upper = float(grid[-1])
    loglike0, coeff0, residual0 = evaluate(0.0)
    dof = max(1, depth.size - design.shape[1])
    chi2_red = float(np.sum((residual0 / error)**2) / dof)
    return {
        "scatter_ppm": best,
        "scatter_2sig_ppm": upper,
        "residual_ppm": residual0,
        "mean_ppm": float(coeff0[0]),
        "detector_offset_ppm": (float(coeff0[1]) if coeff0.size > 1 else 0.0),
        "chi2_red": chi2_red,
        "delta_loglike": float(loglikes[best_index] - loglike0),
    }


def amplitude_in_H(s: dict[str, dict[str, np.ndarray]], delta_ppm: float) -> dict:
    """Profiled spectral modulation and its 2-sigma limit, in scale heights."""
    detectors = [det for det in ("NRS1", "NRS2") if det in s]
    w = np.concatenate([s[det]["wave"] for det in detectors])
    d = np.concatenate([s[det]["depth_ppm"] for det in detectors])
    e = np.concatenate([s[det]["depth_err_ppm"] for det in detectors])
    labels = np.concatenate([
        np.full(s[det]["wave"].size, det, dtype="U4") for det in detectors
    ])
    finite = np.isfinite(w) & np.isfinite(d) & np.isfinite(e) & (e > 0)
    w, d, e, labels = w[finite], d[finite], e[finite], labels[finite]
    o = np.argsort(w)
    w, d, e, labels = w[o], d[o], e[o], labels[o]
    profile = _profile_intrinsic_scatter(d, e, labels == "NRS2")
    beta = max(float(s[det].get("rednoise_beta_worst_visit",
                               s[det].get("rednoise_beta", 1.0))) for det in detectors)
    grazing_old_fit = any(
        float(s[det].get("impact_parameter", np.nan)) > 0.8
        and (not s[det].get("fit_versions") or any(
            tuple(int(x) for x in version.split("-")[0].split(".")) < (1, 5)
            for version in s[det].get("fit_versions", [])))
        for det in detectors
    )
    quality_flags = []
    if any(s[det].get("diagnostics_missing", False) for det in detectors):
        quality_flags.append("missing-diagnostics")
    if beta > QUALITY_BETA_MAX:
        quality_flags.append("beta>3")
    if profile["chi2_red"] > QUALITY_CHI2_MAX:
        quality_flags.append("structured")
    if grazing_old_fit:
        quality_flags.append("grazing-v1.4")
    return {"wave": w, "detector": labels,
            "resid_H": profile["residual_ppm"] / delta_ppm,
            "err_H": e / delta_ppm,
            "mean_ppm": profile["mean_ppm"],
            "detector_offset_ppm": profile["detector_offset_ppm"],
            "chi2_red": profile["chi2_red"],
            "amp_H": profile["scatter_ppm"] / delta_ppm,
            "amp_H_2sig": profile["scatter_2sig_ppm"] / delta_ppm,
            "rednoise_beta": beta, "quality_flags": quality_flags,
            "n": int(w.size), "excess_ppm": profile["scatter_ppm"]}


def plot_panels(data, params, out_dir: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aster_toolkit.data_reduction.juliet import (
        _PLOT_STYLE, _savefig, DATA_COLOR, NRS2_COLOR, MEAN_COLOR, GAP_COLOR)

    items = sorted(data.items(), key=lambda kv: params[kv[0]]["teq"])
    n = len(items)
    ncol, nrow = 3, int(np.ceil(n / 3))
    with plt.rc_context(_PLOT_STYLE):
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 2.5 * nrow),
                                 sharex=True, sharey=False, squeeze=False)
        for ax, (slug, s) in zip(axes.ravel(), items):
            p = params[slug]
            m = amplitude_in_H(s, p["delta_ppm"])
            for det, colour in (("NRS1", DATA_COLOR), ("NRS2", NRS2_COLOR)):
                if det not in s:
                    continue
                sel = m["detector"] == det
                ax.errorbar(m["wave"][sel], m["resid_H"][sel],
                            yerr=m["err_H"][sel], fmt=".", ms=4, lw=0.9,
                            color=colour, alpha=0.85)
            ax.axhline(0, ls="--", lw=0.8, color=MEAN_COLOR)
            ax.axvspan(3.72, 3.82, color=GAP_COLOR, alpha=0.35, lw=0)
            ax.set_title(f"{pretty(slug)}   $T_{{\\rm eq}}$ {p['teq']:.0f} K",
                         fontsize=10.5, loc="left")
            ax.text(0.03, 0.03,
                    f"{p['delta_ppm']:.0f} ppm/H   $A_H$ < {m['amp_H_2sig']:.1f}"
                    f"   $\\chi^2_\\nu$ {m['chi2_red']:.1f}",
                    transform=ax.transAxes, fontsize=8.5, color=MEAN_COLOR)
            if m["quality_flags"]:
                ax.text(0.03, 0.14, ", ".join(m["quality_flags"]),
                        transform=ax.transAxes, fontsize=8.0, color="#A33A32",
                        ha="left")
            ax.margins(y=0.25)
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel(r"Wavelength  [$\mu$m]")
        for row in axes:
            row[0].set_ylabel("$(D-\\langle D\\rangle)\\,/\\,\\delta$   [$H$]")
        fig.suptitle("Patchwork — JWST NIRSpec/G395H in units of atmospheric "
                     "scale height  (H$_2$/He, $\\mu$ = 2.3)", y=1.0)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        return _savefig(fig, out_dir, "patchwork_scale_heights" + STEM_TAG)


def plot_absolute(data, params, out_dir: Path) -> str:
    """Absolute transit depth, one panel per planet, auto-scaled.

    Keeps what a single absolute-depth axis gives you -- where each
    planet actually sits, and its real ppm error bars -- while staying
    readable past the three or four planets at which a shared axis stops
    separating them. Panels are ordered by depth, so the survey's range
    (600 to 13500 ppm) reads down the page instead of being crushed onto
    one axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aster_toolkit.data_reduction.juliet import (
        _PLOT_STYLE, _savefig, DATA_COLOR, NRS2_COLOR, MEAN_COLOR, GAP_COLOR)

    def mean_depth(s):
        d = np.concatenate([s[k]["depth_ppm"] for k in s])
        e = np.concatenate([s[k]["depth_err_ppm"] for k in s])
        return float(np.sum(d / e**2) / np.sum(1 / e**2))

    items = sorted(data.items(), key=lambda kv: -mean_depth(kv[1]))
    n = len(items); ncol, nrow = 3, int(np.ceil(n / 3))
    with plt.rc_context(_PLOT_STYLE):
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 2.5 * nrow),
                                 sharex=True, squeeze=False)
        for ax, (slug, s) in zip(axes.ravel(), items):
            m = mean_depth(s)
            for det, colour in (("NRS1", DATA_COLOR), ("NRS2", NRS2_COLOR)):
                if det not in s:
                    continue
                ax.errorbar(s[det]["wave"], s[det]["depth_ppm"],
                            yerr=s[det]["depth_err_ppm"], fmt=".", ms=4,
                            lw=0.9, color=colour, alpha=0.85)
            ax.axhline(m, ls="--", lw=0.8, color=MEAN_COLOR)
            ax.axvspan(3.72, 3.82, color=GAP_COLOR, alpha=0.35, lw=0)
            ax.set_title(f"{pretty(slug)}   {m:.0f} ppm", fontsize=10.5, loc="left")
            ax.set_ylabel(r"$(R_p/R_\star)^2$ [ppm]", fontsize=9)
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel(r"Wavelength  [$\mu$m]")
        fig.suptitle("Patchwork — JWST NIRSpec/G395H transmission spectra, "
                     "absolute depth", y=1.0)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        return _savefig(fig, out_dir, "patchwork_absolute_depths" + STEM_TAG)


def mu_limit(amp_H_2sig: float) -> float | None:
    """Lower limit on mu from an amplitude limit, or None if unconstrained.

    Amplitude scales as 1/mu, so mu > mu_H2 * A_ref / A_H. That inverts
    into nonsense at both ends and both ends occur in real data:

    * A_H at or above A_ref returns mu below 2.3 -- lighter than H2/He,
      which is unphysical. It means the spectrum scatters MORE than a
      clear solar atmosphere would, i.e. a detection or (far more often
      here) excess noise. Either way it is not a composition limit.
    * A_H at zero returns infinity. Reported as no constraint rather
      than as a spectacular one.
    """
    a = float(amp_H_2sig)
    if not np.isfinite(a) or a <= 0:
        return None
    mu = MU_H2HE * A_REF_SCALE_HEIGHTS / a
    return mu if mu > MU_H2HE else None


def plot_trend(data, params, out_dir: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aster_toolkit.data_reduction.juliet import (
        _PLOT_STYLE, _savefig, DATA_COLOR, MODEL_COLOR, MEAN_COLOR)

    rows = []
    for slug, s in data.items():
        p = params[slug]
        m = amplitude_in_H(s, p["delta_ppm"])
        mu_min = mu_limit(m["amp_H_2sig"])
        if mu_min is None or m["quality_flags"]:
            continue
        rows.append((pretty(slug), p, m, mu_min))
    rows.sort(key=lambda r: r[1]["teq"])

    with plt.rc_context(_PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        # Planet mass first: whether atmospheric modulation changes across
        # the radius-valley population is the comparative question. Do not
        # add a secondary metallicity axis: mu does not map uniquely to
        # abundance without a self-consistent chemistry model.
        for ax, key, xlabel in (
                (axes[0], "mp_me", r"Planet mass  [$M_\oplus$]"),
                (axes[1], "teq", r"Equilibrium temperature  [K]"),
                (axes[2], "density", r"Bulk density  [g cm$^{-3}$]")):
            for name, p, m, mu in rows:
                x = p[key]
                if not np.isfinite(x):
                    continue
                # Arrow: the limit is one-sided.
                ax.errorbar(x, mu, yerr=mu * 0.45, lolims=True, fmt="^", ms=8,
                            lw=1.2, color=DATA_COLOR, alpha=0.9)
                ax.annotate(name, (x, mu), textcoords="offset points",
                            xytext=(6, 4), fontsize=8, color=MEAN_COLOR)
            ax.axhline(MU_H2HE, ls="--", lw=1.2, color=MODEL_COLOR)
            ax.text(0.02, 0.06, r"solar H$_2$/He, $\mu$ = 2.3",
                    transform=ax.transAxes, color=MODEL_COLOR, fontsize=9)
            ax.axhline(18.0, ls=":", lw=1.2, color=MEAN_COLOR)
            ax.text(0.02, 0.90, r"H$_2$O steam, $\mu$ = 18",
                    transform=ax.transAxes, color=MEAN_COLOR, fontsize=9)
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            if key == "mp_me":
                ax.set_xscale("log")
        axes[0].set_ylabel(r"heuristic lower floor on $\mu$  [amu]   (2$\sigma$)")
        fig.suptitle("Patchwork — heuristic mean-molecular-weight floors from "
                     "profiled spectral modulation (or aerosol muting); "
                     "quality-flagged targets withheld",
                     y=0.99)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        return _savefig(fig, out_dir, "patchwork_mmw_trend" + STEM_TAG)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="Directory of per-target Patchwork outputs.")
    ap.add_argument("-o", "--out-dir", default=".")
    ap.add_argument("--suffix", default="",
                    help="Spectrum file suffix, e.g. _R50 for the rebinned set.")
    ap.add_argument("--tag", default="",
                    help="Appended to output filenames, e.g. _R50.")
    ap.add_argument("--inflate-beta", action="store_true",
                    help="Conservative sensitivity test: multiply channel errors by beta.")
    ap.add_argument("--legacy-heuristic-mu", action="store_true",
                    help="Opt in to an uncalibrated historical mu diagnostic, not a composition constraint.")
    args = ap.parse_args()
    TAG = args.tag or args.suffix

    root = Path(os.path.expanduser(args.root))
    out = Path(os.path.expanduser(args.out_dir))
    global STEM_TAG
    STEM_TAG = TAG
    data = load_survey(root, args.suffix,
                       inflate_rednoise=args.inflate_beta)
    params = planet_params([pretty(s) for s in data])
    params = {s: params[pretty(s)] for s in list(data) if pretty(s) in params}
    import json
    from datetime import datetime, timezone
    missing = sorted(set(data) - set(params))
    out.mkdir(parents=True, exist_ok=True)
    (out / ("scale_height_parameters" + STEM_TAG + ".json")).write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "NASA Exoplanet Archive pscomppars; queried by planet name and known alias",
        "mu_reference": MU_H2HE, "parameters": params,
        "omitted_from_scale_heights": missing,
        "status": "reference normalization; parameter uncertainties not propagated"}, indent=2)+"\n")
    print("\n" + plot_absolute(data, params, out))
    if missing:
        print("WARNING: no finite positive archive parameters for " + ", ".join(missing)
              + "; retained in absolute-depth atlas, omitted from scale heights.")
    data = {s: v for s, v in data.items() if s in params}
    if not data:
        print("No spectra with archive parameters found."); return 1

    print(f"{'planet':13s} {'Teq':>5s} {'g':>6s} {'H':>6s} {'ppm/H':>7s} "
          f"{'beta':>5s} {'off':>7s} {'chi2':>6s} {'A_H':>6s} "
          f"{'A_H<2s':>7s} {'mu~>':>7s}  quality")
    for slug in sorted(data, key=lambda s: params[s]["teq"]):
        p, m = params[slug], amplitude_in_H(data[slug], params[slug]["delta_ppm"])
        mu = (mu_limit(m["amp_H_2sig"])
              if args.legacy_heuristic_mu and not m["quality_flags"] else None)
        mu_s = f"{mu:7.1f}" if mu is not None else "      -"
        print(f"{pretty(slug):13s} {p['teq']:5.0f} {p['g']:6.1f} "
              f"{p['H_km']:6.0f} {p['delta_ppm']:7.1f} {m['rednoise_beta']:5.2f} "
              f"{m['detector_offset_ppm']:7.0f} {m['chi2_red']:6.2f} "
              f"{m['amp_H']:6.2f} {m['amp_H_2sig']:7.2f} {mu_s}  "
              f"{','.join(m['quality_flags']) or 'ok'}")
    print(plot_panels(data, params, out))
    if args.legacy_heuristic_mu:
        print("WARNING: heuristic mu floors are uncalibrated and not publication constraints.")
        print(plot_trend(data, params, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
