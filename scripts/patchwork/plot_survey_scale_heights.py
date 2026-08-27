#!/usr/bin/env python
"""Survey comparison in SCALE HEIGHTS, and the mean-molecular-weight trend.

Overplotting a dozen transmission spectra in ppm is not a comparison, it
is a hairball: the same 200 ppm is a fraction of a scale height for
GJ 1214 b and ten scale heights for TOI-836 b, so the eye cannot tell a
flat spectrum from a featured one. Across this survey the depth change
per scale height spans 20-275 ppm, a factor of 14, which is exactly the
dynamic range that makes a shared ppm axis meaningless.

The fix is the normalization comparative surveys have used since
D. K. Sing et al. (2016, Nature 529, 59), who placed ten hot Jupiters on
a common axis as an altitude difference in units of the equilibrium
scale height (dZ/H_eq) rather than in absorption depth:

    H     = k T_eq / (mu g)          atmospheric scale height
    delta = 2 R_p H / R_*^2          depth change per scale height
    A_H   = (D(lambda) - <D>) / delta

A_H is the observable a survey should compare. It also carries the
composition constraint directly: feature amplitude scales as 1/mu, so a
spectrum flat to A_H < X, where a solar-composition atmosphere would
show A_ref, implies mu > mu_H2 * A_ref / X -- a heavier atmosphere, or
an aerosol deck muting it. Both readings matter and neither is
available from a ppm axis.

Writes small multiples (one panel per planet, shared axis, ordered by
equilibrium temperature) plus the mu-constraint trend figure.

    python scripts/patchwork/plot_survey_scale_heights.py ~/Patchwork_Data \
        -o ~/Desktop
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


def planet_params(names: list[str]) -> dict[str, dict]:
    from aster_toolkit.data_acquisition.mast import archive_tap_query

    rows = archive_tap_query(
        ["pl_name in (" + ",".join(f"'{n}'" for n in names) + ")"],
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
        g = G_SI * mp / rp**2
        h = K_B * teq / (MU_H2HE * AMU * g)
        out[r["pl_name"]] = {
            "rp_m": rp, "mp_kg": mp, "teq": teq, "rs_m": rs,
            "g": g, "H_km": h / 1e3,
            # ppm of transit depth per scale height
            "delta_ppm": 2 * rp * h / rs**2 * 1e6,
            "rp_re": float(r["pl_rade"]), "mp_me": float(r["pl_bmasse"]),
            "density": (float(r["pl_dens"]) if r.get("pl_dens") else np.nan),
        }
    return out


def load_survey(root: Path) -> dict[str, dict[str, np.ndarray]]:
    from aster_toolkit.data_reduction.juliet import read_spectrum_csv

    out: dict[str, dict[str, np.ndarray]] = {}
    for f in sorted(glob.glob(str(root / "*" / "combined" /
                                  "combined_*_transmission_spectrum.csv"))):
        slug = Path(f).parts[-3]
        det = "NRS2" if "nrs2" in os.path.basename(f).lower() else "NRS1"
        s = read_spectrum_csv(f)
        out.setdefault(slug, {})[det] = s
    return out


def pretty(slug: str) -> str:
    """Directory slug -> archive planet name."""
    special = {"TOI_836_01": "TOI-836.01", "L_98_59_d": "L 98-59 d",
               "K2_18_b": "K2-18 b"}
    if slug in special:
        return special[slug]
    p = slug.split("_")
    if p[0] in ("GJ", "LTT"):
        return f"{p[0]} {p[1]} {p[-1]}"
    return f"{'-'.join(p[:-1])} {p[-1]}"


def amplitude_in_H(s: dict[str, np.ndarray], delta_ppm: float) -> dict:
    """Spectral amplitude and its 2-sigma limit, in scale heights."""
    w = np.concatenate([s[d]["wave"] for d in s])
    d = np.concatenate([s[d]["depth_ppm"] for d in s])
    e = np.concatenate([s[d]["depth_err_ppm"] for d in s])
    o = np.argsort(w)
    w, d, e = w[o], d[o], e[o]
    mean = np.sum(d / e**2) / np.sum(1 / e**2)
    resid = d - mean
    # Excess scatter beyond the error bars -- the part of the spread that
    # is not photon noise. Negative excess means consistent with flat.
    var_obs = float(np.mean(resid**2))
    var_err = float(np.mean(e**2))
    excess = var_obs - var_err
    amp = np.sqrt(max(excess, 0.0)) / delta_ppm
    # 2-sigma upper limit on a real amplitude, from the error on the
    # variance of n points.
    n = w.size
    sig_var = var_obs * np.sqrt(2.0 / max(n - 1, 1))
    amp_hi = np.sqrt(max(excess + 2 * sig_var, 0.0)) / delta_ppm
    return {"wave": w, "resid_H": resid / delta_ppm, "err_H": e / delta_ppm,
            "mean_ppm": float(mean), "amp_H": float(amp),
            "amp_H_2sig": float(amp_hi), "n": int(n),
            "excess_ppm": float(np.sqrt(max(excess, 0.0)))}


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
                                 sharex=True, sharey=True, squeeze=False)
        for ax, (slug, s) in zip(axes.ravel(), items):
            p = params[slug]
            m = amplitude_in_H(s, p["delta_ppm"])
            for det, colour in (("NRS1", DATA_COLOR), ("NRS2", NRS2_COLOR)):
                if det not in s:
                    continue
                sel = np.isin(m["wave"], s[det]["wave"])
                ax.errorbar(m["wave"][sel], m["resid_H"][sel],
                            yerr=m["err_H"][sel], fmt=".", ms=4, lw=0.9,
                            color=colour, alpha=0.85)
            ax.axhline(0, ls="--", lw=0.8, color=MEAN_COLOR)
            ax.axvspan(3.72, 3.82, color=GAP_COLOR, alpha=0.35, lw=0)
            ax.set_title(f"{pretty(slug)}   $T_{{\\rm eq}}$ {p['teq']:.0f} K",
                         fontsize=10.5, loc="left")
            ax.text(0.03, 0.06,
                    f"{p['delta_ppm']:.0f} ppm/H   $A_H$ < {m['amp_H_2sig']:.1f}",
                    transform=ax.transAxes, fontsize=8.5, color=MEAN_COLOR)
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        axes[-1, 0].set_ylim(-4.5, 4.5)
        for ax in axes[-1]:
            ax.set_xlabel(r"Wavelength  [$\mu$m]")
        for row in axes:
            row[0].set_ylabel("$(D-\\langle D\\rangle)\\,/\\,\\delta$   [$H$]")
        fig.suptitle("Patchwork — JWST NIRSpec/G395H in units of atmospheric "
                     "scale height  (H$_2$/He, $\\mu$ = 2.3)", y=1.0)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        return _savefig(fig, out_dir, "patchwork_scale_heights")


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
        return _savefig(fig, out_dir, "patchwork_absolute_depths")


def metallicity_from_mu(mu: float) -> float:
    """Rough metallicity (x solar) implied by a mean molecular weight.

    Solar composition by number is ~0.855 H2 / 0.145 He, giving mu = 2.30.
    Enriching the metals by Z_rel raises the number fraction x of heavy
    species (taken as water-like, mu_Z ~ 18) roughly linearly, so
    mu ~ (1-x) 2.30 + x 18 with x ~ 5e-4 Z_rel.

    Deliberately crude -- a real number needs a self-consistent chemistry
    grid -- but it puts a mu limit on the axis the sub-Neptune literature
    argues over. For scale it gives ~3 at 100x solar, which is where
    Benneke et al. place K2-18 b, TOI-270 d and GJ 3470 b, and ~10 at
    1000x. Note the Solar System values those are compared against come
    from CH4/H while exoplanet values come mostly from H2O/H, so the
    comparison carries a systematic the plot cannot show.
    """
    mu = float(mu)
    if mu <= MU_H2HE:
        return 1.0
    x = (mu - MU_H2HE) / (18.0 - MU_H2HE)
    return float(min(x / 5e-4, 1e5))


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
        mu_min = MU_H2HE * A_REF_SCALE_HEIGHTS / max(m["amp_H_2sig"], 1e-6)
        rows.append((pretty(slug), p, m, mu_min))
    rows.sort(key=lambda r: r[1]["teq"])

    with plt.rc_context(_PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        # Planet mass first: the mass-metallicity relation is the axis
        # the sub-Neptune literature actually argues over, following the
        # Solar System's CH4 enrichment trend. Benneke et al. place
        # K2-18 b, TOI-270 d and GJ 3470 b near 100x solar; whether that
        # continues, cliffs or plateaus below Neptune mass is the open
        # question a uniform survey is built to answer.
        for ax, key, xlabel in (
                (axes[0], "mp_me", r"Planet mass  [$M_\oplus$]"),
                (axes[1], "teq", r"Equilibrium temperature  [K]"),
                (axes[2], "density", r"Bulk density  [g cm$^{-3}$]")):
            for name, p, m, mu in rows:
                x = p[key]
                if not np.isfinite(x):
                    continue
                # Arrow: the limit is one-sided.
                ax.errorbar(x, mu, yerr=[[mu * 0.45], [0]], fmt="v", ms=8,
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
                # Published sub-Neptune metallicities for context. These
                # are retrieved values, not flatness limits, so they are
                # marked distinctly -- a limit and a measurement are not
                # the same claim.
                for nm, mass, zrel in (("K2-18 b", 8.92, 100.0),
                                       ("TOI-270 d", 4.78, 100.0),
                                       ("GJ 3470 b", 13.9, 100.0)):
                    mu_lit = MU_H2HE + zrel * 5e-4 * (18.0 - MU_H2HE)
                    ax.plot(mass, mu_lit, "o", ms=7, mfc="none", mew=1.6,
                            color=MODEL_COLOR, zorder=5)
                    ax.annotate(nm, (mass, mu_lit), textcoords="offset points",
                                xytext=(6, -12), fontsize=8, color=MODEL_COLOR)
        axes[0].set_ylabel(r"lower limit on $\mu$  [amu]   (2$\sigma$)")
        sec = axes[-1].secondary_yaxis(
            "right", functions=(lambda m: np.clip(
                (np.asarray(m) - MU_H2HE) / (18.0 - MU_H2HE) / 5e-4, 1, 1e5),
                lambda z: MU_H2HE + np.asarray(z) * 5e-4 * (18.0 - MU_H2HE)))
        sec.set_ylabel(r"approx. metallicity  [$\times$ solar]")
        fig.suptitle("Patchwork — mean molecular weight limits from spectral flatness "
                     "(or aerosol muting); open circles are published retrievals",
                     y=0.99)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        return _savefig(fig, out_dir, "patchwork_mmw_trend")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="Directory of per-target Patchwork outputs.")
    ap.add_argument("-o", "--out-dir", default=".")
    args = ap.parse_args()

    root = Path(os.path.expanduser(args.root))
    out = Path(os.path.expanduser(args.out_dir))
    data = load_survey(root)
    params = planet_params([pretty(s) for s in data])
    params = {s: params[pretty(s)] for s in list(data) if pretty(s) in params}
    data = {s: v for s, v in data.items() if s in params}
    if not data:
        print("No spectra with archive parameters found."); return 1

    print(f"{'planet':13s} {'Teq':>5s} {'g':>6s} {'H':>6s} {'ppm/H':>7s} "
          f"{'A_H':>6s} {'A_H<2s':>7s} {'mu >':>7s}")
    for slug in sorted(data, key=lambda s: params[s]["teq"]):
        p, m = params[slug], amplitude_in_H(data[slug], params[slug]["delta_ppm"])
        mu = MU_H2HE * A_REF_SCALE_HEIGHTS / max(m["amp_H_2sig"], 1e-6)
        print(f"{pretty(slug):13s} {p['teq']:5.0f} {p['g']:6.1f} "
              f"{p['H_km']:6.0f} {p['delta_ppm']:7.1f} {m['amp_H']:6.2f} "
              f"{m['amp_H_2sig']:7.2f} {mu:7.1f}")
    print("\n" + plot_absolute(data, params, out))
    print(plot_panels(data, params, out))
    print(plot_trend(data, params, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
