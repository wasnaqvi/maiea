"""Collect the PLATON survey retrievals into a table and survey figures.

    python collect_platon_survey.py ~/scratch/platon -o ~/scratch/platon/survey

Reads <root>/<slug>/{flat,eq,free}/{summary.json,posterior.npz,models.npz}
written by retrieve_patchwork.py. Needs numpy, scipy and matplotlib only,
so it runs on the Mac on copied products as well as on Fir.

Writes:
  platon_survey_table.csv / .md  one row per planet: evidences, Bayes
      factors against the flat null and their Benneke & Seager (2013) sigma,
      logZ, C/O, mu (eq and free), T, cloud top, NRS2 offset, error excess
  platon_spectra_atlas.{pdf,png,svg}   every spectrum with its equilibrium
      retrieval (median, 1 and 2 sigma) and the flat null, coolest first
  platon_posteriors.{pdf,png,svg}  logZ and mu marginals per planet
  platon_mu_teq.{pdf,png,svg}      mu at 1 mbar against T_eq
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import exoplot as xp  # noqa: E402

MODES = ("flat", "eq", "free")
RADIUS_CUT_REARTH = 1.75  # Patchwork I sample definition
DISPLAY_R = 200           # model display resolution in the atlas
EQ_COLOR = "#2c7bb6"
FREE_COLOR = "#d7301f"


def bayes_sigma(ln_b):
    """Benneke & Seager (2013) / Trotta (2008): B = -1/(e p ln p), p -> sigma."""
    from scipy.optimize import brentq
    from scipy.special import erfcinv
    if ln_b is None or not np.isfinite(ln_b):
        return None
    if ln_b <= 0:
        return 0.0  # no preference; the mapping is only defined for B >= 1
    f = lambda lp: -1.0 - lp - np.log(-lp) - ln_b  # ln of -1/(e p ln p) - ln B
    lp = brentq(f, -745.0, -1.0 - 1e-12)          # ln p in (ln p_min, -1)
    return float(np.sqrt(2) * erfcinv(np.exp(lp)))


def load(root):
    planets = []
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir() or d.name in ("logs", "code", "survey"):
            continue
        runs = {}
        for m in MODES:
            s = d / m / "summary.json"
            if s.exists():
                runs[m] = {"summary": json.loads(s.read_text()), "dir": d / m}
        if runs:
            planets.append({"slug": d.name, "runs": runs})
    return planets


def post(run, name):
    z = np.load(run["dir"] / "posterior.npz")
    names = list(z["names"])
    if name == "mu_1mbar":
        return z["mu_1mbar"]
    return z["samples"][:, names.index(name)] if name in names else None


# Parameters whose posterior should NOT pile against a prior edge. Gas VMRs
# are left out: an upper limit against the lower edge is a normal result.
EDGE_CHECKED = ("logZ", "CO_ratio", "Rp", "T", "offset_nrs2", "error_excess")


def edge_flags(run):
    """Flag checked parameters the prior, not the data, is shaping:
    - "<p> 1σ reaches upper/lower prior edge": the 84th (16th) percentile is
      in the outer 5% of the prior, so the posterior is truncated there
      (logZ at 3 is PLATON's 1000x solar grid limit, not a measurement);
    - "<p> unconstrained": the 68% interval spans >= 55% of the prior range
      (a flat prior gives 68%), so the data barely move it."""
    s = run["summary"]
    pri = s.get("priors")
    if not pri:
        return ["no priors in summary (run write_priors.py)"]
    flags = []
    for name in EDGE_CHECKED:
        if name not in pri or name not in s["params"]:
            continue
        lo, hi = pri[name]
        q = s["params"][name]
        p16 = (q["median"] - q["minus"] - lo) / (hi - lo)
        p84 = (q["median"] + q["plus"] - lo) / (hi - lo)
        if p84 - p16 >= 0.55:
            flags.append(f"{name} unconstrained")
            continue
        if p84 > 0.95:
            flags.append(f"{name} 1σ reaches upper prior edge")
        if p16 < 0.05 and name != "error_excess":
            flags.append(f"{name} 1σ reaches lower prior edge")
    return flags


def describe(planet):
    any_run = next(iter(planet["runs"].values()))["summary"]
    p = any_run["planet_params"]
    return {"planet": any_run["planet"], "slug": planet["slug"],
            "Teq_K": any_run["Teq_K"], "rp_rearth": float(p["rp_rearth"]),
            "mp_mearth": float(p["mp_mearth"]),
            "in_sample": float(p["rp_rearth"]) >= RADIUS_CUT_REARTH}


def fmt(par):
    return (f"{par['median']:.3g} -{par['minus']:.2g} +{par['plus']:.2g}"
            if par else "")


def table(planets, out):
    rows = []
    for pl in planets:
        r = describe(pl)
        runs = pl["runs"]
        lnz = {m: runs[m]["summary"]["ln_evidence"] for m in runs}
        for m in ("eq", "free"):
            if m in lnz and "flat" in lnz:
                lnb = lnz[m] - lnz["flat"]
                r[f"lnB_{m}_vs_flat"] = round(lnb, 2)
                r[f"sigma_{m}"] = round(bayes_sigma(lnb), 2)
            r[f"lnZ_{m}"] = lnz.get(m)
        r["lnZ_flat"] = lnz.get("flat")
        for m in ("eq", "free"):
            if m not in runs:
                continue
            s = runs[m]["summary"]
            P = s["params"]
            if m == "eq":
                r["logZ_eq"] = fmt(P.get("logZ"))
                r["CO_eq"] = fmt(P.get("CO_ratio"))
            mu = s.get("mu_1mbar")
            r[f"mu_{m}"] = (f"{mu['median']:.2f} [{mu['p16']:.2f}, {mu['p84']:.2f}]"
                            if mu else "")
            r[f"T_{m}"] = fmt(P.get("T"))
            r[f"logPcloud_{m}"] = fmt(P.get("log_cloudtop_P"))
            r[f"offset_nrs2_ppm_{m}"] = fmt(P.get("offset_nrs2_ppm"))
            r[f"error_excess_ppm_{m}"] = fmt(P.get("error_excess_ppm"))
            r[f"chi2_{m}"] = round(s["chi2_best"], 1)
            r["n_data"] = s["n_data"]
            r[f"flags_{m}"] = "; ".join(edge_flags(runs[m]))
        rows.append(r)
    rows.sort(key=lambda r: r["Teq_K"])
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "platon_survey_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    md = ["| planet | T_eq | R_p | lnB eq | σ eq | lnB free | σ free | logZ (eq) | C/O (eq) | μ eq | μ free | offset NRS2 (ppm) | err. excess (ppm) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| " + " | ".join(str(x) for x in (
            r["planet"] + ("" if r["in_sample"] else " †"), f"{r['Teq_K']:.0f}",
            f"{r['rp_rearth']:.2f}", r.get("lnB_eq_vs_flat", ""), r.get("sigma_eq", ""),
            r.get("lnB_free_vs_flat", ""), r.get("sigma_free", ""),
            r.get("logZ_eq", ""), r.get("CO_eq", ""), r.get("mu_eq", ""),
            r.get("mu_free", ""), r.get("offset_nrs2_ppm_eq", ""),
            r.get("error_excess_ppm_eq", ""))) + " |")
    md.append("")
    md.append("Checks (prior-shaped parameters; see edge_flags):")
    for r in rows:
        for m in ("eq", "free"):
            if r.get(f"flags_{m}"):
                md.append(f"- {r['planet']} ({m}): {r[f'flags_{m}']}")
    md.append("")
    md.append("† below the 1.75 R_E Patchwork I radius cut. lnB = ln Z(model) − "
              "ln Z(flat); σ from Benneke & Seager (2013). μ at 1 mbar, median "
              "[16th, 84th percentile]. logZ is log10 metallicity relative to solar.")
    (out / "platon_survey_table.md").write_text("\n".join(md) + "\n")
    return rows


def binned_model(wl_um, y, edges):
    idx = np.digitize(wl_um, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    sums = np.bincount(idx[ok], y[ok], minlength=len(edges) - 1)
    cnt = np.bincount(idx[ok], minlength=len(edges) - 1)
    mid = np.sqrt(edges[1:] * edges[:-1])
    good = cnt > 0
    return mid[good], sums[good] / cnt[good]


def atlas(planets, rows, out):
    order = [next(p for p in planets if p["slug"] == r["slug"]) for r in rows]
    order = [p for p in order if "eq" in p["runs"]]
    n = len(order)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    w, _ = xp.figsize("aas", "double")
    fig, axes = plt.subplots(nrow, ncol, figsize=(w * 1.35, 1.75 * nrow + 0.4),
                             sharex=True)
    axes = np.atleast_2d(axes)
    for ax, pl in zip(axes.flat, order):
        eqd = pl["runs"]["eq"]["dir"]
        m = np.load(eqd / "models.npz")
        n1 = int(m["n_nrs1"])
        bins_um = 1e6 * m["bins_m"]
        wl = bins_um.mean(axis=1)
        hw = 0.5 * (bins_um[:, 1] - bins_um[:, 0])
        depth = 1e6 * m["depth"].copy()
        depth[n1:] -= 1e6 * float(m["offset_nrs2_median"])  # onto NRS1's scale
        err = 1e6 * m["depth_err"]
        gap = (wl[:n1].max() + hw[:n1].max(), wl[n1:].min() - hw[n1:].min())
        mwl = 1e6 * m["wavelength_m"]
        lo_edge = bins_um.min() - 0.01
        hi_edge = bins_um.max() + 0.01
        edges = xp.constant_r_edges(lo_edge, hi_edge, DISPLAY_R)
        for seg in ((lo_edge, gap[0]), (gap[1], hi_edge)):
            sel = (mwl >= seg[0]) & (mwl <= seg[1])
            if len(m["band_p50"]):
                xs, med = binned_model(mwl[sel], 1e6 * m["band_p50"][sel], edges)
                _, lo = binned_model(mwl[sel], 1e6 * m["band_p16"][sel], edges)
                _, hi = binned_model(mwl[sel], 1e6 * m["band_p84"][sel], edges)
                xp.model_band(ax, xs, med, lo, hi, color=EQ_COLOR)
        if "flat" in pl["runs"]:
            f = np.load(pl["runs"]["flat"]["dir"] / "models.npz")
            ax.axhline(1e6 * float(np.median(f["best_fit_binned"][:int(f["n_nrs1"])])),
                       color=xp.MODEL_GREY, ls="--", lw=0.8, zorder=1)
        xp.data_points(ax, wl, depth, err, xerr=hw, ms=2.6)
        r = next(r for r in rows if r["slug"] == pl["slug"])
        # Name above the panel, never over the data.
        ax.set_title(f"{r['planet']}, {r['Teq_K']:.0f} K", loc="left",
                     fontsize=10, pad=3)
        span = np.percentile(depth, [2, 98])
        pad = 0.3 * (span[1] - span[0]) + 2 * np.median(err)
        ax.set_ylim(span[0] - pad, span[1] + pad)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    # The lowest visible panel of each column carries the wavelength axis.
    for c in range(ncol):
        col = [ax for ax in axes[:, c] if ax.get_visible()]
        if col:
            col[-1].xaxis.set_tick_params(labelbottom=True)
            col[-1].set_xlabel(xp.label("Wavelength", "μm"))
    for ax in axes[:, 0]:
        ax.set_ylabel(xp.label("Transit depth", "ppm"))
    axes[0, 0].set_xlim(2.8, 5.2)
    fig.subplots_adjust(hspace=0.30, wspace=0.32)
    return xp.finalize(fig, str(out / "platon_spectra_atlas"))


def posteriors(planets, rows, out):
    order = [r for r in rows
             if "eq" in next(p for p in planets if p["slug"] == r["slug"])["runs"]]
    names = [r["planet"] + ("" if r["in_sample"] else " †") for r in order]
    w, _ = xp.figsize("aas", "double")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(w * 1.35, 6.2), sharex=True)
    for i, r in enumerate(order):
        runs = next(p for p in planets if p["slug"] == r["slug"])["runs"]
        for ax, key in ((a1, "logZ"), (a2, "mu_1mbar")):
            for m, col, dx in (("eq", EQ_COLOR, -0.18), ("free", FREE_COLOR, 0.18)):
                if m not in runs or (key == "logZ" and m != "eq"):
                    continue
                s = post(runs[m], key)
                if s is None or len(s) < 10:
                    continue
                if key == "mu_1mbar":
                    s = np.log10(s)
                x = i + (0 if key == "logZ" else dx)
                v = ax.violinplot(s, positions=[x], widths=0.34 if key != "logZ" else 0.6,
                                  showextrema=False)
                for b in v["bodies"]:
                    b.set_facecolor(col); b.set_edgecolor(col); b.set_alpha(0.55)
                q = np.percentile(s, [15.865, 50, 84.135])
                ax.plot([x, x], [q[0], q[2]], color="k", lw=0.9)
                ax.plot(x, q[1], "o", mfc="white", mec="k", ms=3.2)
    a1.set_ylabel(r"log$_{10}$ metallicity (× solar)")
    a2.set_ylabel(r"log$_{10}\,\mu$ at 1 mbar")
    xp.zero_line(a1)
    a2.axhline(np.log10(2.3), color=xp.MODEL_GREY, ls="--", lw=0.8)
    a2.plot([], [], "s", color=EQ_COLOR, alpha=0.55, label="equilibrium")
    a2.plot([], [], "s", color=FREE_COLOR, alpha=0.55, label="free chemistry")
    a2.legend(loc="upper center", ncol=2, fontsize=9.5)
    # Headroom so the panel tags and legend sit above every violin.
    a1.set_ylim(-1.3, 3.6)
    top = max(v for c in a2.collections for p in c.get_paths()
              for v in p.vertices[:, 1]) if a2.collections else 1.8
    a2.set_ylim(0.25, top + 0.35)
    a2.set_xticks(range(len(order)))
    a2.set_xticklabels([f"{n}, {r['Teq_K']:.0f} K" for n, r in zip(names, order)],
                       rotation=50, ha="right", rotation_mode="anchor", fontsize=9.5)
    a2.minorticks_off(); a1.tick_params(axis="x", which="minor", bottom=False, top=False)
    a2.set_xlim(-0.6, len(order) - 0.4)
    fig.subplots_adjust(hspace=0.05)
    xp.panel_label(a1, "a"); xp.panel_label(a2, "b")
    return xp.finalize(fig, str(out / "platon_posteriors"))


def mu_teq(planets, rows, out):
    w, _ = xp.figsize("aas", "single")
    fig, ax = plt.subplots(figsize=(w * 1.3, w * 1.0))
    for r in rows:
        runs = next(p for p in planets if p["slug"] == r["slug"])["runs"]
        for m, col, dx in (("eq", EQ_COLOR, 0.985), ("free", FREE_COLOR, 1.015)):
            if m not in runs or not runs[m]["summary"].get("mu_1mbar"):
                continue
            mu = runs[m]["summary"]["mu_1mbar"]
            x = r["Teq_K"] * dx
            ax.errorbar(x, mu["median"], yerr=[[mu["median"] - mu["p16"]],
                                               [mu["p84"] - mu["median"]]],
                        fmt="o" if r["in_sample"] else "s", ms=4.5, color=col,
                        mfc=col if r["in_sample"] else "white", mec=col, lw=0.9,
                        capsize=0)
    ax.axhline(2.3, color=xp.MODEL_GREY, ls="--", lw=0.8)
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ax.yaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20, 30]))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(xp.label(r"$T_{\rm eq}$", "K"))
    ax.set_ylabel(r"$\mu$ at 1 mbar (amu)")
    ax.plot([], [], "o", color=EQ_COLOR, ms=4.5, label="equilibrium")
    ax.plot([], [], "o", color=FREE_COLOR, ms=4.5, label="free chemistry")
    ax.plot([], [], "s", mfc="white", mec="0.3", ms=4.5,
            label=r"$R_{\rm p} < 1.75\,R_\oplus$")
    # Legend above the box: the error bars fill the whole panel.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              fontsize=9.5, handletextpad=0.3, columnspacing=1.0)
    return xp.finalize(fig, str(out / "platon_mu_teq"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="PLATON output root (~/scratch/platon)")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    root = Path(args.root).expanduser()
    out = Path(args.out).expanduser() if args.out else root / "survey"
    out.mkdir(parents=True, exist_ok=True)
    xp.use_style(fontsize=10.5, family="serif")
    planets = load(root)
    if not planets:
        sys.exit(f"no retrievals under {root}")
    rows = table(planets, out)
    print(f"{len(rows)} planets -> {out}")
    for r in rows:
        print(f"  {r['planet']:14s} Teq {r['Teq_K']:5.0f}  lnB eq {r.get('lnB_eq_vs_flat', '-'):>6}"
              f" ({r.get('sigma_eq', '-')}σ)  free {r.get('lnB_free_vs_flat', '-'):>6}"
              f"  logZ {r.get('logZ_eq', '')}  mu_eq {r.get('mu_eq', '')}")
    atlas(planets, rows, out)
    posteriors(planets, rows, out)
    mu_teq(planets, rows, out)


if __name__ == "__main__":
    main()
