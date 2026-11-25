"""PLATON 7 retrieval of one Patchwork G395H transmission spectrum.

    platon-python retrieve_patchwork.py L_98_59_d --mode eq
    platon-python retrieve_patchwork.py L_98_59_d --mode free

Run on a GPU node (retrieve_patchwork.sbatch), never a login node.

Input: the per-detector combined spectra the Patchwork combine step writes,
<root>/<slug>/combined/combined_{nrs1,nrs2}_transmission_spectrum.csv
(wave_um, wave_err_um = half bin width, depth_ppm, depth_err_ppm). Planet
and star come from scripts/patchwork/planet_params_v17.csv. (The Stage 6.5
corrected spectrum is not an input: it has no bin widths or detector split,
and stellar contamination belongs in the retrieval, not divided out first.)

Model choices, and why:
- NRS1 is the reference and NRS2 gets a fitted offset (`offset_nrs2`):
  the combine step only MEASURES the NRS1-NRS2 offset, it never applies it.
- `error_excess` (added in quadrature) is fitted: the combined spectra have
  chi2_red 1.5-2.9 against smooth models, so the quoted errors are too small
  and a retrieval that trusts them gives over-tight posteriors.
- Planet mass is fixed at the catalogue value; Rp is the radius at
  PLATON's 1 bar reference pressure, free over +-30%.
- Isothermal terminator, grey cloud top (log_cloudtop_P in Pa, up to 1e8 =
  effectively cloud-free).
- eq:   logZ (log10 x solar) and C/O in chemical equilibrium.
  free: log VMRs of --gases (default H2O CH4 CO2 CO NH3; add SO2, H2S
        etc. per planet), H2+He filling the rest.
  flat: featureless null -- cloud top at the top of PLATON's pressure grid
        (10^-3.9 Pa), so only Rp, the NRS2 offset and the error excess are
        fitted. ln B = ln Z(eq or free) - ln Z(flat) is the evidence for
        any atmospheric feature in the spectrum.
- The mean molecular weight mu at 1 mbar is computed for posterior draws
  and reported, since mu is what the Patchwork population science uses.

Outputs in --out (default ~/scratch/platon/<slug>/<mode>[_haze]/):
result.pkl (PLATON RetrievalResult), posterior.npz (equal-weight
samples + names), models.npz (data, best fit and 1-sigma model band at
native resolution), summary.json (medians, 1-sigma, ln
evidence, chi2, mu), spectrum_transit.png, corner.png,
contrib_contrib_func.png. Refuses to overwrite an
existing result without --force.
"""
import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_PARAMS = HERE.parent / "patchwork" / "planet_params_v17.csv"
FREE_GASES = "H2O,CH4,CO2,CO,NH3"  # H2-He is appended as the filler
MU_PRESSURE_PA = 100.0  # 1 mbar, where G395H transmission probes
FLAT_LOG_CLOUDTOP_P = -3.9  # log10 Pa: just inside the top of PLATON's grid (1e-4 Pa;
                           # exactly -4 fails its bounds check by rounding)


def slugify(name):
    return name.replace(" ", "_").replace("-", "_").replace(".", "_")


def load_planet(params_csv, slug):
    with open(params_csv) as f:
        for row in csv.DictReader(f):
            if slugify(row["pl_name"]) == slug:
                return row
    sys.exit(f"{slug}: no row in {params_csv}")


def read_csv_spectrum(path):
    rows = [r for r in csv.reader(open(path)) if r and not r[0].startswith("#")]
    header, data = rows[0], np.array(rows[1:], dtype=float)
    col = {name: data[:, i] for i, name in enumerate(header)}
    return col


def load_spectrum(root, slug):
    """Return bins (m), depths, errors (fractional) and the NRS1/NRS2 split."""
    parts = []
    for det in ("nrs1", "nrs2"):
        p = root / slug / "combined" / f"combined_{det}_transmission_spectrum.csv"
        if not p.exists():
            sys.exit(f"missing {p}")
        parts.append(read_csv_spectrum(p))
    n1 = len(parts[0]["wave_um"])
    w = np.concatenate([p["wave_um"] for p in parts])
    hw = np.concatenate([p["wave_err_um"] for p in parts])
    depth = 1e-6 * np.concatenate([p["depth_ppm"] for p in parts])
    err = 1e-6 * np.concatenate([p["depth_err_ppm"] for p in parts])
    bins = 1e-6 * np.column_stack([w - hw, w + hw])
    return bins, depth, err, n1


def build_fit_info(retriever, planet, mode, n1, n_total, haze, gases):
    from platon.constants import R_sun, R_earth, M_earth, AU
    Rs = float(planet["rstar_rsun"]) * R_sun
    Rp = float(planet["rp_rearth"]) * R_earth
    Mp = float(planet["mp_mearth"]) * M_earth
    Teff = float(planet["teff_k"])
    Teq = Teff * np.sqrt(Rs / (2 * float(planet["a_au"]) * AU))
    T_hi = min(4000.0, 1.5 * Teq)
    T0 = min(max(0.8 * Teq, 110.0), T_hi - 1)

    common = dict(Rs=Rs, Mp=Mp, Rp=Rp, T=T0, log_cloudtop_P=4,
                  log_scatt_factor=0, error_excess=0, T_star=Teff,
                  transit_offsets={"offset_nrs2": (n1, n_total)})
    if mode == "flat":
        common["log_cloudtop_P"] = FLAT_LOG_CLOUDTOP_P
        fit_info = retriever.get_default_fit_info(logZ=0, CO_ratio=0.53, **common)
    elif mode == "eq":
        fit_info = retriever.get_default_fit_info(logZ=1, CO_ratio=0.53, **common)
        fit_info.add_uniform_fit_param("logZ", -1, 3)
        fit_info.add_uniform_fit_param("CO_ratio", 0.05, 2.0)
    else:
        fit_info = retriever.get_default_fit_info(logZ=None, CO_ratio=None,
                                                  fit_vmr=True, **common)
        fit_info.add_gases_vmr(gases + ["H2-He"], 1e-12, 10**-0.3)

    # Rp is the radius at 1 bar; with the flat model's deck at 1e-4 Pa the
    # visible radius sits ~20 scale heights higher, so it needs room below.
    fit_info.add_uniform_fit_param("Rp", (0.4 if mode == "flat" else 0.7) * Rp,
                                   1.3 * Rp)
    if mode != "flat":
        fit_info.add_uniform_fit_param("T", 100, T_hi)
        fit_info.add_uniform_fit_param("log_cloudtop_P", -1, 8)
    fit_info.add_uniform_fit_param("offset_nrs2", -300e-6, 300e-6)
    fit_info.add_uniform_fit_param("error_excess", 0, 300e-6)
    if haze and mode != "flat":
        fit_info.add_uniform_fit_param("log_scatt_factor", -2, 6)
    return fit_info, Teq


def mu_posterior(retriever, result, data, n_draws):
    """Mean molecular weight at MU_PRESSURE_PA for posterior draws.

    Re-evaluates the likelihood with ret_best_fit=True, which returns the
    full transit info dict (P_profile, mu_profile); this uses PLATON's
    private _make_ln_like, so re-check it after a PLATON upgrade."""
    ln_like = retriever._make_ln_like(data, result.fit_info, True, (), None, None)
    draws = result.equal_samples
    idx = np.random.default_rng(0).choice(len(draws), min(n_draws, len(draws)),
                                          replace=False)
    mus = []
    for params in draws[idx]:
        ret = ln_like(params, ret_best_fit=True)
        if np.isscalar(ret):
            continue
        info = ret[1]
        # mu_profile is on PLATON's full pressure grid, whose pressures are
        # full_TP_profile[0]; P_profile is only the layers above the cloud
        # deck / below the escape radius, so it is usually shorter.
        mu = np.asarray(info["mu_profile"])
        P = np.asarray(info["full_TP_profile"])[0]
        mus.append(float(np.interp(np.log(MU_PRESSURE_PA), np.log(P), mu)))
    return np.array(mus)


def summarize(samples, names):
    out = {}
    for i, name in enumerate(names):
        lo, med, hi = np.percentile(samples[:, i], [15.865, 50, 84.135])
        out[name] = {"median": float(med), "minus": float(med - lo),
                     "plus": float(hi - med)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("slug", help="target directory name, e.g. L_98_59_d")
    ap.add_argument("--mode", choices=["eq", "free", "flat"], default="eq")
    ap.add_argument("--root", default=os.path.expanduser("~/scratch/patchwork"))
    ap.add_argument("--params", default=str(DEFAULT_PARAMS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-live", type=int, default=1000)
    ap.add_argument("--n-eff", type=int, default=5000,
                    help="nautilus stopping point (effective posterior samples); "
                         "lower it for weakly constrained free retrievals")
    ap.add_argument("--haze", action="store_true",
                    help="also fit a Rayleigh-like scattering amplitude")
    ap.add_argument("--gases", default=FREE_GASES,
                    help="free mode: comma-separated gases to fit (H2-He fills)")
    ap.add_argument("--mu-draws", type=int, default=300)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing result")
    ap.add_argument("--post-only", action="store_true",
                    help="reuse result.pkl: redo summary, mu and plots, no sampling")
    args = ap.parse_args()

    out = Path(args.out or os.path.expanduser(
        f"~/scratch/platon/{args.slug}/{args.mode}"
        + ("_haze" if args.haze else "")))
    have = (out / "result.pkl").exists()
    if args.post_only and not have:
        sys.exit(f"--post-only: no {out}/result.pkl")
    if have and not (args.force or args.post_only):
        sys.exit(f"{out}/result.pkl exists; --force reruns, --post-only redoes outputs")
    out.mkdir(parents=True, exist_ok=True)

    import jax
    import platon
    from platon.combined_retriever import CombinedRetriever
    from platon.plotter import Plotter

    if os.environ.get("CUDA_VISIBLE_DEVICES") and jax.default_backend() != "gpu":
        sys.exit("GPU allocated but jax is on CPU; check platon-python / modules")

    planet = load_planet(args.params, args.slug)
    bins, depths, errors, n1 = load_spectrum(Path(args.root), args.slug)
    retriever = CombinedRetriever()
    gases = [g.strip() for g in args.gases.split(",") if g.strip()]
    fit_info, Teq = build_fit_info(retriever, planet, args.mode, n1,
                                   len(depths), args.haze, gases)
    print(f"{planet['pl_name']}: {n1} NRS1 + {len(depths) - n1} NRS2 channels, "
          f"Teq {Teq:.0f} K, mode {args.mode}, backend {jax.default_backend()}")
    print("fit params:", fit_info.fit_param_names)

    if args.post_only:
        with open(out / "result.pkl", "rb") as f:
            result = pickle.load(f)
        if list(result.fit_info.fit_param_names) != list(fit_info.fit_param_names):
            sys.exit("result.pkl was fit with different parameters; rerun with --force")
        fit_info = result.fit_info
        old = out / "summary.json"
        elapsed = json.loads(old.read_text())["elapsed_s"] if old.exists() else None
    else:
        ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
        t0 = time.time()
        result = retriever.run_nautilus(bins, depths, errors, None, None, None,
                                        fit_info, n_live=args.n_live,
                                        n_eff=args.n_eff, n_train_workers=ncpu)
        elapsed = round(time.time() - t0, 1)
        with open(out / "result.pkl", "wb") as f:
            pickle.dump(result, f)
    data = (bins, depths, errors, None, None, None)
    # The flat model fixes the composition (logZ=0), so its mu means nothing.
    mus = (np.array([]) if args.mode == "flat"
           else mu_posterior(retriever, result, data, args.mu_draws))

    names = list(fit_info.fit_param_names)
    summary = {
        "planet": planet["pl_name"], "slug": args.slug, "mode": args.mode,
        "spectrum": str(Path(args.root) / args.slug / "combined"),
        "haze": args.haze,
        "gases": gases if args.mode == "free" else None,
        "platon_version": platon.__version__, "jax_version": jax.__version__,
        "sampler": "nautilus", "n_live": args.n_live, "n_eff_target": args.n_eff,
        "n_nrs1": n1, "n_nrs2": len(depths) - n1, "Teq_K": Teq,
        "planet_params": planet,
        "ln_evidence": float(np.atleast_1d(result.final_logz)[-1])
        if hasattr(result, "final_logz") else None,
        "chi2_best": float(result.transit_chi_sqr),
        "n_data": int(len(depths)),
        "params": summarize(np.asarray(result.equal_samples), names),
        "priors": {n: [float(fit_info.all_params[n].low_lim),
                       float(fit_info.all_params[n].high_lim)] for n in names},
        "mu_1mbar": ({"median": float(np.median(mus)),
                      "p16": float(np.percentile(mus, 15.865)),
                      "p84": float(np.percentile(mus, 84.135)),
                      "n_draws": int(len(mus))} if len(mus) else None),
        "elapsed_s": elapsed,
        # Peak-to-peak of the binned best fit over NRS1 (no offset there):
        # ~0 for the flat model, the size of the fitted features otherwise.
        "model_span_nrs1_ppm": float(1e6 * np.ptp(
            np.asarray(result.best_fit_transit_depths)[:n1])),
    }
    for k in ("offset_nrs2", "error_excess"):
        if k in summary["params"]:
            summary["params"][k + "_ppm"] = {
                kk: 1e6 * v for kk, v in summary["params"][k].items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # Light products for plotting off Fir (result.pkl needs PLATON to load).
    eq = np.asarray(result.equal_samples)
    np.savez_compressed(out / "posterior.npz", names=np.array(names),
                        samples=eq, mu_1mbar=mus)
    bf = result.best_fit_transit_dict
    rand = np.asarray(result.random_transit_depths)
    off = float(np.median(eq[:, names.index("offset_nrs2")]))
    np.savez_compressed(
        out / "models.npz",
        bins_m=bins, depth=depths, depth_err=errors, n_nrs1=n1,
        offset_nrs2_median=off,
        best_fit_binned=np.asarray(result.best_fit_transit_depths),
        wavelength_m=np.asarray(bf["unbinned_wavelengths"]),
        best_fit=np.asarray(bf["unbinned_depths"]),
        band_p16=np.percentile(rand, 15.865, axis=0) if len(rand) else [],
        band_p50=np.percentile(rand, 50, axis=0) if len(rand) else [],
        band_p84=np.percentile(rand, 84.135, axis=0) if len(rand) else [])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cwd = os.getcwd()
    os.chdir(out)
    try:
        for name, plot in (
                ("spectrum", lambda: Plotter.plot_retrieval_transit_spectrum(
                    result, prefix="spectrum")),
                ("corner", lambda: Plotter.plot_retrieval_corner(
                    result, filename="corner.png")),
                ("contrib", lambda: Plotter.plot_contrib_func(
                    bf, log_scale=True, prefix="contrib"))):
            if name == "contrib" and args.mode == "flat":
                continue  # fully opaque deck: no contribution function
            try:
                plot()
            except Exception as exc:  # one bad plot must not lose the rest
                print(f"plot {name} failed: {exc!r}")
            plt.close("all")
    finally:
        os.chdir(cwd)

    p = summary["params"]
    print(f"sampling took {elapsed} s -> {out}")
    for name in names:
        print(f"  {name:16s} {p[name]['median']:.4g} "
              f"-{p[name]['minus']:.2g} +{p[name]['plus']:.2g}")
    if summary["mu_1mbar"]:
        m = summary["mu_1mbar"]
        print(f"  mu(1 mbar)       {m['median']:.2f} "
              f"[{m['p16']:.2f}, {m['p84']:.2f}]")
    print(f"  chi2 {summary['chi2_best']:.1f} / {summary['n_data']} points, "
          f"ln(evidence) {summary['ln_evidence']:.2f}, "
          f"NRS1 model span {summary['model_span_nrs1_ppm']:.0f} ppm")


if __name__ == "__main__":
    main()
