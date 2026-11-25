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
- The mean molecular weight mu at 1 mbar is computed for posterior draws
  and reported, since mu is what the Patchwork population science uses.

Outputs in --out (default ~/scratch/platon/<slug>/<mode>[_haze]/):
result.pkl (PLATON RetrievalResult), summary.json (medians, 1-sigma, ln
evidence, chi2, mu), spectrum.png, corner.png. Refuses to overwrite an
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
    if mode == "eq":
        fit_info = retriever.get_default_fit_info(logZ=1, CO_ratio=0.53, **common)
        fit_info.add_uniform_fit_param("logZ", -1, 3)
        fit_info.add_uniform_fit_param("CO_ratio", 0.05, 2.0)
    else:
        fit_info = retriever.get_default_fit_info(logZ=None, CO_ratio=None,
                                                  fit_vmr=True, **common)
        fit_info.add_gases_vmr(gases + ["H2-He"], 1e-12, 10**-0.3)

    fit_info.add_uniform_fit_param("Rp", 0.7 * Rp, 1.3 * Rp)
    fit_info.add_uniform_fit_param("T", 100, T_hi)
    fit_info.add_uniform_fit_param("log_cloudtop_P", -1, 8)
    fit_info.add_uniform_fit_param("offset_nrs2", -300e-6, 300e-6)
    fit_info.add_uniform_fit_param("error_excess", 0, 300e-6)
    if haze:
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
    ap.add_argument("--mode", choices=["eq", "free"], default="eq")
    ap.add_argument("--root", default=os.path.expanduser("~/scratch/patchwork"))
    ap.add_argument("--params", default=str(DEFAULT_PARAMS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-live", type=int, default=1000)
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
                                        n_train_workers=ncpu)
        elapsed = round(time.time() - t0, 1)
        with open(out / "result.pkl", "wb") as f:
            pickle.dump(result, f)
    data = (bins, depths, errors, None, None, None)
    mus = mu_posterior(retriever, result, data, args.mu_draws)

    names = list(fit_info.fit_param_names)
    summary = {
        "planet": planet["pl_name"], "slug": args.slug, "mode": args.mode,
        "spectrum": str(Path(args.root) / args.slug / "combined"),
        "haze": args.haze,
        "gases": gases if args.mode == "free" else None,
        "platon_version": platon.__version__, "jax_version": jax.__version__,
        "sampler": "nautilus", "n_live": args.n_live,
        "n_nrs1": n1, "n_nrs2": len(depths) - n1, "Teq_K": Teq,
        "planet_params": planet,
        "ln_evidence": float(np.atleast_1d(result.final_logz)[-1])
        if hasattr(result, "final_logz") else None,
        "chi2_best": float(result.transit_chi_sqr),
        "n_data": int(len(depths)),
        "params": summarize(np.asarray(result.equal_samples), names),
        "mu_1mbar": ({"median": float(np.median(mus)),
                      "p16": float(np.percentile(mus, 15.865)),
                      "p84": float(np.percentile(mus, 84.135)),
                      "n_draws": int(len(mus))} if len(mus) else None),
        "elapsed_s": elapsed,
    }
    for k in ("offset_nrs2", "error_excess"):
        if k in summary["params"]:
            summary["params"][k + "_ppm"] = {
                kk: 1e6 * v for kk, v in summary["params"][k].items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    cwd = os.getcwd()
    os.chdir(out)
    try:
        Plotter.plot_retrieval_transit_spectrum(result, prefix="spectrum")
        Plotter.plot_retrieval_corner(result, filename="corner.png")
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
          f"ln(evidence) {summary['ln_evidence']:.2f}")


if __name__ == "__main__":
    main()
