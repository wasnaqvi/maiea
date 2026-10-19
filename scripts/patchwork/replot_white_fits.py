#!/usr/bin/env python
"""Regenerate white-light fit figures from saved products — no refitting.

Every white fit leaves ``white_lightcurve_residuals.npz`` (and, since
2026-07-31, ``white_lightcurve_series.csv``) holding time, flux, error and
the full juliet model beside ``white_fit_summary.json``. This script
redraws ``white_lightcurve_fit.pdf/.svg`` from those with
``juliet.plot_white_fit_series`` -- the same function the fit uses -- so
figures can be restyled without repeating hours of nested sampling.

    python scripts/patchwork/replot_white_fits.py ~/scratch/patchwork
    python scripts/patchwork/replot_white_fits.py ~/scratch/patchwork --target TOI_270_c

2026-10-20: the data are drawn with the systematics removed (the fitted
transit plus residuals) under the transit model alone. The old figure
drew the FULL model -- transit plus every regressor, which jitters per
integration -- over the raw flux. That band hid whether the transit itself
had been fitted (HD 15337 c o019 NRS2). Time is centred on the fitted
mid-transit, with the ephemeris prediction marked when it is >5 min off.
The residual panel is unchanged.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aster_toolkit.data_reduction.juliet import plot_white_fit_series  # noqa: E402


def _load_series(fit_dir: Path) -> dict[str, np.ndarray] | None:
    npz_path = fit_dir / "white_lightcurve_residuals.npz"
    if npz_path.is_file():
        with np.load(npz_path) as d:
            return {k: np.asarray(d[k], float)
                    for k in ("time", "flux", "flux_err", "model")}
    csv_path = fit_dir / "white_lightcurve_series.csv"
    if csv_path.is_file():
        d = np.genfromtxt(csv_path, delimiter=",", names=True)
        return {"time": d["time_bjd"], "flux": d["flux"],
                "flux_err": d["flux_err"], "model": d["model"]}
    return None


def replot(fit_dir: Path) -> str | None:
    json_path = fit_dir / "white_fit_summary.json"
    series = _load_series(fit_dir)
    if series is None or not json_path.is_file():
        return None
    with json_path.open() as fh:
        s = json.load(fh)
    plot_white_fit_series(fit_dir, s, series["time"], series["flux"],
                          series["flux_err"], series["model"])
    rms = float(np.nanstd(series["flux"] - series["model"]) * 1e6)
    beta = (s.get("rednoise") or {}).get("beta_median")
    offset = (s["t0_p1"]["median"] - s["t0_obs"]) * 1440 if "t0_obs" in s else None
    return (f"{fit_dir}  depth={s['depth_ppm']['median']:.0f} ppm  rms={rms:.0f} ppm"
            + (f"  beta={beta:.2f}" if beta is not None else "")
            + (f"  t0-ephemeris={offset:+.0f} min" if offset is not None else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate white-light fit figures from saved series.")
    ap.add_argument("output_root",
                    help="Patchwork output root, e.g. ~/scratch/patchwork")
    ap.add_argument("--target", default="*",
                    help="Target slug to restrict to (default: all).")
    args = ap.parse_args(argv)

    root = os.path.expanduser(args.output_root)
    pattern = os.path.join(root, args.target, "fits", "*", "nrs*",
                           "white_fit_summary.json")
    dirs = sorted({Path(p).parent for p in glob.glob(pattern)})
    if not dirs:
        print(f"No white_fit_summary.json under {pattern}")
        return 1

    done = 0
    for fit_dir in dirs:
        line = replot(fit_dir)
        if line:
            print(line)
            done += 1
    print(f"\nReplotted {done} figure(s). PDF + SVG written in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
