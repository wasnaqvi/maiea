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
    python scripts/patchwork/replot_white_fits.py ~/scratch/patchwork --out /tmp/preview

2026-10-20: the data are drawn with the systematics removed (the fitted
transit plus residuals) under the transit model alone. The old figure
drew the FULL model -- transit plus every regressor, which jitters per
integration -- over the raw flux. That band hid whether the transit itself
had been fitted (HD 15337 c o019 NRS2). Time is centred on the fitted
mid-transit, with the ephemeris prediction marked when it is >5 min off.
The residual panel is unchanged.

2026-10-23: reads the npz first (the CSV rounded BJD to 86 s before
then), redraws the pass-1 figures too, and recovers the integrations a
fit dropped at tilt steps from pass 1 so they stay on the figure.
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

from aster_toolkit.data_reduction.juliet import (  # noqa: E402
    load_white_fit_series, replot_white_fit)


def replot(fit_dir: Path, out: Path | None = None,
           n_bins: int | None = None) -> str | None:
    if replot_white_fit(fit_dir, out, n_bins=n_bins) is None:
        return None
    s = json.loads((fit_dir / "white_fit_summary.json").read_text())
    series = load_white_fit_series(fit_dir)
    rms = float(np.nanstd(series["flux"] - series["model"]) * 1e6)
    beta = (s.get("rednoise") or {}).get("beta_median")
    offset = (s["t0_p1"]["median"] - s["t0_obs"]) * 1440 if "t0_obs" in s else None
    n_ex = 0 if series["excluded_time"] is None else series["excluded_time"].size
    return (f"{fit_dir}  depth={s['depth_ppm']['median']:.0f} ppm  rms={rms:.0f} ppm"
            + (f"  beta={beta:.2f}" if beta is not None else "")
            + (f"  t0-ephemeris={offset:+.0f} min" if offset is not None else "")
            + (f"  excluded={n_ex}" if n_ex else "")
            + ("" if series["source"].endswith(".npz")
               else f"  [{series['source']}: time rounded]"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate white-light fit figures from saved series.")
    ap.add_argument("output_root",
                    help="Patchwork output root, e.g. ~/scratch/patchwork")
    ap.add_argument("--target", default="*",
                    help="Target slug to restrict to (default: all).")
    ap.add_argument("--no-pass1", action="store_true",
                    help="Skip the pass-1 (pre-scan) figures.")
    ap.add_argument("--out", default=None,
                    help="Write under this root (same tree) instead of in "
                         "place -- preview a restyle before overwriting.")
    ap.add_argument("--bins", type=int, default=None,
                    help="Display bins (default juliet.WHITE_PLOT_BINS).")
    args = ap.parse_args(argv)

    root = Path(os.path.expanduser(args.output_root))
    patterns = [os.path.join(root, args.target, "fits", "*", "nrs*",
                             "white_fit_summary.json")]
    if not args.no_pass1:
        patterns.append(os.path.join(root, args.target, "fits", "*", "nrs*",
                                     "pass1", "white_fit_summary.json"))
    dirs = sorted({Path(p).parent for pat in patterns for p in glob.glob(pat)})
    if not dirs:
        print(f"No white_fit_summary.json under {patterns[0]}")
        return 1

    out_root = Path(os.path.expanduser(args.out)) if args.out else None
    done = 0
    for fit_dir in dirs:
        out = out_root / fit_dir.relative_to(root) if out_root else None
        line = replot(fit_dir, out, args.bins)
        if line:
            print(line)
            done += 1
    where = f"under {out_root}" if out_root else "in place"
    print(f"\nReplotted {done} figure(s). PDF + SVG written {where}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
