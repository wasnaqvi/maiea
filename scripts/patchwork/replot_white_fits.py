#!/usr/bin/env python
"""Regenerate white-light fit figures from saved series — no refitting.

``fit_white_lightcurve`` writes ``white_lightcurve_series.csv``
(time_bjd, flux, flux_err, model) next to every fit precisely so figures
can be restyled without repeating hours of nested sampling. This script
rebuilds ``white_lightcurve_fit.pdf/.svg`` from that CSV plus
``white_fit_summary.json``.

    python scripts/patchwork/replot_white_fits.py ~/scratch/patchwork
    python scripts/patchwork/replot_white_fits.py ~/scratch/patchwork --target TOI_270_c

Differences from the figures produced before 2026-07-31:
  - per-integration error bars are drawn (they were absent entirely),
    as a faint layer (alpha 0.18) under more opaque markers (0.38, was
    0.16) so the binned series stays readable
  - the annotated depth comes from the POSTERIOR (median p^2) instead of
    ``1 - min(model)``, which also contained the systematics trend and
    tilt-step offsets. Those disagree by 2000 ppm on a visit with strong
    systematics (TOI-270 c: 7249 vs 5230 ppm).
  - the red-noise beta is annotated, so a figure carries its own warning
    when the quoted errors need inflating
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
    BINNED_COLOR,
    DATA_COLOR,
    MODEL_COLOR,
    _PLOT_STYLE,
    _bin_series,
    figure_title,
)


def replot(fit_dir: Path) -> str | None:
    csv_path = fit_dir / "white_lightcurve_series.csv"
    json_path = fit_dir / "white_fit_summary.json"
    if not csv_path.is_file() or not json_path.is_file():
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    with json_path.open() as fh:
        s = json.load(fh)

    instrument = s.get("instrument", "")
    t_hr = (d["time_bjd"] - s["t0_obs"]) * 24
    flux, err, model = d["flux"], d["flux_err"], d["model"]
    residual = flux - model
    rms = float(np.nanstd(residual) * 1e6)
    depth_ppm = s["depth_ppm"]["median"]
    beta = (s.get("rednoise") or {}).get("beta_median")

    with plt.rc_context(_PLOT_STYLE):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.0, 6.4), sharex=True,
                                     height_ratios=[3, 1])
        a1.errorbar(t_hr, flux, yerr=err, fmt="none", ecolor=DATA_COLOR,
                    elinewidth=0.5, capsize=0, alpha=0.18, zorder=1)
        a1.plot(t_hr, flux, "o", ms=2.4, mew=0, color=DATA_COLOR, alpha=0.38,
                zorder=2, label="integrations")
        bx, by, be = _bin_series(t_hr, flux)
        a1.errorbar(bx, by, yerr=be, fmt="o", ms=5.0, mew=0,
                    color=BINNED_COLOR, ecolor=BINNED_COLOR, elinewidth=1.2,
                    capsize=0, alpha=0.95, zorder=3, label="binned")
        a1.plot(t_hr, model, color=MODEL_COLOR, lw=2.6, zorder=4,
                solid_capstyle="round", label="juliet transit model",
                path_effects=[pe.Stroke(linewidth=4.6, foreground="white"),
                              pe.Normal()])
        a1.set_ylabel("Relative flux")
        a1.set_title(figure_title(s.get("planet_name", ""), instrument,
                                  program=s.get("program", ""),
                                  visit=s.get("visit", ""),
                                  suffix="white light"), pad=12)
        a1.legend(loc="lower left", ncol=3, columnspacing=1.4,
                  handletextpad=0.5)
        note = f"depth {depth_ppm:.0f} ppm\nresidual rms {rms:.0f} ppm"
        if beta is not None and np.isfinite(beta):
            note += f"\nred-noise $\\beta$ = {beta:.2f}"
        a1.annotate(note, xy=(0.985, 0.06), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=10.5, color="#4E5866")

        a2.errorbar(t_hr, residual * 1e6, yerr=err * 1e6, fmt="none",
                    ecolor=DATA_COLOR, elinewidth=0.5, capsize=0,
                    alpha=0.18, zorder=1)
        a2.plot(t_hr, residual * 1e6, "o", ms=2.4, mew=0, color=DATA_COLOR,
                alpha=0.38, zorder=2)
        rbx, rby, rbe = _bin_series(t_hr, residual * 1e6)
        a2.errorbar(rbx, rby, yerr=rbe, fmt="o", ms=4.5, mew=0,
                    color=BINNED_COLOR, ecolor=BINNED_COLOR, elinewidth=1.1,
                    capsize=0, alpha=0.95, zorder=3)
        a2.axhline(0, color=MODEL_COLOR, lw=1.8, zorder=2)
        a2.set_xlabel("Time from mid-transit  [h]")
        a2.set_ylabel("Residual  [ppm]")
        a2.set_ylim(-4 * rms, 4 * rms)
        fig.tight_layout()
        fig.savefig(fit_dir / "white_lightcurve_fit.pdf")
        fig.savefig(fit_dir / "white_lightcurve_fit.svg")
        plt.close(fig)

    return (f"{fit_dir}  depth={depth_ppm:.0f} ppm  rms={rms:.0f} ppm"
            + (f"  beta={beta:.2f}" if beta is not None else ""))


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
                           "white_lightcurve_series.csv")
    dirs = sorted({Path(p).parent for p in glob.glob(pattern)})
    if not dirs:
        print(f"No white_lightcurve_series.csv under {pattern}")
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
