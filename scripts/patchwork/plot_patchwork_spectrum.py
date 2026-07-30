#!/usr/bin/env python
"""Publication-quality Patchwork transmission-spectrum figure.

Reads the combined per-detector CSVs written by the Patchwork pipeline
(wave_um, wave_err_um, depth_ppm, depth_err_ppm, resid_rms_ppm) and
produces a figure suitable for circulation to collaborators.

Deliberately separate from the pipeline so figures can be restyled and
regenerated in seconds without refitting anything.

Usage
-----
    python plot_patchwork_spectrum.py \
        --nrs1 gj9827d_nrs1_transmission_spectrum.csv \
        --nrs2 gj9827d_nrs2_transmission_spectrum.csv \
        --planet "GJ 9827 d" --program "GO 4098" --visits 2 \
        --out gj9827d_transmission_spectrum
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

# Patchwork house style: no grid, no top/right spines, muted data, one
# strong accent. Grid lines compete with error bars at these amplitudes.
STYLE = {
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 10.5,
    "figure.dpi": 150,
}

NRS1_COLOR = "#1A2F6B"     # deep navy
NRS2_COLOR = "#1B7F79"     # teal
GAP_COLOR = "#B9C2CC"      # cool grey, for the inter-detector gap
MEAN_COLOR = "#8A94A6"


def read_spectrum(path: str | Path) -> dict[str, np.ndarray]:
    w, we, d, de, rms = [], [], [], [], []
    with Path(path).open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("#") or row[0] == "wave_um":
                continue
            w.append(float(row[0])); we.append(float(row[1]))
            d.append(float(row[2])); de.append(float(row[3]))
            rms.append(float(row[4]) if len(row) > 4 and row[4] else np.nan)
    keep = np.isfinite(d) & np.isfinite(de)
    return {k: np.asarray(v)[keep] for k, v in
            zip(("wave", "wave_err", "depth", "depth_err", "rms"),
                (w, we, d, de, rms))}


def plot_spectrum(specs: dict[str, dict], out_stem: str, *, planet: str,
                  program: str, n_visits: int, resolution: int = 100,
                  show_mean: bool = True) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"NRS1": NRS1_COLOR, "NRS2": NRS2_COLOR}

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10.5, 5.2))

        # --- inter-detector gap -------------------------------------
        # NIRSpec G395H has a physical gap between NRS1 and NRS2; shading
        # it stops the eye reading across a wavelength range that was
        # never observed.
        if {"NRS1", "NRS2"} <= set(specs):
            gap_lo = specs["NRS1"]["wave"].max() + specs["NRS1"]["wave_err"][-1]
            gap_hi = specs["NRS2"]["wave"].min() - specs["NRS2"]["wave_err"][0]
            ax.axvspan(gap_lo, gap_hi, color=GAP_COLOR, alpha=0.30,
                       zorder=0, lw=0)
            # Label at the base of the band: the top of the axes belongs
            # to the legend.
            ax.text(0.5 * (gap_lo + gap_hi), 0.035, "detector\ngap",
                    transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=9.5, color="#4E5866",
                    linespacing=1.15)

        all_d, all_e = [], []
        for det in ("NRS1", "NRS2"):
            if det not in specs:
                continue
            s = specs[det]
            all_d.append(s["depth"]); all_e.append(s["depth_err"])
            ax.errorbar(s["wave"], s["depth"],
                        xerr=s["wave_err"], yerr=s["depth_err"],
                        fmt="o", ms=4.5, mew=0, color=colors[det],
                        ecolor=colors[det], elinewidth=1.1, capsize=0,
                        alpha=0.85, zorder=3,
                        label=f"{det}  ({s['wave'].size} channels)")

        # --- weighted mean depth ------------------------------------
        if show_mean and all_d:
            d = np.concatenate(all_d); e = np.concatenate(all_e)
            wmean = np.sum(d / e**2) / np.sum(1 / e**2)
            ax.axhline(wmean, color=MEAN_COLOR, lw=1.2, ls="--",
                       zorder=1, alpha=0.9,
                       label=f"weighted mean  {wmean:.0f} ppm")

        ax.set_xlabel("Wavelength  [μm]")
        ax.set_ylabel(r"Transit depth  $(R_{\rm p}/R_{\star})^{2}$  [ppm]")
        ax.set_title(
            f"{planet}   ·   {program}   ·   JWST NIRSpec/G395H   ·   "
            f"{n_visits} visit{'s' if n_visits != 1 else ''} combined, "
            f"R = {resolution}",
            pad=12,
        )
        ax.legend(loc="upper left", ncol=3, columnspacing=1.4,
                  handletextpad=0.5)

        # Symmetric, generous limits so error bars are not clipped.
        if all_d:
            d = np.concatenate(all_d); e = np.concatenate(all_e)
            lo, hi = (d - e).min(), (d + e).max()
            pad = 0.18 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)

        fig.tight_layout()
        for ext in ("pdf", "svg"):
            fig.savefig(f"{out_stem}.{ext}", bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {out_stem}.pdf and {out_stem}.svg")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nrs1"); p.add_argument("--nrs2")
    p.add_argument("--planet", required=True)
    p.add_argument("--program", default="")
    p.add_argument("--visits", type=int, default=1)
    p.add_argument("--resolution", type=int, default=100)
    p.add_argument("--out", required=True, help="Output stem (no extension).")
    a = p.parse_args(argv)

    specs = {}
    if a.nrs1:
        specs["NRS1"] = read_spectrum(a.nrs1)
    if a.nrs2:
        specs["NRS2"] = read_spectrum(a.nrs2)
    if not specs:
        p.error("Give at least one of --nrs1 / --nrs2.")

    plot_spectrum(specs, a.out, planet=a.planet, program=a.program,
                  n_visits=a.visits, resolution=a.resolution)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
