#!/usr/bin/env python
"""Overlay every Patchwork target's combined transmission spectrum.

The survey-level figure: one colour per planet, both detectors, with the
NIRSpec/G395H inter-detector gap shaded so the eye never reads across a
wavelength range that was not observed.

    python scripts/patchwork/plot_survey_spectra.py ~/scratch/patchwork
    python scripts/patchwork/plot_survey_spectra.py wave1_csvs --relative

Absolute depths (the default) show the sample as observed — useful for
seeing which planets are accessible at all. ``--relative`` subtracts each
planet's inverse-variance-weighted mean, which is what you want to
compare spectral SHAPE across planets of very different size; otherwise a
deep target simply sits far above a shallow one and no feature is
comparable.

Reads ``combined/combined_{nrs1,nrs2}_transmission_spectrum.csv`` under
each target directory. Writes PDF and SVG only, per the Patchwork figure
rule.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aster_toolkit.data_reduction.juliet import (  # noqa: E402
    GAP_COLOR,
    _PLOT_STYLE,
    _savefig,
    read_spectrum_csv,
)

# Okabe-Ito, then extensions: chosen for colour-vision deficiency safety,
# since a survey figure is the one most likely to be projected.
PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
    "#8A3FFC", "#B22222", "#117733", "#882255", "#44AA99", "#DDCC77",
    "#332288", "#AA4499", "#88CCEE", "#999933", "#661100", "#6699CC",
]


def load_target(target_dir: Path) -> dict[str, np.ndarray] | None:
    """Concatenate a target's NRS1 + NRS2 combined spectra."""
    parts = []
    for det in ("nrs1", "nrs2"):
        p = target_dir / "combined" / f"combined_{det}_transmission_spectrum.csv"
        if p.is_file():
            parts.append(read_spectrum_csv(p))
    if not parts:
        return None
    out = {k: np.concatenate([s[k] for s in parts]) for k in
           ("wave", "wave_err", "depth_ppm", "depth_err_ppm")}
    order = np.argsort(out["wave"])
    return {k: v[order] for k, v in out.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Overlay all Patchwork combined transmission spectra.")
    ap.add_argument("root", help="Output root containing per-target dirs.")
    ap.add_argument("-o", "--output", default=None,
                    help="Output stem (default: <root>/patchwork_survey_spectra).")
    ap.add_argument("--relative", action="store_true",
                    help="Subtract each planet's weighted mean depth, to "
                         "compare spectral shape rather than absolute depth.")
    ap.add_argument("--exclude", default="",
                    help="Comma-separated target slugs to omit (e.g. ones "
                         "flagged depth_check.suspect).")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(os.path.expanduser(args.root)).resolve()
    skip = {s.strip() for s in args.exclude.split(",") if s.strip()}

    spectra: dict[str, dict[str, np.ndarray]] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name in skip or d.name.endswith("_asis"):
            continue
        s = load_target(d)
        if s is not None:
            spectra[d.name] = s

    if not spectra:
        print(f"No combined_*_transmission_spectrum.csv found under {root}")
        return 1

    with plt.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(11.5, 6.2))

        # Shade the inter-detector gap using the actual data extent.
        all_w = np.concatenate([s["wave"] for s in spectra.values()])
        lo_side = all_w[all_w < 3.78]
        hi_side = all_w[all_w > 3.78]
        if lo_side.size and hi_side.size:
            ax.axvspan(lo_side.max(), hi_side.min(), color=GAP_COLOR,
                       alpha=0.30, zorder=0, lw=0)
            ax.text(0.5 * (lo_side.max() + hi_side.min()), 0.02,
                    "detector\ngap", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9.5, color="#4E5866",
                    linespacing=1.15)

        for i, (name, s) in enumerate(spectra.items()):
            d = s["depth_ppm"]
            e = s["depth_err_ppm"]
            good = np.isfinite(d) & np.isfinite(e) & (e > 0)
            label = name.replace("_", " ")
            if args.relative:
                wmean = np.sum(d[good] / e[good] ** 2) / np.sum(1 / e[good] ** 2)
                d = d - wmean
                label += f"  ({wmean:.0f} ppm)"
            ax.errorbar(s["wave"][good], d[good], yerr=e[good],
                        xerr=s["wave_err"][good], fmt="o", ms=3.8, mew=0,
                        color=PALETTE[i % len(PALETTE)],
                        ecolor=PALETTE[i % len(PALETTE)], elinewidth=1.0,
                        capsize=0, alpha=0.85, zorder=3 + i, label=label)

        if args.relative:
            ax.axhline(0, color="#8A94A6", lw=1.0, ls="--", zorder=1)
            ax.set_ylabel(r"$\Delta$ transit depth from weighted mean  [ppm]")
        else:
            ax.set_ylabel(r"Transit depth  $(R_{\rm p}/R_{\star})^{2}$  [ppm]")
        ax.set_xlabel("Wavelength  [μm]")
        ax.set_title(
            f"Patchwork — JWST NIRSpec/G395H, {len(spectra)} planet"
            f"{'s' if len(spectra) != 1 else ''}, R = 100", pad=12)
        ncol = 2 if len(spectra) > 6 else 1
        ax.legend(loc="best", ncol=ncol, columnspacing=1.2,
                  handletextpad=0.5, fontsize=9.5)
        fig.tight_layout()

        stem = args.output or str(root / "patchwork_survey_spectra")
        stem_path = Path(stem)
        _savefig(fig, stem_path.parent, stem_path.name)
        plt.close(fig)

    print(f"Plotted {len(spectra)} target(s): {', '.join(spectra)}")
    print(f"Wrote {stem}.pdf and {stem}.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
