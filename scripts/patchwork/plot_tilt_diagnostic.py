#!/usr/bin/env python
"""Tilt-event diagnostic figure: which data a tilt touched, and what a
Heaviside correction does to it.

The standard white-light figure plots one homogeneous series, which is
exactly wrong for a visit containing a tilt event: the flux sits at two
(or four) different levels and the eye is given no way to see it. This
figure makes the segmentation explicit --

  * each inter-step SEGMENT gets its own colour, so a level change is
    visible as a colour change rather than something you have to infer;
  * the integrations dropped at each transition are drawn GREY, because
    the integration straddling a tilt is a blend of two PSF states that
    no Heaviside describes (TILT_TRANSITION_MASK);
  * the lower panel shows the residuals BEFORE and AFTER removing a
    fitted step per event, so the correction can be judged by eye rather
    than through beta alone.

Step times come from a Stage 5.5 ``anomaly_report.json`` (events with
``kind='step'``), from an explicit ``--step-index``, or from both.

    python scripts/patchwork/plot_tilt_diagnostic.py \
        ~/Patchwork_Data/TOI_270_c/fits/o016 --planet "TOI-270 c" \
        --program "GO 4098" --visit o016 -o ~/Desktop
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aster_toolkit.data_reduction.juliet import (  # noqa: E402
    DATA_COLOR, MEAN_COLOR, MODEL_COLOR, _PLOT_STYLE, _savefig, figure_title,
)
from aster_toolkit.data_reduction.lightcurves import (  # noqa: E402
    TILT_TRANSITION_MASK,
)

# One colour per segment between steps. Deliberately not a gradient: the
# point is that these are DIFFERENT instrument states, not a continuum.
SEGMENT_COLORS = [DATA_COLOR, "#1B7F79", "#8C4A9E", "#B5651D", "#2E7D32"]
MASK_COLOR = "#B9C2CC"


def step_indices(fit_dir: Path, detector: str,
                 extra: list[int] | None = None) -> list[int]:
    """Break indices for one detector: Stage 5.5 steps plus any given."""
    idx = list(extra or [])
    report = fit_dir / "anomalies" / "anomaly_report.json"
    if report.exists():
        payload = json.load(report.open())
        for event in payload.get("events", []):
            if event.get("kind") != "step" or not event.get("confirmed"):
                continue
            entry = (event.get("detectors") or {}).get(detector.upper())
            if entry is None:
                continue
            # Centre of the flagged span: a step's detrended feature is
            # antisymmetric about the transition, so the lobes straddle it.
            idx.append(int(round(0.5 * (int(entry["index_start"])
                                        + int(entry["index_end"])))))
    return sorted(set(idx))


def fit_steps(residual: np.ndarray, breaks: list[int],
              keep: np.ndarray) -> np.ndarray:
    """Least-squares Heaviside amplitudes -> the step model.

    Amplitudes are free and unconstrained in sign: a tilt's flux change
    is pixel-dependent and "can be positive or negative" (2405.06737),
    so a sign constraint would be wrong physics, not a safety net.
    """
    if not breaks:
        return np.zeros_like(residual)
    cols = [np.ones(residual.size)]
    for b in breaks:
        c = np.zeros(residual.size)
        c[b:] = 1.0
        cols.append(c)
    design = np.column_stack(cols)
    good = keep & np.isfinite(residual)
    coef, *_ = np.linalg.lstsq(design[good], residual[good], rcond=None)
    return design @ coef


def plot_visit(fit_dir: Path, out_dir: Path, *, planet: str = "",
               program: str = "", visit: str = "",
               extra_steps: list[int] | None = None,
               stem: str = "tilt_diagnostic") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dets = [d for d in ("nrs1", "nrs2")
            if (fit_dir / d / "white_lightcurve_residuals.npz").exists()]
    if not dets:
        raise FileNotFoundError(f"No white_lightcurve_residuals.npz under {fit_dir}")

    with plt.rc_context(_PLOT_STYLE):
        fig, axes = plt.subplots(
            2 * len(dets), 1, figsize=(10, 3.4 * 2 * len(dets)),
            sharex=True, squeeze=False,
            gridspec_kw={"height_ratios": [2.0, 1.25] * len(dets)})
        rows = axes[:, 0]

        for k, det in enumerate(dets):
            d = np.load(fit_dir / det / "white_lightcurve_residuals.npz")
            t = np.asarray(d["time"], float)
            flux = np.asarray(d["flux"], float)
            model = np.asarray(d["model"], float)
            resid = np.asarray(d["residual"], float)
            t0 = float(np.asarray(d["t0_obs"]))
            hrs = (t - t0) * 24

            breaks = step_indices(fit_dir, det, extra_steps)
            masked = np.zeros(t.size, dtype=bool)
            for b in breaks:
                masked[max(0, b - TILT_TRANSITION_MASK):
                       min(t.size, b + TILT_TRANSITION_MASK + 1)] = True
            step_model = fit_steps(resid, breaks, ~masked)
            corrected = resid - step_model

            ax_lc, ax_rs = rows[2 * k], rows[2 * k + 1]
            bounds = [0] + [b for b in breaks] + [t.size]
            for j in range(len(bounds) - 1):
                s = slice(bounds[j], bounds[j + 1])
                sel = np.zeros(t.size, bool); sel[s] = True
                sel &= ~masked
                colour = SEGMENT_COLORS[j % len(SEGMENT_COLORS)]
                ax_lc.plot(hrs[sel], flux[sel] * 1e2, ".", ms=1.8, alpha=0.5,
                           color=colour, zorder=2,
                           label=f"segment {j + 1}" if len(bounds) > 2 else "data")
                ax_rs.plot(hrs[sel], resid[sel] * 1e6, ".", ms=1.6, alpha=0.32,
                           color=colour, zorder=2)
            if masked.any():
                ax_lc.plot(hrs[masked], flux[masked] * 1e2, ".", ms=2.6,
                           color=MASK_COLOR, zorder=3,
                           label=f"dropped at transition (±{TILT_TRANSITION_MASK})")
                ax_rs.plot(hrs[masked], resid[masked] * 1e6, ".", ms=2.4,
                           color=MASK_COLOR, zorder=3)

            ax_lc.plot(hrs, model * 1e2, "-", lw=1.7, color=MODEL_COLOR,
                       zorder=4, label="fitted transit + systematics")
            for b in breaks:
                for ax in (ax_lc, ax_rs):
                    ax.axvline(hrs[b], ls="--", lw=1.0, color=MEAN_COLOR, zorder=1)
            ax_lc.set_ylabel(f"{det.upper()} normalized flux  [%]")

            ax_rs.plot(hrs, step_model * 1e6, "-", lw=1.6, color=MODEL_COLOR,
                       zorder=4, label="fitted Heaviside steps")
            ax_rs.plot(hrs, corrected * 1e6, ".", ms=1.6, alpha=0.45,
                       color="#D9534F", zorder=3,
                       label="residual AFTER step correction")
            ax_rs.axhline(0, lw=0.8, color=MEAN_COLOR)
            rms_b = np.nanstd(resid[~masked]) * 1e6
            rms_a = np.nanstd(corrected[~masked]) * 1e6
            ax_rs.set_ylabel(f"{det.upper()} residual  [ppm]")
            ax_rs.set_title(f"rms {rms_b:.0f} ppm before  ->  {rms_a:.0f} ppm after"
                            f"   ({len(breaks)} step(s))", fontsize=10.5, loc="left")
            for ax in (ax_lc, ax_rs):
                ax.legend(loc="lower right", fontsize=8.5, ncol=2)

        rows[-1].set_xlabel("Time from mid-transit  [h]")
        rows[0].set_title(figure_title(planet, program=program, visit=visit,
                                       suffix="tilt-event diagnostic"))
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        return _savefig(fig, out_dir, stem)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("fit_dir", help="fits/<visit> directory (holds nrs1/, nrs2/).")
    ap.add_argument("-o", "--out-dir", default=".")
    ap.add_argument("--planet", default="")
    ap.add_argument("--program", default="")
    ap.add_argument("--visit", default="")
    ap.add_argument("--step-index", type=int, action="append",
                    help="Extra break index, repeatable.")
    ap.add_argument("--stem", default="tilt_diagnostic")
    args = ap.parse_args()
    path = plot_visit(Path(os.path.expanduser(args.fit_dir)),
                      Path(os.path.expanduser(args.out_dir)),
                      planet=args.planet, program=args.program,
                      visit=args.visit, extra_steps=args.step_index,
                      stem=args.stem)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
