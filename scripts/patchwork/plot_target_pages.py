#!/usr/bin/env python
"""One review page per target: lightcurves, residuals, spectrum.

Everything that decides whether a target is trustworthy on a single
page, rather than spread over five pages of a merged figure book.

  * white lightcurves per visit x detector, offset, with binned points
    over the integrations so the transit shape is readable at a glance
    and the fitted model on top;
  * residuals, offset the same way and labelled with rms and beta --
    this is where a spot crossing, a tilt step or a mis-fit contact
    actually shows, so it gets its own vertical space per series rather
    than being overplotted into an indistinguishable band;
  * any fitted step marked on both panels, since a corrected step is
    invisible in a residual that no longer contains it and the reader
    otherwise cannot tell a clean visit from a repaired one;
  * the combined spectrum at both resolutions -- survey R=100 faint
    behind canonical rebinned R=50 bold. Rebinning averages white noise down by
    sqrt(2) but leaves coherent structure untouched, so a feature
    holding its amplitude between the two is saying something the error
    bars are not.

    python scripts/patchwork/plot_target_pages.py ~/Patchwork_Data -o out/
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
    _PLOT_STYLE, BINNED_COLOR, DATA_COLOR, GAP_COLOR, MEAN_COLOR,
    MODEL_COLOR, NRS2_COLOR, figure_title, read_spectrum_csv,
)

DET_COLOR = {"nrs1": DATA_COLOR, "nrs2": NRS2_COLOR}
STEP_COLOR = "#D9534F"


def _bin(x, y, n=90):
    """Bin to ~n points for legibility over thousands of integrations."""
    m = x.size // max(n, 1)
    if m < 2:
        return x, y
    k = (x.size // m) * m
    return (x[:k].reshape(-1, m).mean(1), y[:k].reshape(-1, m).mean(1))


def target_data(tdir: Path) -> dict:
    out: dict = {"visits": [], "spec": {}, "steps": load_steps(tdir)}
    for npz in sorted(glob.glob(str(tdir / "fits" / "*" / "nrs*" /
                                    "white_lightcurve_residuals.npz"))):
        parts = Path(npz).parts
        d = np.load(npz)
        js = Path(npz).with_name("white_fit_summary.json")
        s = json.load(js.open()) if js.exists() else {}
        out["visits"].append({
            "visit": parts[-3], "det": parts[-2],
            "t": np.asarray(d["time"], float),
            "flux": np.asarray(d["flux"], float),
            "model": np.asarray(d["model"], float),
            "resid": np.asarray(d["residual"], float),
            "index": np.asarray(d["index"], int) if "index" in d
                     else np.arange(len(d["time"])),
            # Centre on the FITTED mid-transit, falling back to the
            # propagated ephemeris. Visits only overlay correctly on the
            # fitted value: the propagated t0 carries whatever drift the
            # archive ephemeris has accumulated, which for K2-18 b is
            # ~45 min and smeared four visits into a transit half an hour
            # wider than any one of them.
            "t0": float((s.get("t0_p1") or {}).get("median")
                        or np.asarray(d["t0_obs"])), "summary": s,
        })
    for tag, suffixes in (("R100", ("",)),
                          ("R50", ("_R50",))):
        for suffix in suffixes:
            for f in sorted(glob.glob(str(
                    tdir / "combined" /
                    f"combined_*_transmission_spectrum{suffix}.csv"))):
                det = "NRS2" if "nrs2" in os.path.basename(f).lower() else "NRS1"
                out["spec"].setdefault(tag, {}).setdefault(
                    det, read_spectrum_csv(f)
                )
    return out


def load_steps(tdir: Path) -> dict[tuple[str, str], list[dict]]:
    """Fitted step corrections, keyed by (visit, detector).

    These live in the TARGET summary rather than the per-detector fit
    summary, which is why an earlier version of this figure silently
    drew none: it looked in white_fit_summary.json, where the key does
    not exist.
    """
    js = tdir / "patchwork_summary.json"
    if not js.exists():
        return {}
    try:
        s = json.load(js.open())
    except Exception:
        return {}
    out: dict[tuple[str, str], list[dict]] = {}
    for vname, v in (s.get("visits") or {}).items():
        for det, f in (v.get("fits") or {}).items():
            out[(vname, det.lower())] = f.get("scan_steps_corrected") or []
    return out


def _grid_bin(hrs_list, y_list, n=110):
    """Bin several series onto one common time grid.

    Visits of the same planet are the same event observed again, so they
    belong on a common axis rather than stacked apart -- overlaying them
    is what makes the repeatability, and the combined precision, visible.
    Different visits do not share a time sampling, so the combination has
    to happen on a grid rather than index-by-index.
    """
    lo = min(float(np.min(h)) for h in hrs_list)
    hi = max(float(np.max(h)) for h in hrs_list)
    edges = np.linspace(lo, hi, n + 1)
    x = 0.5 * (edges[:-1] + edges[1:])
    allh = np.concatenate(hrs_list)
    ally = np.concatenate(y_list)
    idx = np.digitize(allh, edges) - 1
    out = np.full(n, np.nan)
    for k in range(n):
        m = idx == k
        if m.any():
            out[k] = np.nanmean(ally[m])
    good = np.isfinite(out)
    return x[good], out[good]


def draw(name: str, data: dict, pdf_dir: Path, svg_dir: Path, pdf=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vis = sorted(data["visits"], key=lambda x: (x["det"], x["visit"]))
    if not vis:
        return None
    by_det: dict[str, list] = {}
    for v in vis:
        by_det.setdefault(v["det"], []).append(v)
    dets = [d for d in ("nrs1", "nrs2") if d in by_det]

    with plt.rc_context(_PLOT_STYLE):
        fig, axes = plt.subplots(
            3, 1, figsize=(10.5, 10.5),
            gridspec_kw={"height_ratios": [2.6, 1.6, 2.5]})
        ax_lc, ax_rs, ax_sp = axes

        lc_step = 1.45 * max(np.ptp(v["flux"]) for v in vis)
        rs_step = 5.5 * max(np.nanstd(v["resid"]) for v in vis) * 1e6
        for k, det in enumerate(dets):
            group = by_det[det]
            c = DET_COLOR.get(det, DATA_COLOR)
            hrs = [(v["t"] - v["t0"]) * 24 for v in group]
            off, roff = -k * lc_step, -k * rs_step

            for h, v in zip(hrs, group):
                ax_lc.plot(h, v["flux"] + off, ".", ms=1.1, alpha=0.16, color=c)
                ax_rs.plot(h, v["resid"] * 1e6 + roff, ".", ms=1.0,
                           alpha=0.16, color=c)
            bx, by = _grid_bin(hrs, [v["flux"] for v in group])
            ax_lc.plot(bx, by + off, "o", ms=3.0, color=BINNED_COLOR, zorder=5)
            rx, ry = _grid_bin(hrs, [v["resid"] * 1e6 for v in group])
            ax_rs.plot(rx, ry + roff, "o", ms=2.8, color=BINNED_COLOR, zorder=5)
            for h, v in zip(hrs, group):
                ax_lc.plot(h, v["model"] + off, "-", lw=1.2,
                           color=MODEL_COLOR, alpha=0.85, zorder=4)
            ax_rs.axhline(roff, lw=0.7, color=MEAN_COLOR, zorder=1)

            deps = [(v["summary"].get("depth_ppm") or {}).get("median")
                    for v in group]
            deps = [d for d in deps if d]
            betas = [(v["summary"].get("rednoise") or {}).get("beta_median")
                     for v in group]
            betas = [b for b in betas if b]
            nv = len(group)
            lab = f"{det.upper()}   {nv} visit{'s' if nv > 1 else ''}"
            if deps:
                lab += f"   {np.mean(deps):.0f} ppm"
            ax_lc.text(min(h.min() for h in hrs), 1.0 + off + 0.30 * lc_step,
                       lab, fontsize=9.5, color=c, va="bottom",
                       fontweight="bold")
            rl = f"{det.upper()}"
            if betas:
                rl += f"   $\\beta$ range {min(betas):.2f}-{max(betas):.2f}"
            ax_rs.text(min(h.min() for h in hrs), roff + 0.34 * rs_step, rl,
                       fontsize=9, color=c, va="bottom")

            for v, h in zip(group, hrs):
                for e in data["steps"].get((v["visit"], v["det"]), []):
                    i = int(e.get("index", -1))
                    if v["index"].size and v["index"][0] <= i <= v["index"][-1]:
                        event_hour = np.interp(i, v["index"], h)
                        for ax in (ax_lc, ax_rs):
                            ax.axvline(event_hour, ls=":", lw=1.1, color=STEP_COLOR,
                                       alpha=0.7, zorder=2)

        ax_lc.set_ylabel("Relative flux")
        ax_lc.set_yticks([])
        ax_rs.set_ylabel("Residual  [ppm]")
        ax_rs.set_yticks([])
        ax_rs.set_xlabel("Time from mid-transit  [h]")
        if any(data["steps"].values()):
            ax_lc.plot([], [], ls=":", color=STEP_COLOR, label="fitted step")
        ax_lc.plot([], [], "o", ms=3, color=BINNED_COLOR,
                   label="visits combined")
        ax_lc.plot([], [], "-", color=MODEL_COLOR, label="fitted model")
        ax_lc.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02),
                     fontsize=8.5, ncol=3)

        allw, alld, alle = [], [], []
        for tag, st in (("R100", dict(ms=3, alpha=0.30, lw=0.7, zorder=2)),
                        ("R50", dict(ms=5.5, alpha=1.0, lw=1.3, zorder=3))):
            for det, sp in (data["spec"].get(tag) or {}).items():
                c = DET_COLOR["nrs1" if det == "NRS1" else "nrs2"]
                ax_sp.errorbar(sp["wave"], sp["depth_ppm"],
                               yerr=sp["depth_err_ppm"], xerr=sp["wave_err"],
                               fmt=".", color=c, **st,
                               label=(f"{det} R=50" if tag == "R50"
                                      else (f"{det} R=100" if det == "NRS1"
                                            else None)))
                if tag == "R50":
                    allw.append(sp["wave"]); alld.append(sp["depth_ppm"])
                    alle.append(sp["depth_err_ppm"])
        if alld:
            d, e = np.concatenate(alld), np.concatenate(alle)
            mean = float(np.sum(d / e**2) / np.sum(1 / e**2))
            ax_sp.axhline(mean, ls="--", lw=1.0, color=MEAN_COLOR,
                          label=f"weighted mean {mean:.0f} ppm")
        ax_sp.axvspan(3.72, 3.82, color=GAP_COLOR, alpha=0.4, lw=0)
        ax_sp.set_xlabel(r"Wavelength  [$\mu$m]")
        ax_sp.set_ylabel(r"Transit depth $(R_p/R_\star)^2$  [ppm]")
        ax_sp.legend(loc="best", fontsize=8.5, ncol=3)

        s0 = vis[0]["summary"]
        nv = len({v["visit"] for v in vis})
        fig.suptitle(figure_title(
            s0.get("planet_name") or name, program=s0.get("program") or "",
            suffix=f"{nv} visit{'s' if nv > 1 else ''} combined"), y=0.997)
        fig.tight_layout(h_pad=1.3)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        svg_dir.mkdir(parents=True, exist_ok=True)
        if pdf is not None:
            pdf.savefig(fig)
        p = pdf_dir / f"{name}_review.pdf"
        fig.savefig(p)
        fig.savefig(svg_dir / f"{name}_review.svg")
        plt.close(fig)
    return str(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root")
    ap.add_argument("-o", "--out-dir", default=".")
    args = ap.parse_args()
    root = Path(os.path.expanduser(args.root))
    out = Path(os.path.expanduser(args.out_dir))

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    out.mkdir(parents=True, exist_ok=True)
    book = out / "all_targets.pdf"
    n = 0
    with PdfPages(book) as pdf:
        for tdir in sorted(p for p in root.iterdir() if p.is_dir()):
            if draw(tdir.name, target_data(tdir), out / "pdf", out / "svg", pdf):
                n += 1
                print(f"  {tdir.name}")
    print(f"{n} target page(s) -> {book}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
