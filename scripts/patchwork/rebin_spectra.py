#!/usr/bin/env python
"""Rebin combined transmission spectra to a coarser resolution.

No refit. Inverse-variance averaging adjacent constant-R channels is a
useful coarser representation of the fitted spectrum, but it is not
identical to refitting the light curves: neighbouring channels share
instrument systematics, orbital parameters, and common-mode corrections.
Rebinning R=100 -> R=50 costs seconds and approximately halves the channel
count.

Why coarser at all: the features these spectra are searched for -- CO2
at 4.3 um, CH4 at 3.3 um -- are far broader than R=50, while the
per-channel precision (15-90 ppm) is comparable to the feature
amplitudes (20-50 ppm). Resolution beyond what the bands require buys
nothing and costs signal-to-noise, exactly the argument that put the
survey at R=100 rather than the native R~2700.

One caveat this cannot remove: neighbouring channels share instrument
systematics and a common white-light orbit, so their errors are not
strictly independent and the rebinned error is mildly optimistic.
Treat the sqrt(2) as an upper bound on the real gain.

    python scripts/patchwork/rebin_spectra.py ~/Patchwork_Data -R 50
    python scripts/patchwork/rebin_spectra.py ~/Patchwork_Data -R 50 --report
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aster_toolkit.data_reduction.juliet import read_spectrum_csv  # noqa: E402


def rebin(wave, wave_err, depth, err, rms, *, resolution: float,
          min_merged: int = 2):
    """Inverse-variance rebin a spectrum onto constant-R edges."""
    wave = np.asarray(wave, dtype=float)
    wave_err = np.asarray(wave_err, dtype=float)
    depth = np.asarray(depth, dtype=float)
    err = np.asarray(err, dtype=float)
    rms = np.asarray(rms, dtype=float)
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError(f"resolution must be positive; got {resolution!r}")
    if min_merged < 1:
        raise ValueError("min_merged must be at least 1")
    good = (np.isfinite(wave) & np.isfinite(wave_err) & np.isfinite(depth)
            & np.isfinite(err) & (wave > 0) & (wave_err > 0) & (err > 0))
    if good.sum() < min_merged:
        return []
    wave, wave_err, depth, err, rms = (
        array[good] for array in (wave, wave_err, depth, err, rms)
    )
    input_resolution = float(np.nanmedian(wave / (2.0 * wave_err)))
    if resolution >= 0.98 * input_resolution:
        raise ValueError(
            f"Requested R={resolution:g} is not coarser than the input "
            f"spectrum (R~{input_resolution:.1f}); a finer/equal grid needs "
            "a light-curve refit, not rebinning."
        )

    lo = float(np.min(wave - wave_err))
    hi = float(np.max(wave + wave_err))
    edges = [lo]
    while edges[-1] < hi:
        edges.append(edges[-1] * (1 + 1.0 / resolution))
    edges[-1] = hi
    edges = np.asarray(edges)

    out = []
    for i in range(len(edges) - 1):
        m = (wave >= edges[i]) & (wave < edges[i + 1])
        if int(m.sum()) < min_merged:
            continue
        w = 1.0 / err[m] ** 2
        d = float(np.sum(depth[m] * w) / np.sum(w))
        e = float(1.0 / np.sqrt(np.sum(w)))
        # Flux-weighted centre of the channels that actually contributed,
        # not the bin centre: a bin clipped at the detector edge would
        # otherwise report a wavelength no data supports.
        support_lo = float(np.min(wave[m] - wave_err[m]))
        support_hi = float(np.max(wave[m] + wave_err[m]))
        c = 0.5 * (support_lo + support_hi)
        half = 0.5 * (support_hi - support_lo)
        r = float(np.sqrt(np.nanmean(rms[m] ** 2))) if np.isfinite(rms[m]).any() else float("nan")
        out.append({"wave": c, "wave_err": half, "depth": d / 1e6,
                    "depth_err": e / 1e6, "rms_ppm": r, "n_merged": int(m.sum())})
    return out


def write_csv(path: Path, rows, header: str) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"# {header}"])
        w.writerow(["wave_um", "wave_err_um", "depth_ppm", "depth_err_ppm",
                    "resid_rms_ppm", "n_merged"])
        for r in rows:
            w.writerow([f"{r['wave']:.6f}", f"{r['wave_err']:.6f}",
                        f"{r['depth'] * 1e6:.2f}", f"{r['depth_err'] * 1e6:.2f}",
                        f"{r.get('rms_ppm', float('nan')):.1f}", r["n_merged"]])


def write_dat(path: Path, rows, header: str) -> None:
    """TauREx input: wavelength um, fractional depth, error, FULL bin width."""
    with path.open("w") as fh:
        fh.write("# wavelength_um  transit_depth  depth_err  bin_width_um\n")
        fh.write(f"# {header}\n")
        for r in rows:
            fh.write(f"{r['wave']:.6f}  {r['depth']:.8f}  "
                     f"{r['depth_err']:.8f}  {2*r['wave_err']:.6f}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root")
    ap.add_argument("-R", "--resolution", type=float, default=50.0)
    ap.add_argument("--report", action="store_true",
                    help="Print the before/after error comparison only.")
    ap.add_argument("--min-merged", type=int, default=2,
                    help="Minimum input channels per output bin (default: 2).")
    ap.add_argument("--tag", default="",
                    help="Output suffix, e.g. _R50_test (default: _R50).")
    args = ap.parse_args()

    root = Path(os.path.expanduser(args.root))
    R = args.resolution
    tag = args.tag or f"_R{R:.0f}"
    if not tag.startswith("_"):
        tag = "_" + tag
    if "clean" in tag.lower():
        print("warning: rebin_spectra only rebins; a 'clean' tag does not "
              "apply any additional data cleaning")
    pat = str(root / "*" / "combined" / "combined_*_transmission_spectrum.csv")
    files = [f for f in sorted(glob.glob(pat)) if f"_R{R:.0f}" not in f]
    if not files:
        print(f"No combined spectra under {root}"); return 1

    print(f"{'target':13s} {'det':5s} {'n@in':>5s} {'n@R'+str(int(R)):>6s} "
          f"{'err in':>8s} {'err out':>8s} {'gain':>6s}")
    tot_in, tot_out = [], []
    for f in files:
        slug = Path(f).parts[-3]
        det = "NRS2" if "nrs2" in os.path.basename(f).lower() else "NRS1"
        s = read_spectrum_csv(f)
        rows = rebin(s["wave"], s["wave_err"], s["depth_ppm"],
                     s["depth_err_ppm"], s["rms_ppm"], resolution=R,
                     min_merged=args.min_merged)
        if not rows:
            continue
        e_in = float(np.median(s["depth_err_ppm"]))
        e_out = float(np.median([r["depth_err"] * 1e6 for r in rows]))
        tot_in.append(e_in); tot_out.append(e_out)
        print(f"{slug:13s} {det:5s} {s['wave'].size:5d} {len(rows):6d} "
              f"{e_in:6.0f} ppm {e_out:6.0f} ppm {e_in / e_out:5.2f}x")
        if args.report:
            continue
        head = (f"Patchwork combined spectrum rebinned to R={R:.0f} from "
                f"{s['wave'].size} channels ({det}); errors inverse-variance "
                "combined, neighbouring channels share systematics so this "
                "is a mild upper bound on the true gain")
        base = Path(f).with_name(Path(f).stem + tag)
        write_csv(base.with_suffix(".csv"), rows, head)
        write_dat(base.with_suffix(".dat"), rows, head)

    if tot_in:
        print(f"\nmedian channel error {np.median(tot_in):.0f} -> "
              f"{np.median(tot_out):.0f} ppm   "
              f"(median gain {np.median(np.array(tot_in) / np.array(tot_out)):.2f}x)")
    if not args.report:
        print(f"wrote *{tag}.csv and *{tag}.dat (TauREx input) "
              "beside each combined spectrum")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
