#!/usr/bin/env python
"""Rebuild publication diagnostics from preserved Patchwork products, offline.

This does not refit data or interpret excess scatter as atmospheric composition.
All derived files go to a separate output directory with input SHA256 hashes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aster_toolkit.data_reduction.juliet import read_spectrum_csv
from plot_survey_scale_heights import pretty

COLORS = {"nrs1": "#0072B2", "nrs2": "#D55E00"}
BANDS = {"CH4_window": ("nrs1", ((2.95, 3.15), (3.20, 3.50), (3.55, 3.70))),
         "CO2_window": ("nrs2", ((3.95, 4.15), (4.20, 4.50), (4.55, 4.75)))}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.bbox": "tight"})


def binned_noise(time, residual, minutes, phase=0.0):
    """Nonoverlapping, time-based bins; reject bins below 80% occupancy.

    Bin origins are shifted by phase to expose bin-edge sensitivity. RMS is
    scatter of bin means, not a transit-depth uncertainty or channel error.
    """
    good = np.isfinite(time) & np.isfinite(residual)
    t, r = np.asarray(time)[good], np.asarray(residual)[good]
    order = np.argsort(t)
    t, r = t[order], r[order]
    if len(t) < 3:
        return np.nan, 0
    cadence = np.median(np.diff(t)) * 1440
    expected = minutes / cadence
    ids = np.floor((t - t[0]) * 1440 / minutes + phase).astype(int)
    means = [np.mean(r[ids == i]) for i in np.unique(ids)
             if np.sum(ids == i) >= max(2, 0.8 * expected)]
    return (float(np.std(means, ddof=1) * 1e6) if len(means) >= 3
            else np.nan), len(means)


def write_table(path, rows):
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def centered(s):
    valid = np.isfinite(s["depth_ppm"]) & np.isfinite(s["depth_err_ppm"]) & (s["depth_err_ppm"] > 0)
    w, d, e = (s[key][valid] for key in ("wave", "depth_ppm", "depth_err_ppm"))
    mean = np.average(d, weights=1/e**2)
    return w, d-mean, e, mean


def band_contrast(s, windows):
    """Central weighted mean minus equally weighted flanking means.

    Select native channel centers; store exact coefficients rather than imply
    sharp response edges or independent errors after arbitrary interpolation.
    """
    w, d, e = (np.asarray(s[key]) for key in ("wave", "depth_ppm", "depth_err_ppm"))
    good = np.isfinite(w) & np.isfinite(d) & np.isfinite(e) & (e > 0)
    coefficients = np.zeros(len(w))
    counts = []
    for factor, (lo, hi) in zip((-.5, 1., -.5), windows):
        selected = good & (w >= lo) & (w < hi)
        counts.append(int(selected.sum()))
        if not selected.any():
            raise ValueError("A contrast window has no valid channels")
        weights = 1/e[selected]**2
        coefficients[selected] += factor*weights/weights.sum()
    used = coefficients != 0
    return {"contrast_ppm": float(coefficients[used] @ d[used]),
            "error_diagonal_ppm": float(np.sqrt(np.sum((coefficients[used]*e[used])**2))),
            "counts_blue_feature_red": json.dumps(counts),
            "channel_indices": json.dumps(np.flatnonzero(used).tolist()),
            "coefficients": json.dumps(coefficients[used].tolist())}


def run(root, out):
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    inputs = set()
    visits, spectra, inventory = [], [], []
    targets = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.endswith("_asis")
                     and ((p / "fits").exists() or (p / "reductions").exists()))
    for target in targets:
        paths = sorted(target.glob("fits/*/nrs*/white_fit_summary.json"))
        combined = sorted(target.glob("combined/combined_*_transmission_spectrum.csv"))
        inventory.append({"slug": target.name, "planet": pretty(target.name),
                          "final_detector_fits": len(paths),
                          "visits": len({p.parents[1].name for p in paths}),
                          "combined_detectors": len(combined),
                          "status": "provisional spectra" if combined else "no final spectrum"})
        for path in paths:
            inputs.add(path)
            s = json.loads(path.read_text())
            residual_path = path.with_name("white_lightcurve_residuals.npz")
            inputs.add(residual_path)
            with np.load(residual_path) as z:
                t, residual = z["time"], z["residual"]
            row = {"slug": target.name, "planet": pretty(target.name),
                   "visit": path.parents[1].name, "detector": path.parent.name,
                   "fit_version": s.get("fit_version", "unknown"),
                   "depth_ppm": s["depth_ppm"]["median"],
                   "depth_err_ppm": max(s["depth_ppm"]["minus"], s["depth_ppm"]["plus"]),
                   "impact_parameter": s["b_p1"]["median"],
                   "t0_shift_minutes": (s["t0_p1"]["median"]-s["t0_obs"])*1440,
                   "rms_ppm": float(np.std(residual)*1e6),
                   "beta": s.get("rednoise", {}).get("beta_median", np.nan)}
            for minutes in (5, 10, 20):
                measures = [binned_noise(t, residual, minutes, phase)[0]
                            for phase in (0, .25, .5, .75)]
                row[f"rms_{minutes}min_ppm"] = float(np.nanmedian(measures))
                row[f"rms_{minutes}min_phase_min_ppm"] = float(np.nanmin(measures))
                row[f"rms_{minutes}min_phase_max_ppm"] = float(np.nanmax(measures))
            visits.append(row)
        for path in combined:
            inputs.add(path)
            s = read_spectrum_csv(str(path))
            w, residual, e, mean = centered(s)
            stat = float(np.sum((residual/e)**2))
            spectra.append({"slug": target.name, "planet": pretty(target.name),
                            "detector": "nrs2" if "nrs2" in path.name else "nrs1",
                            "channels": len(w), "weighted_depth_ppm": mean,
                            "median_error_ppm": float(np.median(e)),
                            "chi2_flat": stat, "dof": len(w)-1,
                            "p_flat_diagonal": float(chi2.sf(stat, len(w)-1))})
    write_table(out / "target_inventory.csv", inventory)
    write_table(out / "white_noise_diagnostics.csv", visits)
    write_table(out / "spectral_diagnostics.csv", spectra)

    # Absolute spectra keep offsets visible; centered panels do not imply a
    # physically measured detector calibration or molecular abundance.
    fitted = [p for p in targets if list(p.glob("combined/combined_*_transmission_spectrum.csv"))]
    for relative in (False, True):
        fig, axes = plt.subplots(4, 4, figsize=(12, 10), layout="constrained")
        for ax, target in zip(axes.flat, fitted):
            for detector, color in COLORS.items():
                path = target / "combined" / f"combined_{detector}_transmission_spectrum.csv"
                if not path.exists():
                    continue
                s = read_spectrum_csv(str(path))
                w, r, e, mean = centered(s)
                ax.errorbar(w, r if relative else r+mean, yerr=e,
                            fmt=".", ms=3, elinewidth=.6, color=color)
            ax.set_title(pretty(target.name), loc="left", fontsize=10)
            if relative:
                ax.axhline(0, color="0.5", lw=.6)
            ax.set_xlim(2.8, 5.2)
            ax.tick_params(labelsize=8)
        for ax in axes.flat[len(fitted):]:
            ax.set_visible(False)
        fig.supxlabel(r"Wavelength ($\mu$m)")
        fig.supylabel("Depth minus each detector mean (ppm)" if relative else "Transit depth (ppm)")
        stem = "survey_centered" if relative else "survey_absolute"
        fig.savefig(figures / f"{stem}.pdf")
        fig.savefig(figures / f"{stem}.svg")
        fig.savefig(figures / f"{stem}.png", dpi=180)
        plt.close(fig)

    pairs = sorted({(r["slug"], r["visit"]) for r in visits})
    fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True, layout="constrained")
    for det, color in COLORS.items():
        rows = [next((r for r in visits if (r["slug"], r["visit"], r["detector"]) == (s, v, det)), None)
                for s, v in pairs]
        for ax, key in zip(axes, ("beta", "rms_10min_ppm")):
            ax.plot([r[key] if r else np.nan for r in rows], range(len(pairs)),
                    "o", color=color, ms=4, label=det.upper())
    axes[0].set_yticks(range(len(pairs)), [f"{pretty(s)} / {v}" for s, v in pairs], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(r"White-light $\beta$ (saved fit)")
    axes[1].set_xlabel("10-minute residual scatter (ppm)")
    axes[1].legend(frameon=False)
    fig.savefig(figures / "noise_comparison.pdf")
    fig.savefig(figures / "noise_comparison.svg")
    fig.savefig(figures / "noise_comparison.png", dpi=180)
    plt.close(fig)

    # Independent visit differences: profile one constant offset per detector.
    differences = []
    for target in fitted:
        for det in COLORS:
            paths = sorted(target.glob(f"fits/*/{det}/spectro/transmission_spectrum.csv"))
            for i, first in enumerate(paths):
                for second in paths[i+1:]:
                    inputs.update((first, second))
                    a, b = read_spectrum_csv(str(first)), read_spectrum_csv(str(second))
                    matched = [(k, int(np.argmin(abs(b["wave"]-w)))) for k, w in enumerate(a["wave"])]
                    matched = [(k, j) for k, j in matched if abs(a["wave"][k]-b["wave"][j]) < 1e-3*a["wave"][k]]
                    if len({j for k, j in matched}) != len(matched):
                        raise ValueError("Non-unique wavelength matches")
                    d = np.array([a["depth_ppm"][k]-b["depth_ppm"][j] for k, j in matched])
                    e = np.array([np.hypot(a["depth_err_ppm"][k], b["depth_err_ppm"][j]) for k, j in matched])
                    good = np.isfinite(d) & np.isfinite(e) & (e > 0)
                    d, e = d[good], e[good]
                    if len(d) < 3:
                        continue
                    offset = np.average(d, weights=1/e**2)
                    stat = float(np.sum(((d-offset)/e)**2))
                    differences.append({"planet": pretty(target.name), "detector": det,
                                        "visit_a": first.parents[2].name, "visit_b": second.parents[2].name,
                                        "channels": len(d), "offset_a_minus_b_ppm": offset,
                                        "chi2_shape": stat, "dof": len(d)-1,
                                        "p_shape_diagonal": float(chi2.sf(stat, len(d)-1))})
    write_table(out / "visit_consistency.csv", differences)
    contrasts = []
    for target in fitted:
        for band, (det, windows) in BANDS.items():
            paths = [target / "combined" / f"combined_{det}_transmission_spectrum.csv"]
            paths += sorted(target.glob(f"fits/*/{det}/spectro/transmission_spectrum.csv"))
            for path in paths:
                if not path.exists():
                    continue
                inputs.add(path)
                try:
                    result = {"status": "measured", **band_contrast(read_spectrum_csv(str(path)), windows)}
                except ValueError as exc:
                    result = {"status": str(exc), "contrast_ppm": np.nan,
                              "error_diagonal_ppm": np.nan,
                              "counts_blue_feature_red": "[]", "channel_indices": "[]", "coefficients": "[]"}
                contrasts.append({"planet": pretty(target.name), "detector": det,
                                  "visit": "combined" if path.parent.name == "combined" else path.parents[2].name,
                                  "band": band, "center_selection_windows_um": json.dumps(windows),
                                  "source": str(path.relative_to(root)), **result})
    write_table(out / "band_contrasts.csv", contrasts)
    fig, axes = plt.subplots(1, 2, figsize=(9, 7), sharey=True, layout="constrained")
    for ax, band, color in zip(axes, BANDS, COLORS.values()):
        for i, target in enumerate(fitted):
            rows = [r for r in contrasts if r["planet"] == pretty(target.name) and r["band"] == band]
            individual = [r for r in rows if r["visit"] != "combined"]
            shifts = np.linspace(-.15, .15, len(individual)) if len(individual) > 1 else [0.]
            for shift, r in zip(shifts, individual):
                if not np.isfinite(r["contrast_ppm"]):
                    continue
                ax.errorbar(r["contrast_ppm"], i+shift, xerr=r["error_diagonal_ppm"],
                            fmt="o", ms=3, color="0.70", lw=.7, zorder=1)
            r = next(r for r in rows if r["visit"] == "combined")
            if not np.isfinite(r["contrast_ppm"]):
                ax.text(.98, i, "missing flank", transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=8, color="0.4")
                continue
            ax.errorbar(r["contrast_ppm"], i, xerr=r["error_diagonal_ppm"],
                        fmt="o", ms=4, color=color, lw=1, zorder=2)
        ax.axvline(0, color="0.3", lw=.7)
        ax.set_xlabel("Band contrast (ppm)")
    axes[0].set_title(r"3.20--3.50 $\mu$m window", fontsize=11)
    axes[1].set_title(r"4.20--4.50 $\mu$m window", fontsize=11)
    axes[0].set_yticks(range(len(fitted)), [pretty(p.name) for p in fitted])
    axes[0].invert_yaxis()
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([], [], color=COLORS["nrs2"], marker="o", lw=1,
                                  label="Combined"),
                           Line2D([], [], color="0.70", marker="o", lw=.7,
                                  label="Individual visits")],
                   loc="upper right", frameon=False, fontsize=8)
    fig.savefig(figures / "band_contrasts.pdf", bbox_inches="tight")
    fig.savefig(figures / "band_contrasts.svg", bbox_inches="tight")
    fig.savefig(figures / "band_contrasts.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    gj = root / "GJ_3090_b"
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    gj_rows = []
    for j, det in enumerate(COLORS):
        for visit, color in (("o013", "#0072B2"), ("o014", "#CC79A7")):
            path = gj / "fits" / visit / det / "spectro" / "transmission_spectrum.csv"
            inputs.add(path)
            w, r, e, mean = centered(read_spectrum_csv(str(path)))
            axes[0,j].errorbar(w, r, yerr=e, fmt=".", color=color, alpha=.8, label=visit)
            axes[0,j].set_title(det.upper())
            gj_rows.append({"visit": visit, "detector": det, "mean_ppm": mean,
                            "median_error_ppm": float(np.median(e)),
                            "chi2_flat": float(np.sum((r/e)**2)), "dof": len(w)-1})
            path = gj / "fits" / visit / det / "white_lightcurve_residuals.npz"
            with np.load(path) as z:
                durations = np.geomspace(1, 30, 24)
                noise = [np.nanmedian([binned_noise(z["time"], z["residual"], m, phase)[0]
                                       for phase in (0, .25, .5, .75)]) for m in durations]
            axes[1,j].loglog(durations, noise, color=color, label=visit)
        axes[0,j].axhline(0, color="0.5", lw=.6)
        axes[0,j].set_xlabel(r"Wavelength ($\mu$m)")
        axes[1,j].set_xlabel("Bin duration (minutes)")
        axes[0,j].legend(frameon=False)
    axes[0,0].set_ylabel("Depth minus visit mean (ppm)")
    axes[1,0].set_ylabel("Residual scatter (ppm)")
    fig.savefig(figures / "gj3090_visit_validation.pdf")
    fig.savefig(figures / "gj3090_visit_validation.svg")
    fig.savefig(figures / "gj3090_visit_validation.png", dpi=180)
    plt.close(fig)
    write_table(out / "gj3090_leave_one_visit.csv", gj_rows)
    provenance = [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
                   "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(inputs)]
    (out / "input_manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps({"targets": len(targets), "fitted_targets": len(fitted),
                      "detector_fits": len(visits), "output": str(out)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
