#!/usr/bin/env python
"""Check every fitted target against the CAMPAIGN.md after-fit checklist.

Reads ``<root>/<SLUG>/patchwork_summary.json`` and prints one block per
target: per visit x detector the depth, depth check, betas, channel count,
tilt and spot events, then ``ok`` or the list of failed checks. Physics,
not exit codes: a fit that ran to completion can still be the wrong fit.

    python scripts/patchwork/gate_fits.py ~/scratch/patchwork
    python scripts/patchwork/gate_fits.py ~/scratch/patchwork HD_15337_c TOI_451_c

Exit status is 1 when any target has a STOP-level failure (wrong version,
suspect depth, prior-returning fit, scan never ran), so it can gate a
script. Read-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aster_toolkit.data_reduction.juliet import (  # noqa: E402
    GP_KERNEL,
    GP_RHO_MIN_DAYS,
    JOINT_SHARED_PARAMS,
    PATCHWORK_FIT_VERSION,
)

# Fewer channels than this means many were dropped (expected ~29 / ~24).
MIN_CHANNELS = {"NRS1": 24, "NRS2": 20}
BETA_AFTER_GP_FLAG = 1.5
FRAC_MASKED_FLAG = 0.15


def _med(x):
    return x.get("median") if isinstance(x, dict) else x


def check_target(summary: dict) -> tuple[list[str], bool]:
    """Report lines for one target, and whether it hit a STOP condition."""
    lines, stop = [], False
    offset = summary.get("nrs1_nrs2_offset_ppm")
    lines.append(f"  NRS1-NRS2 offset {offset:+.0f} ppm" if offset is not None
                 else "  NRS1-NRS2 offset not measured")
    for vname, visit in sorted((summary.get("visits") or {}).items()):
        fits = visit.get("fits") or {}
        if not fits:
            continue
        lines.append(f"  -- {vname}"
                     + ("  (t0 shared)" if visit.get("t0_shared_across_detectors") else ""))
        geometry = {}
        for det, f in sorted(fits.items()):
            stops, notes = [], []
            w = f.get("white_depth_ppm") or {}
            depth = w.get("median") or 0.0
            err = max(w.get("plus") or 0.0, w.get("minus") or 0.0)
            dc = f.get("depth_check") or {}
            beta = (f.get("rednoise") or {}).get("beta_median")
            beta_pre = (f.get("rednoise_before_gp") or {}).get("beta_median")
            noise = f.get("noise_model") or {}
            joint = f.get("joint_fit") or {}
            th = f.get("tilt_handling") or {}
            sh = f.get("spot_handling") or {}
            nch = f.get("n_channels") or 0
            geometry[det] = tuple(_med(f.get(k)) for k in ("t0_p1", "b_p1", "a_p1"))

            if f.get("fit_version") != PATCHWORK_FIT_VERSION:
                stops.append(f"version {f.get('fit_version')} != {PATCHWORK_FIT_VERSION}")
            if dc.get("status") == "suspect":
                stops.append("depth SUSPECT")
            if depth and err > 0.5 * depth:
                stops.append("error > depth/2 (prior-returning fit)")
            if not sh.get("scan_ran"):
                stops.append("spot scan did not run")
            if (joint.get("n_detectors") or 0) < 2:
                notes.append("not a joint fit")
            if noise.get("kind") != GP_KERNEL:
                notes.append(f"noise model {noise.get('kind')}")
            rho_min = noise.get("GP_rho_min")
            if rho_min is not None and rho_min < 1.5 * GP_RHO_MIN_DAYS * 1440:
                notes.append(f"GP rho {rho_min:.1f} min at its lower bound")
            if beta is not None and beta > BETA_AFTER_GP_FLAG:
                notes.append(f"beta after GP {beta:.2f} > {BETA_AFTER_GP_FLAG}")
            if f.get("ld_source") != "exotic-ld":
                notes.append(f"ld {f.get('ld_source')}")
            if f.get("priors_source") not in (None, "archive"):
                notes.append(f"priors {f.get('priors_source')}")
            if nch < MIN_CHANNELS.get(det, 20):
                notes.append(f"only {nch} channels")
            if not th.get("searched_diagnostics"):
                notes.append("tilt search flux-only")
            if th.get("rate_warning"):
                notes.append(f"tilt rate: {th['rate_warning']}")
            if (sh.get("frac_masked") or 0) > FRAC_MASKED_FLAG:
                notes.append(f"masked {sh['frac_masked']:.0%}")

            lines.append(
                f"     {det}: {depth:.0f} ± {err:.0f} ppm  "
                f"check={dc.get('status')}/{dc.get('band')}"
                + (f" ({dc['difference_frac'] * 100:+.1f}%)"
                   if dc.get("difference_frac") is not None else "")
                + f"  beta {beta_pre if beta_pre is None else round(beta_pre, 2)}"
                f" -> {beta if beta is None else round(beta, 2)} after GP"
                + (f" (rho {rho_min:.0f} min)" if rho_min is not None else "")
                + f"  chans={nch}")
            for e in th.get("events") or []:
                src = e.get("sources") or [e.get("source")]
                weak = (e.get("n_sources") or len(src)) < 2 or "trace_fwhm" not in src
                lines.append(
                    f"        tilt {e.get('hours_from_mid') or 0:+.2f} h "
                    f"{e.get('amplitude_ppm') or 0:+.0f} ppm"
                    + ("  IN TRANSIT" if e.get("in_transit") else "")
                    + f"  [{', '.join(map(str, src))}]"
                    + ("  <- look: no trace_fwhm / <2 series" if weak else ""))
            for e in sh.get("events") or []:
                lines.append(
                    f"        spot {e.get('kind')} {e.get('hours_from_mid') or 0:+.2f} h "
                    f"{e.get('amplitude_ppm') or 0:+.0f} ppm  confirmed={e.get('confirmed')}"
                    f"  achromatic={e.get('achromatic')}")
            if stops:
                stop = True
                lines.append("        STOP: " + "; ".join(stops)
                             + ("; " + "; ".join(notes) if notes else ""))
            else:
                lines.append("        " + ("look: " + "; ".join(notes) if notes else "ok"))
        if len(geometry) == 2 and len(set(geometry.values())) > 1:
            stop = True
            lines.append("     STOP: detectors disagree on the shared geometry "
                         f"{geometry} (stale per-detector summary?)")
    contam = summary.get("contamination")
    if not contam:
        lines.append("  Stage 6.5: not run")
    elif contam.get("error"):
        lines.append(f"  Stage 6.5: FAILED {contam['error']}")
    else:
        lines.append(
            f"  Stage 6.5: delta_BIC {contam.get('delta_bic', 0):+.1f}, "
            f"f_het < {contam.get('f_het_95pct_upper', float('nan')):.3f} (95%)"
            + ("  CONTAMINATION FAVOURED" if contam.get("detected") else "")
            + ("  railed" if contam.get("railed") else ""))
    return lines, stop


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    root = Path(argv[0]).expanduser()
    slugs = argv[1:] or sorted(p.name for p in root.iterdir()
                               if (p / "patchwork_summary.json").is_file())
    any_stop = False
    print(f"Survey fit version: {PATCHWORK_FIT_VERSION}; shared geometry: "
          f"{', '.join(JOINT_SHARED_PARAMS)}")
    for slug in slugs:
        path = root / slug / "patchwork_summary.json"
        if not path.is_file():
            print(f"\n=== {slug}: no patchwork_summary.json")
            any_stop = True
            continue
        with path.open() as handle:
            summary = json.load(handle)
        lines, stop = check_target(summary)
        any_stop |= stop
        print(f"\n=== {slug}" + ("   [STOP]" if stop else ""))
        print("\n".join(lines))
    return 1 if any_stop else 0


if __name__ == "__main__":
    raise SystemExit(main())
