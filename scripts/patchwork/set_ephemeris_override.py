#!/usr/bin/env python
"""Write a measured/updated ephemeris into a target manifest.

The NASA Exoplanet Archive is CURRENT, not correct. Its ephemeris for a
planet discovered years ago carries a period good to a few seconds, and
a few seconds per epoch over several hundred epochs is hours of drift.
When that drift exceeds the exposure window the fit-time transit-in-
window guard refuses, which is the correct outcome and not something to
work around by loosening the guard.

Measured on the 2026-08-27 Patchwork run:

    TOI-125 b o101   archive predicts +24.71 h from the window centre
    TOI-125 c o201   archive predicts -25.22 h

Both windows contain an obvious transit. The archive periods are wrong
by -181.7 s and +378.1 s per epoch respectively (ExoClock Project IV,
arXiv:2511.14407); over 345 and 196 epochs that is 17.4 h and 20.6 h of
accumulated drift. With the updated values both land inside the window
(+0.52 h and +0.92 h) and the guard passes.

    # published update
    python scripts/patchwork/set_ephemeris_override.py \\
        ~/patchwork/manifests/TOI_125_b.json \\
        --t0 2458978.6849 --period 4.651717 \\
        --source "ExoClock Project IV, arXiv:2511.14407"

    # or measure it from the visit itself, when no update exists
    python scripts/patchwork/set_ephemeris_override.py \\
        ~/patchwork/manifests/TOI_270_b.json --measure \\
        ~/scratch/patchwork/TOI_270_b/reductions/o017

Every override is recorded in the fit summary as
``priors_source: archive+override(...)``, so a target fitted on a
non-archive ephemeris can never be mistaken for a default one.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def measure(reduction_dir: str) -> dict[str, float]:
    """Mid-transit measured from the white lightcurve of each detector."""
    import numpy as np
    from aster_toolkit.data_reduction.lightcurves import (
        N_REFPIX_COLS, estimate_transit_midpoint, load_stage3_spectra,
    )

    out = {}
    for det in ("nrs1", "nrs2"):
        hits = sorted(glob.glob(os.path.join(
            reduction_dir, det, "**", "*_spectra_fullres.fits"), recursive=True))
        if not hits:
            continue
        sp = load_stage3_spectra(hits[0])
        flux = np.nansum(sp["flux"][:, N_REFPIX_COLS:-N_REFPIX_COLS], axis=1)
        r = estimate_transit_midpoint(np.asarray(sp["time"], float), flux)
        if not r["found"]:
            print(f"  {det.upper()}: {r['note']}")
            continue
        print(f"  {det.upper()}: t0={r['t0']:.5f}  depth={r['depth_ppm']:.0f} ppm  "
              f"dur={r['duration_hr']:.2f} h  partial={r['partial']}")
        if r["partial"]:
            print(f"  {det.upper()}: PARTIAL transit -- ingress or egress fell "
                  "outside the exposure, so this t0 is not a measurement.")
            continue
        out[det.upper()] = float(r["t0"])
    if len(out) == 2:
        gap = abs(out["NRS1"] - out["NRS2"]) * 24 * 60
        print(f"  detector agreement: {gap:.1f} min " + ("OK" if gap < 5 else
              "<-- DISAGREE, not a transit"))
        if gap >= 5:
            raise SystemExit("Detectors disagree on t0; refusing to write an override.")
    if not out:
        raise SystemExit("No usable transit measured; nothing written.")
    return {"t0": sum(out.values()) / len(out)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("manifest")
    ap.add_argument("--t0", type=float, help="Mid-transit epoch, BJD_TDB.")
    ap.add_argument("--period", type=float, help="Orbital period, days.")
    ap.add_argument("--duration-hr", type=float)
    ap.add_argument("--measure", metavar="REDUCTION_DIR",
                    help="Measure t0 from this visit's Stage 3 products.")
    ap.add_argument("--source", default="",
                    help="Where the values came from — recorded in the manifest.")
    ap.add_argument("--clear", action="store_true", help="Remove the override.")
    args = ap.parse_args()

    path = os.path.expanduser(args.manifest)
    manifest = json.load(open(path))

    if args.clear:
        manifest.pop("priors_override", None)
        manifest.pop("priors_override_source", None)
    else:
        override = dict(manifest.get("priors_override") or {})
        if args.measure:
            override.update(measure(os.path.expanduser(args.measure)))
        for key, value in (("t0", args.t0), ("period", args.period),
                           ("duration_hr", args.duration_hr)):
            if value is not None:
                override[key] = value
        if not override:
            ap.error("Nothing to write: give --t0/--period/--duration-hr or --measure.")
        manifest["priors_override"] = override
        if args.source:
            manifest["priors_override_source"] = args.source

    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n{os.path.basename(path)}: "
          f"priors_override = {manifest.get('priors_override')}")
    if manifest.get("priors_override_source"):
        print(f"  source: {manifest['priors_override_source']}")
    print("Refit with FORCE_REFIT=1; the summary will record "
          "priors_source='archive+override(...)'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
