#!/usr/bin/env python
"""Restamp v1.4 Patchwork fits as v1.5 where v1.5 would not change them.

Fit v1.5 changed exactly one numerical thing: the per-channel limb
darkening is FIXED to ExoTiC-LD when the white-light impact parameter
exceeds ``LD_FIX_IMPACT_PARAM`` (0.80). Below that, the white fit and
every channel fit run the same code path as v1.4. A refit would differ
only by sampler noise.
The other v1.5 changes are bookkeeping: ``spot_handling`` / ``tilt_handling``
metadata, two new summary fields, and a ``.dat`` format fix (the 4th column is
now the FULL bin width; v1.4 wrote the half width).

So a v1.4 target whose every white fit has b <= 0.80 is restamped
rather than refit:

  * ``fit_version`` 1.4 -> 1.5 in ``patchwork_summary.json`` and every
    ``white_fit_summary.json`` (pass 1 included; suffixes such as
    ``-noscan`` are kept);
  * ``ld_fixed_per_channel: false``, ``impact_parameter`` and, if absent,
    ``resolution`` added to the summary, as a v1.5 fit records them;
  * the spectrum CSV / DAT headers and the combined CSV headers read
    ``v1.5 (restamped from v1.4)``, so a file travelling alone still says
    what it is;
  * the ``.dat`` 4th column is doubled to the full width (only when its
    header still says ``bin_half_width_um``);
  * a ``fit_restamp`` record in the summary names the rule, the largest b
    and every file touched.

A target with ANY v1.4 white fit above the threshold (or with no b
recorded) is refused whole and must be refit at v1.5 instead -- for those the LD
change is the whole point. Targets are all-or-nothing; nothing is written
for a refused target.

Derived files (``*_R50*`` rebins, retrieval exports) are not touched:
regenerate them with ``rebin_spectra.py`` afterwards.

Dry-run by default. ``--write`` keeps a ``.pre-restamp`` copy of every
file it changes. Re-running is safe: restamped files are skipped.

    python scripts/patchwork/restamp_fit_version.py ~/scratch/patchwork
    python scripts/patchwork/restamp_fit_version.py ~/scratch/patchwork --write
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aster_toolkit.data_reduction.juliet import (  # noqa: E402
    LD_FIX_IMPACT_PARAM,
    PATCHWORK_FIT_VERSION,
)
from aster_toolkit.data_reduction.lightcurves import DEFAULT_RESOLUTION  # noqa: E402
from aster_toolkit.data_reduction.survey import _write_json_atomic  # noqa: E402

FROM_VERSION = "1.4"
TO_VERSION = "1.5"
BACKUP_SUFFIX = ".pre-restamp"
RESTAMP_NOTE = f"restamped from v{FROM_VERSION}"
# "fit v1.4" in per-visit headers, "Patchwork v1.4" in combined headers.
_HEADER_STAMP = re.compile(
    r"(Patchwork (?:juliet fit )?v)" + re.escape(FROM_VERSION) + r"(?![\d.])")


class Refused(Exception):
    """This target cannot be restamped; it needs a v1.5 refit."""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Refused(f"{path} is not a JSON object")
    return value


def bump_stamp(stamp: str) -> str:
    """'1.4' -> '1.5', '1.4-noscan' -> '1.5-noscan'; 1.5 stamps unchanged."""
    base, sep, suffix = str(stamp).partition("-")
    if base == TO_VERSION:
        return stamp
    if base != FROM_VERSION:
        raise Refused(f"fit_version {stamp!r} is neither v{FROM_VERSION} nor "
                      f"v{TO_VERSION}; only a v{FROM_VERSION} fit can be restamped")
    return TO_VERSION + sep + suffix


def restamp_header(text: str) -> str:
    return _HEADER_STAMP.sub(rf"\g<1>{TO_VERSION} ({RESTAMP_NOTE})", text, count=1)


def restamp_csv(text: str) -> str:
    """Restamp the first (comment) line of a Patchwork spectrum CSV."""
    first, sep, rest = text.partition("\n")  # first keeps any "\r"
    return restamp_header(first) + sep + rest


def restamp_dat(text: str) -> str:
    """Restamp a spectrum DAT; convert a half-width 4th column to full width."""
    lines = text.splitlines(keepends=True)
    half_width = bool(lines) and "bin_half_width_um" in lines[0]
    out = []
    for i, line in enumerate(lines):
        if line.startswith("#"):
            if i == 0 and half_width:
                line = line.replace("bin_half_width_um", "bin_width_um")
            out.append(restamp_header(line))
            continue
        cols = line.split()
        if half_width and len(cols) == 4:
            cols[3] = f"{2 * float(cols[3]):.6f}"
            line = "  ".join(cols) + line[len(line.rstrip("\r\n")):]
        out.append(line)
    return "".join(out)


def plan_target(target_dir: Path) -> dict[str, Any]:
    """Validate one target and build its changes in memory.

    Returns {"status": "restamp" | "current" | "skip", ...}. Raises
    Refused when any fit rules a restamp out.
    """
    summary_path = target_dir / "patchwork_summary.json"
    whites = sorted(target_dir.glob("fits/*/nrs*/white_fit_summary.json"))
    if not summary_path.exists() or not whites:
        return {"status": "skip", "reason": "no fits"}
    summary = _read_json(summary_path)

    impacts: dict[str, float] = {}
    changes: dict[Path, str] = {}
    for white_path in whites:
        visit, det = white_path.parent.parent.name, white_path.parent.name
        white = _read_json(white_path)
        b = (white.get("b_p1") or {}).get("median")
        if b is None:
            raise Refused(f"{visit} {det.upper()}: no white b_p1 recorded")
        b = float(b)
        native = str(white.get("fit_version", "")).partition("-")[0] == TO_VERSION
        if b > LD_FIX_IMPACT_PARAM and not native:
            raise Refused(f"{visit} {det.upper()}: b = {b:.2f} > "
                          f"{LD_FIX_IMPACT_PARAM:.2f}; v1.5 fixes its limb "
                          "darkening, so refit it")
        impacts[f"{visit}/{det.upper()}"] = b
        for path in (white_path, white_path.parent / "pass1" / white_path.name):
            if not path.exists():
                continue
            payload = white if path == white_path else _read_json(path)
            stamp = payload.get("fit_version", "")
            new = bump_stamp(stamp)
            if new != stamp:
                payload = dict(payload, fit_version=new, fit_restamp={
                    "from": stamp, "rule": f"b <= {LD_FIX_IMPACT_PARAM:.2f}"})
                changes[path] = json.dumps(payload, indent=2) + "\n"
        spectro = white_path.parent / "spectro"
        for path, fn in ((spectro / "transmission_spectrum.csv", restamp_csv),
                         (spectro / "transmission_spectrum.dat", restamp_dat)):
            if path.exists():
                old = _read_text(path)
                new = fn(old)
                if new != old:
                    changes[path] = new
    for path in sorted(target_dir.glob("combined/combined_nrs?_transmission_spectrum.csv")):
        old = _read_text(path)
        new = restamp_csv(old)
        if new != old:
            changes[path] = new

    # The rollup: every fit record must be a v1.4 (or already v1.5) fit
    # whose white summary we just checked.
    new_summary = json.loads(json.dumps(summary))
    for visit, record in (new_summary.get("visits") or {}).items():
        for det, fit in (record.get("fits") or {}).items():
            key = f"{visit}/{det.upper()}"
            if key not in impacts:
                raise Refused(f"{key}: in the summary but has no white_fit_summary.json")
            fit["fit_version"] = bump_stamp(fit.get("fit_version", ""))
            fit.setdefault("ld_fixed_per_channel", False)
            fit.setdefault("impact_parameter", impacts[key])
    if "fit_version" in new_summary:
        new_summary["fit_version"] = bump_stamp(new_summary["fit_version"])
    new_summary.setdefault("resolution", float(DEFAULT_RESOLUTION))

    if not changes and new_summary == summary:
        return {"status": "current", "max_b": max(impacts.values())}

    restamp = {
        "from": FROM_VERSION, "to": TO_VERSION,
        "date": _dt.date.today().isoformat(),
        "source": "scripts/patchwork/restamp_fit_version.py",
        "rule": (f"every white fit has b <= {LD_FIX_IMPACT_PARAM:.2f}, where "
                 "v1.5 runs the v1.4 code path; not refit"),
        "max_impact_parameter": max(impacts.values()),
        "files": sorted(str(p.relative_to(target_dir)) for p in changes)
                 + ["patchwork_summary.json"],
    }
    new_summary.setdefault("fit_restamps", []).append(restamp)
    changes[summary_path] = json.dumps(new_summary, indent=2) + "\n"
    return {"status": "restamp", "changes": changes,
            "max_b": restamp["max_impact_parameter"], "n_fits": len(impacts)}


def _read_text(path: Path) -> str:
    # newline="" keeps csv.writer's CRLF endings, so only the header changes.
    with path.open(newline="") as handle:
        return handle.read()


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", newline="") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def apply_changes(changes: dict[Path, str]) -> None:
    for path in changes:
        backup = path.with_name(path.name + BACKUP_SUFFIX)
        if backup.exists():
            raise FileExistsError(f"Backup already exists: {backup}")
    for path, text in changes.items():
        shutil.copy2(path, path.with_name(path.name + BACKUP_SUFFIX))
        if path.suffix == ".json":
            _write_json_atomic(path, json.loads(text))
        else:
            _write_text_atomic(path, text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", help="Directory containing target output directories.")
    parser.add_argument("--write", action="store_true",
                        help="Apply the restamp (keeps .pre-restamp copies).")
    args = parser.parse_args(argv)
    if PATCHWORK_FIT_VERSION != TO_VERSION:
        raise SystemExit(f"This script encodes the v{FROM_VERSION}->v{TO_VERSION} "
                         f"delta; the code is now v{PATCHWORK_FIT_VERSION}.")
    root = Path(os.path.expanduser(args.root))
    counts = {"restamp": 0, "current": 0, "refused": 0}
    for target_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            plan = plan_target(target_dir)
        except Refused as exc:
            counts["refused"] += 1
            print(f"{target_dir.name:14s} REFUSED  {exc}")
            continue
        if plan["status"] == "skip":
            continue
        counts[plan["status"]] += 1
        if plan["status"] == "current":
            print(f"{target_dir.name:14s} current  already v{TO_VERSION} "
                  f"(max b {plan['max_b']:.2f})")
            continue
        verb = "restamped" if args.write else "would restamp"
        print(f"{target_dir.name:14s} {verb}  {plan['n_fits']} fit(s), "
              f"max b {plan['max_b']:.2f}, {len(plan['changes'])} file(s)")
        if args.write:
            apply_changes(plan["changes"])
    print(f"\n{counts['restamp']} {'restamped' if args.write else 'to restamp'}, "
          f"{counts['current']} already v{TO_VERSION}, "
          f"{counts['refused']} refused (refit at v{TO_VERSION})")
    if not args.write and counts["restamp"]:
        print("Dry run. Re-run with --write to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
