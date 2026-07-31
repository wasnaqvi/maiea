#!/usr/bin/env python
"""Merge a Patchwork figure tree into one bookmarked PDF.

Walks an output root (or a downloaded figure tree) and concatenates the
per-target figures in reading order:

    <target>  white light NRS1 -> NRS2
              transmission spectrum NRS1 -> NRS2
              combined spectrum

Each target becomes a top-level PDF bookmark, so a 20+ page review
document is navigable.

    python scripts/patchwork/merge_figures.py wave1_figs -o wave1_review.pdf

Needs pypdf (``pip install pypdf``). Pure PDF plumbing — it does not
re-render, so figures keep whatever styling they were written with.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

# Reading order within a target. Each entry is (glob suffix, label).
_ORDER = [
    ("fits/*/nrs1/white_lightcurve_fit.pdf", "white light NRS1"),
    ("fits/*/nrs2/white_lightcurve_fit.pdf", "white light NRS2"),
    ("fits/*/nrs1/spectro/transmission_spectrum.pdf", "spectrum NRS1"),
    ("fits/*/nrs2/spectro/transmission_spectrum.pdf", "spectrum NRS2"),
    ("combined/combined_transmission_spectrum.pdf", "combined spectrum"),
]


def collect(root: Path) -> list[tuple[str, str, Path]]:
    """(target, label, path) triples in review order."""
    targets = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    out: list[tuple[str, str, Path]] = []
    for target in targets:
        for suffix, label in _ORDER:
            for p in sorted(glob.glob(str(root / target / suffix))):
                # Distinguish visits when a target has several.
                m = re.search(r"/fits/([^/]+)/", p)
                visit = f" {m.group(1)}" if m else ""
                out.append((target, f"{label}{visit}", Path(p)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Merge Patchwork figures into one bookmarked PDF.")
    ap.add_argument("root", help="Figure tree (e.g. wave1_figs).")
    ap.add_argument("-o", "--output", default=None,
                    help="Output PDF (default: <root>/patchwork_figures.pdf).")
    args = ap.parse_args(argv)

    try:
        from pypdf import PdfWriter, PdfReader
    except ImportError:
        print("pypdf is required:  pip install pypdf")
        return 1

    root = Path(os.path.expanduser(args.root)).resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return 1

    items = collect(root)
    if not items:
        print(f"No Patchwork figures found under {root}")
        return 1

    writer = PdfWriter()
    page_no = 0
    current_target = None
    target_bookmark = None

    for target, label, path in items:
        reader = PdfReader(str(path))
        if target != current_target:
            target_bookmark = writer.add_outline_item(
                target.replace("_", " "), page_no)
            current_target = target
        writer.add_outline_item(label, page_no, parent=target_bookmark)
        for page in reader.pages:
            writer.add_page(page)
            page_no += 1
        print(f"  {target:14s} {label:24s} {path.name}")

    out_path = Path(args.output) if args.output else root / "patchwork_figures.pdf"
    with out_path.open("wb") as fh:
        writer.write(fh)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nWrote {out_path}  ({page_no} pages, {size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
