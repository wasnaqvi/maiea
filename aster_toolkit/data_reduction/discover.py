"""Patchwork discovery — turn a raw JWST download tree into run manifests.

The download tree is untrustworthy as an index: ASTER fell back to obsid
folder names when no planet label was attached, one folder name is five
planet labels concatenated, GJ 9827 has no named folder at all, and two
visits exist in duplicate under both a named and a numeric folder. So
this module ignores directory names entirely and indexes on **FITS
headers**, which cannot be malformed by a labeling bug.

    scan_raw_tree      walk the tree, one header read per exposure group
    group_visits       group by (program, observation, visit) exposure
    resolve_planet     which planet does this visit actually transit?
    write_manifests    emit per-planet manifests for survey.py

Planet identification is the part worth doing carefully. MAST
``TARGPROP`` is host-level ("GJ9827"), and a host may have several
planets, so a folder label like ``TOI_836_GO2512`` does not say which
planet was observed — and the planet name drives the archive ephemeris
query, so guessing it wrong yields a confidently wrong transit fit.
Instead each visit is matched by:

1. cone search on the header ``TARG_RA``/``TARG_DEC`` against the NASA
   Exoplanet Archive -> every known planet of that host, then
2. a **transit-window test**: propagate each candidate's ephemeris into
   the observation window (header ``EXPSTART``/``EXPEND``) and keep the
   ones that actually transit during it.

A visit whose planet cannot be resolved this way is reported as
unresolved rather than guessed, and is written into the manifest with a
``planet_name`` of null for you to fill in.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from orchestral.tools.base.tool import BaseTool
    from orchestral.tools.base.field_utils import RuntimeField, StateField
except ModuleNotFoundError:
    class BaseTool:
        """Fallback that keeps discovery importable without Orchestral."""

    def RuntimeField(default=None, description=None):
        return default

    def StateField(default=None, description=None):
        return default


# jw{program:5}{observation:3}{visit:3}_{exposure_tag:5}_{exp:5}[-seg{n:3}]_{detector}_uncal.fits
UNCAL_RE = re.compile(
    r"^jw(?P<program>\d{5})(?P<observation>\d{3})(?P<visit>\d{3})"
    r"_(?P<exp_tag>\d{5})_(?P<exp_num>\d{5})"
    r"(?:-seg(?P<segment>\d{3}))?"
    r"_(?P<detector>nrs1|nrs2|nis|mirimage|nrca\w+|nrcb\w+)"
    r"_uncal\.fits$",
    re.IGNORECASE,
)

# Patchwork survey definition. A visit must match all of these to be a
# science target of the uniform sample.
SURVEY_FILTER = {
    "GRATING": "G395H",
    "FILTER": "F290LP",
    "SUBARRAY": "SUB2048",
    "EXP_TYPE": "NRS_BRIGHTOBJ",
}

MJD_TO_BJD_OFFSET = 2400000.5
CONE_RADIUS_DEG = 0.02
# A candidate counts as transiting during a visit when its transit
# actually *overlaps* the exposure window: |t_pred - t_window_centre| <
# half the window + half the transit duration. A visit catching only
# ingress or only egress still identifies the planet, while a transit
# falling outside the window does not — a fixed generous pad instead of
# this overlap test lets a short-period inner planet false-match on a
# neighbouring epoch. The tolerance below only absorbs ephemeris drift.
EPHEMERIS_TOLERANCE_HR = 0.5


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


# -------------------- scanning --------------------


def parse_uncal_name(path: str) -> dict[str, Any] | None:
    """Parse a JWST uncal filename. None if it is not one."""
    m = UNCAL_RE.match(os.path.basename(path))
    if not m:
        return None
    d = m.groupdict()
    return {
        "path": path,
        "program": d["program"],
        "observation": d["observation"],
        "visit": d["visit"],
        "visit_prefix": f"jw{d['program']}{d['observation']}{d['visit']}",
        "exp_tag": d["exp_tag"],
        "exp_num": d["exp_num"],
        "segment": int(d["segment"]) if d["segment"] else 1,
        "detector": d["detector"].upper(),
    }


HEADER_KEYS = (
    "TARGPROP", "TARGNAME", "TARG_RA", "TARG_DEC", "GRATING", "FILTER",
    "SUBARRAY", "EXP_TYPE", "DETECTOR", "NINTS", "NGROUPS", "EXSEGNUM",
    "EXSEGTOT", "EXPSTART", "EXPEND", "DATE-OBS", "PROGRAM", "PI_NAME",
    "TITLE", "READPATT",
)


def scan_raw_tree(root: str | os.PathLike[str]) -> dict[str, Any]:
    """Index every uncal file under ``root`` by exposure group.

    One header read per (visit_prefix, exp_tag, detector) group — not per
    file — so a tree with thousands of segments is still cheap to index.
    Files whose header cannot be read (evicted / partial downloads) are
    recorded in ``unreadable`` and excluded, never silently dropped.
    """
    from astropy.io import fits

    files: list[dict[str, Any]] = []
    unreadable: list[str] = []
    skipped_names: list[str] = []

    for dirpath, _dirnames, filenames in os.walk(str(root)):
        for name in filenames:
            if not name.endswith("_uncal.fits"):
                continue
            rec = parse_uncal_name(os.path.join(dirpath, name))
            if rec is None:
                skipped_names.append(os.path.join(dirpath, name))
                continue
            files.append(rec)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for rec in files:
        groups[(rec["visit_prefix"], rec["exp_tag"], rec["detector"])].append(rec)

    exposures: list[dict[str, Any]] = []
    for key, recs in sorted(groups.items()):
        recs.sort(key=lambda r: (r["exp_num"], r["segment"]))
        header = None
        for rec in recs:
            try:
                h = fits.getheader(rec["path"])
            except Exception:
                unreadable.append(rec["path"])
                continue
            header = {k: h.get(k) for k in HEADER_KEYS}
            break
        if header is None:
            continue
        visit_prefix, exp_tag, detector = key
        segments = sorted({r["segment"] for r in recs})
        expected = int(header.get("EXSEGTOT") or 0)
        exposures.append({
            "visit_prefix": visit_prefix,
            "exp_tag": exp_tag,
            "detector": detector,
            "program": recs[0]["program"],
            "observation": recs[0]["observation"],
            "visit": recs[0]["visit"],
            "files": [r["path"] for r in recs],
            "directories": sorted({os.path.dirname(r["path"]) for r in recs}),
            "segments_found": segments,
            "segments_expected": expected,
            "missing_segments": [s for s in range(1, expected + 1)
                                 if s not in segments],
            "complete": bool(expected) and len(segments) == expected,
            "header": header,
        })

    return {"root": str(root), "exposures": exposures,
            "unreadable": unreadable, "unparsed": skipped_names}


def _is_survey_exposure(header: dict[str, Any]) -> bool:
    return all(
        str(header.get(key) or "").upper() == want
        for key, want in SURVEY_FILTER.items()
    )


def group_visits(scan: dict[str, Any]) -> list[dict[str, Any]]:
    """Collapse exposure groups into visits (one science TSO exposure per
    visit, both detectors together).

    Only exposures passing the survey filter (G395H / F290LP / SUB2048 /
    NRS_BRIGHTOBJ) are kept — target-acquisition exposures and other
    gratings drop out on EXP_TYPE and GRATING rather than on a guessed
    exposure-tag convention. When a visit somehow has several qualifying
    exposure tags, the one with the most integrations wins.
    """
    by_visit: dict[str, list[dict]] = defaultdict(list)
    for exp in scan["exposures"]:
        if _is_survey_exposure(exp["header"]):
            by_visit[exp["visit_prefix"]].append(exp)

    visits = []
    for visit_prefix, exps in sorted(by_visit.items()):
        by_tag: dict[str, list[dict]] = defaultdict(list)
        for e in exps:
            by_tag[e["exp_tag"]].append(e)
        sci_tag = max(
            by_tag,
            key=lambda t: max(int(e["header"].get("NINTS") or 0) for e in by_tag[t]),
        )
        sci = by_tag[sci_tag]
        h = sci[0]["header"]

        # A visit present under more than one directory is a duplicate
        # download (the malformed folder plus a numeric obsid folder).
        dirs = sorted({d for e in sci for d in e["directories"]})
        detectors = {e["detector"]: e for e in sci}

        visits.append({
            "visit_prefix": visit_prefix,
            "sci_tag": sci_tag,
            "program": sci[0]["program"],
            "observation": sci[0]["observation"],
            "visit": sci[0]["visit"],
            "target": h.get("TARGPROP"),
            "ra": h.get("TARG_RA"),
            "dec": h.get("TARG_DEC"),
            "date_obs": h.get("DATE-OBS"),
            "expstart_mjd": h.get("EXPSTART"),
            "expend_mjd": h.get("EXPEND"),
            "nints": h.get("NINTS"),
            "ngroups": h.get("NGROUPS"),
            "readpatt": h.get("READPATT"),
            "pi": h.get("PI_NAME"),
            "detectors": {
                det: {
                    "n_files": len(e["files"]),
                    "segments_found": e["segments_found"],
                    "segments_expected": e["segments_expected"],
                    "missing_segments": e["missing_segments"],
                    "complete": e["complete"],
                }
                for det, e in sorted(detectors.items())
            },
            "directories": dirs,
            "duplicate_download": len(dirs) > 1,
            "complete": (
                set(detectors) >= {"NRS1", "NRS2"}
                and all(e["complete"] for e in detectors.values())
            ),
            "other_exposure_tags": sorted(set(by_tag) - {sci_tag}),
        })
    return visits


# -------------------- planet resolution --------------------


def _archive_planets_at(ra: float, dec: float,
                        radius_deg: float = CONE_RADIUS_DEG) -> list[dict]:
    from aster_toolkit.data_acquisition.mast import archive_tap_query

    dra = radius_deg / max(1e-6, abs(math.cos(math.radians(dec))))
    conditions = [
        f"ra BETWEEN {ra - dra} AND {ra + dra}",
        f"dec BETWEEN {dec - radius_deg} AND {dec + radius_deg}",
    ]
    return archive_tap_query(
        conditions,
        columns=["pl_name", "hostname", "ra", "dec", "pl_orbper",
                 "pl_tranmid", "pl_trandur", "pl_rade"],
    )


def resolve_planet_from_rows(
    rows: list[dict], expstart_mjd: float, expend_mjd: float,
) -> dict[str, Any]:
    """Transit-window test: of the archive planets at this pointing,
    which actually transit during the exposure window?

    Returns the transiting candidates (``matches``, nearest mid-transit
    first), every planet of the host (``host_planets``), and a
    ``confident`` flag true only when exactly one candidate matches.
    """
    result: dict[str, Any] = {"matches": [], "host_planets": [],
                              "confident": False, "note": ""}
    if not rows:
        result["note"] = "No archive planet at this pointing."
        return result

    result["host_planets"] = sorted({r["pl_name"] for r in rows})
    t_start = float(expstart_mjd) + MJD_TO_BJD_OFFSET
    t_end = float(expend_mjd) + MJD_TO_BJD_OFFSET
    t_mid = 0.5 * (t_start + t_end)
    half_window_hr = 0.5 * (t_end - t_start) * 24

    for row in rows:
        try:
            period = float(row["pl_orbper"])
            tranmid = float(row["pl_tranmid"])
        except (TypeError, ValueError, KeyError):
            continue
        if period <= 0:
            continue
        try:
            duration_hr = float(row["pl_trandur"])
        except (TypeError, ValueError, KeyError):
            duration_hr = 0.0

        n = round((t_mid - tranmid) / period)
        t_pred = tranmid + n * period
        offset_hr = (t_pred - t_mid) * 24
        reach_hr = half_window_hr + 0.5 * duration_hr + EPHEMERIS_TOLERANCE_HR
        if abs(offset_hr) > reach_hr:
            continue

        # Fraction of the transit that actually falls inside the window —
        # 1.0 for a fully covered transit, less for a partial one.
        if duration_hr > 0:
            covered = (
                min(offset_hr + 0.5 * duration_hr, half_window_hr)
                - max(offset_hr - 0.5 * duration_hr, -half_window_hr)
            )
            coverage = max(0.0, min(1.0, covered / duration_hr))
        else:
            coverage = None

        result["matches"].append({
            "pl_name": row["pl_name"],
            "predicted_tmid_bjd": t_pred,
            "hours_from_window_centre": offset_hr,
            "period_d": period,
            "duration_hr": duration_hr or None,
            "transit_coverage": coverage,
            "pl_rade": row.get("pl_rade") or None,
        })

    result["matches"].sort(key=lambda m: abs(m["hours_from_window_centre"]))
    if len(result["matches"]) == 1:
        result["confident"] = True
    elif not result["matches"]:
        result["note"] = (
            "No archive planet of this host transits during the exposure "
            "window — stale ephemeris, an eclipse/phase-curve observation, "
            "or an unlisted planet."
        )
    else:
        result["note"] = (
            f"{len(result['matches'])} planets transit during this window; "
            "pick one manually."
        )
    return result


def resolve_planet(
    ra: float, dec: float, expstart_mjd: float, expend_mjd: float,
    *, radius_deg: float = CONE_RADIUS_DEG,
) -> dict[str, Any]:
    """Cone-search the archive at a pointing, then run the transit-window
    test. Convenience wrapper over ``_archive_planets_at`` +
    ``resolve_planet_from_rows`` for one-off use."""
    if ra is None or dec is None:
        return {"matches": [], "host_planets": [], "confident": False,
                "note": "No TARG_RA/TARG_DEC in header."}
    try:
        rows = _archive_planets_at(float(ra), float(dec), radius_deg)
    except Exception as exc:
        return {"matches": [], "host_planets": [], "confident": False,
                "note": f"Archive query failed: {exc}"}
    return resolve_planet_from_rows(rows, expstart_mjd, expend_mjd)


# -------------------- manifests --------------------


def build_manifests(
    root: str | os.PathLike[str],
    *,
    resolve: bool = True,
    planet_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Scan a raw tree and build one Patchwork manifest per planet.

    ``planet_overrides`` maps either a visit prefix (``jw04098010001``)
    or a header target name (``GJ9827``) to a planet name, for the cases
    the transit-window test cannot settle on its own.
    """
    overrides = planet_overrides or {}
    # Manifests are copied to the cluster and read from other working
    # directories, so record an absolute raw_root.
    root = os.path.abspath(str(root))
    scan = scan_raw_tree(root)
    visits = group_visits(scan)

    # The cone search depends only on the pointing, so it is cached per
    # pointing; the transit-window test depends on the visit and is redone
    # for each one.
    pointing_cache: dict[str, tuple[list[dict], str | None]] = {}

    for v in visits:
        override = overrides.get(v["visit_prefix"]) or overrides.get(v["target"] or "")
        if override:
            v["planet_name"] = override
            v["planet_source"] = "override"
            continue
        if not resolve or v["ra"] is None or v["dec"] is None:
            v["planet_name"] = None
            v["planet_source"] = "unresolved"
            v["resolution_note"] = (
                "resolution disabled" if not resolve else "no pointing in header"
            )
            continue

        key = f"{float(v['ra']):.4f},{float(v['dec']):.4f}"
        if key not in pointing_cache:
            try:
                pointing_cache[key] = (
                    _archive_planets_at(float(v["ra"]), float(v["dec"])), None
                )
            except Exception as exc:
                # A failed query must not masquerade as "this pointing has
                # no known planets" — that would silently drop a target.
                pointing_cache[key] = ([], f"Archive query failed: {exc}")
        rows, query_error = pointing_cache[key]
        res = resolve_planet_from_rows(rows, v["expstart_mjd"], v["expend_mjd"])
        v["host_planets"] = res["host_planets"]
        v["planet_matches"] = res["matches"]
        v["resolution_note"] = query_error or res["note"]
        if res["confident"]:
            v["planet_name"] = res["matches"][0]["pl_name"]
            v["planet_source"] = "transit-window"
        elif res["matches"]:
            v["planet_name"] = res["matches"][0]["pl_name"]
            v["planet_source"] = "ambiguous"
        else:
            v["planet_name"] = None
            v["planet_source"] = "unresolved"

    by_planet: dict[str, list[dict]] = defaultdict(list)
    unresolved = []
    for v in visits:
        if v["planet_name"]:
            by_planet[v["planet_name"]].append(v)
        else:
            unresolved.append(v)

    manifests = {}
    for planet, vs in sorted(by_planet.items()):
        letter = planet.split()[-1] if len(planet.split()[-1]) == 1 else "b"
        manifests[planet] = {
            "planet_name": planet,
            "planet_letter": letter,
            "visits": {
                f"o{v['observation']}": {
                    "raw_root": str(root),
                    "visit_prefix": v["visit_prefix"],
                    "sci_tag": v["sci_tag"],
                }
                for v in sorted(vs, key=lambda v: v["visit_prefix"])
            },
            "_provenance": {
                "planet_source": vs[0].get("planet_source"),
                "host_planets": vs[0].get("host_planets", []),
                "visits": [
                    {"visit_prefix": v["visit_prefix"], "date_obs": v["date_obs"],
                     "nints": v["nints"], "complete": v["complete"],
                     "duplicate_download": v["duplicate_download"]}
                    for v in vs
                ],
            },
        }

    return {"scan": scan, "visits": visits, "manifests": manifests,
            "unresolved": unresolved}


def write_manifests(result: dict[str, Any],
                    output_dir: str | os.PathLike[str]) -> list[str]:
    """Write one ``{slug}.json`` manifest per planet. Unresolved visits go
    into ``_unresolved.json`` with a null planet_name to fill in."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for planet, manifest in result["manifests"].items():
        path = out / f"{_slug(planet)}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        written.append(str(path))
    if result["unresolved"]:
        path = out / "_unresolved.json"
        path.write_text(json.dumps({
            "planet_name": None,
            "_comment": "Fill planet_name per visit, then split into manifests.",
            "visits": [
                {"visit_prefix": v["visit_prefix"], "target": v["target"],
                 "date_obs": v["date_obs"], "ra": v["ra"], "dec": v["dec"],
                 "host_planets": v.get("host_planets", []),
                 "note": v.get("resolution_note", "")}
                for v in result["unresolved"]
            ],
        }, indent=2) + "\n")
        written.append(str(path))
    return written


# -------------------- reporting --------------------


def format_discovery_report(result: dict[str, Any]) -> str:
    scan = result["scan"]
    visits = result["visits"]
    lines = [
        f"Patchwork discovery — {scan['root']}",
        f"  {len(scan['exposures'])} exposure group(s); "
        f"{len(visits)} G395H BOTS science visit(s) "
        f"({len(result['manifests'])} planet(s) resolved).",
    ]
    if scan["unreadable"]:
        lines.append(f"  !! {len(scan['unreadable'])} unreadable file(s) — "
                     "excluded (evicted or partial download):")
        for f in scan["unreadable"][:5]:
            lines.append(f"       {f}")
    if scan["unparsed"]:
        lines.append(f"  {len(scan['unparsed'])} file(s) with unrecognized names, skipped.")

    lines.append("")
    lines.append(f"{'visit':>14}  {'target':<12} {'date':<11} {'nints':>6} "
                 f"{'det':<10} {'planet':<14} source")
    for v in visits:
        dets = "+".join(sorted(v["detectors"]))
        flag = "" if v["complete"] else "  [INCOMPLETE]"
        if v["duplicate_download"]:
            flag += "  [DUPLICATE]"
        lines.append(
            f"{v['visit_prefix']:>14}  {str(v['target'] or '?'):<12} "
            f"{str(v['date_obs'] or '?'):<11} {str(v['nints'] or '?'):>6} "
            f"{dets:<10} {str(v['planet_name'] or '—'):<14} "
            f"{v.get('planet_source', '?')}{flag}"
        )

    ambiguous = [v for v in visits if v.get("planet_source") == "ambiguous"]
    if ambiguous:
        lines.append("")
        lines.append("Ambiguous (several planets transit the window — confirm):")
        for v in ambiguous:
            names = ", ".join(
                f"{m['pl_name']} ({m['hours_from_window_centre']:+.2f} h"
                + (f", {m['transit_coverage'] * 100:.0f}% covered)"
                   if m.get("transit_coverage") is not None else ")")
                for m in v["planet_matches"]
            )
            lines.append(f"  {v['visit_prefix']}: {names}")

    if result["unresolved"]:
        lines.append("")
        lines.append("Unresolved (planet_name must be supplied):")
        for v in result["unresolved"]:
            hosts = ", ".join(v.get("host_planets", [])) or "no archive match"
            lines.append(f"  {v['visit_prefix']} ({v['target']}): {hosts}")
            if v.get("resolution_note"):
                lines.append(f"      {v['resolution_note']}")

    incomplete = [v for v in visits if not v["complete"]]
    if incomplete:
        lines.append("")
        lines.append("Incomplete segment sets — NOT science-usable as-is:")
        for v in incomplete:
            for det, d in v["detectors"].items():
                if not d["complete"]:
                    lines.append(
                        f"  {v['visit_prefix']} {det}: segments "
                        f"{d['segments_found']} of {d['segments_expected']}, "
                        f"missing {d['missing_segments']}"
                    )
            missing_det = {"NRS1", "NRS2"} - set(v["detectors"])
            if missing_det:
                lines.append(f"  {v['visit_prefix']}: no {'/'.join(sorted(missing_det))} data")

    dups = [v for v in visits if v["duplicate_download"]]
    if dups:
        lines.append("")
        lines.append("Duplicate downloads (same visit under several directories):")
        for v in dups:
            lines.append(f"  {v['visit_prefix']}:")
            for d in v["directories"]:
                lines.append(f"      {d}")
    return "\n".join(lines)


# -------------------- CLI --------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m aster_toolkit.data_reduction.discover",
        description="Index a JWST raw tree and emit Patchwork manifests.",
    )
    parser.add_argument("--raw-root", required=True,
                        help="Root of the uncal download tree.")
    parser.add_argument("--manifest-dir", default=None,
                        help="Write manifests here (default: report only).")
    parser.add_argument("--no-resolve", action="store_true",
                        help="Skip archive planet resolution (offline).")
    parser.add_argument("--override", action="append", default=[],
                        metavar="KEY=PLANET",
                        help="Force a planet name, keyed by visit prefix or "
                             "header target (repeatable).")
    args = parser.parse_args(argv)

    overrides = {}
    for item in args.override:
        key, _, value = item.partition("=")
        if not value:
            parser.error(f"--override needs KEY=PLANET, got {item!r}")
        overrides[key.strip()] = value.strip()

    result = build_manifests(args.raw_root, resolve=not args.no_resolve,
                             planet_overrides=overrides)
    print(format_discovery_report(result))
    if args.manifest_dir:
        written = write_manifests(result, args.manifest_dir)
        print("\nManifests written:")
        for p in written:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -------------------- orchestral tool --------------------


class DiscoverPatchworkVisits(BaseTool):
    """
    Index a JWST raw download tree and emit ready-to-run Patchwork
    manifests, one per planet.

    Directory names are ignored entirely — everything is read from FITS
    headers, so malformed multi-target folder names, obsid-only folders,
    and missing planet labels do not matter. Visits are grouped by
    program/observation/visit, filtered to the survey definition (G395H,
    F290LP, SUB2048, NRS_BRIGHTOBJ — target-acquisition and other
    gratings drop out automatically), and checked for segment
    completeness and duplicate downloads.

    Each visit's planet is identified by cone-searching the NASA
    Exoplanet Archive at the header pointing and then testing which
    candidate actually transits during the exposure window. Visits whose
    planet cannot be pinned down are reported as unresolved, never
    guessed; supply them with ``overrides``.

    Run this FIRST on the cluster, before any reduction — the manifests
    it writes are the input to ``RunPatchworkTarget`` /
    ``GeneratePatchworkFirJob``.

    Example
    -------
        DiscoverPatchworkVisits(
            raw_root="/project/def-ncowan/wasi/jwst_raw",
            manifest_dir="patchwork/manifests",
        )
    """

    raw_root: str = RuntimeField(
        description="Root of the uncal download tree (walked recursively)."
    )
    manifest_dir: str | None = RuntimeField(
        default=None,
        description="Directory to write per-planet manifests into. "
                    "Omit for a report-only dry run.",
    )
    resolve_planets: bool = RuntimeField(
        default=True,
        description="Identify planets via the archive + transit-window test. "
                    "Set False when offline.",
    )
    overrides: str | None = RuntimeField(
        default=None,
        description='JSON dict forcing planet names, keyed by visit prefix or '
                    'header target, e.g. {"jw04098010001": "GJ 9827 d"}.',
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        raw_root = self.raw_root
        if not os.path.isabs(raw_root):
            raw_root = os.path.join(self.base_directory, raw_root)

        overrides = self.overrides
        if isinstance(overrides, str) and overrides.strip():
            overrides = json.loads(overrides)
        elif not isinstance(overrides, dict):
            overrides = {}

        result = build_manifests(raw_root, resolve=self.resolve_planets,
                                 planet_overrides=overrides)
        report = format_discovery_report(result)
        if self.manifest_dir:
            manifest_dir = self.manifest_dir
            if not os.path.isabs(manifest_dir):
                manifest_dir = os.path.join(self.base_directory, manifest_dir)
            written = write_manifests(result, manifest_dir)
            report += "\n\nManifests written:\n" + "\n".join(
                f"  {p}" for p in written
            )
            report += (
                "\n\nNext: RunPatchworkTarget (local subset test) or "
                "GeneratePatchworkFirJob (full run on Fir) per manifest."
            )
        return report
