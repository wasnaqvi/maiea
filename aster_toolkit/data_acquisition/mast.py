from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import ast
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

try:
    try:  # shared base + field helpers; see aster_toolkit/tool_base.py
        from aster_toolkit.tool_base import (AsterBaseTool as BaseTool, OptionalRuntimeField, RuntimeField, StateField)
    except ModuleNotFoundError as _exc:  # loaded by file path, outside the package
        if _exc.name != "aster_toolkit":
            raise
        import sys as _sys, pathlib as _pl
        _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
        from aster_toolkit.tool_base import (AsterBaseTool as BaseTool, OptionalRuntimeField, RuntimeField, StateField)
except ModuleNotFoundError:
    class BaseTool:
        """Fallback that keeps plain MAST wrappers importable without Orchestral."""

    def RuntimeField(default=None, description=None):
        return default

    def StateField(default=None, description=None):
        return default

    def OptionalRuntimeField(description=None, **kwargs):
        return None


# After the block above, which puts the repo on sys.path when this file is
# loaded by path (tests/scripts/test_mast.py).
from aster_toolkit.data_acquisition import archive_tap as _archive_tap


# ASTER_MAST_BASE_URL exists for end-to-end tests (tests/e2e) that point the
# real agent at a local fake MAST; leave it unset in normal use.
MAST_BASE_URL = os.environ.get("ASTER_MAST_BASE_URL", "https://mast.stsci.edu").rstrip("/")
MAST_INVOKE_URL = f"{MAST_BASE_URL}/api/v0/invoke"
MAST_DOWNLOAD_URL = f"{MAST_BASE_URL}/api/v0.1/Download/file"

# Authoritative per-proposal program metadata (Cycle, title, PI, status).
# MAST CAOM does NOT expose the observing cycle, and cycle is NOT derivable
# from proposal_id arithmetic (e.g. GO 2372 is Cycle 1; GO 3557 and GO 4098
# are Cycle 2). STScI's program-info page is the source of truth.
STSCI_PROGRAM_INFO_URL = (
    "https://www.stsci.edu/jwst-program-info/program/?program={proposal_id}"
)
# Default on-disk cache filename (relative to a workspace/base directory).
# proposal_id -> cycle never changes, so cached entries are valid forever.
JWST_PROGRAM_INFO_CACHE = "mast/jwst_program_info_cache.json"

DEFAULT_ARCHIVE_COLUMNS = [
    "pl_name", "hostname", "ra", "dec",
    "pl_radj", "pl_rade", "pl_bmassj", "pl_bmasse",
    "pl_orbper", "pl_orbsmax", "pl_eqt", "pl_dens", "pl_insol",
    "pl_orbeccen", "pl_orbincl", "pl_trandep", "pl_imppar",
    # Transit ephemeris — required by the transit/eclipse/phase-curve
    # classifier (pl_tranmid is BJD_TDB, pl_trandur is hours).
    "pl_tranmid", "pl_trandur",
    "st_rad", "st_teff", "st_mass", "st_logg", "st_met", "st_age",
    "sy_dist", "sy_vmag", "sy_kmag",
    "discoverymethod", "disc_year",
]

JWST_OBSERVATION_COLUMNS = [
    "obsid",
    "obs_id",
    "target_name",
    "s_ra",
    "s_dec",
    "obs_collection",
    "instrument_name",
    "dataproduct_type",
    "calib_level",
    "filters",
    "t_min",
    "t_max",
    "proposal_id",
    "proposal_pi",
    "intentType",
    # When the observation's files go public (MJD) and whether they are.
    "t_obs_release",
    "dataRights",
]

RAW_PRODUCT_SUBGROUPS = ("UNCAL",)
SCIENCE_PRODUCT_TYPES = ("SCIENCE",)
FITS_EXTENSIONS = (".fits", ".fit", ".fits.gz")

JWST_INSTRUMENTS = ("NIRSpec", "NIRCam", "MIRI", "NIRISS", "FGS")
JWST_DATAPRODUCT_TYPES = ("spectrum", "timeseries", "image", "cube")
JWST_PRODUCT_SUBGROUPS = (
    "UNCAL",      # raw uncalibrated ramps (stage 0 input)
    "RATE",       # stage 1 countrate per exposure
    "RATEINTS",   # stage 1 countrate per integration (time series)
    "CAL",        # stage 2 calibrated image/spectrum per exposure
    "CALINTS",    # stage 2 calibrated per integration
    "X1D",        # stage 3 extracted 1-D spectrum
    "X1DINTS",    # stage 3 extracted 1-D spectrum per integration
    "S2D",        # stage 3 resampled 2-D spectrum
    "S3D",        # stage 3 IFU cube
    "WHTLT",      # white-light curve
)

# Official MAST CAOM ``instrument_name`` vocabulary for JWST (instrument +
# configuration), per MAST Docs "JWST Instrument Names" (MASTDOCS, page
# 176435458). MAST discrete filters are exact-match, so every real mode must
# be enumerated; a missing mode silently drops those observations.
#
# History: the previous table omitted NIRSPEC/BOTS (the bright-object
# time-series mode used by essentially *all* NIRSpec exoplanet transit/eclipse
# observations) and used MIRI/MRS + MIRI/LRS, which are not CAOM values (the
# real ones are MIRI/IFU for MRS and MIRI/SLIT / MIRI/SLITLESS for LRS). Any
# "NIRSpec" or "MIRI" search therefore excluded exactly the TSO data this
# toolkit exists to find.
JWST_INSTRUMENT_ALIASES = {
    "NIRSPEC": [
        "NIRSPEC/BOTS", "NIRSPEC/SLIT", "NIRSPEC/IFU", "NIRSPEC/MSA",
        "NIRSPEC/IMAGE",
    ],
    "NIRCAM": [
        "NIRCAM/IMAGE", "NIRCAM/GRISM", "NIRCAM/WFSS", "NIRCAM/CORON",
        "NIRCAM/TARGACQ",
    ],
    "NIRISS": ["NIRISS/SOSS", "NIRISS/IMAGE", "NIRISS/WFSS", "NIRISS/AMI"],
    "MIRI": [
        "MIRI/SLITLESS", "MIRI/SLIT", "MIRI/IFU", "MIRI/IMAGE", "MIRI/CORON",
        "MIRI/TARGACQ",
    ],
    # FGS is archived unaugmented ("FGS"); "FGS/FGS" kept defensively for any
    # rows that carry the augmented form.
    "FGS": ["FGS", "FGS/FGS"],
    # Convenience shorthand for common exoplanet observing modes.
    "BOTS": ["NIRSPEC/BOTS"],
    "SOSS": ["NIRISS/SOSS"],
    "MRS": ["MIRI/IFU"],
    "LRS": ["MIRI/SLIT", "MIRI/SLITLESS"],
}

# Canonical exoplanet population categories → Exoplanet Archive WHERE conditions.
# Radii use pl_rade (Earth radii); masses use pl_bmasse where needed. Bounds
# follow the commonly cited definitions (e.g. Fulton gap ~1.5-2 R_E, sub-Neptune
# 1.75-4 R_E, giant > 6 R_E). Agents should use these presets instead of
# hand-translating category names — picking the wrong radius column
# (pl_radj vs pl_rade) silently flips the population from sub-Neptunes to hot
# Jupiters.
POPULATION_PRESETS: dict[str, list[str]] = {
    "terrestrial":   ["pl_rade < 1.5"],
    "super_earth":   ["pl_rade >= 1.25", "pl_rade < 2.0"],
    "subneptune":    ["pl_rade >= 1.75", "pl_rade <= 4.0"],
    "sub_neptune":   ["pl_rade >= 1.75", "pl_rade <= 4.0"],
    "neptune":       ["pl_rade > 4.0", "pl_rade <= 6.0"],
    "sub_saturn":    ["pl_rade > 6.0", "pl_rade <= 8.0"],
    "saturn":        ["pl_rade > 8.0", "pl_rade <= 10.0"],
    "jupiter":       ["pl_radj >= 0.8", "pl_radj <= 1.5"],
    "hot_jupiter":   ["pl_radj >= 0.8", "pl_eqt >= 1000"],
    "warm_jupiter":  ["pl_radj >= 0.8", "pl_eqt >= 500", "pl_eqt < 1000"],
    "cold_jupiter":  ["pl_radj >= 0.8", "pl_eqt < 500"],
    "ultra_hot_jupiter": ["pl_radj >= 0.8", "pl_eqt >= 2200"],
    "inflated_jupiter":  ["pl_radj > 1.5"],
    "brown_dwarf":   ["pl_bmassj >= 13", "pl_bmassj <= 80"],
}


def _resolve_population_preset(preset: str | None) -> list[str]:
    """
    Map a categorical population name (e.g. 'subneptune') to ADQL WHERE
    conditions. Returns [] for None/empty; raises ValueError on unknown names.
    """
    if preset is None or not str(preset).strip():
        return []
    key = str(preset).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in POPULATION_PRESETS:
        valid = ", ".join(sorted(set(POPULATION_PRESETS)))
        raise ValueError(
            f"Unknown population_preset {preset!r}. Valid options: {valid}."
        )
    return list(POPULATION_PRESETS[key])


def _mast_query(
    request: dict[str, Any],
    *,
    session: requests.Session | None = None,
    timeout: float = 60.0,
    max_polls: int = 6,
    poll_wait: float = 3.0,
) -> dict[str, Any]:
    """Submit a request to the MAST Mashup API.

    The Mashup API returns HTTP 200 even for failed or still-running queries,
    with the real state in the payload ``status`` field. Previously an
    ``ERROR`` payload (bad column/filter) or an ``EXECUTING`` payload (large
    query still materializing server-side) fell through ``_extract_rows`` as
    an empty list and rendered as "No JWST observations found" — masking real
    failures as empty results. Now: ``ERROR`` raises with the MAST message,
    and ``EXECUTING`` re-polls (MAST caches the query and returns COMPLETE
    once ready, the same strategy astroquery.mast uses).
    """
    client = session or requests
    polls = 0
    while True:
        response = client.post(
            MAST_INVOKE_URL,
            data=f"request={quote(json.dumps(request))}",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()

        status = str(payload.get("status", "COMPLETE")).upper()
        if status == "ERROR":
            raise RuntimeError(
                f"MAST query failed (service={request.get('service')}): "
                f"{payload.get('msg') or 'no error message returned'}"
            )
        if status == "EXECUTING":
            polls += 1
            if polls > max_polls:
                raise TimeoutError(
                    f"MAST query still EXECUTING after {max_polls} polls "
                    f"(service={request.get('service')}). Retry later or "
                    "narrow the query."
                )
            time.sleep(poll_wait)
            continue
        return payload


def _extract_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("data", [])
    if isinstance(rows, list):
        return rows
    return []


def _as_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped[0] in "[(" and stripped[-1:] in "])":
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                return [value]
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
            return [parsed]
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_int_list(value: Any) -> list[int] | None:
    values = _as_list(value)
    if values is None:
        return None
    return [int(v) for v in values]


def _clean_str(value: Any) -> str | None:
    """Normalize LLM-supplied string args: '' / whitespace / None -> None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_float_or_none(value: Any, name: str) -> float | None:
    """Coerce LLM-supplied numeric args ('359.8386', 359.8386, '') to float/None."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}.") from exc


def _as_int_or_none(value: Any, name: str) -> int | None:
    """Coerce LLM-supplied integer args ('10', 10, '') to int/None."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


def _as_bool(value: Any, name: str, default: bool = False) -> bool:
    """Coerce LLM-supplied booleans ('True', 'false', True, '') to bool.

    Guards the ``bool('False') is True`` footgun: string args are matched
    against explicit true/false vocabularies instead of Python truthiness.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    if text in ("true", "1", "yes", "y", "on"):
        return True
    if text in ("false", "0", "no", "n", "off"):
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}.")


def _normalize_jwst_instruments(value: Any) -> list[str] | None:
    values = _as_list(value)
    if values is None:
        return None

    normalized: list[str] = []
    for item in values:
        instrument = str(item).strip()
        if not instrument:
            continue

        key = instrument.upper()
        aliases = JWST_INSTRUMENT_ALIASES.get(key)
        if aliases:
            normalized.extend(aliases)
        else:
            normalized.append(key if "/" in key else instrument)

    return normalized or None


def _sanitize_path_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "target"


def _build_jwst_observation_filters(
    *,
    instruments: list[str] | tuple[str, ...] | str | None = None,
    dataproduct_types: list[str] | tuple[str, ...] | str | None = None,
    calib_levels: list[int] | tuple[int, ...] | int | None = None,
    target_name: str | None = None,
    proposal_id: str | int | None = None,
) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = [
        {"paramName": "obs_collection", "values": ["JWST"]},
    ]

    instrument_values = _normalize_jwst_instruments(instruments)
    if instrument_values:
        filters.append({"paramName": "instrument_name", "values": instrument_values})

    data_values = _as_list(dataproduct_types)
    if data_values:
        filters.append({"paramName": "dataproduct_type", "values": data_values})

    level_values = _as_int_list(calib_levels)
    if level_values:
        filters.append({"paramName": "calib_level", "values": level_values})

    if target_name:
        # Documented Mashup free-text shape: freeText is a sibling of an empty
        # values list ({"paramName": ..., "values": [], "freeText": "%X%"}),
        # not a dict inside values. The previous in-values form was silently
        # ignored or errored server-side.
        filters.append(
            {
                "paramName": "target_name",
                "values": [],
                "freeText": f"%{target_name}%",
            }
        )

    if proposal_id is not None and str(proposal_id).strip() != "":
        filters.append({"paramName": "proposal_id", "values": [str(proposal_id)]})

    return filters


def resolve_target_coordinates(
    target_name: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> tuple[float, float]:
    """
    Resolve a target name to sky coordinates using the MAST name lookup service.

    Planet names are accepted when MAST can resolve them directly. For cases
    where the planet name is not resolvable, pass host-star coordinates directly
    to ``search_jwst_observations``.
    """
    request = {
        "service": "Mast.Name.Lookup",
        "params": {
            "input": target_name,
            "format": "json",
        },
    }
    payload = _mast_query(request, session=session, timeout=timeout)
    resolved = payload.get("resolvedCoordinate")
    if isinstance(resolved, list) and resolved:
        row = resolved[0]
    elif isinstance(resolved, dict):
        row = resolved
    else:
        raise ValueError(f"MAST could not resolve target name '{target_name}'.")

    try:
        return float(row["ra"]), float(row["decl"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"MAST returned malformed coordinates for '{target_name}'.") from exc


def search_all_jwst_observations(
    *,
    instruments: list[str] | tuple[str, ...] | str | None = None,
    dataproduct_types: list[str] | tuple[str, ...] | str | None = None,
    calib_levels: list[int] | tuple[int, ...] | int | None = None,
    target_name: str | None = None,
    proposal_id: str | int | None = None,
    columns: list[str] | tuple[str, ...] | None = None,
    pagesize: int = 50000,
    page: int = 1,
    max_pages: int = 20,
    session: requests.Session | None = None,
    timeout: float = 120.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Filter-only MAST search for JWST observations (no cone / no position).

    Use when you want a population-level / demographics query that scans every
    JWST observation matching the filters, not centered on a single target.
    Returns (rows, filters_used) so callers can echo the filters that produced
    the result set.

    Pagination correctness (IMPORTANT HISTORY)
    ------------------------------------------
    MAST ``Mast.Caom.Filtered`` pagination has NO stable row ordering, so a
    multi-page read can both duplicate and silently MISS rows at page
    boundaries — and differently on every run. Observed live 2026-07-10 on
    the Q2 sub-Neptune demographics query (~171k rows / 4 pages): two
    identical queries returned sets differing by 41 and 5 observations, each
    with dozens of duplicated rows. This — not instrument vocabulary — was
    the root cause of the "missing NIRSpec visits" incidents.

    Defenses, in order:
      1. Rows are always deduplicated by ``obsid`` (unique per CAOM
         observation) while paging.
      2. The Mashup ``paging`` block reports the server-side total
         (``rowsFiltered``/``rowsTotal``). If, after paging, fewer unique
         rows were collected than the server reported, the query is retried
         ONCE as a single page sized to hold the entire result set (a single
         page has no boundaries to lose rows across).
      3. If the count still disagrees, ``RuntimeError`` is raised — an
         incomplete demographics table must never be returned silently.
    """
    selected_columns = _as_list(columns) or list(JWST_OBSERVATION_COLUMNS)
    filters = _build_jwst_observation_filters(
        instruments=instruments,
        dataproduct_types=dataproduct_types,
        calib_levels=calib_levels,
        target_name=target_name,
        proposal_id=proposal_id,
    )

    def _fetch(page_number: int, size: int) -> tuple[list[dict[str, Any]], int | None]:
        request = {
            "service": "Mast.Caom.Filtered",
            "params": {
                "columns": ",".join(selected_columns),
                "filters": filters,
            },
            "format": "json",
            "pagesize": size,
            "page": page_number,
        }
        payload = _mast_query(request, session=session, timeout=timeout)
        rows = _extract_rows(payload)
        total: int | None = None
        paging = payload.get("paging")
        if isinstance(paging, dict):
            for key in ("rowsFiltered", "rowsTotal"):
                value = paging.get(key)
                if isinstance(value, int) and value >= 0:
                    total = value
                    break
        return rows, total

    def _collect(pages: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        seen: set[Any] = set()
        unique: list[dict[str, Any]] = []
        for rows in pages:
            for row in rows:
                key = row.get("obsid")
                if key is None:
                    unique.append(row)  # never drop a row we cannot key
                    continue
                if key in seen:
                    continue
                seen.add(key)
                unique.append(row)
        return unique

    pages: list[list[dict[str, Any]]] = []
    expected_total: int | None = None
    current_page = page
    for _ in range(max_pages):
        rows, total = _fetch(current_page, pagesize)
        if total is not None:
            expected_total = total
        pages.append(rows)
        if len(rows) < pagesize:
            break
        if expected_total is not None and sum(len(p) for p in pages) >= expected_total:
            break
        current_page += 1

    all_rows = _collect(pages)

    if expected_total is not None and len(all_rows) < expected_total:
        # Unstable page ordering dropped rows between pages. Refetch the
        # whole result set as ONE page — no boundaries, no loss.
        if expected_total > 1_000_000:
            raise RuntimeError(
                f"MAST reports {expected_total} matching observations but "
                f"paging only returned {len(all_rows)} unique rows, and the "
                "result set is too large for a single-page retry. Narrow the "
                "filters (e.g. calib_level, dataproduct_type) and rerun."
            )
        retry_rows, retry_total = _fetch(1, expected_total + 1000)
        if retry_total is not None:
            expected_total = retry_total
        all_rows = _collect([retry_rows])
        # Judge completeness by rows RECEIVED, not unique rows: MAST can
        # return one obsid twice and count both copies in its total (live
        # 2026-10-15: NIRISS timeseries, 306 rows / 305 unique, obsid
        # 337745841 repeated verbatim). A full-length single page with fewer
        # unique rows is duplication; nothing is missing.
        if expected_total is not None and len(retry_rows) < expected_total:
            raise RuntimeError(
                f"MAST pagination returned an incomplete result set: server "
                f"reports {expected_total} observations, received "
                f"{len(all_rows)} unique rows even after a single-page "
                "retry. Refusing to return silently-incomplete demographics "
                "data — rerun or narrow the filters."
            )

    return all_rows, filters


def search_jwst_observations(
    planet_name: str | None,
    *,
    ra: float | None = None,
    dec: float | None = None,
    radius_deg: float = 0.02,
    instruments: list[str] | tuple[str, ...] | str | None = None,
    dataproduct_types: list[str] | tuple[str, ...] | str | None = None,
    calib_levels: list[int] | tuple[int, ...] | int | None = None,
    target_name_filter: bool = False,
    proposal_id: str | int | None = None,
    columns: list[str] | tuple[str, ...] | None = None,
    pagesize: int = 2000,
    page: int = 1,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """
    Search MAST for JWST observations around an individual planet target.

    MAST observations are usually filed under host-star target names, so the
    default search is coordinate-centered. Set ``target_name_filter=True`` when
    the MAST target name is known to contain the planet name.
    """
    if (ra is None) != (dec is None):
        raise ValueError("Pass both ra and dec, or pass neither.")

    if ra is None or dec is None:
        if not planet_name or not str(planet_name).strip():
            raise ValueError(
                "Provide a planet_name for name resolution, or pass ra and dec."
            )
        ra, dec = resolve_target_coordinates(
            planet_name,
            session=session,
            timeout=timeout,
        )

    selected_columns = _as_list(columns) or list(JWST_OBSERVATION_COLUMNS)
    filters = _build_jwst_observation_filters(
        instruments=instruments,
        dataproduct_types=dataproduct_types,
        calib_levels=calib_levels,
        target_name=planet_name if target_name_filter else None,
        proposal_id=proposal_id,
    )

    request = {
        "service": "Mast.Caom.Filtered.Position",
        "params": {
            "columns": ",".join(selected_columns),
            "filters": filters,
            "position": f"{ra}, {dec}, {radius_deg}",
        },
        "format": "json",
        "pagesize": pagesize,
        "page": page,
    }

    return _extract_rows(_mast_query(request, session=session, timeout=timeout))


def _instrument_matches(row_value: Any, requested: list[Any]) -> bool:
    """Prefix-tolerant instrument match: 'NIRSpec' matches 'NIRSPEC/BOTS'."""
    if not requested:
        return True
    value = str(row_value or "").upper()
    for item in requested:
        want = str(item).strip().upper()
        if not want:
            continue
        if "/" in want:
            if value == want:
                return True
        elif value == want or value.startswith(want + "/"):
            return True
    return False


def _row_matches_filters(
    row: dict[str, Any],
    *,
    instruments: list[Any] | None,
    dataproduct_types: list[Any] | None,
    calib_levels: list[int] | None,
    proposal_id: str | None,
) -> bool:
    """Client-side equivalent of the server-side observation filters."""
    if instruments and not _instrument_matches(row.get("instrument_name"), instruments):
        return False
    if dataproduct_types:
        wanted = {str(v).strip().lower() for v in dataproduct_types}
        if str(row.get("dataproduct_type") or "").strip().lower() not in wanted:
            return False
    if calib_levels:
        try:
            level = int(row.get("calib_level"))
        except (TypeError, ValueError):
            return False
        if level not in calib_levels:
            return False
    if proposal_id is not None:
        if str(row.get("proposal_id") or "").strip() != str(proposal_id).strip():
            return False
    return True


def _summarize_distinct(
    rows: list[dict[str, Any]], key: str, limit: int = 12
) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) if row.get(key) is not None else "?")
        counts[value] = counts.get(value, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    body = ", ".join(f"{name} (x{count})" for name, count in ranked[:limit])
    if len(ranked) > limit:
        body += f", ... {len(ranked) - limit} more"
    return body or "none"


def _diagnose_empty_cone_search(
    planet_name: str | None,
    *,
    ra: float,
    dec: float,
    radius_deg: float,
    instruments: list[Any] | None,
    dataproduct_types: list[Any] | None,
    calib_levels: list[int] | None,
    proposal_id: str | None,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> tuple[list[dict[str, Any]], str]:
    """
    Self-diagnosis after a filtered cone search returns 0 rows.

    Runs ONE unfiltered JWST cone query at the same position and re-applies
    the requested filters client-side with prefix-tolerant instrument
    matching. This recovers observations lost to instrument_name vocabulary
    drift (exact-match server filters silently drop modes missing from
    ``JWST_INSTRUMENT_ALIASES``) and, when nothing matches, reports what the
    cone actually contains so the caller can fix the query instead of
    concluding "no data". If the cone is empty and a proposal_id was given, a
    no-position proposal probe reports where that program's observations are.

    Returns (recovered_rows, diagnostics_text).
    """
    try:
        cone_rows = search_jwst_observations(
            planet_name,
            ra=ra,
            dec=dec,
            radius_deg=radius_deg,
            session=session,
            timeout=timeout,
        )
    except Exception as exc:  # diagnostics must never replace the primary answer
        return [], f"(Diagnostic relaxed cone query failed: {exc})"

    if not cone_rows:
        lines = [
            "Diagnostics: no JWST observations of ANY kind within "
            f"{radius_deg} deg of RA={ra}, Dec={dec}. The filters are not the "
            "problem — re-check the coordinates and radius."
        ]
        if proposal_id:
            try:
                program_rows, _ = search_all_jwst_observations(
                    proposal_id=proposal_id,
                    session=session,
                    timeout=timeout,
                    pagesize=2000,
                    max_pages=1,
                )
            except Exception as exc:
                program_rows = []
                lines.append(f"(No-position probe for proposal {proposal_id} failed: {exc})")
            if program_rows:
                lines.append(
                    f"Proposal {proposal_id} DOES exist in MAST with "
                    f"{len(program_rows)} JWST observation(s) — "
                    f"targets: {_summarize_distinct(program_rows, 'target_name')}; "
                    f"instruments: {_summarize_distinct(program_rows, 'instrument_name')}. "
                    "Re-check your ra/dec, or search by proposal_id alone "
                    "(omit planet_name/ra/dec)."
                )
            else:
                lines.append(
                    f"A no-position search for proposal {proposal_id} also "
                    "returned nothing; the proposal id itself may be wrong."
                )
        return [], "\n".join(lines)

    matched = [
        row
        for row in cone_rows
        if _row_matches_filters(
            row,
            instruments=instruments,
            dataproduct_types=dataproduct_types,
            calib_levels=calib_levels,
            proposal_id=proposal_id,
        )
    ]
    if matched:
        return matched, (
            f"Note: the strict server-side query returned 0 rows, but {len(matched)} "
            "observation(s) in the cone match your criteria. They were recovered by "
            "an unfiltered cone search + client-side matching — the MAST "
            "instrument_name vocabulary likely contains a mode missing from "
            "JWST_INSTRUMENT_ALIASES (worth updating the table)."
        )

    return [], "\n".join(
        [
            f"Diagnostics: the cone DOES contain {len(cone_rows)} JWST "
            "observation(s), but none match your filters. Present in cone:",
            f"  instrument_name:  {_summarize_distinct(cone_rows, 'instrument_name')}",
            f"  dataproduct_type: {_summarize_distinct(cone_rows, 'dataproduct_type')}",
            f"  calib_level:      {_summarize_distinct(cone_rows, 'calib_level')}",
            f"  proposal_id:      {_summarize_distinct(cone_rows, 'proposal_id')}",
            "Relax the mismatched filter(s) and retry.",
        ]
    )


def get_observation_products(
    obsid: str | int,
    *,
    product_types: list[str] | tuple[str, ...] | str | None = SCIENCE_PRODUCT_TYPES,
    product_subgroups: list[str] | tuple[str, ...] | str | None = None,
    raw_only: bool = False,
    extensions: list[str] | tuple[str, ...] | str | None = FITS_EXTENSIONS,
    skip_proprietary: bool = True,
    product_dataproduct_types: list[str] | tuple[str, ...] | str | None = None,
    drop_auxiliary_exposures: bool = True,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """Return downloadable MAST products for one observation id.

    ``drop_auxiliary_exposures`` (default) drops exposures taken through a
    different optical element than the observation itself — in practice
    target acquisition, e.g. NIRSpec ``F110W;MIRROR`` exposures listed under
    a ``F290LP;G395H`` time-series observation. See
    ``_observation_level_filters``.
    """
    request = {
        "service": "Mast.Caom.Products",
        "params": {"obsid": str(obsid)},
        "format": "json",
    }
    products = _extract_rows(_mast_query(request, session=session, timeout=timeout))
    return filter_products(
        products,
        product_types=product_types,
        product_subgroups=product_subgroups,
        raw_only=raw_only,
        extensions=extensions,
        skip_proprietary=skip_proprietary,
        product_dataproduct_types=product_dataproduct_types,
        exposure_filters=(_observation_level_filters(products)
                          if drop_auxiliary_exposures else None),
    )


# Observation-level (stage 3) data products: jw01366-o003_t001_..., also
# association/candidate forms (-a3001, -c1000) and source ids (_s00001_).
# Per-exposure names (jw01366003001_04101_...) never match, and neither do
# the association files (jw01366-o003_20261011t215903_image2_00001_asn.json),
# which MAST lists for the TA exposures too, with the TA's filters.
_OBSERVATION_LEVEL_NAME_RE = re.compile(r"^jw\d{5}-[a-z]\d{3,}_[ts]\d{3,}_", re.IGNORECASE)


def _observation_level_filters(products: list[dict[str, Any]]) -> str | None:
    """The observation's own optical elements, read off its stage-3 products.

    MAST lists a time-series observation's target-acquisition exposures in
    its product list with ``dataproduct_type='timeseries'`` — the SAME type
    as the science segments (verified live 2026-10-15, ERS 1366 obsid
    233644122: ``_02101_`` TA uncals 60-72 kB, ``F110W;MIRROR``; ``_04101_``
    science 1.4 GB, ``F290LP;G395H``). So the type cannot separate them, and
    exoTEDRF's ``download_observations`` (which filters on exactly that)
    downloads the TA files too. Each product row does carry its own
    ``filters``, and the observation-level products carry the
    observation's. Returns None — no filtering — when there are no
    observation-level products or they disagree, so an unprocessed
    observation keeps every exposure rather than losing its science.
    """
    values = {
        str(p.get("filters") or "").strip().upper()
        for p in products
        if _OBSERVATION_LEVEL_NAME_RE.match(str(p.get("productFilename") or ""))
        and str(p.get("productType") or "").upper() != "INFO"
        and str(p.get("filters") or "").strip()
    }
    return values.pop() if len(values) == 1 else None


def filter_products(
    products: list[dict[str, Any]],
    *,
    product_types: list[str] | tuple[str, ...] | str | None = SCIENCE_PRODUCT_TYPES,
    product_subgroups: list[str] | tuple[str, ...] | str | None = None,
    raw_only: bool = False,
    extensions: list[str] | tuple[str, ...] | str | None = FITS_EXTENSIONS,
    skip_proprietary: bool = True,
    product_dataproduct_types: list[str] | tuple[str, ...] | str | None = None,
    exposure_filters: str | None = None,
) -> list[dict[str, Any]]:
    """Filter MAST products by science type, JWST subgroup, raw status, file
    extension, and proprietary-access status.

    ``exposure_filters`` (e.g. 'F290LP;G395H') drops products whose own
    ``filters`` lack any of its tokens — the target-acquisition exposures
    of a time-series observation. Products without the field pass.

    ``product_dataproduct_types`` filters on each PRODUCT's own
    ``dataproduct_type`` (e.g. ['timeseries']). A JWST time-series
    observation's product list carries everything typed 'timeseries' —
    including its target-acquisition exposures, so this does NOT remove TA
    (``exposure_filters`` does). Products without the field pass.

    ``skip_proprietary=True`` (default) drops products whose ``dataRights``
    is not ``PUBLIC`` — typically ``EXCLUSIVE_ACCESS`` files that would
    otherwise return HTTP 401 from the MAST download endpoint. Set to False
    only when the caller is authenticated with appropriate MAST credentials.
    """
    type_values = {value.upper() for value in _as_list(product_types) or []}
    subgroup_values = {value.upper() for value in _as_list(product_subgroups) or []}
    extension_values = tuple(value.lower() for value in (_as_list(extensions) or []))
    dptype_values = {
        str(v).lower() for v in (_as_list(product_dataproduct_types) or []) if str(v).strip()
    }

    if raw_only and not subgroup_values:
        subgroup_values = set(RAW_PRODUCT_SUBGROUPS)

    selected: list[dict[str, Any]] = []
    for product in products:
        product_type = str(product.get("productType", "")).upper()
        subgroup = str(product.get("productSubGroupDescription", "")).upper()
        filename = str(product.get("productFilename", "")).lower()
        data_rights = str(product.get("dataRights", "")).upper()

        if type_values and product_type not in type_values:
            continue

        if subgroup_values and subgroup not in subgroup_values:
            continue

        if raw_only and not is_raw_jwst_product(product):
            continue

        if extension_values and not filename.endswith(extension_values):
            continue

        if skip_proprietary and data_rights and data_rights != "PUBLIC":
            continue

        product_dptype = str(product.get("dataproduct_type") or "").lower()
        if dptype_values and product_dptype and product_dptype not in dptype_values:
            continue

        if (exposure_filters and product.get("filters")
                and not _filters_match(product.get("filters"), [exposure_filters])):
            continue

        selected.append(product)

    return selected


# One Mast.Caom.Products request lists the products of this many
# observations (comma-separated obsids; each product names its observation
# in ``parent_obsid``). Above RAW_CHECK_MAX_OBS the default raw check is
# skipped rather than turning a broad search into minutes of requests.
RAW_CHECK_BATCH = 50
RAW_CHECK_MAX_OBS = 1000
RAW_ROW_KEYS = frozenset({"raw_data", "raw_files", "raw_public_files", "release_date"})


def raw_data_by_obsid(
    obsids: Any,
    *,
    session: requests.Session | None = None,
    timeout: float = 120.0,
) -> dict[str, dict[str, int]]:
    """Raw (UNCAL) science files per observation, and how many are public.

    Counts the files ``DownloadMastJwstProducts(raw_only=True)`` would fetch
    (science UNCAL FITS, target acquisition dropped), but keeps exclusive-
    access ones so a proprietary observation reads as "not yet public"
    rather than "no raw data". One request per RAW_CHECK_BATCH observations.
    """
    ids = list(dict.fromkeys(str(o) for o in (obsids or []) if o not in (None, "")))
    result: dict[str, dict[str, int]] = {}
    for start in range(0, len(ids), RAW_CHECK_BATCH):
        chunk = ids[start:start + RAW_CHECK_BATCH]
        request = {
            "service": "Mast.Caom.Products",
            "params": {"obsid": ",".join(chunk)},
            "format": "json",
        }
        products = _extract_rows(_mast_query(request, session=session, timeout=timeout))
        by_obs: dict[str, list[dict[str, Any]]] = {}
        for product in products:
            parent = product.get("parent_obsid")
            if parent in (None, ""):
                parent = chunk[0] if len(chunk) == 1 else product.get("obsID")
            by_obs.setdefault(str(parent), []).append(product)
        for obsid in chunk:
            listed = by_obs.get(obsid, [])
            raw = filter_products(
                listed, raw_only=True, skip_proprietary=False,
                exposure_filters=_observation_level_filters(listed),
            )
            result[obsid] = {
                "raw_files": len(raw),
                "raw_public_files": sum(
                    1 for p in raw if str(p.get("dataRights") or "").upper() == "PUBLIC"
                ),
            }
    return result


def _mjd_to_date(value: Any) -> str | None:
    try:
        mjd = float(value)
    except (TypeError, ValueError):
        return None
    if mjd != mjd:  # NaN
        return None
    return (datetime(1858, 11, 17) + timedelta(days=mjd)).date().isoformat()


def annotate_rows_with_raw_data(rows: list[dict[str, Any]]) -> list[str]:
    """Add raw_data / raw_files / raw_public_files / release_date in place.

    ``raw_data`` is 'public' (every raw file public), 'exclusive_access'
    (none yet), 'partly_public', or 'none' (MAST lists no UNCAL science
    files). Returns summary lines for the tool output.
    """
    obsids = list(dict.fromkeys(str(r.get("obsid")) for r in rows
                                if r.get("obsid") not in (None, "")))
    if len(obsids) > RAW_CHECK_MAX_OBS:
        return [f"Raw (UNCAL) data not checked: {len(obsids)} observations is more "
                f"than {RAW_CHECK_MAX_OBS}. Narrow the search (instruments, filters, "
                "population_preset, proposal_id) to see which have raw data."]
    info = raw_data_by_obsid(obsids)
    for row in rows:
        counts = info.get(str(row.get("obsid")), {"raw_files": 0, "raw_public_files": 0})
        total, public = counts["raw_files"], counts["raw_public_files"]
        row["raw_files"] = total
        row["raw_public_files"] = public
        row["raw_data"] = ("none" if total == 0 else "public" if public == total
                           else "exclusive_access" if public == 0 else "partly_public")
        row["release_date"] = _mjd_to_date(row.get("t_obs_release"))

    per_obs = {str(r.get("obsid")): r for r in rows if r.get("obsid") not in (None, "")}
    status: dict[str, int] = {}
    for row in per_obs.values():
        status[row["raw_data"]] = status.get(row["raw_data"], 0) + 1
    public_files = sum(r["raw_public_files"] for r in per_obs.values())
    lines = [
        f"Raw (UNCAL) data: {status.get('public', 0)} of {len(per_obs)} observations "
        f"public ({public_files} raw files); {status.get('partly_public', 0)} partly "
        f"public; {status.get('exclusive_access', 0)} exclusive access; "
        f"{status.get('none', 0)} with no UNCAL files listed."
    ]
    pending = sorted(
        {r["release_date"] for r in per_obs.values()
         if r["raw_data"] in ("exclusive_access", "partly_public") and r["release_date"]}
    )
    if pending:
        lines.append(f"  Exclusive-access release dates: {', '.join(pending[:8])}"
                     + (" ..." if len(pending) > 8 else ""))
    planets = {r.get("pl_name") for r in rows if r.get("pl_name")}
    if planets:
        with_raw = {r.get("pl_name") for r in rows
                    if r.get("pl_name") and r["raw_data"] in ("public", "partly_public")}
        lines.append(f"Distinct planets: {len(planets)}; with public raw data: {len(with_raw)}.")
    return lines


def is_raw_jwst_product(product: dict[str, Any]) -> bool:
    """Return True for JWST uncalibrated/raw products."""
    subgroup = str(product.get("productSubGroupDescription", "")).upper()
    filename = str(product.get("productFilename", "")).lower()
    data_uri = str(product.get("dataURI", "")).lower()
    return (
        subgroup in RAW_PRODUCT_SUBGROUPS
        or filename.endswith("_uncal.fits")
        or filename.endswith("_uncal.fits.gz")
        or "_uncal." in data_uri
    )


class ProprietaryProductError(RuntimeError):
    """Raised when the MAST download endpoint refuses an exclusive-access file.

    Carries the offending dataURI so batch downloaders can log and skip
    without aborting the whole run.
    """

    def __init__(self, data_uri: str, status_code: int) -> None:
        super().__init__(
            f"MAST refused proprietary product (HTTP {status_code}): {data_uri}"
        )
        self.data_uri = data_uri
        self.status_code = status_code


def download_mast_product(
    data_uri: str,
    output_directory: str | os.PathLike[str],
    *,
    filename: str | None = None,
    session: requests.Session | None = None,
    timeout: float | tuple[float, float] = (30.0, 600.0),
    retries: int = 6,
    retry_backoff: float = 5.0,
) -> Path:
    """Download a single MAST product by dataURI and return the local path.

    ``timeout`` is a ``(connect, read)`` tuple by default — 30 s to establish
    the TCP/TLS connection, 600 s of inactivity tolerance per-chunk while the
    file streams. Large JWST products (NIRSpec X1DINTS, UNCAL) routinely take
    minutes to start streaming; the previous 120 s flat timeout caused false
    failures.

    On a transient network error (``ConnectionError``, ``Timeout``,
    ``ChunkedEncodingError``, ``ProtocolError``/``IncompleteRead``) the function
    retries up to ``retries`` times with a linear backoff of ``retry_backoff``
    seconds. Each retry **resumes** from the bytes already on disk via an HTTP
    ``Range`` request (like ``wget -c``) instead of restarting at byte 0, so a
    multi-GB UNCAL ramp that drops mid-transfer eventually completes across
    several flaky attempts. If MAST ignores the ``Range`` header (returns 200
    instead of 206) the file is restarted cleanly.

    Raises ``ProprietaryProductError`` if MAST returns 401/403 (file is
    proprietary / under exclusive-access embargo). Callers can catch this
    to skip and continue.
    """
    if not data_uri:
        raise ValueError("data_uri is required.")

    client = session or requests
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    local_name = filename or os.path.basename(data_uri)
    if not local_name:
        raise ValueError("Could not infer a filename from data_uri.")

    destination = output_path / local_name

    # MAST drops on large files surface as urllib3 ProtocolError/IncompleteRead,
    # which requests usually (but not always) re-wraps as ChunkedEncodingError.
    # Catch the raw urllib3 forms too so a mid-stream truncation triggers a
    # resume rather than aborting the whole download.
    try:
        from urllib3.exceptions import ProtocolError as _ProtocolError
        from urllib3.exceptions import IncompleteRead as _U3IncompleteRead
        _extra_transient: tuple[type[BaseException], ...] = (
            _ProtocolError, _U3IncompleteRead,
        )
    except Exception:  # pragma: no cover - urllib3 always present with requests
        _extra_transient = ()
    transient_errors = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
    ) + _extra_transient

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        # Resume from whatever already landed on disk.
        resume_pos = destination.stat().st_size if destination.exists() else 0
        headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}
        try:
            response = client.get(
                MAST_DOWNLOAD_URL,
                params={"uri": data_uri},
                headers=headers,
                stream=True,
                timeout=timeout,
            )
            if response.status_code in (401, 403):
                response.close()
                raise ProprietaryProductError(data_uri, response.status_code)

            # A resumed request whose start offset sits at/beyond EOF returns
            # 416 (Requested Range Not Satisfiable). Per wget -c semantics that
            # means the file already fully landed on disk from a prior run.
            # A 416 SHOULD carry "Content-Range: bytes */<total>"; verify the
            # on-disk size against it. Match (or header absent) => complete,
            # so return it. Mismatch => local file is corrupt/oversized, so
            # delete and let the loop restart the download cleanly from byte 0.
            if resume_pos and response.status_code == 416:
                content_range = response.headers.get("Content-Range", "")
                response.close()
                tail = content_range.rsplit("/", 1)[-1].strip()
                remote_total = int(tail) if tail.isdigit() else None
                if remote_total is None or destination.stat().st_size == remote_total:
                    return destination
                destination.unlink(missing_ok=True)
                continue

            response.raise_for_status()

            # 206 => server honored the Range, append. Anything else (200) =>
            # it sent the whole file, so overwrite from the start.
            if resume_pos and response.status_code == 206:
                mode = "ab"
            else:
                mode = "wb"
                resume_pos = 0

            # Total expected size, to catch a silent short read.
            expected_total: int | None = None
            content_range = response.headers.get("Content-Range")
            if content_range and "/" in content_range:
                try:
                    expected_total = int(content_range.rsplit("/", 1)[1])
                except ValueError:
                    expected_total = None
            elif response.headers.get("Content-Length") is not None:
                try:
                    expected_total = resume_pos + int(response.headers["Content-Length"])
                except ValueError:
                    expected_total = None

            with destination.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

            # A truncated stream that didn't raise still leaves a short file;
            # treat that as transient so the next attempt resumes the rest.
            if expected_total is not None and destination.stat().st_size < expected_total:
                raise requests.exceptions.ChunkedEncodingError(
                    f"Incomplete download: {destination.stat().st_size} of "
                    f"{expected_total} bytes for {local_name}."
                )
            return destination
        except ProprietaryProductError:
            raise
        except transient_errors as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(retry_backoff * attempt)
                continue
            raise

    # Unreachable — loop either returns or re-raises — but appease type checkers.
    raise RuntimeError(f"Download retry loop exited without result: {last_exc!r}")


# ---- exoTEDRF-style selection: grating/filter tokens and visit numbers ----

_VISIT_PREFIX_RE = re.compile(r"^jw\d{11}$")


def _filter_tokens(value: Any) -> set[str]:
    """'F290LP;G395H' -> {'F290LP', 'G395H'} (CAOM joins elements with ';')."""
    return {t for t in re.split(r"[;,/\s]+", str(value or "").upper()) if t}


def _is_filter_wheel_token(token: str) -> bool:
    """'F290LP', 'F322W2', 'CLEAR' are filter-wheel elements; 'G395H',
    'PRISM', 'GR700XD', 'GRISMR', 'P750L' are dispersers / pupils."""
    return bool(re.fullmatch(r"F\d{3}[A-Z0-9]*", token)) or token in {"CLEAR", "OPAQUE"}


def _filters_match(row_value: Any, requested: list[Any] | None) -> bool:
    """True if the row's CAOM filters satisfy the requested tokens.

    ``['G395H']`` matches 'F290LP;G395H'; ``['G395H', 'G395M']`` matches
    either grating; ``['F290LP', 'G395H']`` requires BOTH, because single
    tokens are OR-ed only within their kind (filter-wheel vs disperser) and
    AND-ed across kinds — treating that list as any-of let every G395M row
    in through its F290LP. A joined value (``'F290LP;G395H'``) is an exact
    combination that matches on its own. Applied client-side on purpose:
    MAST discrete filters are exact-match on the full joined string, so a
    server-side 'G395H' would silently drop 'F290LP;G395H' rows.
    """
    if not requested:
        return True
    have = _filter_tokens(row_value)
    singles: dict[bool, set[str]] = {}
    for value in requested:
        tokens = _filter_tokens(value)
        if len(tokens) > 1 and tokens <= have:
            return True
        if len(tokens) == 1:
            (token,) = tokens
            singles.setdefault(_is_filter_wheel_token(token), set()).add(token)
    return bool(singles) and all(group & have for group in singles.values())


def _visit_prefix(product: dict[str, Any]) -> str | None:
    """jwPPPPPOOOVVV visit prefix of a per-exposure product filename."""
    head = str(product.get("productFilename") or "").split("_", 1)[0].lower()
    return head if _VISIT_PREFIX_RE.match(head) else None


def _normalize_visits(visits: Any) -> list[Any] | None:
    """LLM input -> list of 1-based ints and/or 'jw...' visit prefixes."""
    values = _as_list(visits)
    if not values:
        return None
    out: list[Any] = []
    for v in values:
        text = str(v).strip().lower()
        if not text:
            continue
        if text.startswith("jw"):
            if not _VISIT_PREFIX_RE.match(text):
                raise ValueError(
                    f"Visit prefix {v!r} must look like 'jw01366004001' "
                    "(jw + 5-digit program + 3-digit observation + 3-digit visit)."
                )
            out.append(text)
        else:
            try:
                number = int(float(text))
            except ValueError:
                raise ValueError(
                    f"visits entries must be 1-based numbers or 'jw...' prefixes, got {v!r}."
                ) from None
            if number < 1:
                raise ValueError("Visit numbers are 1-based.")
            out.append(number)
    return out or None


def select_products_by_visit(
    products_by_obs: list[tuple[Any, list[dict[str, Any]]]],
    visits: Any,
) -> tuple[list[tuple[Any, list[dict[str, Any]]]], list[str], list[str]]:
    """Keep only products from the requested visits (exoTEDRF ``visit_nos``).

    Visit numbers are 1-based positions in the SORTED list of distinct
    ``jwPPPPPOOOVVV`` prefixes across ALL selected products. exoTEDRF's
    ``download_observations`` numbers visits in the order MAST happens to
    return products instead; that usually matches the sorted order but is
    not guaranteed, so this numbering is the deterministic version of it.
    Literal prefixes are accepted too and are unambiguous — prefer them for
    anything that must be reproducible. Products without a per-visit
    prefix (observation-level stage-3 files) are dropped when a visit is
    requested. Returns (filtered products_by_obs, all prefixes, selected).
    """
    wanted = _normalize_visits(visits)
    all_prefixes = sorted({
        p for _, products in products_by_obs for p in map(_visit_prefix, products) if p
    })
    if wanted is None:
        return products_by_obs, all_prefixes, all_prefixes
    selected: list[str] = []
    for v in wanted:
        if isinstance(v, int):
            if v > len(all_prefixes):
                raise ValueError(
                    f"You are trying to retrieve visit {v}, but only "
                    f"{len(all_prefixes)} visit(s) exist for this selection: "
                    f"{all_prefixes}."
                )
            prefix = all_prefixes[v - 1]
        else:
            if v not in all_prefixes:
                raise ValueError(
                    f"Visit {v} is not in this selection. Available: {all_prefixes}."
                )
            prefix = v
        if prefix not in selected:
            selected.append(prefix)
    keep = set(selected)
    filtered = [
        (key, [p for p in products if _visit_prefix(p) in keep])
        for key, products in products_by_obs
    ]
    return filtered, all_prefixes, selected


def _download_product_sets(
    products_by_obs: list[tuple[Any, list[dict[str, Any]]]],
    target_dir: Path,
    *,
    dir_name: Any,
    session: requests.Session | None,
    download_timeout: float | tuple[float, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Download (key, products) pairs; return (downloaded, skipped_proprietary).

    ``dir_name(key)`` names the per-observation subdirectory; ``key`` is
    echoed into each manifest entry by the caller-supplied ``entry`` dict.
    """
    downloaded: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for key, products in products_by_obs:
        obs_dir = target_dir / _sanitize_path_component(str(dir_name(key)))
        for product in products:
            data_uri = product.get("dataURI")
            if not data_uri:
                continue
            try:
                local_path = download_mast_product(
                    str(data_uri),
                    obs_dir,
                    filename=product.get("productFilename"),
                    session=session,
                    timeout=download_timeout,
                )
            except ProprietaryProductError as exc:
                skipped.append({"key": key, "product": product, "reason": str(exc)})
                continue
            downloaded.append({"key": key, "product": product, "path": str(local_path)})
    return downloaded, skipped


def download_observations_products(
    obsids: list[str | int] | tuple[str | int, ...],
    output_directory: str | os.PathLike[str],
    *,
    product_types: list[str] | tuple[str, ...] | str | None = SCIENCE_PRODUCT_TYPES,
    product_subgroups: list[str] | tuple[str, ...] | str | None = None,
    raw_only: bool = False,
    extensions: list[str] | tuple[str, ...] | str | None = FITS_EXTENSIONS,
    max_products_per_obs: int | None = None,
    visits: Any = None,
    obs_dir_names: dict[str, str] | None = None,
    product_dataproduct_types: list[str] | tuple[str, ...] | str | None = None,

    session: requests.Session | None = None,
    timeout: float = 120.0,
    download_timeout: float | tuple[float, float] = (30.0, 600.0),
    label: str = "aggregate",
) -> dict[str, Any]:
    """
    Batch-download products for a fixed list of MAST obsids.

    Writes files under ``{output_directory}/{label}/{obs_dir}/`` (``obs_dir``
    is ``obs_dir_names[obsid]`` when given, else the obsid) and a
    ``manifest.json`` capturing every download. ``visits`` restricts to
    exoTEDRF-numbered visits (see ``select_products_by_visit``); products
    are listed for every obsid first so the numbering is global.
    """
    target_dir = Path(output_directory) / _sanitize_path_component(label)
    target_dir.mkdir(parents=True, exist_ok=True)

    obsid_list = [str(o) for o in obsids if o is not None]
    products_by_obs = [
        (
            obsid,
            get_observation_products(
                obsid,
                product_types=product_types,
                product_subgroups=product_subgroups,
                raw_only=raw_only,
                extensions=extensions,
                product_dataproduct_types=product_dataproduct_types,
                session=session,
                timeout=timeout,
            ),
        )
        for obsid in obsid_list
    ]
    products_by_obs, all_visits, selected_visits = select_products_by_visit(
        products_by_obs, visits
    )
    if max_products_per_obs is not None:
        products_by_obs = [(k, p[:max_products_per_obs]) for k, p in products_by_obs]

    names = obs_dir_names or {}
    downloaded, skipped = _download_product_sets(
        products_by_obs,
        target_dir,
        dir_name=lambda obsid: names.get(obsid) or obsid,
        session=session,
        download_timeout=download_timeout,
    )
    for entry in downloaded + skipped:
        entry["obsid"] = entry.pop("key")

    manifest = {
        "label": label,
        "obsids": obsid_list,
        "downloaded": downloaded,
        "skipped_proprietary": skipped,
    }
    if visits is not None:
        manifest["visits_available"] = all_visits
        manifest["visits_selected"] = selected_visits

    with (target_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest


def download_planet_jwst_products(
    planet_name: str,
    output_directory: str | os.PathLike[str],
    *,
    ra: float | None = None,
    dec: float | None = None,
    radius_deg: float = 0.02,
    instruments: list[str] | tuple[str, ...] | str | None = None,
    dataproduct_types: list[str] | tuple[str, ...] | str | None = None,
    calib_levels: list[int] | tuple[int, ...] | int | None = None,
    proposal_id: str | int | None = None,
    filters: list[str] | tuple[str, ...] | str | None = None,
    product_types: list[str] | tuple[str, ...] | str | None = SCIENCE_PRODUCT_TYPES,
    product_subgroups: list[str] | tuple[str, ...] | str | None = None,
    raw_only: bool = False,
    extensions: list[str] | tuple[str, ...] | str | None = FITS_EXTENSIONS,
    max_observations: int | None = None,
    max_products: int | None = None,
    visits: Any = None,
    session: requests.Session | None = None,
    timeout: float = 120.0,
    download_timeout: float | tuple[float, float] = (30.0, 600.0),
) -> dict[str, Any]:
    """
    Search for JWST observations of one planet target and download selected products.

    ``proposal_id`` / ``instruments`` / ``dataproduct_types`` filter the MAST
    query; ``filters`` (e.g. 'G395H') and ``visits`` select client-side, as
    exoTEDRF's ``download_observations`` does. The returned manifest is
    JSON-serializable and records the observation rows, product metadata,
    and local file paths.
    """
    observations = search_jwst_observations(
        planet_name,
        ra=ra,
        dec=dec,
        radius_deg=radius_deg,
        instruments=instruments,
        dataproduct_types=dataproduct_types,
        calib_levels=calib_levels,
        proposal_id=proposal_id,
        session=session,
        timeout=timeout,
    )
    requested_filters = _as_list(filters)
    observations = [
        o for o in observations if _filters_match(o.get("filters"), requested_filters)
    ]
    if max_observations is not None:
        observations = observations[:max_observations]

    target_dir = Path(output_directory) / _sanitize_path_component(planet_name)
    products_by_obs = [
        (
            index,
            get_observation_products(
                observation["obsid"],
                product_types=product_types,
                product_subgroups=product_subgroups,
                raw_only=raw_only,
                extensions=extensions,
                product_dataproduct_types=dataproduct_types,
                session=session,
                timeout=timeout,
            ),
        )
        for index, observation in enumerate(observations)
        if observation.get("obsid") is not None
    ]
    products_by_obs, all_visits, selected_visits = select_products_by_visit(
        products_by_obs, visits
    )
    if max_products is not None:
        products_by_obs = [(k, p[:max_products]) for k, p in products_by_obs]

    downloaded, skipped = _download_product_sets(
        products_by_obs,
        target_dir,
        dir_name=lambda i: observations[i].get("obs_id") or observations[i]["obsid"],
        session=session,
        download_timeout=download_timeout,
    )
    for entry in downloaded + skipped:
        entry["observation"] = observations[entry.pop("key")]

    manifest = {
        "planet_name": planet_name,
        "observations": observations,
        "downloaded": downloaded,
        "skipped_proprietary": skipped,
    }
    if visits is not None:
        manifest["visits_available"] = all_visits
        manifest["visits_selected"] = selected_visits

    target_dir.mkdir(parents=True, exist_ok=True)
    with (target_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest


def _format_filters_block(
    filters: list[dict[str, Any]] | None,
    *,
    extra: dict[str, Any] | None = None,
) -> str:
    """Echo the MAST filters and query parameters used for a search."""
    if not filters and not extra:
        return ""

    lines = ["Filters used:"]
    for entry in filters or []:
        param = entry.get("paramName", "?")
        values = entry.get("values", [])
        rendered: list[str] = []
        for value in values:
            if isinstance(value, dict) and "freeText" in value:
                rendered.append(f"freeText='{value['freeText']}'")
            else:
                rendered.append(str(value))
        if entry.get("freeText"):
            rendered.append(f"freeText='{entry['freeText']}'")
        lines.append(f"  - {param}: [{', '.join(rendered)}]")
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            lines.append(f"  - {key}: {value}")
    return "\n".join(lines) + "\n"


def _format_observations_summary(
    rows: list[dict[str, Any]],
    limit: int = 50,
    *,
    filters: list[dict[str, Any]] | None = None,
    query_extra: dict[str, Any] | None = None,
) -> str:
    """Format observation rows as a compact, LLM-readable summary."""
    prefix = _format_filters_block(filters, extra=query_extra)
    if not rows:
        return f"{prefix}No JWST observations found for the given query."

    header = f"Found {len(rows)} JWST observation(s). Showing first {min(len(rows), limit)}:\n"
    lines = [prefix + header] if prefix else [header]
    for i, row in enumerate(rows[:limit], start=1):
        obsid = row.get("obsid", "?")
        obs_id = row.get("obs_id", "?")
        instrument = row.get("instrument_name", "?")
        dptype = row.get("dataproduct_type", "?")
        filters = row.get("filters", "?")
        target = row.get("target_name", "?")
        proposal = row.get("proposal_id", "?")
        pi = row.get("proposal_pi", "?")
        calib = row.get("calib_level", "?")
        lines.append(
            f"{i:3}. obsid={obsid} obs_id={obs_id} target={target} "
            f"inst={instrument} type={dptype} filters={filters} "
            f"calib_level={calib} proposal_id={proposal} pi={pi}"
            + (f" raw={row['raw_data']}" if row.get("raw_data") else "")
        )
    if len(rows) > limit:
        lines.append(f"... ({len(rows) - limit} more truncated)")
    return "\n".join(lines)


def _format_products_summary(products: list[dict[str, Any]], limit: int = 100) -> str:
    """Format product rows as a compact, LLM-readable summary."""
    if not products:
        return "No matching products."

    header = f"Found {len(products)} product(s). Showing first {min(len(products), limit)}:\n"
    lines = [header]
    for i, p in enumerate(products[:limit], start=1):
        subgroup = p.get("productSubGroupDescription", "?")
        filename = p.get("productFilename", "?")
        ptype = p.get("productType", "?")
        size = p.get("size", "?")
        uri = p.get("dataURI", "?")
        lines.append(
            f"{i:3}. subgroup={subgroup} type={ptype} size={size} "
            f"file={filename} uri={uri}"
        )
    if len(products) > limit:
        lines.append(f"... ({len(products) - limit} more truncated)")
    return "\n".join(lines)


def _format_download_manifest(manifest: dict[str, Any]) -> str:
    """Format download manifest as a compact, LLM-readable summary."""
    downloaded = manifest.get("downloaded", [])
    skipped = manifest.get("skipped_proprietary", [])

    if "planet_name" in manifest:
        planet = manifest.get("planet_name", "?")
        observations = manifest.get("observations", [])
        lines = [
            f"Downloaded JWST data for {planet}.",
            f"Observations matched: {len(observations)}",
            f"Files downloaded: {len(downloaded)}",
            f"Skipped (proprietary / 401): {len(skipped)}",
            "",
            "Local paths:",
        ]
    else:
        label = manifest.get("label", "aggregate")
        obsids = manifest.get("obsids", [])
        lines = [
            f"Downloaded JWST data for batch '{label}'.",
            f"Obsids requested: {len(obsids)}",
            f"Files downloaded: {len(downloaded)}",
            f"Skipped (proprietary / 401): {len(skipped)}",
            "",
            "Local paths:",
        ]
    if "visits_selected" in manifest:
        lines.insert(4, f"Visits selected: {manifest['visits_selected']} "
                        f"(of {manifest.get('visits_available')})")
    for entry in downloaded:
        lines.append(f"  - {entry.get('path', '?')}")
    if not downloaded:
        lines.append("  (none)")
    if skipped:
        lines.append("")
        lines.append("Proprietary products skipped:")
        for entry in skipped[:20]:
            prod = entry.get("product") or {}
            lines.append(
                f"  - obsid={entry.get('obsid','?')} "
                f"file={prod.get('productFilename','?')}"
            )
        if len(skipped) > 20:
            lines.append(f"  ... ({len(skipped) - 20} more skipped, see manifest.json)")
    return "\n".join(lines)


def _diagnose_empty_demographics(
    *,
    instruments: Any,
    dataproduct_types: Any,
    calib_levels: Any,
    proposal_id: str | None,
    target_name: str | None,
    session: requests.Session | None = None,
    timeout: float = 120.0,
) -> str:
    """
    Explain an empty no-position (demographics) JWST search so the agent does
    not conclude "no such data exists" when the FILTERS are simply too strict.

    Strategy: re-run the same search with the filters progressively relaxed
    (drop calib_levels, then dataproduct_types, then instruments) and report the
    first relaxation that yields rows, summarizing what is actually present. If a
    proposal_id was given, probe that proposal alone. Every branch tells the
    caller which filter to change rather than dead-ending.
    """
    lines = [
        "Diagnostics (demographics mode): the strict query matched nothing. "
        "This usually means the FILTER combination is too narrow, NOT that the "
        "data is absent. Findings:",
    ]
    applied = {
        "instruments": _as_list(instruments),
        "dataproduct_types": _as_list(dataproduct_types),
        "calib_levels": _as_int_list(calib_levels),
        "proposal_id": proposal_id,
    }
    applied = {k: v for k, v in applied.items() if v}
    lines.append(f"  filters applied: {applied or '(none)'}")

    # Proposal probe first — cheapest and most specific.
    if proposal_id:
        try:
            prog_rows, _ = search_all_jwst_observations(
                proposal_id=proposal_id, session=session, timeout=timeout,
                pagesize=2000, max_pages=1,
            )
        except Exception as exc:
            prog_rows = []
            lines.append(f"  (proposal probe failed: {exc})")
        if prog_rows:
            lines.append(
                f"  proposal {proposal_id} has {len(prog_rows)} observation(s), "
                "but your OTHER filters excluded them. Present in that proposal:"
            )
            lines.append(f"    instrument_name:  {_summarize_distinct(prog_rows, 'instrument_name')}")
            lines.append(f"    dataproduct_type: {_summarize_distinct(prog_rows, 'dataproduct_type')}")
            lines.append(f"    calib_level:      {_summarize_distinct(prog_rows, 'calib_level')}")
            lines.append("  Relax the mismatched filter(s) above and retry.")
            return "\n".join(lines)
        lines.append(
            f"  a search for proposal {proposal_id} alone ALSO returned nothing "
            "— the proposal id itself may be wrong."
        )
        return "\n".join(lines)

    # No proposal_id to scope a cheap probe. Live relaxation is deliberately
    # AVOIDED here: a filter-free demographics search spans ~170k rows and the
    # completeness guard would refetch the whole set — a diagnostic must never
    # cost more than the query. Give targeted static guidance instead.
    culprits = []
    if applied.get("calib_levels"):
        culprits.append(
            "calib_levels — the most common culprit; JWST science is calib_level "
            "3 and calib_level=1 matches almost nothing. Try removing it first."
        )
    if applied.get("dataproduct_types"):
        culprits.append(
            "dataproduct_types — transit/eclipse data is usually 'timeseries', "
            "not 'spectrum'; a 'spectrum'-only filter drops it."
        )
    if applied.get("instruments"):
        culprits.append(
            "instruments — check spelling/mode (e.g. NIRSpec transit data is "
            "NIRSPEC/BOTS; a bare 'NIRSpec' is expanded automatically)."
        )
    if culprits:
        lines.append("  Remove filters ONE AT A TIME and retry — likely culprits:")
        lines.extend(f"    - {c}" for c in culprits)
    else:
        lines.append(
            "  no filters were applied, yet nothing returned — check "
            "instrument/target spelling, or the data may genuinely not exist."
        )
    lines.append(
        "  Do NOT conclude the data does not exist from this empty result "
        "until the filters above have been relaxed."
    )
    return "\n".join(lines)


# Row columns produced by the transit/eclipse classifier.
OBS_TYPE_KEYS = frozenset({
    "obs_type", "n_transits_in_window", "n_eclipses_in_window", "obs_type_note",
})
OBS_TYPES = ("transit", "eclipse", "phase_curve", "baseline", "unknown")
# Columns that only exist once observations are joined to planets.
PLANET_ROW_KEYS = (frozenset(DEFAULT_ARCHIVE_COLUMNS) - {"ra", "dec"}) | OBS_TYPE_KEYS
SEARCH_ROW_KEYS = (
    frozenset(JWST_OBSERVATION_COLUMNS) | frozenset(DEFAULT_ARCHIVE_COLUMNS)
    | OBS_TYPE_KEYS | RAW_ROW_KEYS | {"cycle_number"}
)
# Names models reach for that are not columns. 'event' is the one that
# mattered: the old crossmatch preview printed obs_type as "event=", and a
# model then grouped by a column that did not exist and got None for every
# row instead of an error.
ROW_KEY_ALIASES = {
    "event": "obs_type", "event_type": "obs_type", "obs_event": "obs_type",
    "cycle": "cycle_number", "instrument": "instrument_name",
    "planet": "pl_name", "grating": "filters", "pi": "proposal_pi",
    "program": "proposal_id", "proposal": "proposal_id", "target": "target_name",
}


def _resolve_row_keys(value: Any, notes: list[str]) -> list[str]:
    keys = []
    for key in _as_list(value) or []:
        key = str(key).strip()
        if not key:
            continue
        alias = ROW_KEY_ALIASES.get(key.lower())
        if alias:
            notes.append(f"'{key}' is not a column; read it as '{alias}'.")
            key = alias
        keys.append(key)
    return keys


def _check_row_keys(keys: list[str], available: set[str] | frozenset[str],
                    *, where: str) -> None:
    missing = [k for k in keys if k not in available]
    if missing:
        raise ValueError(
            f"No column {missing} in {where}. Available columns: "
            f"{sorted(available)}. Transit/eclipse is 'obs_type'."
        )


def _normalize_obs_types(value: Any) -> list[str]:
    types = [str(v).strip().lower().replace("-", "_").replace(" ", "_")
             for v in (_as_list(value) or []) if _clean_str(v)]
    types = ["transit" if t == "transits" else "eclipse" if t == "eclipses" else t
             for t in types]
    bad = [t for t in types if t not in OBS_TYPES]
    if bad:
        raise ValueError(f"Unknown obs_types {bad}. Valid: {list(OBS_TYPES)}.")
    return types


def _reject_observation_columns(conditions: list[str]) -> None:
    for condition in conditions:
        if re.search(r"\b(event|obs_type)\b", condition, re.IGNORECASE):
            raise ValueError(
                f"archive_conditions {condition!r}: the Exoplanet Archive has no "
                "event column — transit vs eclipse is a property of each JWST "
                "observation. Pass obs_types=['transit'] instead."
            )


def _event_classification_lines(rows: list[dict[str, Any]], classified: int) -> list[str]:
    lines = [
        f"Event classification: {classified}/{len(rows)} rows typed "
        "(transit/eclipse/phase_curve/baseline); 'unknown' rows lack "
        "pl_tranmid/pl_orbper — see obs_type_note."
    ]
    notes: dict[str, int] = {}
    for row in rows:
        note = row.get("obs_type_note")
        if note:
            notes[str(note)] = notes.get(str(note), 0) + 1
    if notes:
        lines.append("obs_type_note breakdown (read before trusting eclipse/baseline calls):")
        lines.extend(
            f"  - (x{count}) {note}"
            for note, count in sorted(notes.items(), key=lambda kv: -kv[1])
        )
    return lines


class SearchMastJwstObservations(BaseTool):
    """
    Search MAST for JWST observations — the one tool for every "which JWST
    data exists" question, from one planet to a whole population, including
    counts.

    USE WHEN: who observed a planet, what instrument/grating, which program
    or PI; every observation of a population ("sub-Neptunes with NIRSpec
    G395H transits"); or a count/breakdown ("how many per planet / cycle /
    instrument"). Answer in ONE call: put the population, the transit filter
    and the grouping on this call rather than chaining tools.

    NOT FOR: a planet's physical parameters or a planet list with no
    observations (`QueryExoplanetArchive`); files (`DownloadMastJwstProducts`,
    ``list_only=True`` to list them); release dates (`AsterAlert`).

    How a query runs — MAST first
    -----------------------------
    1. MAST is queried for JWST observations: a cone around ``planet_name``
       (or ``ra``/``dec``), or every observation matching the filters when no
       position is given.
    2. ``filters`` narrows by grating, client-side.
    3. Only if planets are needed (``population_preset``,
       ``archive_conditions``, ``obs_types``, or a planet column in
       ``group_by``) are the observations joined to Exoplanet Archive planets
       by sky position, giving one row per (planet, observation) with the
       planet's parameters, ``obs_type`` (transit / eclipse / phase_curve /
       baseline / unknown, from the planet's ephemeris) and, for populations,
       ``cycle_number``. With ``planet_name`` the planets are that planet (or
       every planet of that host).
    4. ``obs_types`` keeps rows of those types.
    5. By default every remaining observation is checked for raw data: its
       UNCAL science files (what ``DownloadMastJwstProducts(raw_only=True)``
       fetches), how many are public, and the release date. Rows get
       ``raw_data`` ('public' / 'exclusive_access' / 'partly_public' /
       'none'), ``raw_files``, ``raw_public_files``, ``release_date``, and
       the output says how many observations (and planets) have public raw
       data. Batched, one MAST request per 50 observations; skipped above
       1000 observations. ``check_raw=False`` turns it off.
    6. ``group_by`` counts the rows (``raw_data`` works as a key).

    The full rows are always written to CSV in population mode (cone mode
    only with ``output_csv``); that CSV feeds
    ``DownloadMastJwstProducts(rows_path=...)``, or this tool's own
    ``rows_path`` to regroup it without querying MAST again.

    Do not pass ``calib_levels`` unless the user asks for a level: raw data
    lives under every level, and a level filter only drops observations.

    Examples
    --------
    Sub-Neptunes with NIRSpec G395H transits, counted per planet:
        SearchMastJwstObservations(
            population_preset="subneptune",
            instruments=["NIRSpec"], dataproduct_types=["timeseries"],
            filters=["G395H"], obs_types=["transit"], group_by=["pl_name"],
        )
    Who observed WASP-39 b with NIRSpec:
        SearchMastJwstObservations(planet_name="WASP-39 b", instruments=["NIRSpec"])
    JWST time series per instrument (no planets needed):
        SearchMastJwstObservations(dataproduct_types=["timeseries"],
                                   group_by=["instrument_name"])
    Regroup a CSV an earlier call wrote:
        SearchMastJwstObservations(rows_path="mast/demographics/search_....csv",
                                   group_by=["cycle_number"])

    Populations
    -----------
    Prefer ``population_preset`` ('subneptune', 'super_earth', 'neptune',
    'hot_jupiter', 'warm_jupiter', ... ) to hand-written radius cuts: sub-
    Neptune is ``pl_rade`` 1.75-4.0 (Earth radii), NOT ``pl_radj`` (Jupiter
    radii, which gives hot Jupiters). ``archive_conditions`` are ADQL WHERE
    clauses on pscomppars (e.g. ``"sy_dist < 50"``), AND-ed with the preset.
    Transit/eclipse is NOT an archive column — use ``obs_types``.

    JWST instruments
    ----------------
    Bare names expand to every MAST mode of that instrument ("NIRSpec" →
    BOTS, SLIT, IFU, MSA, IMAGE; "MIRI" → SLITLESS, SLIT, IFU, IMAGE, ...).
    Full mode strings pass through ("NIRSPEC/BOTS"); shorthands "BOTS",
    "SOSS", "MRS", "LRS" map to their modes.

    Gratings / filters
    ------------------
    ``['G395H']`` keeps 'F290LP;G395H' and drops G395M and PRISM;
    ``['G395H', 'G395M']`` keeps either grating; ``['F290LP', 'G395H']``
    requires both (filter-wheel and disperser tokens are AND-ed, tokens of
    one kind OR-ed).

    Calibration levels
    ------------------
    ``calib_levels`` is the OBSERVATION's processing stage (JWST science is
    3), NOT a raw-vs-calibrated file selector. Never use ``[1]`` to find raw
    data: UNCAL ramps are products under calib-level 3 observations — search
    without calib_levels, then ``DownloadMastJwstProducts(obsids=...,
    raw_only=True)``.

    Zero results
    ------------
    An empty cone is re-run unfiltered and your filters re-applied
    client-side; an empty population search names the filter that emptied
    it. Report which stage was empty, never "no data exists".
    """

    planet_name: str | None = OptionalRuntimeField(
        description=(
            "Exoplanet or host-star target name, e.g. 'WASP-39 b' (cone search). "
            "Leave unset for a population / all-JWST search."
        ),
    )
    ra: float | str | None = OptionalRuntimeField(
        description="Right ascension in degrees. Optional if MAST can resolve planet_name.",
    )
    dec: float | str | None = OptionalRuntimeField(
        description="Declination in degrees. Optional if MAST can resolve planet_name.",
    )
    radius_deg: float | str = RuntimeField(
        default=0.02,
        description=(
            "Cone radius in degrees, for the planet_name/ra/dec search and for "
            "matching observations to planets."
        ),
    )
    population_preset: str | None = OptionalRuntimeField(
        description=(
            "Planet population to keep, e.g. 'subneptune', 'super_earth', "
            "'neptune', 'hot_jupiter', 'warm_jupiter'. Joins each observation "
            "to its planet (radius, Teq, ephemeris, obs_type, cycle). Use this "
            "rather than hand-written radius cuts."
        ),
    )
    archive_conditions: list | str | None = OptionalRuntimeField(
        description=(
            "Extra Exoplanet Archive (pscomppars) ADQL conditions, AND-ed with "
            "population_preset, e.g. ['sy_dist < 50']. pl_rade is Earth radii, "
            "pl_radj Jupiter radii. NOT for transit/eclipse — use obs_types."
        ),
    )
    obs_types: list | str | None = OptionalRuntimeField(
        description=(
            "Keep only these observation types, classified per planet from its "
            "ephemeris: 'transit', 'eclipse', 'phase_curve', 'baseline', "
            "'unknown'. Needs planet_name or a population."
        ),
    )
    instruments: list | str | None = OptionalRuntimeField(
        description="JWST instrument filters, e.g. ['NIRSpec', 'NIRCam', 'MIRI'].",
    )
    dataproduct_types: list | str | None = OptionalRuntimeField(
        description="MAST dataproduct filters, e.g. ['spectrum', 'timeseries', 'image'].",
    )
    calib_levels: list | str | None = OptionalRuntimeField(
        description=(
            "Leave unset unless the user names a level. It is the observation's "
            "processing stage, not raw vs calibrated: raw UNCAL files live under "
            "level 2/3 observations, and the raw check reports them by default."
        ),
    )
    filters: list | str | None = OptionalRuntimeField(
        description=(
            "Grating/filter tokens, e.g. ['G395H']. Gratings are any-of; a "
            "filter plus a grating (['F290LP', 'G395H']) must both match."
        ),
    )
    proposal_id: str | None = OptionalRuntimeField(
        description="Optional JWST proposal id filter.",
    )
    target_name: str | None = OptionalRuntimeField(
        description="Optional free-text MAST target_name filter (population mode).",
    )
    group_by: list | str | None = OptionalRuntimeField(
        description=(
            "Count the rows by these columns, e.g. ['pl_name'], "
            "['instrument_name', 'filters'], ['cycle_number'], ['obs_type']. "
            "Planet columns (pl_name, obs_type, pl_rade, ...) need planet_name "
            "or a population."
        ),
    )
    distinct_fields: list | str | None = OptionalRuntimeField(
        description=(
            "With group_by: count distinct values of these columns per group, "
            "e.g. ['instrument_name'] per pl_name for multi-instrument planets."
        ),
    )
    rows_path: str | None = OptionalRuntimeField(
        description=(
            "A rows CSV an earlier search wrote: filter (filters, obs_types) "
            "and group it again WITHOUT querying MAST."
        ),
    )
    output_csv: str | None = OptionalRuntimeField(
        description=(
            "Optional CSV path (relative to the workspace) for the full rows. "
            "Population mode always writes one (auto-named if unset)."
        ),
    )
    check_raw: bool | str = RuntimeField(
        default=True,
        description=(
            "Check each observation for raw (UNCAL) files and whether they are "
            "public (default True; skipped above 1000 observations)."
        ),
    )
    base_directory: str | None = StateField(default=None)

    def _save_rows(self, rows: list[dict[str, Any]], *, always: bool,
                   hint_parts: list[Any]) -> str:
        """Write the full observation rows to CSV; return a result-string line."""
        output_csv = _clean_str(self.output_csv)
        if not rows or not self.base_directory or not (always or output_csv):
            return ""
        path = self._rows_csv_path(hint_parts)
        _write_rows_csv(path, rows)
        return (
            f"Full rows ({len(rows)}) written to: {path}\n"
            "(Fetch files with DownloadMastJwstProducts(rows_path=...); regroup "
            "with SearchMastJwstObservations(rows_path=..., group_by=[...]).)\n\n"
        )

    def _rows_csv_path(self, hint_parts: list[Any]) -> Path:
        output_csv = _clean_str(self.output_csv)
        if output_csv:
            return Path(self.base_directory or ".") / output_csv
        return _autoname_csv_path(self.base_directory or ".", kind="search",
                                  hint_parts=hint_parts)

    def _annotate_cycles(self, rows: list[dict[str, Any]]) -> list[str]:
        return annotate_rows_with_cycles(
            rows, cache_path=Path(self.base_directory or ".") / JWST_PROGRAM_INFO_CACHE,
        )

    def _grouped(self, rows: list[dict[str, Any]], group_keys: list[str],
                 distinct_keys: list[str], hint: Any) -> str:
        """Count rows by group_keys; write the table; return its summary."""
        if not group_keys:
            return ""
        groups = aggregate_observations(rows, group_by=group_keys,
                                        distinct_fields=distinct_keys)
        csv_path = _autoname_csv_path(self.base_directory or ".", kind="aggregate",
                                      hint_parts=[group_keys, distinct_keys, hint])
        _write_rows_csv(csv_path, groups)
        return _format_aggregate_summary(
            groups, group_by=group_keys, distinct_fields=distinct_keys,
            total_rows=len(rows), csv_path=csv_path,
        ) + "\n\n"

    def _planets(self, population: list[str], planet_name: str | None
                 ) -> tuple[list[dict[str, Any]], list[str]]:
        """Archive planets to join the MAST observations to."""
        if population:
            return archive_tap_query(population, columns=DEFAULT_ARCHIVE_COLUMNS), population
        quoted = planet_name.replace("'", "''")
        for conditions in ([f"pl_name = '{quoted}'"], [f"hostname = '{quoted}'"]):
            planets = archive_tap_query(conditions, columns=DEFAULT_ARCHIVE_COLUMNS)
            if planets:
                return planets, conditions
        raise ValueError(
            f"The Exoplanet Archive has no planet or host named {planet_name!r}, "
            "so obs_type cannot be computed. Use the archive's name (e.g. "
            "'WASP-39 b') or pass archive_conditions."
        )

    def _run(self) -> str:
        # LLM callers routinely pass "" for unused fields and numbers as
        # strings; normalize before choosing a mode ("" used to be treated as
        # a real planet name, sending garbage to the MAST name resolver).
        planet_name = _clean_str(self.planet_name)
        target_name = _clean_str(self.target_name)
        proposal_id = _clean_str(self.proposal_id)
        ra = _as_float_or_none(self.ra, "ra")
        dec = _as_float_or_none(self.dec, "dec")
        radius_deg = _as_float_or_none(self.radius_deg, "radius_deg")
        radius_deg = 0.02 if radius_deg is None else radius_deg
        grating_filters = [
            f for f in (_as_list(self.filters) or []) if _clean_str(f)
        ] or None

        def by_grating(rows):
            return [r for r in rows if _filters_match(r.get("filters"), grating_filters)]

        grating_extra = {"filters (client-side)": grating_filters} if grating_filters else {}

        population_preset = _clean_str(self.population_preset)
        archive_conditions = [str(c).strip() for c in (_as_list(self.archive_conditions) or [])
                              if _clean_str(c)]
        _reject_observation_columns(archive_conditions)
        population = _resolve_population_preset(population_preset) + archive_conditions
        obs_types = _normalize_obs_types(self.obs_types)
        key_notes: list[str] = []
        group_keys = _resolve_row_keys(self.group_by, key_notes)
        distinct_keys = _resolve_row_keys(self.distinct_fields, key_notes)
        if distinct_keys and not group_keys:
            raise ValueError("distinct_fields needs group_by.")
        requested_keys = set(group_keys) | set(distinct_keys)
        key_note = "".join(f"NOTE: {n}\n" for n in key_notes)
        check_raw = _as_bool(self.check_raw, "check_raw", default=True)
        if requested_keys & RAW_ROW_KEYS and not check_raw:
            raise ValueError(f"group_by {sorted(requested_keys & RAW_ROW_KEYS)} needs check_raw=True.")

        rows_path = _clean_str(self.rows_path)
        if rows_path:
            return key_note + self._run_on_saved_rows(
                rows_path, by_grating, grating_filters, obs_types,
                group_keys, distinct_keys,
            )

        # Unknown columns fail here, before any query, instead of grouping
        # every row under None.
        _check_row_keys(sorted(requested_keys), SEARCH_ROW_KEYS, where="search rows")
        wants_planets = bool(population or obs_types or requested_keys & PLANET_ROW_KEYS)
        if wants_planets and not population and planet_name is None:
            raise ValueError(
                "obs_types and planet columns (pl_name, obs_type, pl_rade, ...) "
                "need planets to classify against: pass population_preset (e.g. "
                "'subneptune') or archive_conditions, or a planet_name."
            )

        # calib_level=1 is a common trap: callers use it hoping to get raw /
        # uncalibrated data, but JWST science observations are almost never
        # filed at calib_level 1 (they are calib_level 3, and the UNCAL ramps
        # are PRODUCTS under them). The empty result then reads as "no raw data
        # exists", which is false. Attach a correction to every return path.
        calib_list = _as_int_list(self.calib_levels)
        raw_note = ""
        if calib_list and 1 in calib_list:
            raw_note = (
                "\n\nNOTE ON calib_level=1: JWST science observations are almost "
                "never filed at calib_level 1, so this filter usually matches "
                "nothing. Raw/uncalibrated (UNCAL) data is NOT obtained this way "
                "— it exists as PRODUCTS under calib_level 2/3 observations. To "
                "get raw data: re-run this search WITHOUT calib_levels, then call "
                "DownloadMastJwstProducts(obsids=..., raw_only=True) (add "
                "list_only=True to preview). An empty result "
                "here does NOT mean uncalibrated data is unavailable."
            )

        # ---- 1. MAST ----
        demographics = planet_name is None and ra is None and dec is None
        diagnostics = ""
        if demographics:
            observations, filters = search_all_jwst_observations(
                instruments=self.instruments,
                dataproduct_types=self.dataproduct_types,
                calib_levels=self.calib_levels,
                target_name=target_name,
                proposal_id=proposal_id,
            )
            observations = by_grating(observations)
            query_extra = {"mode": "demographics (Mast.Caom.Filtered, no position)",
                           **grating_extra}
            hint_parts = [population_preset, archive_conditions, self.instruments,
                          self.dataproduct_types, self.calib_levels, proposal_id,
                          target_name, grating_filters, obs_types]
            if not observations:
                # Empty demographics result used to dead-end as a bare "No JWST
                # observations found", inviting a false "no such data" verdict.
                # Diagnose which filter was too strict instead.
                diagnostics = _diagnose_empty_demographics(
                    instruments=self.instruments,
                    dataproduct_types=self.dataproduct_types,
                    calib_levels=self.calib_levels,
                    proposal_id=proposal_id,
                    target_name=target_name,
                )
                if grating_filters:
                    diagnostics += (
                        f"\nAlso check the grating filter {grating_filters}: "
                        "rerun without `filters` to see which gratings exist."
                    )
        else:
            if (ra is None) != (dec is None):
                raise ValueError(
                    "Pass both ra and dec (or neither). A planet_name alone is "
                    "also fine — it is resolved via MAST Name Lookup."
                )
            observations = by_grating(search_jwst_observations(
                planet_name,
                ra=ra,
                dec=dec,
                radius_deg=radius_deg,
                instruments=self.instruments,
                dataproduct_types=self.dataproduct_types,
                calib_levels=self.calib_levels,
                proposal_id=proposal_id,
            ))
            filters = _build_jwst_observation_filters(
                instruments=self.instruments,
                dataproduct_types=self.dataproduct_types,
                calib_levels=self.calib_levels,
                proposal_id=proposal_id,
            )
            query_extra = {
                "mode": "per-planet cone search (Mast.Caom.Filtered.Position)",
                "planet_name": planet_name,
                "radius_deg": radius_deg,
                **grating_extra,
            }
            hint_parts = [planet_name, population_preset, grating_filters, obs_types]
            if not observations:
                # Strict query came back empty: self-diagnose instead of dead-ending.
                if ra is None or dec is None:
                    ra, dec = resolve_target_coordinates(planet_name)
                recovered, diagnostics = _diagnose_empty_cone_search(
                    planet_name,
                    ra=ra,
                    dec=dec,
                    radius_deg=radius_deg,
                    instruments=_as_list(self.instruments),
                    dataproduct_types=_as_list(self.dataproduct_types),
                    calib_levels=_as_int_list(self.calib_levels),
                    proposal_id=proposal_id,
                )
                observations = by_grating(recovered)

        if not wants_planets or not observations:
            # No planets wanted — or MAST found nothing, so there is nothing
            # to join and the archive is never queried.
            cycle_warnings = []
            if observations and "cycle_number" in requested_keys:
                cycle_warnings = self._annotate_cycles(observations)
            raw_lines = (annotate_rows_with_raw_data(observations)
                         if observations and check_raw else [])
            summary = _format_observations_summary(
                observations, limit=10 if group_keys else 50,
                filters=filters, query_extra=query_extra,
            )
            saved = self._save_rows(observations, always=demographics,
                                    hint_parts=hint_parts)
            if demographics:
                body = saved + summary + (f"\n\n{diagnostics}" if diagnostics else "")
            elif not observations:
                body = f"{summary}\n\n{diagnostics}"
            else:
                body = saved + (f"{diagnostics}\n\n" if diagnostics else "") + summary
            if raw_lines:
                body += "\n\n" + "\n".join(raw_lines)
            if wants_planets:
                body += ("\n\nNo observations, so no planets were looked up "
                         "(MAST is queried first).")
            return (key_note + self._grouped(observations, group_keys, distinct_keys, hint_parts)
                    + body + _format_cycle_warnings(cycle_warnings) + raw_note)

        # ---- 2. Join the observations to their planets ----
        planets, conditions = self._planets(population, planet_name)
        rows = crossmatch_observations_to_planets(
            observations, planets, radius_deg=radius_deg,
        )
        classified = annotate_rows_with_event_types(rows)
        type_lines = _event_classification_lines(rows, classified) if rows else []
        type_counts: dict[str, int] = {}
        for row in rows:
            type_counts[row["obs_type"]] = type_counts.get(row["obs_type"], 0) + 1
        pairs = len(rows)
        if obs_types:
            rows = [r for r in rows if r.get("obs_type") in obs_types]
            type_lines.append(f"obs_types={obs_types}: kept {len(rows)} of {pairs} pairs "
                              f"(all pairs by type: {type_counts}).")
        cycle_warnings = []
        if rows and (population or "cycle_number" in requested_keys):
            cycle_warnings = self._annotate_cycles(rows)
        if rows and check_raw:
            type_lines.extend(annotate_rows_with_raw_data(rows))

        csv_path = None
        if self.base_directory and (demographics or _clean_str(self.output_csv)):
            csv_path = self._rows_csv_path(hint_parts)
            _write_rows_csv(csv_path, rows)

        def _has_coords(p: dict[str, Any]) -> bool:
            try:
                return (_as_float_or_none(p.get("ra"), "ra") is not None
                        and _as_float_or_none(p.get("dec"), "dec") is not None)
            except ValueError:
                return False

        summary = _format_crossmatch_summary(
            rows,
            planet_count=len(planets),
            obs_count=len(observations),
            planets_with_coords=sum(1 for p in planets if _has_coords(p)),
            radius_deg=radius_deg,
            csv_path=csv_path,
            limit=10 if group_keys else 20,
            preset=population_preset,
            conditions=conditions,
            obs_filters={
                "instruments": _as_list(self.instruments),
                "dataproduct_types": _as_list(self.dataproduct_types),
                "calib_levels": _as_int_list(self.calib_levels),
                "filters": grating_filters,
                "proposal_id": proposal_id,
            },
            obs_types=obs_types,
            pairs_before_type_filter=pairs,
        )
        body = (f"{diagnostics}\n\n" if diagnostics else "") + summary
        if type_lines:
            body += "\n\n" + "\n".join(type_lines)
        return (key_note + self._grouped(rows, group_keys, distinct_keys, hint_parts)
                + body + _format_cycle_warnings(cycle_warnings) + raw_note)

    def _run_on_saved_rows(self, rows_path, by_grating, grating_filters,
                           obs_types, group_keys, distinct_keys) -> str:
        """Filter / regroup a rows CSV from an earlier search. No MAST query."""
        path = Path(self.base_directory or ".") / rows_path
        rows = _read_rows_from_path(path)
        n_read = len(rows)
        lines = [f"Re-read {n_read} row(s) from {path} — no MAST query."]
        ignored = [name for name in (
            "planet_name", "ra", "dec", "instruments", "dataproduct_types",
            "calib_levels", "proposal_id", "target_name", "population_preset",
            "archive_conditions",
        ) if getattr(self, name, None) not in (None, "", [], ())]
        if ignored:
            lines.append(f"rows_path mode re-reads that file; ignored {ignored} "
                         "(drop rows_path to query MAST).")
        rows = by_grating(rows)
        columns = set().union(*(r.keys() for r in rows)) if rows else set()
        requested = set(group_keys) | set(distinct_keys)
        type_keys = (requested & OBS_TYPE_KEYS) | ({"obs_type"} if obs_types else set())
        if rows and type_keys - columns:
            if "pl_tranmid" not in columns:
                raise ValueError(
                    f"{rows_path} has no planet ephemeris columns, so obs_type "
                    "cannot be computed. Rerun the search with population_preset "
                    "or planet_name instead of rows_path."
                )
            lines.extend(_event_classification_lines(rows, annotate_rows_with_event_types(rows)))
            columns |= OBS_TYPE_KEYS
        if obs_types:
            rows = [r for r in rows if r.get("obs_type") in obs_types]
        cycle_warnings = []
        if rows and "cycle_number" in requested and "cycle_number" not in columns:
            cycle_warnings = self._annotate_cycles(rows)
            columns.add("cycle_number")
        if rows and requested & RAW_ROW_KEYS - columns:
            lines.extend(annotate_rows_with_raw_data(rows))
            columns |= RAW_ROW_KEYS
        if rows:
            _check_row_keys(sorted(requested), columns, where=rows_path)
        if grating_filters or obs_types:
            out_path = self._rows_csv_path([rows_path, grating_filters, obs_types])
            _write_rows_csv(out_path, rows)
            lines.append(f"Kept {len(rows)} of {n_read} rows "
                         f"(filters={grating_filters}, obs_types={obs_types}); "
                         f"written to: {out_path}")
        head = "\n".join(lines) + "\n\n"
        if group_keys:
            body = self._grouped(rows, group_keys, distinct_keys, rows_path).rstrip()
        elif "pl_name" in columns:
            body = "\n".join(
                f"{i:3}. pl_name={r.get('pl_name')} obsid={r.get('obsid')} "
                f"inst={r.get('instrument_name')} filters={r.get('filters')} "
                f"obs_type={r.get('obs_type', '?')} raw={r.get('raw_data', '?')} "
                f"proposal_id={r.get('proposal_id')}"
                for i, r in enumerate(rows[:20], start=1)
            ) + (f"\n... ({len(rows) - 20} more rows in the CSV)" if len(rows) > 20 else "")
        else:
            body = _format_observations_summary(rows)
        return head + body + _format_cycle_warnings(cycle_warnings)


def _format_cycle_warnings(warnings: list[str]) -> str:
    if not warnings:
        return ""
    return (
        "\n\nWARNING — cycle_number left empty for some proposals "
        "(no guessed values were written):\n"
        + "\n".join(f"  - {w}" for w in warnings)
    )


# (Listing products for an obsid is DownloadMastJwstProducts(list_only=True);
#  the merged download tool is defined after download_demographic_products.)


# -------------------- JWST program info (observing cycle) --------------------


# In-process memo so one crossmatch never fetches or re-reads the same
# proposal twice. Keyed by str(proposal_id).
_PROGRAM_INFO_MEMO: dict[str, dict[str, Any]] = {}

# Tag-tolerant field extractors for the STScI program-info page. The raw HTML
# interleaves tags with the labels (e.g. "<b>Cycle:</b> 1"), so each pattern
# skips any run of tags/entities/whitespace between "Label:" and the value.
_TAG_RUN = r"(?:</?[^>]+>|&nbsp;|\s)*"
_PROGRAM_INFO_PATTERNS = {
    "cycle": re.compile(rf"Cycle:{_TAG_RUN}(\d+)"),
    "title": re.compile(rf"Title:{_TAG_RUN}([^<\r\n]+)"),
    "pi": re.compile(rf"Principal\s+Investigator:{_TAG_RUN}([^<\r\n]+)"),
    "status": re.compile(rf"Program\s+Status:{_TAG_RUN}([^<\r\n]+)"),
    "exclusive_access_period": re.compile(
        rf"Exclusive\s+Access\s+Period:{_TAG_RUN}([^<\r\n]+)"
    ),
}
_PROPOSAL_TYPE_PATTERN = re.compile(r"#types\"[^>]*>\s*([A-Z][A-Z/ ]{0,15}?)\s*<")


def _load_program_info_cache(cache_path: Path | None) -> dict[str, dict[str, Any]]:
    if cache_path is None or not cache_path.is_file():
        return {}
    try:
        data = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_program_info_cache(
    cache_path: Path | None, cache: dict[str, dict[str, Any]]
) -> None:
    if cache_path is None:
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))
    except OSError:
        # A read-only workspace must not break the query itself; the
        # in-process memo still prevents duplicate fetches this run.
        pass


def get_jwst_program_info(
    proposal_id: str | int,
    *,
    cache_path: str | os.PathLike[str] | None = None,
    refresh: bool = False,
    session: requests.Session | None = None,
    timeout: float = 30.0,
    request_pause: float = 0.5,
) -> dict[str, Any]:
    """
    Fetch authoritative JWST program metadata (observing cycle, title, PI,
    proposal type, status, exclusive-access period) from STScI's program-info
    page, with permanent on-disk caching.

    Why this exists: MAST CAOM has no cycle column, and cycle CANNOT be
    inferred from proposal_id (GO 2372 is Cycle 1; GO 3557 / GO 4098 are
    Cycle 2). Guessing produced wrong catalogs; this fetches the real value.

    ``cache_path`` points at a JSON file mapping proposal_id -> info dict.
    A proposal's cycle never changes, so cache entries never expire (title /
    status can drift, pass ``refresh=True`` to force a refetch). Raises
    ``ValueError`` if the page has no parseable Cycle field (bad proposal id
    or page-layout change) — never silently returns a guess.
    """
    pid = str(proposal_id).strip()
    if not pid or not pid.isdigit():
        raise ValueError(f"proposal_id must be a positive integer, got {proposal_id!r}.")

    resolved_cache_path = Path(cache_path) if cache_path is not None else None

    if not refresh:
        memo_hit = _PROGRAM_INFO_MEMO.get(pid)
        if memo_hit is not None:
            return dict(memo_hit)
        disk = _load_program_info_cache(resolved_cache_path)
        disk_hit = disk.get(pid)
        if isinstance(disk_hit, dict) and disk_hit.get("cycle") is not None:
            _PROGRAM_INFO_MEMO[pid] = disk_hit
            return dict(disk_hit)

    client = session or requests
    response = client.get(
        STSCI_PROGRAM_INFO_URL.format(proposal_id=pid), timeout=timeout
    )
    response.raise_for_status()
    html = response.text

    cycle_match = _PROGRAM_INFO_PATTERNS["cycle"].search(html)
    if cycle_match is None:
        raise ValueError(
            f"STScI program-info page for proposal {pid} has no 'Cycle:' field. "
            "Either the proposal id is wrong, or the page layout changed and "
            "_PROGRAM_INFO_PATTERNS needs updating. Refusing to guess."
        )

    info: dict[str, Any] = {
        "proposal_id": pid,
        "cycle": int(cycle_match.group(1)),
        "source": STSCI_PROGRAM_INFO_URL.format(proposal_id=pid),
    }
    for key in ("title", "pi", "status", "exclusive_access_period"):
        match = _PROGRAM_INFO_PATTERNS[key].search(html)
        info[key] = match.group(1).strip() if match else None
    type_match = _PROPOSAL_TYPE_PATTERN.search(html)
    info["proposal_type"] = type_match.group(1).strip() if type_match else None

    _PROGRAM_INFO_MEMO[pid] = info
    disk = _load_program_info_cache(resolved_cache_path)
    disk[pid] = info
    _save_program_info_cache(resolved_cache_path, disk)

    if request_pause:
        time.sleep(request_pause)  # politeness between cache-miss fetches
    return dict(info)


def annotate_rows_with_cycles(
    rows: list[dict[str, Any]],
    *,
    cache_path: str | os.PathLike[str] | None = None,
    session: requests.Session | None = None,
    timeout: float = 30.0,
) -> list[str]:
    """
    Add a ``cycle_number`` key to every row that has a ``proposal_id``,
    resolved via ``get_jwst_program_info`` (one lookup per unique proposal,
    memo + disk cached).

    Returns a list of human-readable warnings for proposals whose cycle could
    not be resolved; those rows get ``cycle_number=None`` rather than a guess,
    and one failure never aborts the whole annotation.
    """
    warnings: list[str] = []
    cycles: dict[str, int | None] = {}
    for row in rows:
        pid = str(row.get("proposal_id") or "").strip()
        if not pid:
            row["cycle_number"] = None
            continue
        if pid not in cycles:
            try:
                cycles[pid] = get_jwst_program_info(
                    pid, cache_path=cache_path, session=session, timeout=timeout
                )["cycle"]
            except Exception as exc:  # network / bad pid / layout drift
                cycles[pid] = None
                warnings.append(
                    f"cycle_number unresolved for proposal {pid}: {exc}"
                )
        row["cycle_number"] = cycles[pid]
    return warnings


# ---------------- observation event type (transit / eclipse / phase curve) ----------------


def _bjd_to_mjd(value: float) -> float | None:
    """
    Normalize an epoch to MJD. Accepts full BJD (~2.4e6, the Exoplanet
    Archive ``pl_tranmid`` convention) or an already-MJD value (30000–80000).
    Returns None for anything ambiguous — better unknown than misclassified.
    The ~minutes-level BJD_TDB vs MJD_UTC timescale difference is negligible
    for event classification (transit durations are hours).
    """
    if value > 2_400_000.0:
        return value - 2_400_000.5
    if 30_000.0 <= value <= 80_000.0:
        return value
    return None


def _count_events_in_window(
    t_min: float, t_max: float, first_center: float, period: float, half_width: float
) -> int:
    """Count periodic event centers (± half_width) inside [t_min, t_max]."""
    lo = math.ceil((t_min - half_width - first_center) / period)
    hi = math.floor((t_max + half_width - first_center) / period)
    return max(0, hi - lo + 1)


def classify_jwst_observation_event(
    t_min: Any,
    t_max: Any,
    *,
    tranmid_bjd: Any,
    period_days: Any,
    duration_hours: Any = None,
    eccentricity: Any = None,
    phase_curve_fraction: float = 0.8,
) -> dict[str, Any]:
    """
    Classify one JWST observation window against a planet's transit ephemeris.

    MAST/CAOM carries no transit/eclipse label — but ``t_min``/``t_max``
    (observation window, MJD) plus the Exoplanet Archive ephemeris
    (``pl_tranmid`` BJD, ``pl_orbper`` days, ``pl_trandur`` hours) determine
    it: a window containing a transit center (orbital phase 0) is a transit,
    one containing phase 0.5 is an eclipse, one spanning ≥
    ``phase_curve_fraction`` of an orbit (or both event types) is a phase
    curve, and one containing neither is baseline (e.g. the ~0.5 h MIRI
    background exposures).

    Returns a dict with:
      * ``obs_type``  — 'transit' | 'eclipse' | 'phase_curve' | 'baseline'
                        | 'unknown'
      * ``n_transits_in_window`` / ``n_eclipses_in_window``
      * ``obs_type_note`` — caveats (missing ephemeris, high eccentricity)

    Caveats: eclipse timing assumes a circular orbit; for e > 0.1 the true
    eclipse phase shifts away from 0.5, so eclipse/baseline calls are flagged
    approximate in the note. Event centers are counted within the window
    padded by half the transit duration, so partial (grazing-coverage)
    events still count.
    """
    unknown = {
        "obs_type": "unknown",
        "n_transits_in_window": None,
        "n_eclipses_in_window": None,
        "obs_type_note": None,
    }

    tmin = _as_float_or_none(t_min, "t_min")
    tmax = _as_float_or_none(t_max, "t_max")
    t0_raw = _as_float_or_none(tranmid_bjd, "tranmid_bjd")
    period = _as_float_or_none(period_days, "period_days")
    dur_h = _as_float_or_none(duration_hours, "duration_hours")
    ecc = _as_float_or_none(eccentricity, "eccentricity")

    missing = [
        name
        for name, val in (
            ("t_min/t_max", tmin if tmax is not None else None),
            ("pl_tranmid", t0_raw),
            ("pl_orbper", period),
        )
        if val is None
    ]
    if missing or period <= 0 or tmax < tmin:
        unknown["obs_type_note"] = (
            f"missing/invalid ephemeris or window: {', '.join(missing) or 't order'}"
        )
        return unknown

    t0 = _bjd_to_mjd(t0_raw)
    if t0 is None:
        unknown["obs_type_note"] = (
            f"pl_tranmid={t0_raw} is neither full BJD (~2.4e6) nor MJD; refusing to guess"
        )
        return unknown

    half_width = (dur_h / 24.0 / 2.0) if dur_h else 0.0
    n_transits = _count_events_in_window(tmin, tmax, t0, period, half_width)
    n_eclipses = _count_events_in_window(tmin, tmax, t0 + period / 2.0, period, half_width)

    note = None
    if ecc is not None and ecc > 0.1:
        note = (
            f"e={ecc:.2f} > 0.1: eclipse phase assumed 0.5 (circular); "
            "eclipse/baseline call is approximate"
        )

    span = tmax - tmin
    if span >= phase_curve_fraction * period or (n_transits > 0 and n_eclipses > 0):
        obs_type = "phase_curve"
    elif n_transits > 0:
        obs_type = "transit"
    elif n_eclipses > 0:
        obs_type = "eclipse"
    else:
        obs_type = "baseline"

    return {
        "obs_type": obs_type,
        "n_transits_in_window": n_transits,
        "n_eclipses_in_window": n_eclipses,
        "obs_type_note": note,
    }


def annotate_rows_with_event_types(rows: list[dict[str, Any]]) -> int:
    """
    Add ``obs_type`` / ``n_transits_in_window`` / ``n_eclipses_in_window`` /
    ``obs_type_note`` to crossmatch rows in place, using each row's own
    planet ephemeris. Because rows are (planet, observation) pairs, the same
    observation is classified per planet — a K2-18 b transit visit is
    correctly 'baseline' on the K2-18 c row unless c also transits in window.

    Returns the number of rows classified as something other than 'unknown'.
    """
    classified = 0
    for row in rows:
        result = classify_jwst_observation_event(
            row.get("t_min"),
            row.get("t_max"),
            tranmid_bjd=row.get("pl_tranmid"),
            period_days=row.get("pl_orbper"),
            duration_hours=row.get("pl_trandur"),
            eccentricity=row.get("pl_orbeccen"),
        )
        row.update(result)
        if result["obs_type"] != "unknown":
            classified += 1
    return classified


# -------------------- crossmatch + aggregate helpers --------------------


def _haversine_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle separation (degrees) between two sky positions."""
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    a = (
        math.sin((d2 - d1) / 2) ** 2
        + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2) ** 2
    )
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(a))))


def archive_tap_query(
    conditions: list[str] | str,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    table: str = "pscomppars",
    limit: int | None = None,
    session: requests.Session | None = None,
    timeout: float = 120.0,
) -> list[dict[str, Any]]:
    """
    NASA Exoplanet Archive TAP query; every matching row as a dict.

    A thin wrapper over ``archive_tap.query_archive``, which owns the ADQL,
    the request, timeouts and error handling for every archive caller
    (Patchwork discovery/fits, ASTER Alert, ``SearchMastJwstObservations``,
    ``QueryExoplanetArchive``). ``limit`` of 0 or None means no limit.
    """
    return _archive_tap.query_archive(
        _as_list(columns) or list(DEFAULT_ARCHIVE_COLUMNS),
        conditions,
        table=table,
        limit=int(limit) if limit else None,
        session=session,
        timeout=timeout,
    )


def crossmatch_observations_to_planets(
    observations: list[dict[str, Any]],
    planets: list[dict[str, Any]],
    *,
    radius_deg: float = 0.02,
    obs_ra_key: str = "s_ra",
    obs_dec_key: str = "s_dec",
    planet_ra_key: str = "ra",
    planet_dec_key: str = "dec",
    obs_keep_keys: tuple[str, ...] = (
        "obsid", "obs_id", "instrument_name", "dataproduct_type",
        "calib_level", "proposal_id", "proposal_pi", "target_name",
        "filters", "t_min", "t_max", "t_obs_release",
    ),
) -> list[dict[str, Any]]:
    """
    Cone-match JWST observations against a planet population by RA/Dec.

    Each output row is one (planet, observation) pair: planet attributes
    are copied verbatim, then a fixed set of MAST observation fields are
    appended. Observations or planets missing RA/Dec are skipped. Each
    (planet, obsid) pair is emitted at most once — duplicate observation
    rows (e.g. from MAST pagination overlap) cannot duplicate output rows.
    """
    rows: list[dict[str, Any]] = []
    emitted: set[tuple[str, Any]] = set()
    # Pre-coerce planets once.
    planet_coords: list[tuple[dict[str, Any], float, float]] = []
    for planet in planets:
        try:
            pra = float(planet.get(planet_ra_key))
            pdec = float(planet.get(planet_dec_key))
        except (TypeError, ValueError):
            continue
        planet_coords.append((planet, pra, pdec))

    for obs in observations:
        try:
            ora = float(obs.get(obs_ra_key))
            odec = float(obs.get(obs_dec_key))
        except (TypeError, ValueError):
            continue
        for planet, pra, pdec in planet_coords:
            if _haversine_deg(ora, odec, pra, pdec) <= radius_deg:
                pair = (str(planet.get("pl_name")), obs.get("obsid"))
                if pair[1] is not None and pair in emitted:
                    continue
                emitted.add(pair)
                row = dict(planet)
                for k in obs_keep_keys:
                    row[k] = obs.get(k)
                rows.append(row)
    return rows


def aggregate_observations(
    rows: list[dict[str, Any]],
    group_by: list[str] | str,
    *,
    distinct_fields: list[str] | str | None = None,
) -> list[dict[str, Any]]:
    """
    Group rows by ``group_by`` keys and return per-group counts plus
    optional distinct-value counts for ``distinct_fields``.

    Each output dict has the group-by key/value pairs, ``count``, and
    ``{field}_distinct`` + ``{field}_values`` entries for every distinct
    field requested.
    """
    group_keys = _as_list(group_by) or []
    distinct_keys = _as_list(distinct_fields) or []
    if not group_keys:
        raise ValueError("group_by must contain at least one key.")

    buckets: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        bucket = buckets.setdefault(
            key,
            {
                "_key": key,
                "count": 0,
                "_distinct_sets": {k: set() for k in distinct_keys},
            },
        )
        bucket["count"] += 1
        for k in distinct_keys:
            value = row.get(k)
            if value is None or value == "":
                continue
            bucket["_distinct_sets"][k].add(value)

    output: list[dict[str, Any]] = []
    for key, bucket in buckets.items():
        entry: dict[str, Any] = dict(zip(group_keys, key))
        entry["count"] = bucket["count"]
        for k in distinct_keys:
            values = sorted(bucket["_distinct_sets"][k], key=lambda v: str(v))
            entry[f"{k}_distinct"] = len(values)
            entry[f"{k}_values"] = values
        output.append(entry)

    output.sort(key=lambda e: (-e["count"], tuple(str(e.get(k)) for k in group_keys)))
    return output


def _read_rows_from_path(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load row dicts from a CSV or JSON file written by an earlier tool."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"rows file not found: {p}")
    suffix = p.suffix.lower()
    if suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        with p.open() as fh:
            return list(csv.DictReader(fh, delimiter=delimiter))
    if suffix == ".json":
        with p.open() as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "rows" in data:
            data = data["rows"]
        if not isinstance(data, list):
            raise ValueError(f"JSON at {p} is not a list of rows.")
        return data
    raise ValueError(f"Unsupported rows file extension: {suffix}")


def _slugify_for_filename(value: Any, max_len: int = 40) -> str:
    """Lower-case, filesystem-safe slug for embedding in autogenerated filenames."""
    text = str(value) if value is not None else ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:max_len] or "query"


def _autoname_csv_path(
    base_directory: str | os.PathLike[str],
    *,
    kind: str,
    hint_parts: list[Any] | None = None,
    subdir: str = "mast/demographics",
) -> Path:
    """
    Build a deterministic-ish CSV path under ``base_directory`` so every
    crossmatch/aggregate call leaves a complete on-disk record even when the
    caller forgot to pass ``output_csv``.

    Name format: ``{subdir}/{kind}_{stamp}_{slug}_{hash}.csv``
      * ``stamp``   – UTC ``YYYYMMDDTHHMMSS`` so reruns don't overwrite.
      * ``slug``    – short text snippet from ``hint_parts`` (e.g. archive
                      conditions or group keys).
      * ``hash``    – 6-char sha1 of the full hint payload, so two distinct
                      queries land in distinct files even when their slugs
                      collide.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    hint_repr = json.dumps(hint_parts or [], sort_keys=True, default=str)
    digest = hashlib.sha1(hint_repr.encode("utf-8")).hexdigest()[:6]
    slug_source = " ".join(str(p) for p in (hint_parts or []) if p)
    slug = _slugify_for_filename(slug_source) if slug_source else "query"
    filename = f"{kind}_{stamp}_{slug}_{digest}.csv"
    return Path(base_directory) / subdir / filename


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write row dicts to CSV using the union of all keys (preserves first-seen order)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    seen: list[str] = []
    seen_set: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen_set:
                seen_set.add(k)
                seen.append(k)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=seen)
        writer.writeheader()
        for row in rows:
            # CSV writer can't serialize lists/sets → stringify
            writer.writerow({k: _csv_value(row.get(k)) for k in seen})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(v) for v in value)
    return value


def _format_crossmatch_summary(
    rows: list[dict[str, Any]],
    *,
    planet_count: int,
    obs_count: int,
    radius_deg: float,
    csv_path: Path | None,
    limit: int = 20,
    preset: str | None = None,
    conditions: list[str] | None = None,
    planets_with_coords: int | None = None,
    obs_filters: dict[str, Any] | None = None,
    obs_types: list[str] | None = None,
    pairs_before_type_filter: int | None = None,
) -> str:
    head: list[str] = []
    if csv_path is not None:
        head.append(
            f"FULL RESULTS ({len(rows)} rows) saved to CSV: {csv_path}"
        )
        head.append(
            "Read that file (e.g. pandas.read_csv) for the complete row set. "
            "The preview below is TRUNCATED — do NOT reconstruct downstream "
            "files from it."
        )
        head.append("")
    head.append(f"Cross-matched {len(rows)} (planet, JWST observation) pair(s).")
    if preset:
        head.append(f"Population preset: {preset}")
    if conditions:
        head.append(f"Archive conditions: {conditions}")
    head.extend(
        [
            f"Population planets considered: {planet_count}",
            f"JWST observations considered: {obs_count}",
            f"Cone radius: {radius_deg} deg",
            "",
        ]
    )
    if not rows:
        head.append("No (planet, observation) pairs matched. WHY — check the stage "
                    "that was empty before concluding anything:")
        active_filters = {k: v for k, v in (obs_filters or {}).items() if v}
        if planet_count == 0:
            head.append(
                "  • 0 population planets matched your archive conditions/preset. "
                "The ARCHIVE query is empty — this says NOTHING about JWST data. "
                "Loosen archive_conditions or check the population_preset name."
            )
        elif planets_with_coords == 0:
            head.append(
                f"  • {planet_count} planets matched, but NONE have usable RA/Dec, "
                "so the cone-match had nothing to match against. Ensure "
                "the archive rows include 'ra' and 'dec'."
            )
        if obs_count == 0:
            head.append(
                "  • 0 JWST observations matched your instrument/dataproduct/calib/"
                f"proposal filters ({active_filters or 'none'}). The MAST search is "
                "empty — loosen those filters (e.g. DROP calib_levels; JWST science "
                "is calib_level 3, and calib_level=1 matches almost nothing)."
            )
        if obs_types and pairs_before_type_filter:
            head.append(
                f"  • {pairs_before_type_filter} (planet, observation) pairs matched, "
                f"but none is of type {obs_types}. Rerun without obs_types (or "
                "group_by=['obs_type']) to see what they are; 'unknown' means the "
                "planet has no transit ephemeris."
            )
        elif planet_count > 0 and planets_with_coords and obs_count > 0:
            head.append(
                f"  • Both sides are non-empty ({planet_count} planets, {obs_count} "
                f"observations) but none fall within {radius_deg} deg of each other. "
                "Try a LARGER radius_deg (e.g. 0.05), or these two populations "
                "genuinely do not overlap on-sky."
            )
        head.append(
            "  Do NOT report 'no planets have JWST data' from an empty result "
            "without identifying which stage above was empty."
        )
        return "\n".join(head)

    head.append(f"Preview — first {min(len(rows), limit)} of {len(rows)} rows:")
    for i, row in enumerate(rows[:limit], start=1):
        head.append(
            f"{i:3}. pl_name={row.get('pl_name','?')} "
            f"inst={row.get('instrument_name','?')} "
            f"filters={row.get('filters','?')} "
            f"obsid={row.get('obsid','?')} "
            f"proposal_id={row.get('proposal_id','?')} "
            f"cycle={row.get('cycle_number','?')} "
            f"obs_type={row.get('obs_type','?')} "
            + (f"raw={row['raw_data']} " if row.get("raw_data") else "")
            + f"pi={row.get('proposal_pi','?')}"
        )
    if len(rows) > limit:
        head.append(
            f"... ({len(rows) - limit} more rows in CSV — not shown here)"
        )
    return "\n".join(head)


def _format_aggregate_summary(
    groups: list[dict[str, Any]],
    *,
    group_by: list[str],
    distinct_fields: list[str],
    total_rows: int,
    limit: int = 40,
    csv_path: Path | None = None,
) -> str:
    head: list[str] = []
    if csv_path is not None:
        head.append(
            f"FULL RESULTS ({len(groups)} groups) saved to CSV: {csv_path}"
        )
        head.append(
            "Read that file (e.g. pandas.read_csv) for the complete group "
            "table. The preview below is TRUNCATED — do NOT reconstruct "
            "downstream files from it."
        )
        head.append("")
    head.extend(
        [
            f"Aggregated {total_rows} row(s) by {group_by}.",
            f"Distinct fields tracked: {distinct_fields or '[]'}",
            f"Groups: {len(groups)}",
            "",
        ]
    )
    if not groups:
        head.append("No groups.")
        return "\n".join(head)

    head.append(
        f"Preview — top {min(len(groups), limit)} of {len(groups)} groups "
        f"(by count):"
    )
    for i, g in enumerate(groups[:limit], start=1):
        key_part = " ".join(f"{k}={g.get(k)!r}" for k in group_by)
        distinct_part = " ".join(
            f"{f}_distinct={g.get(f + '_distinct')}" for f in distinct_fields
        )
        line = f"{i:3}. {key_part} count={g['count']}"
        if distinct_part:
            line += f"  {distinct_part}"
        head.append(line)
    if len(groups) > limit:
        head.append(
            f"... ({len(groups) - limit} more groups in CSV — not shown here)"
        )
    return "\n".join(head)


def download_demographic_products(
    rows: list[dict[str, Any]],
    output_directory: str | os.PathLike[str],
    *,
    label: str = "aggregate",
    product_types: list[str] | tuple[str, ...] | str | None = SCIENCE_PRODUCT_TYPES,
    product_subgroups: list[str] | tuple[str, ...] | str | None = None,
    raw_only: bool = False,
    extensions: list[str] | tuple[str, ...] | str | None = FITS_EXTENSIONS,
    max_planets: int | None = None,
    max_obs_per_planet: int | None = None,
    max_products_per_obs: int | None = None,
    product_dataproduct_types: list[str] | tuple[str, ...] | str | None = None,
    session: requests.Session | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """
    Download JWST products for an entire demographic population, organized per
    planet on disk.

    ``rows`` must contain ``pl_name`` and ``obsid`` columns (the schema written
    by a ``SearchMastJwstObservations`` population search). Rows are grouped by planet name; for each
    planet, the unique obsids are downloaded under
    ``{output_directory}/{label}/{pl_name}/{obsid}/``. A per-planet
    ``manifest.json`` is written by the underlying batch downloader; a global
    ``demographic_manifest.json`` summarizing the whole demographic is written
    at the label root.
    """
    by_planet: dict[str, list[str]] = {}
    for row in rows:
        pl = row.get("pl_name") or row.get("planet_name")
        obsid = row.get("obsid")
        if not pl or obsid in (None, ""):
            continue
        bucket = by_planet.setdefault(str(pl), [])
        if str(obsid) not in bucket:
            bucket.append(str(obsid))

    if max_planets is not None:
        by_planet = dict(list(by_planet.items())[:max_planets])

    root = Path(output_directory) / _sanitize_path_component(label)
    root.mkdir(parents=True, exist_ok=True)

    per_planet_summary: list[dict[str, Any]] = []
    total_files = 0
    for pl_name, obsids in by_planet.items():
        if max_obs_per_planet is not None:
            obsids = obsids[:max_obs_per_planet]

        sub_manifest = download_observations_products(
            obsids,
            root,
            product_types=product_types,
            product_subgroups=product_subgroups,
            raw_only=raw_only,
            extensions=extensions,
            max_products_per_obs=max_products_per_obs,
            product_dataproduct_types=product_dataproduct_types,
            session=session,
            timeout=timeout,
            label=pl_name,
        )
        files = len(sub_manifest.get("downloaded", []))
        total_files += files
        per_planet_summary.append(
            {
                "pl_name": pl_name,
                "obsid_count": len(obsids),
                "files_downloaded": files,
                "directory": str(root / _sanitize_path_component(pl_name)),
            }
        )

    manifest = {
        "label": label,
        "planet_count": len(per_planet_summary),
        "total_files_downloaded": total_files,
        "per_planet": per_planet_summary,
    }
    with (root / "demographic_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def _format_demographic_summary(manifest: dict[str, Any], limit: int = 40) -> str:
    label = manifest.get("label", "?")
    planet_count = manifest.get("planet_count", 0)
    total_files = manifest.get("total_files_downloaded", 0)
    per_planet = manifest.get("per_planet", [])

    lines = [
        f"Downloaded JWST data for demographic '{label}'.",
        f"Planets processed: {planet_count}",
        f"Total files downloaded: {total_files}",
        "",
        f"Per-planet (first {min(len(per_planet), limit)}):",
    ]
    for entry in per_planet[:limit]:
        lines.append(
            f"  - {entry['pl_name']} obsids={entry['obsid_count']} "
            f"files={entry['files_downloaded']}  dir={entry['directory']}"
        )
    if len(per_planet) > limit:
        lines.append(f"... ({len(per_planet) - limit} more truncated)")
    return "\n".join(lines)


# -------------------- BaseTools --------------------


def _obsids_by_planet(
    rows: list[dict[str, Any]],
    *,
    max_planets: int | None = None,
    max_obs_per_planet: int | None = None,
) -> dict[str, list[str]]:
    """Group the unique obsids of crossmatch rows by planet (first-seen order).

    Rows without a ``pl_name`` / ``planet_name`` column are grouped under the
    empty key ``""`` so a plain ``SearchMastJwstObservations`` CSV (no planet
    attribution) still downloads, as one flat batch.
    """
    by_planet: dict[str, list[str]] = {}
    for row in rows:
        obsid = row.get("obsid")
        if obsid in (None, ""):
            continue
        pl = row.get("pl_name") or row.get("planet_name") or ""
        bucket = by_planet.setdefault(str(pl), [])
        if str(obsid) not in bucket:
            bucket.append(str(obsid))
    if max_planets is not None:
        by_planet = dict(list(by_planet.items())[:max_planets])
    if max_obs_per_planet is not None:
        by_planet = {k: v[:max_obs_per_planet] for k, v in by_planet.items()}
    return by_planet


def _format_product_listing(
    groups: list[tuple[str, list[str]]],
    *,
    product_subgroups: Any,
    raw_only: bool,
    max_products: int | None,
    limit_per_obs: int = 40,
    prefetched: dict[Any, list[dict[str, Any]]] | None = None,
    product_dataproduct_types: Any = None,
) -> str:
    """List (do not download) MAST products for obsids, grouped by a heading."""
    lines: list[str] = ["LIST ONLY — nothing was downloaded."]
    total_files = 0
    total_obs = 0
    for heading, obsids in groups:
        if heading:
            lines.append(f"\n=== {heading} ===")
        for obsid in obsids:
            total_obs += 1
            if prefetched is not None:
                products = prefetched.get((heading, obsid), [])
            else:
                products = get_observation_products(
                    obsid,
                    product_subgroups=product_subgroups,
                    raw_only=raw_only,
                    product_dataproduct_types=product_dataproduct_types,
                )
            if max_products is not None:
                products = products[:max_products]
            total_files += len(products)
            lines.append(f"\nobsid={obsid}:")
            lines.append(_format_products_summary(products, limit=limit_per_obs))
    lines.insert(
        1,
        f"Observations listed: {total_obs}; matching product files: {total_files}. "
        "Re-run with list_only=False (same arguments) to download them.",
    )
    return "\n".join(lines)


class DownloadMastJwstProducts(BaseTool):
    """
    The ONE tool for JWST product FILES on MAST: list them, or download them —
    for a target and/or program, an obsid list, or every row of a population CSV.

    USE WHEN: the user wants to see which FITS products (UNCAL, RATEINTS,
    X1DINTS, ...) exist for JWST observations, or wants them on disk —
    including "get the uncals for program 1366 / WASP-39 b / G395H" style
    requests (this tool supersedes exoTEDRF's ``download_observations``).

    NOT FOR: discovering/browsing observations (`SearchMastJwstObservations`),
    or reduced Exoplanet-Archive transit spectra (`DownloadDataset`).

    Input — supply ONE selection (precedence if several: rows_path > obsids >
    query; ignored inputs are reported in the result):
      * **query** — ``planet_name`` and/or ``proposal_id``, narrowed by
        ``instruments``, ``dataproduct_types``, ``filters`` and ``visits``:
          - with ``planet_name``: cone search around the resolved target
            (``ra``/``dec``/``radius_deg`` to override), optionally limited to
            ``proposal_id``. Writes ``{output_dir}/{planet_name}/{obs_id}/``.
          - ``proposal_id`` alone: every observation in the program. Writes
            ``{output_dir}/{label or program_<id>}/{obs_id}/``.
      * ``obsids`` — explicit MAST obsid list (one obsid is fine). Writes
        ``{output_dir}/{label}/{obsid}/``.
      * ``rows_path`` — a CSV/JSON written by
        ``SearchMastJwstObservations`` (needs an ``obsid`` column). Rows with
        ``pl_name`` download per planet under
        ``{output_dir}/{label}/{pl_name}/{obsid}/`` (+ a top-level
        ``demographic_manifest.json``); rows without it download as one flat
        batch under ``{output_dir}/{label}/{obsid}/``.
    Every run writes a ``manifest.json``. One directory per observation is
    deliberate: ``ReduceNirspecG395hTso`` takes ONE observation's segments
    per ``input_dir``, and ``DiscoverPatchworkVisits`` reads headers anyway.

    Selection
    ---------
      * ``instruments`` — bare names expand to every MAST mode ("NIRSpec");
        full modes pass through ("NIRSPEC/SLIT", "NIRSPEC/BOTS").
      * ``filters`` — grating/filter tokens matched against each
        observation's CAOM ``filters`` value, e.g. ['G395H'] keeps
        'F290LP;G395H' and drops PRISM/CLEAR; ['G395H', 'G395M'] keeps either.
      * ``visits`` — like exoTEDRF ``visit_nos``: 1-based numbers into the
        sorted distinct ``jwPPPPPOOOVVV`` prefixes of the whole selection
        (listed in the result), or literal prefixes like 'jw01366004001'
        (unambiguous; prefer these). query and
        obsids modes only. Observation-level stage-3 files have no visit
        prefix and are dropped when ``visits`` is set.
      * ``raw_only=True`` — only UNCAL ramps (what exoTEDRF downloads);
        ``product_subgroups`` for other stages (RATE, RATEINTS, CAL, CALINTS,
        X1D, X1DINTS, S2D, S3D, WHTLT). For retrievals prefer X1DINTS.

    ``list_only=True`` lists the matching files (subgroup, size, filename,
    dataURI) for the same selection WITHOUT downloading.

    Caps (always set them for UNCAL — multi-GB per observation)
    -----------------------------------------------------------
      * ``max_planets``      — rows_path mode: number of planets.
      * ``max_observations`` — observations per target/program (query and
                               rows_path modes) or first N obsids.
      * ``max_products``     — product files per observation.

    Downloads resume interrupted files, and proprietary (exclusive-access)
    products are skipped and listed in the manifest rather than aborting.

    exoTEDRF equivalence
    --------------------
        download_observations(proposal_id='1366', objectname='WASP-39b',
                              instrument_name='NIRSPEC/SLIT', filters='G395H')
      is
        DownloadMastJwstProducts(proposal_id="1366", planet_name="WASP-39 b",
                                 instruments=["NIRSPEC/SLIT"], filters=["G395H"],
                                 dataproduct_types=["timeseries"], raw_only=True)
      with files under mast/WASP-39_b/<obs_id>/ instead of ./DMS_uncal.
      Target-acquisition exposures (MAST types them 'timeseries' too) are
      dropped in every mode: any exposure whose own filters differ from the
      observation's (e.g. F110W;MIRROR under F290LP;G395H) is left out.

    More examples
    -------------
        DownloadMastJwstProducts(obsids="98765", raw_only=True, list_only=True)
        DownloadMastJwstProducts(proposal_id="1366", filters=["G395H"],
                                 dataproduct_types=["timeseries"],
                                 raw_only=True, visits=[1])
        DownloadMastJwstProducts(rows_path="mast/demographics/subneptunes.csv",
                                 label="subneptunes", raw_only=True,
                                 max_observations=1, max_products=2)
    """

    planet_name: str | None = OptionalRuntimeField(
        description="query: target name, e.g. 'WASP-39 b' (exoTEDRF objectname).",
    )
    proposal_id: str | None = OptionalRuntimeField(
        description="query: JWST program id, e.g. '1366'. Alone, selects the whole program.",
    )
    obsids: list | str | None = OptionalRuntimeField(
        description="obsids mode: one or more MAST obsids, e.g. ['87602456'].",
    )
    rows_path: str | None = OptionalRuntimeField(
        description=(
            "rows_path mode: CSV/JSON (relative to the workspace) from "
            "SearchMastJwstObservations; must have "
            "an obsid column (pl_name enables the per-planet layout)."
        ),
    )
    list_only: bool | str = RuntimeField(
        default=False,
        description="If True, list matching product files without downloading.",
    )
    output_dir: str | None = OptionalRuntimeField(
        description=(
            "Output directory relative to the workspace. Default 'mast' "
            "(query / obsids) or 'mast/demographics_raw' (rows_path)."
        ),
    )
    label: str | None = OptionalRuntimeField(
        description=(
            "Subdirectory label for obsids, rows_path and proposal-only query "
            "modes. Default 'aggregate' (or 'program_<id>')."
        ),
    )
    ra: float | str | None = OptionalRuntimeField(
        description="query: RA in degrees if MAST cannot resolve planet_name.",
    )
    dec: float | str | None = OptionalRuntimeField(
        description="query: Dec in degrees if MAST cannot resolve planet_name.",
    )
    radius_deg: float | str = RuntimeField(
        default=0.02,
        description="query: cone-search radius in degrees around planet_name.",
    )
    instruments: list | str | None = OptionalRuntimeField(
        description="query: instruments or modes, e.g. ['NIRSpec'] or ['NIRSPEC/SLIT'].",
    )
    dataproduct_types: list | str | None = OptionalRuntimeField(
        description=(
            "Dataproduct filter, e.g. ['timeseries']. Narrows the query AND each "
            "product's own type; the product-level part applies in every mode. "
            "Target-acquisition exposures are dropped automatically regardless."
        ),
    )
    filters: list | str | None = OptionalRuntimeField(
        description="query: grating/filter tokens, e.g. ['G395H'] (any-of).",
    )
    visits: list | str | None = OptionalRuntimeField(
        description=(
            "query/obsids: 1-based visit numbers (exoTEDRF visit_nos) or "
            "prefixes like 'jw01366004001'."
        ),
    )
    product_subgroups: list | str | None = OptionalRuntimeField(
        description="JWST product subgroup filters, e.g. ['X1DINTS'] or ['UNCAL'].",
    )
    raw_only: bool | str = RuntimeField(
        default=False,
        description="If True, only raw JWST UNCAL files.",
    )
    max_planets: int | str | None = OptionalRuntimeField(
        description="rows_path mode: cap on the number of planets.",
    )
    max_observations: int | str | None = OptionalRuntimeField(
        description="Cap on observations per target/program (or on obsids in obsids mode).",
    )
    max_products: int | str | None = OptionalRuntimeField(
        description="Cap on product files per observation.",
    )
    base_directory: str = StateField()

    def _query_observations(self, planet_name, proposal_id, ra, dec, radius_deg,
                            filters, max_observations):
        """Observation rows for query mode (planet cone and/or program)."""
        if planet_name:
            rows = search_jwst_observations(
                planet_name, ra=ra, dec=dec, radius_deg=radius_deg,
                instruments=self.instruments,
                dataproduct_types=self.dataproduct_types,
                proposal_id=proposal_id,
            )
        else:
            rows, _ = search_all_jwst_observations(
                instruments=self.instruments,
                dataproduct_types=self.dataproduct_types,
                proposal_id=proposal_id,
            )
        rows = [r for r in rows if _filters_match(r.get("filters"), filters)]
        if max_observations is not None:
            rows = rows[:max_observations]
        return rows

    def _run(self) -> str:
        # Normalize LLM-supplied args: '' -> None, 'True' -> True, '10' -> 10.
        planet_name = _clean_str(self.planet_name)
        proposal_id = _clean_str(self.proposal_id)
        rows_path = _clean_str(self.rows_path)
        obsids_list = [o for o in (_as_list(self.obsids) or []) if _clean_str(o)]
        list_only = _as_bool(self.list_only, "list_only", default=False)
        raw_only = _as_bool(self.raw_only, "raw_only", default=False)
        max_planets = _as_int_or_none(self.max_planets, "max_planets")
        max_observations = _as_int_or_none(self.max_observations, "max_observations")
        max_products = _as_int_or_none(self.max_products, "max_products")
        radius_deg = _as_float_or_none(self.radius_deg, "radius_deg")
        if radius_deg is None:
            radius_deg = 0.02
        filters = [f for f in (_as_list(self.filters) or []) if _clean_str(f)] or None
        visits = _normalize_visits(self.visits)
        if proposal_id is not None:
            if not re.fullmatch(r"\d{1,5}", proposal_id):
                raise ValueError(
                    f"proposal_id must be a JWST program number, got {proposal_id!r}."
                )
            # MAST stores proposal_id without leading zeros ('1366', not
            # '01366'); the server-side filter is exact-match.
            proposal_id = str(int(proposal_id))

        query_inputs = [n for n, v in (("planet_name", planet_name),
                                       ("proposal_id", proposal_id)) if v]
        supplied = [
            name for name, value in (
                ("rows_path", rows_path),
                ("obsids", obsids_list),
                ("query", query_inputs),
            ) if value
        ]
        if not supplied:
            raise ValueError(
                "Provide one of: planet_name and/or proposal_id (query), obsids "
                "(explicit list), or rows_path (population CSV). All are missing."
            )
        mode = supplied[0]
        ignored = [("+".join(query_inputs) if n == "query" else n) for n in supplied[1:]]
        if mode != "query":
            ignored += [n for n, v in (("filters", filters),) if v]
        if mode == "rows_path" and visits:
            raise ValueError(
                "visits is not supported with rows_path (visit numbers would be "
                "ambiguous across planets). Use obsids or a query instead."
            )
        note = (
            f"NOTE: ran in {mode} mode; ignored {', '.join(ignored)}.\n\n"
            if ignored else ""
        )

        default_dir = "mast/demographics_raw" if mode == "rows_path" else "mast"
        output_root = os.path.join(
            self.base_directory, _clean_str(self.output_dir) or default_dir
        )
        label = _clean_str(self.label)
        # Applied to each PRODUCT's dataproduct_type as well as to the
        # observation query. (Target acquisition is removed separately, by
        # get_observation_products: MAST types it 'timeseries' as well.)
        dptypes = [d for d in (_as_list(self.dataproduct_types) or []) if _clean_str(d)] or None

        def listing(groups):
            if visits:
                pairs = [
                    (heading, obsid, get_observation_products(
                        obsid, product_subgroups=self.product_subgroups,
                        raw_only=raw_only, product_dataproduct_types=dptypes))
                    for heading, ids in groups for obsid in ids
                ]
                kept, available, chosen = select_products_by_visit(
                    [((h, o), p) for h, o, p in pairs], visits)
                lookup = {key: prods for key, prods in kept}
                text = _format_product_listing(
                    groups, product_subgroups=self.product_subgroups,
                    raw_only=raw_only, max_products=max_products,
                    prefetched=lookup,
                )
                return text + (f"\n\nVisits available: {available}"
                               f"\nVisits selected:  {chosen}")
            return _format_product_listing(
                groups, product_subgroups=self.product_subgroups,
                raw_only=raw_only, max_products=max_products,
                product_dataproduct_types=dptypes,
            )

        if mode == "rows_path":
            label = label or "aggregate"
            rows = _read_rows_from_path(Path(self.base_directory) / rows_path)
            if rows and "obsid" not in rows[0]:
                raise ValueError(
                    f"{rows_path} has no 'obsid' column (columns: "
                    f"{sorted(rows[0])[:12]}...). Pass the CSV written by "
                    "SearchMastJwstObservations."
                )
            by_planet = _obsids_by_planet(
                rows, max_planets=max_planets, max_obs_per_planet=max_observations
            )
            if list_only:
                return note + listing([(pl or label, ids) for pl, ids in by_planet.items()])
            if set(by_planet) == {""}:
                manifest = download_observations_products(
                    by_planet[""], output_root,
                    product_subgroups=self.product_subgroups, raw_only=raw_only,
                    max_products_per_obs=max_products, label=label,
                    product_dataproduct_types=dptypes,
                )
                return note + _format_download_manifest(manifest)
            manifest = download_demographic_products(
                rows, output_root, label=label,
                product_subgroups=self.product_subgroups, raw_only=raw_only,
                max_planets=max_planets, max_obs_per_planet=max_observations,
                max_products_per_obs=max_products,
                product_dataproduct_types=dptypes,
            )
            return note + _format_demographic_summary(manifest)

        if mode == "obsids":
            label = label or "aggregate"
            obsids_list = [str(o) for o in obsids_list]
            if max_observations is not None:
                obsids_list = obsids_list[:max_observations]
            if list_only:
                return note + listing([("", obsids_list)])
            manifest = download_observations_products(
                obsids_list, output_root,
                product_subgroups=self.product_subgroups, raw_only=raw_only,
                max_products_per_obs=max_products, visits=visits,
                product_dataproduct_types=dptypes, label=label,
            )
            return note + _format_download_manifest(manifest)

        # query mode: planet_name and/or proposal_id
        ra = _as_float_or_none(self.ra, "ra")
        dec = _as_float_or_none(self.dec, "dec")
        describe = " / ".join(
            ([planet_name] if planet_name else [])
            + ([f"program {proposal_id}"] if proposal_id else [])
        )
        if list_only or not planet_name:
            observations = self._query_observations(
                planet_name, proposal_id, ra, dec, radius_deg, filters, max_observations
            )
            obsids_found = [
                str(o["obsid"]) for o in observations if o.get("obsid") is not None
            ]
            if not obsids_found:
                return note + (
                    f"No JWST observations found for {describe} with these "
                    "filters. Run SearchMastJwstObservations for diagnostics."
                )
            if list_only:
                return note + listing([(describe, obsids_found)])
            manifest = download_observations_products(
                obsids_found, output_root,
                product_subgroups=self.product_subgroups, raw_only=raw_only,
                max_products_per_obs=max_products, visits=visits,
                product_dataproduct_types=dptypes,
                obs_dir_names={
                    str(o["obsid"]): str(o.get("obs_id") or o["obsid"])
                    for o in observations if o.get("obsid") is not None
                },
                label=label or f"program_{proposal_id}",
            )
            return note + _format_download_manifest(manifest)

        manifest = download_planet_jwst_products(
            planet_name, output_root,
            ra=ra, dec=dec, radius_deg=radius_deg,
            instruments=self.instruments,
            dataproduct_types=self.dataproduct_types,
            proposal_id=proposal_id, filters=filters,
            product_subgroups=self.product_subgroups, raw_only=raw_only,
            max_observations=max_observations, max_products=max_products,
            visits=visits,
        )
        return note + _format_download_manifest(manifest)


class GetJwstProgramInfo(BaseTool):
    """
    Look up authoritative JWST program metadata — observing Cycle, title, PI,
    proposal type (GO/GTO/DD/CAL), status, exclusive-access period — for one
    or more proposal ids, from STScI's program-info pages.

    USE WHEN: you have a `proposal_id` (e.g. from `SearchMastJwstObservations`)
    and need its cycle, title, PI, or status.

    NOT FOR: finding observations (use `SearchMastJwstObservations`) or planet
    parameters (use `QueryExoplanetArchive`). Requires a proposal id, not a
    planet name.

    Why this tool exists
    --------------------
    MAST CAOM metadata has NO cycle field, and the cycle CANNOT be computed
    from the proposal id (GO 2372 is Cycle 1; GO 3557 and GO 4098 are
    Cycle 2). Use this tool whenever a task needs cycle numbers — never guess
    them. ``SearchMastJwstObservations`` calls the same lookup automatically for
    its ``cycle_number`` column; use this tool for ad-hoc questions ("which
    cycle is program 5959?") or to pre-warm the cache.

    Caching
    -------
    Results are cached permanently in
    ``{base_directory}/mast/jwst_program_info_cache.json`` (a proposal's
    cycle never changes). Only the first lookup per proposal touches the
    network. Set ``refresh=True`` to force refetch (e.g. to update a
    program's completion status).

    Example
    -------
        GetJwstProgramInfo(proposal_ids=[2372, 3557, 4098])
    """

    proposal_ids: list | str = RuntimeField(
        description=(
            "One or more JWST proposal ids, e.g. [2372, 3557, 4098] or '2372'."
        ),
    )
    refresh: bool | str = RuntimeField(
        default=False,
        description="If True, bypass the cache and refetch from STScI.",
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        ids = _as_list(self.proposal_ids)
        if not ids:
            raise ValueError("proposal_ids is required (one or more JWST proposal ids).")
        refresh = _as_bool(self.refresh, "refresh", default=False)
        cache_path = Path(self.base_directory or ".") / JWST_PROGRAM_INFO_CACHE

        lines: list[str] = []
        failures: list[str] = []
        for raw in ids:
            pid = _clean_str(raw)
            if pid is None:
                continue
            try:
                info = get_jwst_program_info(
                    pid, cache_path=cache_path, refresh=refresh
                )
            except Exception as exc:
                failures.append(f"  - {pid}: {exc}")
                continue
            lines.append(
                f"  - {info['proposal_id']}: Cycle {info['cycle']} | "
                f"{info.get('proposal_type') or '?'} | "
                f"PI {info.get('pi') or '?'} | "
                f"{info.get('title') or '?'} | "
                f"status: {info.get('status') or '?'}"
            )

        out: list[str] = []
        if lines:
            out.append(f"JWST program info ({len(lines)} proposal(s)):")
            out.extend(lines)
            out.append(f"(cached in {cache_path})")
        if failures:
            out.append("Lookups FAILED (no cycle available — do not guess):")
            out.extend(failures)
        return "\n".join(out) or "No valid proposal ids supplied."
