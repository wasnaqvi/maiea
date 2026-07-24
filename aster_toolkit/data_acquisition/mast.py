from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import ast
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

try:
    from orchestral.tools.base.tool import BaseTool
    from orchestral.tools.base.field_utils import RuntimeField, StateField
    from pydantic import Field as _PydanticField

    def OptionalRuntimeField(description=None, **kwargs):
        """LLM-visible field that is OPTIONAL in the tool schema.

        Orchestral's SchemaGenerator marks a runtime field as required unless it
        has a non-None default or a default_factory. ``RuntimeField(default=None)``
        therefore produces a REQUIRED field, and ``BaseTool.execute()`` rejects
        calls that omit it ("Missing Required Fields"). Since RuntimeField
        auto-injects ``default=None`` (which pydantic forbids alongside
        ``default_factory``), we build the pydantic Field directly with the
        ``runtime: True`` marker and a factory returning None.
        """
        kwargs.pop("default", None)
        return _PydanticField(
            json_schema_extra={"runtime": True},
            default_factory=lambda: None,
            description=description,
            **kwargs,
        )
except ModuleNotFoundError:
    class BaseTool:
        """Fallback that keeps plain MAST wrappers importable without Orchestral."""

    def RuntimeField(default=None, description=None):
        return default

    def StateField(default=None, description=None):
        return default

    def OptionalRuntimeField(description=None, **kwargs):
        return None


MAST_INVOKE_URL = "https://mast.stsci.edu/api/v0/invoke"
MAST_DOWNLOAD_URL = "https://mast.stsci.edu/api/v0.1/Download/file"
EXOARCHIVE_TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

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
        if expected_total is not None and len(all_rows) < expected_total:
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
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """Return downloadable MAST products for one observation id."""
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
    )


def filter_products(
    products: list[dict[str, Any]],
    *,
    product_types: list[str] | tuple[str, ...] | str | None = SCIENCE_PRODUCT_TYPES,
    product_subgroups: list[str] | tuple[str, ...] | str | None = None,
    raw_only: bool = False,
    extensions: list[str] | tuple[str, ...] | str | None = FITS_EXTENSIONS,
    skip_proprietary: bool = True,
) -> list[dict[str, Any]]:
    """Filter MAST products by science type, JWST subgroup, raw status, file
    extension, and proprietary-access status.

    ``skip_proprietary=True`` (default) drops products whose ``dataRights``
    is not ``PUBLIC`` — typically ``EXCLUSIVE_ACCESS`` files that would
    otherwise return HTTP 401 from the MAST download endpoint. Set to False
    only when the caller is authenticated with appropriate MAST credentials.
    """
    type_values = {value.upper() for value in _as_list(product_types) or []}
    subgroup_values = {value.upper() for value in _as_list(product_subgroups) or []}
    extension_values = tuple(value.lower() for value in (_as_list(extensions) or []))

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

        selected.append(product)

    return selected


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


def download_observations_products(
    obsids: list[str | int] | tuple[str | int, ...],
    output_directory: str | os.PathLike[str],
    *,
    product_types: list[str] | tuple[str, ...] | str | None = SCIENCE_PRODUCT_TYPES,
    product_subgroups: list[str] | tuple[str, ...] | str | None = None,
    raw_only: bool = False,
    extensions: list[str] | tuple[str, ...] | str | None = FITS_EXTENSIONS,
    max_products_per_obs: int | None = None,
    session: requests.Session | None = None,
    timeout: float = 120.0,
    download_timeout: float | tuple[float, float] = (30.0, 600.0),
    label: str = "aggregate",
) -> dict[str, Any]:
    """
    Batch-download products for a fixed list of MAST obsids.

    Writes files under ``{output_directory}/{label}/{obs_id_or_obsid}/`` and a
    ``manifest.json`` capturing every download. Used by demographics workflows
    where obsids come from a no-position filtered search rather than a cone
    search around one planet.
    """
    target_dir = Path(output_directory) / _sanitize_path_component(label)
    target_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[dict[str, Any]] = []
    skipped_proprietary: list[dict[str, Any]] = []
    for obsid in obsids:
        if obsid is None:
            continue

        products = get_observation_products(
            obsid,
            product_types=product_types,
            product_subgroups=product_subgroups,
            raw_only=raw_only,
            extensions=extensions,
            session=session,
            timeout=timeout,
        )

        if max_products_per_obs is not None:
            products = products[:max_products_per_obs]

        obs_dir_name = _sanitize_path_component(str(obsid))
        for product in products:
            data_uri = product.get("dataURI")
            if not data_uri:
                continue

            try:
                local_path = download_mast_product(
                    str(data_uri),
                    target_dir / obs_dir_name,
                    filename=product.get("productFilename"),
                    session=session,
                    timeout=download_timeout,
                )
            except ProprietaryProductError as exc:
                skipped_proprietary.append(
                    {
                        "obsid": str(obsid),
                        "product": product,
                        "reason": str(exc),
                    }
                )
                continue
            downloaded.append(
                {
                    "obsid": str(obsid),
                    "product": product,
                    "path": str(local_path),
                }
            )

    manifest = {
        "label": label,
        "obsids": [str(o) for o in obsids if o is not None],
        "downloaded": downloaded,
        "skipped_proprietary": skipped_proprietary,
    }

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
    product_types: list[str] | tuple[str, ...] | str | None = SCIENCE_PRODUCT_TYPES,
    product_subgroups: list[str] | tuple[str, ...] | str | None = None,
    raw_only: bool = False,
    extensions: list[str] | tuple[str, ...] | str | None = FITS_EXTENSIONS,
    max_observations: int | None = None,
    max_products: int | None = None,
    session: requests.Session | None = None,
    timeout: float = 120.0,
    download_timeout: float | tuple[float, float] = (30.0, 600.0),
) -> dict[str, Any]:
    """
    Search for JWST observations of one planet target and download selected products.

    The returned manifest is JSON-serializable and records the observation rows,
    product metadata, and local file paths.
    """
    observations = search_jwst_observations(
        planet_name,
        ra=ra,
        dec=dec,
        radius_deg=radius_deg,
        instruments=instruments,
        dataproduct_types=dataproduct_types,
        calib_levels=calib_levels,
        session=session,
        timeout=timeout,
    )
    if max_observations is not None:
        observations = observations[:max_observations]

    target_dir = Path(output_directory) / _sanitize_path_component(planet_name)
    downloaded: list[dict[str, Any]] = []
    skipped_proprietary: list[dict[str, Any]] = []

    for observation in observations:
        obsid = observation.get("obsid")
        if obsid is None:
            continue

        products = get_observation_products(
            obsid,
            product_types=product_types,
            product_subgroups=product_subgroups,
            raw_only=raw_only,
            extensions=extensions,
            session=session,
            timeout=timeout,
        )

        if max_products is not None:
            products = products[:max_products]

        obs_dir_name = _sanitize_path_component(str(observation.get("obs_id") or obsid))
        for product in products:
            data_uri = product.get("dataURI")
            if not data_uri:
                continue

            try:
                local_path = download_mast_product(
                    str(data_uri),
                    target_dir / obs_dir_name,
                    filename=product.get("productFilename"),
                    session=session,
                    timeout=download_timeout,
                )
            except ProprietaryProductError as exc:
                skipped_proprietary.append(
                    {
                        "observation": observation,
                        "product": product,
                        "reason": str(exc),
                    }
                )
                continue
            downloaded.append(
                {
                    "observation": observation,
                    "product": product,
                    "path": str(local_path),
                }
            )

    manifest = {
        "planet_name": planet_name,
        "observations": observations,
        "downloaded": downloaded,
        "skipped_proprietary": skipped_proprietary,
    }

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


class SearchMastJwstObservations(BaseTool):
    """
    Search MAST for JWST observations (the "who/what observed" tool).

    USE WHEN: the user asks who observed a planet, what instrument/mode was used,
    which JWST program or PI, or wants to discover JWST data for a target or a
    whole population. This is the FIRST step for any observation question.

    NOT FOR: a planet's physical parameters (`GetExoplanetParameters`). It returns
    observation rows (obsid, instrument, filters, proposal_id, PI) — not files.
    To get files for an obsid, call `GetMastObservationProducts` next.

    Three modes:
      * **Per-planet cone search** — supply ``planet_name`` (resolved to RA/Dec
        via MAST Name Lookup) or supply ``ra`` and ``dec`` directly.
      * **Population / demographics search** — omit ``planet_name``, ``ra`` and
        ``dec`` entirely. The tool runs a no-position ``Mast.Caom.Filtered``
        query so every JWST observation matching the filters is returned. Use
        for "all NIRSpec spectra" / "every JWST timeseries at calib level 3"
        style questions.
      * **Target-name filter** — supply ``target_name`` (free-text) without any
        coordinates to constrain by MAST target_name without resolving a planet.

    Workflow
    --------
    1. Call this tool to discover JWST observations (returns obsid + metadata).
    2. Pick an obsid of interest and call ``GetMastObservationProducts`` to list files.
    3. Call ``DownloadMastJwstProducts`` (one-shot) OR fetch specific products directly.

    Why coordinate-centered (per-planet mode)
    -----------------------------------------
    MAST target names are usually host-star names (e.g. 'WASP-39'), not planet names
    ('WASP-39 b'). The per-planet mode resolves the planet name via MAST Name
    Lookup and runs a cone-search by RA/Dec. If MAST cannot resolve the planet
    name, pass RA and Dec directly (look them up via the exoarchive tools or
    Simbad), or drop ``planet_name`` and use the demographics mode.

    JWST instruments
    ----------------
    Bare instrument names expand to ALL of that instrument's MAST
    instrument_name configurations (exact-match server-side vocabulary):
        - "NIRSpec"  : BOTS, SLIT, IFU, MSA, IMAGE (BOTS = bright-object time
                       series — virtually all NIRSpec transit/eclipse data)
        - "NIRCam"   : IMAGE, GRISM, WFSS, CORON, TARGACQ
        - "MIRI"     : SLITLESS, SLIT, IFU (MRS), IMAGE, CORON, TARGACQ
                       (LRS TSO data is MIRI/SLITLESS)
        - "NIRISS"   : SOSS, IMAGE, WFSS, AMI
        - "FGS"      : fine guidance sensor (rarely needed for science)
    Full mode strings pass through unchanged (e.g. "NIRSPEC/BOTS"), and the
    shorthands "BOTS", "SOSS", "MRS", "LRS" map to their modes.

    Zero-result behavior
    --------------------
    If the strict server-side query returns nothing, the tool re-runs the cone
    unfiltered and re-applies your filters client-side (recovering data hidden
    by MAST vocabulary drift), and otherwise reports what instruments /
    proposals / product types ARE present in the cone — or where a given
    proposal's observations actually live — so the next query can be corrected.

    Data product types
    ------------------
    Valid values for ``dataproduct_types``:
        - "spectrum"   : 1-D extracted spectra
        - "timeseries" : time-series exposures (transits/eclipses)
        - "image"      : 2-D images
        - "cube"       : 3-D IFU cubes

    Calibration levels
    ------------------
    ``calib_level`` is the OBSERVATION's highest processing stage — NOT a way to
    select raw vs calibrated FILES. Valid values:
        - 1 : uncalibrated observation (JWST science obs are almost NEVER filed here)
        - 2 : per-exposure calibrated products
        - 3 : combined / extracted science-ready products (the usual JWST value)
        - 4 : community / contributed products

    IMPORTANT — getting RAW / uncalibrated (UNCAL) data:
        Do NOT filter ``calib_levels=[1]`` to find raw data. A calib_level-3
        observation still CONTAINS the raw UNCAL ramps as products. To fetch
        them, search WITHOUT a calib_level filter, then call
        ``GetMastObservationProducts(obsid, raw_only=True)`` (or
        ``DownloadMastJwstProducts(..., raw_only=True)``). Filtering by
        calib_level=1 will return nothing and falsely imply "no raw data exists".

    Returns
    -------
    A line-per-observation summary including obsid, instrument, dataproduct_type,
    filters, calib_level, proposal_id, and target_name. Use the obsid values
    with ``GetMastObservationProducts`` or ``DownloadMastJwstProducts``.

    Examples
    --------
    Per-planet cone search:
        SearchMastJwstObservations(
            planet_name="WASP-39 b",
            instruments=["NIRSpec"],
            dataproduct_types=["spectrum", "timeseries"],
            calib_levels=[3],
        )

    Population / demographics search (no planet_name, no ra/dec):
        SearchMastJwstObservations(
            instruments=["NIRSpec", "NIRCam", "MIRI", "NIRISS"],
            dataproduct_types=["spectrum", "timeseries"],
            calib_levels=[3],
        )
    """

    planet_name: str | None = OptionalRuntimeField(
        description=(
            "Exoplanet or host-star target name, e.g. 'WASP-39 b'. Leave None "
            "to run a no-position demographics query over every JWST observation "
            "matching the filters."
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
        description="Cone-search radius in degrees (only used when planet_name or ra/dec are set).",
    )
    instruments: list | str | None = OptionalRuntimeField(
        description="JWST instrument filters, e.g. ['NIRSpec', 'NIRCam', 'MIRI'].",
    )
    dataproduct_types: list | str | None = OptionalRuntimeField(
        description="MAST dataproduct filters, e.g. ['spectrum', 'timeseries', 'image'].",
    )
    calib_levels: list | str | None = OptionalRuntimeField(
        description=(
            "Optional MAST calibration levels (observation processing stage; "
            "JWST science is usually 3). Do NOT use calib_levels=[1] to find raw "
            "data — UNCAL products live under calib_level 2/3 observations; fetch "
            "them with raw_only=True on the products/download tools instead."
        ),
    )
    proposal_id: str | None = OptionalRuntimeField(
        description="Optional JWST proposal id filter.",
    )
    target_name: str | None = OptionalRuntimeField(
        description=(
            "Optional free-text target-name filter (MAST target_name). Used "
            "directly in demographics mode, or combined with cone search via "
            "target_name_filter=True semantics."
        ),
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
                "GetMastObservationProducts(obsid, raw_only=True) or "
                "DownloadMastJwstProducts(..., raw_only=True). An empty result "
                "here does NOT mean uncalibrated data is unavailable."
            )

        if planet_name is None and ra is None and dec is None:
            observations, filters = search_all_jwst_observations(
                instruments=self.instruments,
                dataproduct_types=self.dataproduct_types,
                calib_levels=self.calib_levels,
                target_name=target_name,
                proposal_id=proposal_id,
            )
            summary = _format_observations_summary(
                observations,
                filters=filters,
                query_extra={"mode": "demographics (Mast.Caom.Filtered, no position)"},
            )
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
                summary = f"{summary}\n\n{diagnostics}"
            return summary + raw_note

        if (ra is None) != (dec is None):
            raise ValueError(
                "Pass both ra and dec (or neither). A planet_name alone is "
                "also fine — it is resolved via MAST Name Lookup."
            )

        observations = search_jwst_observations(
            planet_name,
            ra=ra,
            dec=dec,
            radius_deg=radius_deg,
            instruments=self.instruments,
            dataproduct_types=self.dataproduct_types,
            calib_levels=self.calib_levels,
            proposal_id=proposal_id,
        )
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
        }
        if observations:
            return _format_observations_summary(
                observations, filters=filters, query_extra=query_extra
            ) + raw_note

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
        if recovered:
            summary = _format_observations_summary(
                recovered, filters=filters, query_extra=query_extra
            )
            return f"{diagnostics}\n\n{summary}{raw_note}"
        empty_summary = _format_observations_summary(
            [], filters=filters, query_extra=query_extra
        )
        return f"{empty_summary}\n\n{diagnostics}{raw_note}"


class GetMastObservationProducts(BaseTool):
    """
    List downloadable MAST product files for one JWST observation id (obsid).

    USE WHEN: you already have an `obsid` (from `SearchMastJwstObservations`) and
    want to see the available files/pipeline stages before downloading.

    NOT FOR: discovering observations (use `SearchMastJwstObservations`) or
    actually downloading (use `DownloadMastJwstProducts`). Requires a real obsid.

    Workflow
    --------
    Call ``SearchMastJwstObservations`` first to obtain obsid values, then
    invoke this tool to inspect available product files before downloading.

    JWST product subgroups
    ----------------------
    Choose ``product_subgroups`` based on the pipeline stage you need:

    Raw (pipeline input):
        - "UNCAL"     : raw uncalibrated ramps (use ``raw_only=True`` shortcut)

    Stage 1 (countrate):
        - "RATE"      : countrate per exposure
        - "RATEINTS"  : countrate per integration (time series)

    Stage 2 (calibrated per-exposure):
        - "CAL"       : calibrated image / spectrum per exposure
        - "CALINTS"   : calibrated per integration

    Stage 3 (science-ready):
        - "X1D"       : extracted 1-D spectrum (per exposure)
        - "X1DINTS"   : extracted 1-D spectrum per integration (transit/eclipse)
        - "S2D"       : resampled 2-D spectrum
        - "S3D"       : IFU spectral cube
        - "WHTLT"     : white-light curve

    For atmospheric retrievals, prefer ``X1DINTS`` (time-resolved) or ``X1D``.
    For full-reduction-from-scratch workflows, use ``raw_only=True`` for UNCAL.

    Returns
    -------
    A line-per-product summary with subgroup, file size, filename, and dataURI.
    Pass any dataURI to a download helper to fetch the file.

    Example
    -------
        GetMastObservationProducts(obsid="98765", product_subgroups=["X1DINTS"])
        GetMastObservationProducts(obsid="98765", raw_only=True)  # UNCAL shortcut
    """

    obsid: str = RuntimeField(
        description="MAST observation id from a JWST observation search result."
    )
    product_subgroups: list | str | None = OptionalRuntimeField(
        description="Optional JWST product subgroup filters, e.g. ['UNCAL', 'RATEINTS', 'X1DINTS'].",
    )
    raw_only: bool | str = RuntimeField(
        default=False,
        description="If True, return only raw JWST UNCAL products.",
    )

    def _run(self) -> str:
        obsid = _clean_str(self.obsid)
        if obsid is None:
            raise ValueError("obsid is required (from a JWST observation search result).")
        products = get_observation_products(
            obsid,
            product_subgroups=self.product_subgroups,
            raw_only=_as_bool(self.raw_only, "raw_only", default=False),
        )
        return _format_products_summary(products)


class DownloadMastJwstProducts(BaseTool):
    """
    One-shot download of JWST FITS products for ONE planet or an obsid list.

    USE WHEN: the user wants JWST FITS files (X1DINTS, UNCAL, …) for a single
    target, or for a specific list of obsids.

    NOT FOR: reduced Exoplanet-Archive transit spectra (use `DownloadDataset`),
    or bulk downloads across many planets from a crossmatch/aggregate CSV (use
    `DownloadDemographicJwstProducts`).

    Two modes:
      * **Per-planet** — supply ``planet_name``. Runs search + product listing
        + download for that target. Writes under
        ``{base_directory}/{output_dir}/{planet_name}/{obs_id}/`` with a
        ``manifest.json``.
      * **Batch by obsid** — supply ``obsids`` (list of MAST obsid strings,
        typically obtained from a demographics ``SearchMastJwstObservations``
        call). Writes under ``{base_directory}/{output_dir}/{label}/{obsid}/``
        with a ``manifest.json``.

    Workflow
    --------
    1. (Optional) Use ``SearchMastJwstObservations`` first to preview matches.
    2. Call this tool with the same filters (or with ``obsids=[...]``) to
       actually download files.
    3. Use ``max_observations`` and ``max_products`` to bound download size.

    Important warnings
    ------------------
    - Raw JWST UNCAL files are large (several GB per integration set). Always
      cap with ``max_observations`` and ``max_products`` when using ``raw_only=True``.
    - For atmospheric retrievals, you usually do NOT need raw data. Filter to
      ``product_subgroups=['X1DINTS']`` (stage-3 extracted) instead.

    JWST instruments / dataproduct types / subgroups
    ------------------------------------------------
    Same vocabulary as ``SearchMastJwstObservations`` and
    ``GetMastObservationProducts``. See those tools' docstrings for the full enum.

    Returns
    -------
    A summary listing observations matched, files downloaded, and local paths
    of every saved file. Pass any path to retrieval / analysis tools.

    Examples
    --------
    Stage-3 extracted spectra (small, retrieval-ready):
        DownloadMastJwstProducts(
            planet_name="WASP-39 b",
            instruments=["NIRSpec"],
            product_subgroups=["X1DINTS"],
            max_observations=2,
        )

    Raw UNCAL for reprocessing (cap aggressively):
        DownloadMastJwstProducts(
            planet_name="WASP-39 b",
            instruments=["NIRSpec"],
            raw_only=True,
            max_observations=1,
            max_products=2,
        )
    """

    planet_name: str | None = OptionalRuntimeField(
        description=(
            "Exoplanet or host-star target name, e.g. 'WASP-39 b'. Leave None "
            "when downloading by an explicit ``obsids`` list."
        ),
    )
    obsids: list | str | None = OptionalRuntimeField(
        description=(
            "Optional list of MAST obsids to download in batch (e.g. obsids "
            "from a demographics SearchMastJwstObservations result). When set, "
            "planet_name / ra / dec / radius_deg are ignored."
        ),
    )
    label: str = RuntimeField(
        default="aggregate",
        description="Subdirectory label used in batch (obsids) mode.",
    )
    output_dir: str = RuntimeField(
        default="mast",
        description="Output directory relative to the ASTER workspace.",
    )
    ra: float | str | None = OptionalRuntimeField(
        description="Right ascension in degrees. Optional if MAST can resolve planet_name.",
    )
    dec: float | str | None = OptionalRuntimeField(
        description="Declination in degrees. Optional if MAST can resolve planet_name.",
    )
    radius_deg: float | str = RuntimeField(
        default=0.02,
        description="Cone-search radius in degrees.",
    )
    instruments: list | str | None = OptionalRuntimeField(
        description="Optional JWST instrument filters.",
    )
    dataproduct_types: list | str | None = OptionalRuntimeField(
        description="Optional dataproduct filters.",
    )
    product_subgroups: list | str | None = OptionalRuntimeField(
        description="Optional JWST product subgroup filters.",
    )
    raw_only: bool | str = RuntimeField(
        default=False,
        description="If True, download only raw JWST UNCAL FITS files.",
    )
    max_observations: int | str | None = OptionalRuntimeField(
        description="Optional maximum number of observations to process.",
    )
    max_products: int | str | None = OptionalRuntimeField(
        description="Optional maximum number of products per observation.",
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        # Normalize LLM-supplied args: '' -> None, 'True' -> True, '10' -> 10.
        planet_name = _clean_str(self.planet_name)
        label = _clean_str(self.label) or "aggregate"
        output_dir = _clean_str(self.output_dir) or "mast"
        raw_only = _as_bool(self.raw_only, "raw_only", default=False)
        max_observations = _as_int_or_none(self.max_observations, "max_observations")
        max_products = _as_int_or_none(self.max_products, "max_products")
        radius_deg = _as_float_or_none(self.radius_deg, "radius_deg")
        if radius_deg is None:
            radius_deg = 0.02

        obsids_list = _as_list(self.obsids)
        if obsids_list:
            manifest = download_observations_products(
                obsids_list,
                os.path.join(self.base_directory, output_dir),
                product_subgroups=self.product_subgroups,
                raw_only=raw_only,
                max_products_per_obs=max_products,
                label=label,
            )
            return _format_download_manifest(manifest)

        if planet_name is None:
            raise ValueError(
                "Provide either planet_name (per-planet mode) or obsids "
                "(batch mode). Both are missing."
            )

        manifest = download_planet_jwst_products(
            planet_name,
            os.path.join(self.base_directory, output_dir),
            ra=_as_float_or_none(self.ra, "ra"),
            dec=_as_float_or_none(self.dec, "dec"),
            radius_deg=radius_deg,
            instruments=self.instruments,
            dataproduct_types=self.dataproduct_types,
            product_subgroups=self.product_subgroups,
            raw_only=raw_only,
            max_observations=max_observations,
            max_products=max_products,
        )
        return _format_download_manifest(manifest)


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
    Minimal NASA Exoplanet Archive TAP query.

    Used by ``CrossmatchJwstToPlanets`` so mast.py stays standalone (no
    coupling to exoarchive.py / orchestral imports). For richer archive
    queries use ``FindExoplanetsByCondition`` in exoarchive.py.
    """
    if isinstance(conditions, str):
        conditions = [conditions]

    selected_columns = _as_list(columns) or list(DEFAULT_ARCHIVE_COLUMNS)
    where_clause = " AND ".join(f"({c})" for c in conditions)
    top_clause = f"TOP {int(limit)} " if limit else ""
    adql = (
        f"SELECT {top_clause}{', '.join(selected_columns)} "
        f"FROM {table} "
        f"WHERE {where_clause}"
    )

    client = session or requests
    response = client.get(
        EXOARCHIVE_TAP_SYNC_URL,
        params={"query": adql, "format": "csv"},
        timeout=timeout,
    )
    response.raise_for_status()
    return list(csv.DictReader(io.StringIO(response.text)))


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
        "filters", "t_min", "t_max",
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
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
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
                "archive_columns includes 'ra' and 'dec'."
            )
        if obs_count == 0:
            head.append(
                "  • 0 JWST observations matched your instrument/dataproduct/calib/"
                f"proposal filters ({active_filters or 'none'}). The MAST search is "
                "empty — loosen those filters (e.g. DROP calib_levels; JWST science "
                "is calib_level 3, and calib_level=1 matches almost nothing)."
            )
        if planet_count > 0 and planets_with_coords and obs_count > 0:
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
            f"event={row.get('obs_type','?')} "
            f"pi={row.get('proposal_pi','?')}"
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
    session: requests.Session | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """
    Download JWST products for an entire demographic population, organized per
    planet on disk.

    ``rows`` must contain ``pl_name`` and ``obsid`` columns (the schema written
    by ``CrossmatchJwstToPlanets``). Rows are grouped by planet name; for each
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


class CrossmatchJwstToPlanets(BaseTool):
    """
    Cross-match a JWST demographics search against an Exoplanet-Archive planet
    population by RA/Dec. (Population pipeline, step 2 of 3.)

    USE WHEN: a population study needs each JWST observation joined to its
    planet's archive parameters — e.g. "which sub-Neptunes have NIRSpec data,
    with their radii". Combines a MAST demographics search + an archive TAP
    query in one call and writes a rows CSV.

    NOT FOR: a single target (use `SearchMastJwstObservations`), just counting
    rows you already have (use `AggregateJwstObservations`), or a planet-list
    with no observations (use `FindExoplanetsByCondition`). Feed the output CSV
    to `AggregateJwstObservations` or `DownloadDemographicJwstProducts`.

    Workflow
    --------
    1. Tool runs an Exoplanet Archive TAP query for planets matching
       ``archive_conditions`` (e.g. ``["pl_bmassj > 0.3", "pl_eqt > 500"]``)
       and/or a ``population_preset`` (e.g. ``"subneptune"``).
    2. Tool runs a no-position MAST JWST search with the given instrument /
       dataproduct / calib filters (same vocabulary as
       ``SearchMastJwstObservations``). The MAST search auto-paginates so
       populations larger than a single page are not silently truncated.
    3. Each MAST observation is cone-matched (default 0.02 deg) against every
       returned planet. Each match becomes one CSV row: all planet columns
       followed by ``obsid``, ``obs_id``, ``instrument_name``,
       ``dataproduct_type``, ``calib_level``, ``proposal_id``, ``proposal_pi``,
       ``target_name``, ``filters``, ``t_min``, ``t_max``, and (by default)
       ``cycle_number`` — the JWST observing cycle fetched from STScI's
       program-info pages (MAST has no cycle column, and cycle is NOT
       derivable from proposal_id arithmetic: GO 2372 is Cycle 1, GO 3557
       and GO 4098 are Cycle 2). Lookups are one-per-unique-proposal and
       cached permanently on disk, so only the first run pays any network
       cost. Unresolvable proposals get ``cycle_number`` empty plus a warning
       in the summary — never a guessed value. Also by default each row is
       classified against the planet's transit ephemeris into ``obs_type``
       ('transit' / 'eclipse' / 'phase_curve' / 'baseline' / 'unknown') with
       ``n_transits_in_window`` / ``n_eclipses_in_window`` /
       ``obs_type_note`` columns (local math, no extra queries; see
       ``add_obs_type``).
    4. **The full row table is ALWAYS written to disk.** Pass ``output_csv``
       to choose the path, or omit it to get an auto-named file under
       ``{base_directory}/mast/demographics/crossmatch_*.csv``. The returned
       summary string only previews the first ~20 rows — agents must read the
       CSV (e.g. ``pandas.read_csv``) to get the complete list. Do NOT try to
       reconstruct the table from the preview; rows past the preview are
       suppressed in the text output.

    Use this tool to answer questions like:
      * "Compile every JWST NIRSpec spectrum of a warm or hot Jupiter."
      * "Which sub-Neptunes have been observed by JWST, and with which PI?"
      * Pipe its CSV into ``AggregateJwstObservations`` to count
        observations per instrument / per filter / per planet.

    UNITS WARNING — pl_radj vs pl_rade
    ----------------------------------
    The Exoplanet Archive stores planet radius in BOTH Jupiter radii
    (``pl_radj``, 1 R_J ≈ 11.2 R_E) and Earth radii (``pl_rade``). Categorical
    names refer to Earth radii:
      * sub-Neptune  ≈ 1.75–4.0 R_E   (NOT 1.5–4.0 R_J — that is hot Jupiters!)
      * Neptune      ≈ 4.0–6.0 R_E
      * Saturn       ≈ 8.0–10.0 R_E
      * Jupiter      ≈ 0.8–1.5 R_J  ≈ 9–17 R_E
    Hand-translating "sub-Neptune" to ``pl_radj 1.5-4.0`` returns inflated hot
    Jupiters and brown-dwarf companions, which is the wrong population. **Prefer
    the ``population_preset`` argument** for canonical categories — it picks
    the correct radius column for you.

    Returns
    -------
    A summary string (matched row count, planets / obs scanned, radius, csv
    path, first rows). The CSV path is the canonical source for downstream
    use — always read it back instead of parsing the preview text.

    Examples
    --------
        # Canonical category — preset handles units correctly.
        CrossmatchJwstToPlanets(
            population_preset="subneptune",
            instruments=["NIRSpec", "NIRCam", "MIRI", "NIRISS"],
            dataproduct_types=["spectrum", "timeseries"],
            calib_levels=[3],
            output_csv="mast/demographics/subneptunes_jwst.csv",
        )

        # Custom conditions — explicit units, free-form ADQL.
        CrossmatchJwstToPlanets(
            archive_conditions=["pl_bmassj > 0.3", "pl_eqt > 500"],
            instruments=["NIRSpec", "NIRCam", "MIRI", "NIRISS"],
            dataproduct_types=["spectrum", "timeseries"],
            calib_levels=[3],
            output_csv="mast/demographics/warm_hot_jupiters_jwst.csv",
        )
    """

    archive_conditions: list | str | None = OptionalRuntimeField(
        description=(
            "List of ADQL WHERE conditions for the NASA Exoplanet Archive "
            "(pscomppars). E.g. ['pl_bmassj > 0.3', 'pl_eqt > 500']. "
            "Remember pl_rade is Earth radii, pl_radj is Jupiter radii — "
            "sub-Neptune = pl_rade 1.75-4.0, not pl_radj. Optional if "
            "population_preset is set; combined with it via AND."
        ),
    )
    population_preset: str | None = OptionalRuntimeField(
        description=(
            "Canonical exoplanet population name. Maps to the right "
            "Exoplanet Archive WHERE conditions with correct radius units "
            "(use this instead of hand-coding category bounds). Valid: "
            "'terrestrial', 'super_earth', 'subneptune', 'neptune', "
            "'sub_saturn', 'saturn', 'jupiter', 'hot_jupiter', "
            "'warm_jupiter', 'cold_jupiter', 'ultra_hot_jupiter', "
            "'inflated_jupiter', 'brown_dwarf'. Combined with "
            "archive_conditions via AND."
        ),
    )
    archive_columns: list | str | None = OptionalRuntimeField(
        description=(
            "Optional archive columns to return. Defaults to a broad common set "
            "(pl_name, ra, dec, pl_radj/rade, masses, pl_eqt, pl_orbper, stellar "
            "params, sy_dist, discoverymethod, disc_year)."
        ),
    )
    archive_table: str = RuntimeField(
        default="pscomppars",
        description="TAP table name. 'pscomppars' (composite) recommended.",
    )
    archive_limit: int | str | None = OptionalRuntimeField(
        description="Optional row cap on the archive query.",
    )
    instruments: list | str | None = OptionalRuntimeField(
        description="Optional JWST instrument filters, e.g. ['NIRSpec', 'NIRCam'].",
    )
    dataproduct_types: list | str | None = OptionalRuntimeField(
        description="Optional dataproduct filters, e.g. ['spectrum', 'timeseries'].",
    )
    calib_levels: list | str | None = OptionalRuntimeField(
        description="Optional MAST calibration levels, e.g. [3].",
    )
    proposal_id: str | None = OptionalRuntimeField(
        description="Optional JWST proposal id filter.",
    )
    radius_deg: float | str = RuntimeField(
        default=0.02,
        description="Cone-match radius in degrees between obs and planet sky position.",
    )
    output_csv: str | None = OptionalRuntimeField(
        description=(
            "Optional CSV path relative to base_directory. The crossmatch "
            "tool ALWAYS writes a CSV — if this is left unset, the tool "
            "auto-generates a timestamped path under "
            "'mast/demographics/crossmatch_*.csv' and reports it in the "
            "result string. Set this only when you need a specific filename."
        ),
    )
    add_cycle_number: bool | str = RuntimeField(
        default=True,
        description=(
            "If True (default), add a cycle_number column with each "
            "proposal's JWST observing cycle, resolved from STScI program "
            "info (cached on disk; one lookup per unique proposal, first "
            "run only). Set False to skip the lookups entirely."
        ),
    )
    add_obs_type: bool | str = RuntimeField(
        default=True,
        description=(
            "If True (default), classify each (planet, observation) row as "
            "'transit' / 'eclipse' / 'phase_curve' / 'baseline' / 'unknown' "
            "by comparing the observation window (t_min/t_max) with the "
            "planet's transit ephemeris (pl_tranmid, pl_orbper, pl_trandur). "
            "Adds obs_type, n_transits_in_window, n_eclipses_in_window, "
            "obs_type_note columns. Purely local math — no extra queries. "
            "Requires the ephemeris columns in archive_columns (included in "
            "the defaults)."
        ),
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        population_preset = _clean_str(self.population_preset)
        proposal_id = _clean_str(self.proposal_id)
        output_csv = _clean_str(self.output_csv)
        archive_table = _clean_str(self.archive_table) or "pscomppars"
        archive_limit = _as_int_or_none(self.archive_limit, "archive_limit")
        radius_deg = _as_float_or_none(self.radius_deg, "radius_deg")
        if radius_deg is None:
            radius_deg = 0.02

        preset_conditions = _resolve_population_preset(population_preset)
        user_conditions = _as_list(self.archive_conditions) or []
        merged_conditions = preset_conditions + user_conditions
        if not merged_conditions:
            raise ValueError(
                "CrossmatchJwstToPlanets requires either archive_conditions "
                "or population_preset (or both)."
            )

        # The cone-match keys planets on RA/Dec. If the caller's archive_columns
        # omit them, EVERY planet is silently dropped and the tool reports "no
        # matches" — a classic false negative. Force ra/dec into the column set.
        archive_columns = _as_list(self.archive_columns) or list(DEFAULT_ARCHIVE_COLUMNS)
        for required in ("ra", "dec"):
            if required not in archive_columns:
                archive_columns.append(required)

        planets = archive_tap_query(
            merged_conditions,
            columns=archive_columns,
            table=archive_table,
            limit=archive_limit,
        )

        def _has_coords(p: dict[str, Any]) -> bool:
            try:
                return (_as_float_or_none(p.get("ra"), "ra") is not None
                        and _as_float_or_none(p.get("dec"), "dec") is not None)
            except ValueError:
                return False
        planets_with_coords = sum(1 for p in planets if _has_coords(p))

        observations, _filters = search_all_jwst_observations(
            instruments=self.instruments,
            dataproduct_types=self.dataproduct_types,
            calib_levels=self.calib_levels,
            proposal_id=proposal_id,
        )

        rows = crossmatch_observations_to_planets(
            observations,
            planets,
            radius_deg=radius_deg,
        )

        cycle_warnings: list[str] = []
        if _as_bool(self.add_cycle_number, "add_cycle_number", default=True) and rows:
            cycle_warnings = annotate_rows_with_cycles(
                rows,
                cache_path=Path(self.base_directory or ".") / JWST_PROGRAM_INFO_CACHE,
            )

        obs_type_line = None
        if _as_bool(self.add_obs_type, "add_obs_type", default=True) and rows:
            classified = annotate_rows_with_event_types(rows)
            obs_type_line = (
                f"Event classification: {classified}/{len(rows)} rows typed "
                "(transit/eclipse/phase_curve/baseline); 'unknown' rows lack "
                "pl_tranmid/pl_orbper — see obs_type_note."
            )

        if output_csv:
            csv_path = Path(self.base_directory) / output_csv
        else:
            csv_path = _autoname_csv_path(
                self.base_directory,
                kind="crossmatch",
                hint_parts=[
                    population_preset,
                    merged_conditions,
                    self.instruments,
                    self.dataproduct_types,
                    self.calib_levels,
                    proposal_id,
                ],
            )
        _write_rows_csv(csv_path, rows)

        summary = _format_crossmatch_summary(
            rows,
            planet_count=len(planets),
            obs_count=len(observations),
            planets_with_coords=planets_with_coords,
            radius_deg=radius_deg,
            csv_path=csv_path,
            preset=population_preset,
            conditions=merged_conditions,
            obs_filters={
                "instruments": _as_list(self.instruments),
                "dataproduct_types": _as_list(self.dataproduct_types),
                "calib_levels": _as_int_list(self.calib_levels),
                "proposal_id": proposal_id,
            },
        )
        if obs_type_line:
            summary += f"\n\n{obs_type_line}"
        if cycle_warnings:
            summary += (
                "\n\nWARNING — cycle_number left empty for some proposals "
                "(no guessed values were written):\n"
                + "\n".join(f"  - {w}" for w in cycle_warnings)
            )
        return summary


class AggregateJwstObservations(BaseTool):
    """
    Group JWST rows by one or more keys and count. (Population pipeline, step 3 of 3.)

    USE WHEN: the answer is a count or breakdown — "how many planets per
    instrument", "observations by cycle". Either aggregates a rows CSV from
    `CrossmatchJwstToPlanets`, or runs its own demographics search first.

    NOT FOR: discovering observations (`SearchMastJwstObservations`) or joining
    to planet params (`CrossmatchJwstToPlanets`).

    Input modes (provide exactly one):
      * ``rows_path`` — path (relative to ``base_directory``) to a CSV or JSON
        of rows from an earlier ``CrossmatchJwstToPlanets`` or other dump.
        Use this to aggregate crossmatched (planet, obs) rows, e.g. to find
        planets observed with multiple instruments.
      * MAST filter fields (``instruments``, ``dataproduct_types``,
        ``calib_levels``, ``proposal_id``, ``target_name``) — tool runs a fresh
        no-position demographics search and aggregates the raw observation
        rows (no planet attribution).

    Useful aggregations
    -------------------
    * Per instrument:
        AggregateJwstObservations(
            instruments=['NIRSpec','NIRCam','MIRI','NIRISS'],
            dataproduct_types=['spectrum','timeseries'],
            calib_levels=[3],
            group_by=['instrument_name'],
        )

    * Per instrument × filter:
        AggregateJwstObservations(
            ... same filters ...
            group_by=['instrument_name', 'filters'],
        )

    * Planets observed with multiple instruments (chain with crossmatch CSV):
        AggregateJwstObservations(
            rows_path='mast/demographics/warm_hot_jupiters_jwst.csv',
            group_by=['pl_name'],
            distinct_fields=['instrument_name', 'filters', 'proposal_id'],
        )
        # rows with instrument_name_distinct > 1 are multi-instrument targets

    * Observation event types (transit / eclipse / phase_curve / baseline):
        AggregateJwstObservations(
            rows_path='mast/demographics/subneptunes_jwst.csv',
            group_by=['pl_name', 'obs_type'],
        )
        # If the CSV has no obs_type column yet, it is derived on the fly
        # from each row's ephemeris (t_min/t_max vs pl_tranmid / pl_orbper /
        # pl_trandur) whenever group_by or distinct_fields mention obs_type,
        # n_transits_in_window, n_eclipses_in_window, or obs_type_note.
        # Set add_obs_type=True to force classification and ALSO persist the
        # annotated rows to '<rows_path stem>_typed.csv'. Rows without an
        # ephemeris (non-transiting RV planets) classify as 'unknown' with
        # the reason in obs_type_note; eclipse calls on e > 0.1 orbits are
        # flagged approximate (circular phase-0.5 assumption).

    Output
    ------
    Returns a summary listing the top groups by count, with distinct counts.
    **The full grouped table is ALWAYS written to CSV.** Pass ``output_csv``
    for a chosen path, otherwise the tool auto-names a file under
    ``{base_directory}/mast/demographics/aggregate_*.csv``. The preview text
    truncates after ~40 groups — always read the CSV for the complete table.
    """

    group_by: list | str = RuntimeField(
        description="One or more row keys to group by, e.g. ['instrument_name', 'filters'].",
    )
    distinct_fields: list | str | None = OptionalRuntimeField(
        description=(
            "Optional row keys whose distinct values are tracked per group. "
            "E.g. ['instrument_name'] to count how many instruments observed "
            "each planet when group_by=['pl_name']."
        ),
    )
    rows_path: str | None = OptionalRuntimeField(
        description=(
            "Path (relative to base_directory) to a CSV or JSON dump of rows "
            "to aggregate. Mutually exclusive with the MAST filter fields."
        ),
    )
    add_obs_type: bool | str = RuntimeField(
        default=False,
        description=(
            "rows_path mode only. If True, classify every row as transit / "
            "eclipse / phase_curve / baseline / unknown from its ephemeris "
            "columns (t_min/t_max vs pl_tranmid/pl_orbper/pl_trandur) before "
            "aggregating, and write the annotated rows to "
            "'<rows_path stem>_typed.csv'. Classification also happens "
            "automatically (without writing that file) whenever group_by / "
            "distinct_fields request obs_type columns missing from the file."
        ),
    )
    instruments: list | str | None = OptionalRuntimeField(
        description="Demographics-mode JWST instrument filters.",
    )
    dataproduct_types: list | str | None = OptionalRuntimeField(
        description="Demographics-mode dataproduct filters.",
    )
    calib_levels: list | str | None = OptionalRuntimeField(
        description="Demographics-mode MAST calibration levels.",
    )
    proposal_id: str | None = OptionalRuntimeField(
        description="Demographics-mode JWST proposal id filter.",
    )
    target_name: str | None = OptionalRuntimeField(
        description="Demographics-mode free-text target_name filter.",
    )
    output_csv: str | None = OptionalRuntimeField(
        description=(
            "Optional CSV path (relative to base_directory) for the grouped "
            "table. The aggregate tool ALWAYS writes a CSV — if this is left "
            "unset, the tool auto-generates a timestamped path under "
            "'mast/demographics/aggregate_*.csv' and reports it in the "
            "result string."
        ),
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        group_keys = _as_list(self.group_by) or []
        distinct_keys = _as_list(self.distinct_fields) or []
        if not group_keys:
            raise ValueError("group_by must contain at least one key.")

        rows_path = _clean_str(self.rows_path)
        target_name = _clean_str(self.target_name)
        proposal_id = _clean_str(self.proposal_id)
        output_csv = _clean_str(self.output_csv)

        obs_type_lines: list[str] = []
        if rows_path:
            full_path = Path(self.base_directory) / rows_path
            rows = _read_rows_from_path(full_path)

            obs_type_keys = {
                "obs_type", "n_transits_in_window",
                "n_eclipses_in_window", "obs_type_note",
            }
            force = _as_bool(self.add_obs_type, "add_obs_type", default=False)
            requested = (set(group_keys) | set(distinct_keys)) & obs_type_keys
            needs_annotation = rows and (
                force or (requested and any(k not in rows[0] for k in requested))
            )
            if needs_annotation:
                classified = annotate_rows_with_event_types(rows)
                obs_type_lines.append(
                    f"Event classification: {classified}/{len(rows)} rows "
                    "typed (transit/eclipse/phase_curve/baseline); 'unknown' "
                    "rows lack pl_tranmid/pl_orbper — see obs_type_note."
                )
                notes: dict[str, int] = {}
                for row in rows:
                    note = row.get("obs_type_note")
                    if note:
                        notes[str(note)] = notes.get(str(note), 0) + 1
                if notes:
                    obs_type_lines.append(
                        "obs_type_note breakdown (read before trusting "
                        "eclipse/baseline calls):"
                    )
                    obs_type_lines.extend(
                        f"  - (x{count}) {note}"
                        for note, count in sorted(
                            notes.items(), key=lambda kv: -kv[1]
                        )
                    )
                if force:
                    typed_path = full_path.with_name(f"{full_path.stem}_typed.csv")
                    _write_rows_csv(typed_path, rows)
                    obs_type_lines.append(f"Annotated rows saved to: {typed_path}")
        else:
            observations, _filters = search_all_jwst_observations(
                instruments=self.instruments,
                dataproduct_types=self.dataproduct_types,
                calib_levels=self.calib_levels,
                target_name=target_name,
                proposal_id=proposal_id,
            )
            rows = observations

        groups = aggregate_observations(
            rows,
            group_by=group_keys,
            distinct_fields=distinct_keys,
        )

        if output_csv:
            csv_path = Path(self.base_directory) / output_csv
        else:
            csv_path = _autoname_csv_path(
                self.base_directory,
                kind="aggregate",
                hint_parts=[
                    group_keys,
                    distinct_keys,
                    self.instruments,
                    self.dataproduct_types,
                    self.calib_levels,
                    proposal_id,
                    target_name,
                    rows_path,
                ],
            )
        _write_rows_csv(csv_path, groups)

        summary = _format_aggregate_summary(
            groups,
            group_by=group_keys,
            distinct_fields=distinct_keys,
            total_rows=len(rows),
            csv_path=csv_path,
        )
        if obs_type_lines:
            summary += "\n\n" + "\n".join(obs_type_lines)
        return summary


class DownloadDemographicJwstProducts(BaseTool):
    """
    Bulk-download JWST products for EVERY planet in a demographic (from a CSV).

    USE WHEN: you have a `CrossmatchJwstToPlanets` rows CSV and want to fetch
    files for all planets in it at once.

    NOT FOR: one planet or a hand-picked obsid list (use
    `DownloadMastJwstProducts`), or reduced archive spectra (use `DownloadDataset`).

    Reads a crossmatch dump (the CSV/JSON written by
    ``CrossmatchJwstToPlanets`` — must contain ``pl_name`` and ``obsid``
    columns), groups obsids by planet, and downloads each planet's products
    under ``{base_directory}/{output_dir}/{label}/{pl_name}/{obsid}/*.fits``.

    Per-planet directories contain a ``manifest.json``; a top-level
    ``demographic_manifest.json`` summarizes the entire run.

    Why this exists
    ---------------
    ``DownloadMastJwstProducts`` either downloads one planet's products
    (per-planet mode) or one flat obsid bucket (batch mode). Neither lays out
    the result tree per planet across a whole population. This tool closes
    that gap so an agent can do:

        CrossmatchJwstToPlanets(...)             # -> crossmatch.csv
        DownloadDemographicJwstProducts(
            rows_path='crossmatch.csv',
            label='warm_hot_jupiters',
            product_subgroups=['X1DINTS'],
        )

    Important caps
    --------------
    UNCAL ramps are multi-GB per obs. Always cap with one or more of
    ``max_planets``, ``max_obs_per_planet``, ``max_products_per_obs`` when
    using ``raw_only=True``.

    Example
    -------
        DownloadDemographicJwstProducts(
            rows_path='mast/demographics/warm_hot_jupiters_jwst.csv',
            output_dir='mast/raw',
            label='warm_hot_jupiters',
            raw_only=True,
            max_obs_per_planet=1,
            max_products_per_obs=2,
        )
    """

    rows_path: str = RuntimeField(
        description=(
            "Path (relative to base_directory) to a CSV or JSON dump from "
            "CrossmatchJwstToPlanets. Must contain pl_name and obsid columns."
        ),
    )
    output_dir: str = RuntimeField(
        default="mast/demographics_raw",
        description="Output directory relative to base_directory.",
    )
    label: str = RuntimeField(
        default="aggregate",
        description="Subdirectory under output_dir grouping this demographic's downloads.",
    )
    product_subgroups: list | str | None = OptionalRuntimeField(
        description=(
            "Optional JWST product subgroup filters, e.g. ['X1DINTS'] for "
            "stage-3 time-resolved spectra. Leave None to keep all SCIENCE FITS."
        ),
    )
    raw_only: bool | str = RuntimeField(
        default=False,
        description="If True, download only raw JWST UNCAL FITS files (large).",
    )
    max_planets: int | str | None = OptionalRuntimeField(
        description="Optional cap on the number of planets to process.",
    )
    max_obs_per_planet: int | str | None = OptionalRuntimeField(
        description="Optional cap on the number of obsids downloaded per planet.",
    )
    max_products_per_obs: int | str | None = OptionalRuntimeField(
        description="Optional cap on the number of products per observation.",
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        rows_path = _clean_str(self.rows_path)
        if rows_path is None:
            raise ValueError("rows_path is required (CSV/JSON from CrossmatchJwstToPlanets).")
        rows = _read_rows_from_path(Path(self.base_directory) / rows_path)
        manifest = download_demographic_products(
            rows,
            os.path.join(self.base_directory, _clean_str(self.output_dir) or "mast/demographics_raw"),
            label=_clean_str(self.label) or "aggregate",
            product_subgroups=self.product_subgroups,
            raw_only=_as_bool(self.raw_only, "raw_only", default=False),
            max_planets=_as_int_or_none(self.max_planets, "max_planets"),
            max_obs_per_planet=_as_int_or_none(self.max_obs_per_planet, "max_obs_per_planet"),
            max_products_per_obs=_as_int_or_none(self.max_products_per_obs, "max_products_per_obs"),
        )
        return _format_demographic_summary(manifest)


class GetJwstProgramInfo(BaseTool):
    """
    Look up authoritative JWST program metadata — observing Cycle, title, PI,
    proposal type (GO/GTO/DD/CAL), status, exclusive-access period — for one
    or more proposal ids, from STScI's program-info pages.

    USE WHEN: you have a `proposal_id` (e.g. from `SearchMastJwstObservations`)
    and need its cycle, title, PI, or status.

    NOT FOR: finding observations (use `SearchMastJwstObservations`) or planet
    parameters (use `GetExoplanetParameters`). Requires a proposal id, not a
    planet name.

    Why this tool exists
    --------------------
    MAST CAOM metadata has NO cycle field, and the cycle CANNOT be computed
    from the proposal id (GO 2372 is Cycle 1; GO 3557 and GO 4098 are
    Cycle 2). Use this tool whenever a task needs cycle numbers — never guess
    them. ``CrossmatchJwstToPlanets`` calls the same lookup automatically for
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
