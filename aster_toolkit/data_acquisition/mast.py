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
except ModuleNotFoundError:
    class BaseTool:
        """Fallback that keeps plain MAST wrappers importable without Orchestral."""

    def RuntimeField(default=None, description=None):
        return default

    def StateField(default=None, description=None):
        return default


MAST_INVOKE_URL = "https://mast.stsci.edu/api/v0/invoke"
MAST_DOWNLOAD_URL = "https://mast.stsci.edu/api/v0.1/Download/file"
EXOARCHIVE_TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

DEFAULT_ARCHIVE_COLUMNS = [
    "pl_name", "hostname", "ra", "dec",
    "pl_radj", "pl_rade", "pl_bmassj", "pl_bmasse",
    "pl_orbper", "pl_orbsmax", "pl_eqt", "pl_dens", "pl_insol",
    "pl_orbeccen", "pl_orbincl", "pl_trandep", "pl_imppar",
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

JWST_INSTRUMENT_ALIASES = {
    "NIRSPEC": ["NIRSPEC/SLIT", "NIRSPEC/IFU", "NIRSPEC/MSA", "NIRSPEC/IMAGE"],
    "NIRCAM": ["NIRCAM/IMAGE", "NIRCAM/GRISM"],
    "NIRISS": ["NIRISS/SOSS", "NIRISS/IMAGE", "NIRISS/WFSS", "NIRISS/AMI"],
    "MIRI": ["MIRI/IMAGE", "MIRI/IFU", "MIRI/MRS", "MIRI/LRS"],
    "FGS": ["FGS/FGS"],
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
) -> dict[str, Any]:
    """Submit a request to the MAST Mashup API."""
    client = session or requests
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
    return response.json()


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
        filters.append(
            {
                "paramName": "target_name",
                "values": [{"freeText": target_name}],
            }
        )

    if proposal_id is not None:
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

    Pages through the MAST Caom.Filtered API until a short page is returned or
    ``max_pages`` is reached. The previous single-page implementation silently
    capped at ``pagesize`` (50000) and missed observations whenever the
    population exceeded that number.
    """
    selected_columns = list(columns or JWST_OBSERVATION_COLUMNS)
    filters = _build_jwst_observation_filters(
        instruments=instruments,
        dataproduct_types=dataproduct_types,
        calib_levels=calib_levels,
        target_name=target_name,
        proposal_id=proposal_id,
    )

    all_rows: list[dict[str, Any]] = []
    current_page = page
    pages_fetched = 0
    while pages_fetched < max_pages:
        request = {
            "service": "Mast.Caom.Filtered",
            "params": {
                "columns": ",".join(selected_columns),
                "filters": filters,
            },
            "format": "json",
            "pagesize": pagesize,
            "page": current_page,
        }
        rows = _extract_rows(_mast_query(request, session=session, timeout=timeout))
        all_rows.extend(rows)
        pages_fetched += 1
        if len(rows) < pagesize:
            break
        current_page += 1

    return all_rows, filters


def search_jwst_observations(
    planet_name: str,
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
        ra, dec = resolve_target_coordinates(
            planet_name,
            session=session,
            timeout=timeout,
        )

    selected_columns = list(columns or JWST_OBSERVATION_COLUMNS)
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
    retries: int = 3,
    retry_backoff: float = 5.0,
) -> Path:
    """Download a single MAST product by dataURI and return the local path.

    ``timeout`` is a ``(connect, read)`` tuple by default — 30 s to establish
    the TCP/TLS connection, 600 s of inactivity tolerance per-chunk while the
    file streams. Large JWST products (NIRSpec X1DINTS, UNCAL) routinely take
    minutes to start streaming; the previous 120 s flat timeout caused false
    failures.

    On a transient network error (``ConnectionError``, ``Timeout``,
    ``ChunkedEncodingError``) the function retries up to ``retries`` times
    with a linear backoff of ``retry_backoff`` seconds.

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
    transient_errors = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
    )

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.get(
                MAST_DOWNLOAD_URL,
                params={"uri": data_uri},
                stream=True,
                timeout=timeout,
            )
            if response.status_code in (401, 403):
                response.close()
                raise ProprietaryProductError(data_uri, response.status_code)
            response.raise_for_status()

            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
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


class SearchMastJwstObservations(BaseTool):
    """
    Search MAST for JWST observations.

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
    Valid values for ``instruments``:
        - "NIRSpec"  : near-IR spectrograph (most transit/eclipse spectroscopy)
        - "NIRCam"   : near-IR imager + grism (also used for TSO spectroscopy)
        - "MIRI"     : mid-IR imager + LRS/MRS spectrographs
        - "NIRISS"   : near-IR imager + SOSS slitless spectroscopy
        - "FGS"      : fine guidance sensor (rarely needed for science)

    Data product types
    ------------------
    Valid values for ``dataproduct_types``:
        - "spectrum"   : 1-D extracted spectra
        - "timeseries" : time-series exposures (transits/eclipses)
        - "image"      : 2-D images
        - "cube"       : 3-D IFU cubes

    Calibration levels
    ------------------
    Valid values for ``calib_levels``:
        - 1 : raw / minimal calibration
        - 2 : per-exposure calibrated products
        - 3 : combined / extracted science-ready products (recommended for retrievals)
        - 4 : community / contributed products

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

    planet_name: str | None = RuntimeField(
        default=None,
        description=(
            "Exoplanet or host-star target name, e.g. 'WASP-39 b'. Leave None "
            "to run a no-position demographics query over every JWST observation "
            "matching the filters."
        ),
    )
    ra: float | None = RuntimeField(
        default=None,
        description="Right ascension in degrees. Optional if MAST can resolve planet_name.",
    )
    dec: float | None = RuntimeField(
        default=None,
        description="Declination in degrees. Optional if MAST can resolve planet_name.",
    )
    radius_deg: float = RuntimeField(
        default=0.02,
        description="Cone-search radius in degrees (only used when planet_name or ra/dec are set).",
    )
    instruments: list | None = RuntimeField(
        default=None,
        description="JWST instrument filters, e.g. ['NIRSpec', 'NIRCam', 'MIRI'].",
    )
    dataproduct_types: list | None = RuntimeField(
        default=None,
        description="MAST dataproduct filters, e.g. ['spectrum', 'timeseries', 'image'].",
    )
    calib_levels: list | None = RuntimeField(
        default=None,
        description="Optional MAST calibration levels.",
    )
    proposal_id: str | None = RuntimeField(
        default=None,
        description="Optional JWST proposal id filter.",
    )
    target_name: str | None = RuntimeField(
        default=None,
        description=(
            "Optional free-text target-name filter (MAST target_name). Used "
            "directly in demographics mode, or combined with cone search via "
            "target_name_filter=True semantics."
        ),
    )

    def _run(self) -> str:
        if self.planet_name is None and self.ra is None and self.dec is None:
            observations, filters = search_all_jwst_observations(
                instruments=self.instruments,
                dataproduct_types=self.dataproduct_types,
                calib_levels=self.calib_levels,
                target_name=self.target_name,
                proposal_id=self.proposal_id,
            )
            return _format_observations_summary(
                observations,
                filters=filters,
                query_extra={"mode": "demographics (Mast.Caom.Filtered, no position)"},
            )

        if self.planet_name is None:
            raise ValueError(
                "planet_name is required when ra or dec is provided. "
                "Provide both ra and dec with a planet_name, or omit all three "
                "to run a demographics search."
            )

        observations = search_jwst_observations(
            self.planet_name,
            ra=self.ra,
            dec=self.dec,
            radius_deg=self.radius_deg,
            instruments=self.instruments,
            dataproduct_types=self.dataproduct_types,
            calib_levels=self.calib_levels,
            proposal_id=self.proposal_id,
        )
        filters = _build_jwst_observation_filters(
            instruments=self.instruments,
            dataproduct_types=self.dataproduct_types,
            calib_levels=self.calib_levels,
            proposal_id=self.proposal_id,
        )
        return _format_observations_summary(
            observations,
            filters=filters,
            query_extra={
                "mode": "per-planet cone search (Mast.Caom.Filtered.Position)",
                "planet_name": self.planet_name,
                "radius_deg": self.radius_deg,
            },
        )


class GetMastObservationProducts(BaseTool):
    """
    List downloadable MAST products for one JWST observation id.

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
    product_subgroups: list | None = RuntimeField(
        default=None,
        description="Optional JWST product subgroup filters, e.g. ['UNCAL', 'RATEINTS', 'X1DINTS'].",
    )
    raw_only: bool = RuntimeField(
        default=False,
        description="If True, return only raw JWST UNCAL products.",
    )

    def _run(self) -> str:
        products = get_observation_products(
            self.obsid,
            product_subgroups=self.product_subgroups,
            raw_only=self.raw_only,
        )
        return _format_products_summary(products)


class DownloadMastJwstProducts(BaseTool):
    """
    One-shot download of JWST products.

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

    planet_name: str | None = RuntimeField(
        default=None,
        description=(
            "Exoplanet or host-star target name, e.g. 'WASP-39 b'. Leave None "
            "when downloading by an explicit ``obsids`` list."
        ),
    )
    obsids: list | None = RuntimeField(
        default=None,
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
    ra: float | None = RuntimeField(
        default=None,
        description="Right ascension in degrees. Optional if MAST can resolve planet_name.",
    )
    dec: float | None = RuntimeField(
        default=None,
        description="Declination in degrees. Optional if MAST can resolve planet_name.",
    )
    radius_deg: float = RuntimeField(
        default=0.02,
        description="Cone-search radius in degrees.",
    )
    instruments: list | None = RuntimeField(
        default=None,
        description="Optional JWST instrument filters.",
    )
    dataproduct_types: list | None = RuntimeField(
        default=None,
        description="Optional dataproduct filters.",
    )
    product_subgroups: list | None = RuntimeField(
        default=None,
        description="Optional JWST product subgroup filters.",
    )
    raw_only: bool = RuntimeField(
        default=False,
        description="If True, download only raw JWST UNCAL FITS files.",
    )
    max_observations: int | None = RuntimeField(
        default=None,
        description="Optional maximum number of observations to process.",
    )
    max_products: int | None = RuntimeField(
        default=None,
        description="Optional maximum number of products per observation.",
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        obsids_list = _as_list(self.obsids)
        if obsids_list:
            manifest = download_observations_products(
                obsids_list,
                os.path.join(self.base_directory, self.output_dir),
                product_subgroups=self.product_subgroups,
                raw_only=self.raw_only,
                max_products_per_obs=self.max_products,
                label=self.label,
            )
            return _format_download_manifest(manifest)

        if self.planet_name is None:
            raise ValueError(
                "Provide either planet_name (per-planet mode) or obsids "
                "(batch mode). Both are missing."
            )

        manifest = download_planet_jwst_products(
            self.planet_name,
            os.path.join(self.base_directory, self.output_dir),
            ra=self.ra,
            dec=self.dec,
            radius_deg=self.radius_deg,
            instruments=self.instruments,
            dataproduct_types=self.dataproduct_types,
            product_subgroups=self.product_subgroups,
            raw_only=self.raw_only,
            max_observations=self.max_observations,
            max_products=self.max_products,
        )
        return _format_download_manifest(manifest)


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

    selected_columns = list(columns or DEFAULT_ARCHIVE_COLUMNS)
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
    appended. Observations or planets missing RA/Dec are skipped.
    """
    rows: list[dict[str, Any]] = []
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
        head.append("No matches.")
        return "\n".join(head)

    head.append(f"Preview — first {min(len(rows), limit)} of {len(rows)} rows:")
    for i, row in enumerate(rows[:limit], start=1):
        head.append(
            f"{i:3}. pl_name={row.get('pl_name','?')} "
            f"inst={row.get('instrument_name','?')} "
            f"filters={row.get('filters','?')} "
            f"obsid={row.get('obsid','?')} "
            f"proposal_id={row.get('proposal_id','?')} "
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
    population by RA/Dec.

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
       ``target_name``, ``filters``, ``t_min``, ``t_max``.
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

    archive_conditions: list | None = RuntimeField(
        default=None,
        description=(
            "List of ADQL WHERE conditions for the NASA Exoplanet Archive "
            "(pscomppars). E.g. ['pl_bmassj > 0.3', 'pl_eqt > 500']. "
            "Remember pl_rade is Earth radii, pl_radj is Jupiter radii — "
            "sub-Neptune = pl_rade 1.75-4.0, not pl_radj. Optional if "
            "population_preset is set; combined with it via AND."
        ),
    )
    population_preset: str | None = RuntimeField(
        default=None,
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
    archive_columns: list | None = RuntimeField(
        default=None,
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
    archive_limit: int | None = RuntimeField(
        default=None,
        description="Optional row cap on the archive query.",
    )
    instruments: list | None = RuntimeField(
        default=None,
        description="Optional JWST instrument filters, e.g. ['NIRSpec', 'NIRCam'].",
    )
    dataproduct_types: list | None = RuntimeField(
        default=None,
        description="Optional dataproduct filters, e.g. ['spectrum', 'timeseries'].",
    )
    calib_levels: list | None = RuntimeField(
        default=None,
        description="Optional MAST calibration levels, e.g. [3].",
    )
    proposal_id: str | None = RuntimeField(
        default=None,
        description="Optional JWST proposal id filter.",
    )
    radius_deg: float = RuntimeField(
        default=0.02,
        description="Cone-match radius in degrees between obs and planet sky position.",
    )
    output_csv: str | None = RuntimeField(
        default=None,
        description=(
            "Optional CSV path relative to base_directory. The crossmatch "
            "tool ALWAYS writes a CSV — if this is left unset, the tool "
            "auto-generates a timestamped path under "
            "'mast/demographics/crossmatch_*.csv' and reports it in the "
            "result string. Set this only when you need a specific filename."
        ),
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        preset_conditions = _resolve_population_preset(self.population_preset)
        user_conditions = list(self.archive_conditions or [])
        merged_conditions = preset_conditions + user_conditions
        if not merged_conditions:
            raise ValueError(
                "CrossmatchJwstToPlanets requires either archive_conditions "
                "or population_preset (or both)."
            )

        planets = archive_tap_query(
            merged_conditions,
            columns=self.archive_columns,
            table=self.archive_table,
            limit=self.archive_limit,
        )

        observations, _filters = search_all_jwst_observations(
            instruments=self.instruments,
            dataproduct_types=self.dataproduct_types,
            calib_levels=self.calib_levels,
            proposal_id=self.proposal_id,
        )

        rows = crossmatch_observations_to_planets(
            observations,
            planets,
            radius_deg=self.radius_deg,
        )

        if self.output_csv:
            csv_path = Path(self.base_directory) / self.output_csv
        else:
            csv_path = _autoname_csv_path(
                self.base_directory,
                kind="crossmatch",
                hint_parts=[
                    self.population_preset,
                    merged_conditions,
                    self.instruments,
                    self.dataproduct_types,
                    self.calib_levels,
                    self.proposal_id,
                ],
            )
        _write_rows_csv(csv_path, rows)

        return _format_crossmatch_summary(
            rows,
            planet_count=len(planets),
            obs_count=len(observations),
            radius_deg=self.radius_deg,
            csv_path=csv_path,
            preset=self.population_preset,
            conditions=merged_conditions,
        )


class AggregateJwstObservations(BaseTool):
    """
    Group JWST rows by one or more keys and count, optionally tracking the
    distinct values of additional fields per group.

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

    Output
    ------
    Returns a summary listing the top groups by count, with distinct counts.
    **The full grouped table is ALWAYS written to CSV.** Pass ``output_csv``
    for a chosen path, otherwise the tool auto-names a file under
    ``{base_directory}/mast/demographics/aggregate_*.csv``. The preview text
    truncates after ~40 groups — always read the CSV for the complete table.
    """

    group_by: list = RuntimeField(
        description="One or more row keys to group by, e.g. ['instrument_name', 'filters'].",
    )
    distinct_fields: list | None = RuntimeField(
        default=None,
        description=(
            "Optional row keys whose distinct values are tracked per group. "
            "E.g. ['instrument_name'] to count how many instruments observed "
            "each planet when group_by=['pl_name']."
        ),
    )
    rows_path: str | None = RuntimeField(
        default=None,
        description=(
            "Path (relative to base_directory) to a CSV or JSON dump of rows "
            "to aggregate. Mutually exclusive with the MAST filter fields."
        ),
    )
    instruments: list | None = RuntimeField(
        default=None,
        description="Demographics-mode JWST instrument filters.",
    )
    dataproduct_types: list | None = RuntimeField(
        default=None,
        description="Demographics-mode dataproduct filters.",
    )
    calib_levels: list | None = RuntimeField(
        default=None,
        description="Demographics-mode MAST calibration levels.",
    )
    proposal_id: str | None = RuntimeField(
        default=None,
        description="Demographics-mode JWST proposal id filter.",
    )
    target_name: str | None = RuntimeField(
        default=None,
        description="Demographics-mode free-text target_name filter.",
    )
    output_csv: str | None = RuntimeField(
        default=None,
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

        if self.rows_path:
            full_path = Path(self.base_directory) / self.rows_path
            rows = _read_rows_from_path(full_path)
        else:
            observations, _filters = search_all_jwst_observations(
                instruments=self.instruments,
                dataproduct_types=self.dataproduct_types,
                calib_levels=self.calib_levels,
                target_name=self.target_name,
                proposal_id=self.proposal_id,
            )
            rows = observations

        groups = aggregate_observations(
            rows,
            group_by=group_keys,
            distinct_fields=distinct_keys,
        )

        if self.output_csv:
            csv_path = Path(self.base_directory) / self.output_csv
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
                    self.proposal_id,
                    self.target_name,
                    self.rows_path,
                ],
            )
        _write_rows_csv(csv_path, groups)

        return _format_aggregate_summary(
            groups,
            group_by=group_keys,
            distinct_fields=distinct_keys,
            total_rows=len(rows),
            csv_path=csv_path,
        )


class DownloadDemographicJwstProducts(BaseTool):
    """
    Download JWST products for every planet in a demographic, on disk per
    planet.

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
    product_subgroups: list | None = RuntimeField(
        default=None,
        description=(
            "Optional JWST product subgroup filters, e.g. ['X1DINTS'] for "
            "stage-3 time-resolved spectra. Leave None to keep all SCIENCE FITS."
        ),
    )
    raw_only: bool = RuntimeField(
        default=False,
        description="If True, download only raw JWST UNCAL FITS files (large).",
    )
    max_planets: int | None = RuntimeField(
        default=None,
        description="Optional cap on the number of planets to process.",
    )
    max_obs_per_planet: int | None = RuntimeField(
        default=None,
        description="Optional cap on the number of obsids downloaded per planet.",
    )
    max_products_per_obs: int | None = RuntimeField(
        default=None,
        description="Optional cap on the number of products per observation.",
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        rows = _read_rows_from_path(Path(self.base_directory) / self.rows_path)
        manifest = download_demographic_products(
            rows,
            os.path.join(self.base_directory, self.output_dir),
            label=self.label,
            product_subgroups=self.product_subgroups,
            raw_only=self.raw_only,
            max_planets=self.max_planets,
            max_obs_per_planet=self.max_obs_per_planet,
            max_products_per_obs=self.max_products_per_obs,
        )
        return _format_demographic_summary(manifest)
