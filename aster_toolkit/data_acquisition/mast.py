from __future__ import annotations

import json
import os
import re
import ast
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
    )


def filter_products(
    products: list[dict[str, Any]],
    *,
    product_types: list[str] | tuple[str, ...] | str | None = SCIENCE_PRODUCT_TYPES,
    product_subgroups: list[str] | tuple[str, ...] | str | None = None,
    raw_only: bool = False,
    extensions: list[str] | tuple[str, ...] | str | None = FITS_EXTENSIONS,
) -> list[dict[str, Any]]:
    """Filter MAST products by science type, JWST subgroup, raw status, and file extension."""
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

        if type_values and product_type not in type_values:
            continue

        if subgroup_values and subgroup not in subgroup_values:
            continue

        if raw_only and not is_raw_jwst_product(product):
            continue

        if extension_values and not filename.endswith(extension_values):
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


def download_mast_product(
    data_uri: str,
    output_directory: str | os.PathLike[str],
    *,
    filename: str | None = None,
    session: requests.Session | None = None,
    timeout: float = 120.0,
) -> Path:
    """Download a single MAST product by dataURI and return the local path."""
    if not data_uri:
        raise ValueError("data_uri is required.")

    client = session or requests
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    local_name = filename or os.path.basename(data_uri)
    if not local_name:
        raise ValueError("Could not infer a filename from data_uri.")

    response = client.post(
        MAST_DOWNLOAD_URL,
        data=data_uri,
        stream=True,
        timeout=timeout,
    )
    response.raise_for_status()

    destination = output_path / local_name
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)

    return destination


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

            local_path = download_mast_product(
                str(data_uri),
                target_dir / obs_dir_name,
                filename=product.get("productFilename"),
                session=session,
                timeout=timeout,
            )
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
    }

    target_dir.mkdir(parents=True, exist_ok=True)
    with (target_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest


def _format_observations_summary(rows: list[dict[str, Any]], limit: int = 50) -> str:
    """Format observation rows as a compact, LLM-readable summary."""
    if not rows:
        return "No JWST observations found for the given query."

    header = f"Found {len(rows)} JWST observation(s). Showing first {min(len(rows), limit)}:\n"
    lines = [header]
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
    planet = manifest.get("planet_name", "?")
    observations = manifest.get("observations", [])
    downloaded = manifest.get("downloaded", [])

    lines = [
        f"Downloaded JWST data for {planet}.",
        f"Observations matched: {len(observations)}",
        f"Files downloaded: {len(downloaded)}",
        "",
        "Local paths:",
    ]
    for entry in downloaded:
        lines.append(f"  - {entry.get('path', '?')}")
    if not downloaded:
        lines.append("  (none)")
    return "\n".join(lines)


class SearchMastJwstObservations(BaseTool):
    """
    Search MAST for JWST observations around a single exoplanet target.

    Workflow
    --------
    1. Call this tool to discover JWST observations for a planet (returns obsid + metadata).
    2. Pick an obsid of interest and call ``GetMastObservationProducts`` to list files.
    3. Call ``DownloadMastJwstProducts`` (one-shot) OR fetch specific products directly.

    Why coordinate-centered
    -----------------------
    MAST target names are usually host-star names (e.g. 'WASP-39'), not planet names
    ('WASP-39 b'). This tool resolves the planet name via MAST Name Lookup and
    then runs a cone-search by RA/Dec. If MAST cannot resolve the planet name,
    pass RA and Dec directly (look them up via the exoarchive tools or Simbad).

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

    Example
    -------
        SearchMastJwstObservations(
            planet_name="WASP-39 b",
            instruments=["NIRSpec"],
            dataproduct_types=["spectrum", "timeseries"],
            calib_levels=[3],
        )
    """

    planet_name: str = RuntimeField(
        description="Exoplanet or host-star target name, e.g. 'WASP-39 b'."
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

    def _run(self) -> str:
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
        return _format_observations_summary(observations)


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
    One-shot download of JWST products for an individual exoplanet target.

    Combines search + product listing + download into a single call. Writes
    files under ``{base_directory}/{output_dir}/{planet_name}/{obs_id}/`` and
    saves a ``manifest.json`` describing observations + local paths.

    Workflow
    --------
    1. (Optional) Use ``SearchMastJwstObservations`` first to preview matches.
    2. Call this tool with the same filters to actually download files.
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

    planet_name: str = RuntimeField(
        description="Exoplanet or host-star target name, e.g. 'WASP-39 b'."
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
