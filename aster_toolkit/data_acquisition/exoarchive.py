import os
import json
import io
import csv
import requests
import numpy as np
import pandas as pd
from astropy.io import ascii
from tqdm import tqdm

try:
    from orchestral.tools.base.tool import BaseTool
    from orchestral.tools.base.field_utils import RuntimeField, StateField
    from pydantic import Field as _PydanticField

    def OptionalRuntimeField(description=None, **kwargs):
        """LLM-visible OPTIONAL field (see mast.py: orchestral treats
        RuntimeField(default=None) as REQUIRED)."""
        kwargs.pop("default", None)
        return _PydanticField(json_schema_extra={"runtime": True},
                              default_factory=lambda: None,
                              description=description, **kwargs)
except ModuleNotFoundError:
    # Fallback so the pure query helpers (get_exoplanet_params_tap,
    # find_exoplanets_by_condition) stay importable/testable without the full
    # Orchestral stack. Mirrors the shim in mast.py. Tool subclasses become
    # inert stubs (fields collapse to their defaults) but do not crash import.
    class BaseTool:  # type: ignore[no-redef]
        pass

    def RuntimeField(default=None, description=None):  # type: ignore[misc]
        return default

    def StateField(default=None, description=None):  # type: ignore[misc]
        return default

    def OptionalRuntimeField(description=None, **kwargs):  # type: ignore[misc]
        return None


def process_wgets_file(base_directory, wgets_file_path: str, data_path: str) -> None:
    # Read the wgets file and build a dictionary mapping planet names to their data file URLs
    full_wgets_path = os.path.join(base_directory, wgets_file_path)
    with open(full_wgets_path, 'r') as file:
        lines = file.readlines()

    names_to_urls = {}
    manifest_url = None  # keep this in case needed it later

    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace and newline characters

        # Skip empty lines, comments, and shebang
        if not line or line.startswith('#') or line.startswith('//') or line.startswith('!'):
            continue

        # Parse wget command: wget -O filename URL
        parts = line.split()
        if len(parts) != 4 or parts[0] != 'wget' or parts[1] != '-O':
            continue  # Skip malformed lines

        _, _, file_name, url = parts
        planet_name = file_name.split('.')[0]

        if planet_name == 'spectra':
            # do we need this?
            manifest_url = url
            continue

        if planet_name not in names_to_urls:
            names_to_urls[planet_name] = [url]
        else:
            names_to_urls[planet_name].append(url)

    print('Found data files for planets:')
    for i, name in enumerate(names_to_urls, start=1):
        print(f'{i:4} - {name:16}: With {len(names_to_urls[name])} file(s)')

    # Fetch and save planet data
    for planet_name, urls in names_to_urls.items():
        planet_dir = os.path.join(base_directory, data_path, planet_name)
        os.makedirs(planet_dir, exist_ok=True)

        for url in urls:
            response = requests.get(url)
            response.raise_for_status()

            contents = response.text
            tbl = ascii.read(contents, format="ipac")
            df = tbl.to_pandas()      # type: ignore

            file_name = url.split('/')[-1]
            file_path = os.path.join(planet_dir, file_name.replace('.tbl', '.csv'))
            df.to_csv(file_path, index=False)
            print(f'Saved data for {planet_name} to {file_path}')


def process_downloads(base_directory, data_path: str, output_dir: str) -> None:
    '''
    Process the downloaded data files, extract meaningful spectral data and metadata, and save them in a structured format.
    Parameters:
    - base_directory: The base directory for the project.
    - data_path: The relative path to the base_directory containing the raw downloaded data files.
    - output_dir: The relative path to the base_directory where the processed data will be saved.
    Note that the agent never needs to think about the base_directory, thus the need for data and output paths to be relative paths.
    '''
    def extract_meaningful_data(df):
        # Essential spectral data
        central_wavelength = df.CENTRALWAVELNG.values

        # Check if this is transit or eclipse data
        # Note: Some files may have both columns, so we check which has actual data
        has_transit = 'PL_TRANDEP' in df.columns
        has_eclipse = 'ESPECLIPDEP' in df.columns

        if has_transit:
            # Check if transit column has non-null values
            transit_values = df.PL_TRANDEP.dropna()
            if len(transit_values) > 0:
                # Transit observation with actual data
                transit_depth = df.PL_TRANDEP.values / 100  # Modulation fraction (not in percent)
                transit_depth_error = df.PL_TRANDEPERR1.values / 100  # Error in modulation fraction
                authors = df.PL_TRANDEP_AUTHORS.values[0]
                url = df.PL_TRANDEP_URL.values[0]
                observation_type = 'transit'
            elif has_eclipse:
                # Transit column exists but is empty, try eclipse
                raise ValueError('This sample contains an Eclipse observation rather than a Transit observation. Please filter for only transit observations on the Exoplanet Archive website and re-download the data.')
            else:
                raise ValueError(f"Transit depth column exists but contains no data. Available columns: {df.columns.tolist()}")
        elif has_eclipse:
            # Eclipse observation
            raise ValueError('This sample contains an Eclipse observation rather than a Transit observation. Please filter for only transit observations on the Exoplanet Archive website and re-download the data.')
        else:
            raise ValueError(f"Cannot find transit depth column in dataframe. Available columns: {df.columns.tolist()}")

        spec = np.column_stack([
                    central_wavelength,
                    transit_depth,
                    transit_depth_error
                ])

        # Metadata
        facility = 'JWST'
        instrument = 'NIRSpec'

        metadata = {
            'observation_type': observation_type,
            'authors': authors,
            'url': url,
            'facility': facility,
            'instrument': instrument,
            'units': {
                'central_wavelength': '[μm]',
                'transit_depth': 'fraction: [R_P^2/R_S^2]' if observation_type == 'transit' else 'fraction: eclipse depth',
                'transit_depth_error': 'fraction: [R_P^2/R_S^2]' if observation_type == 'transit' else 'fraction: eclipse depth',
            }
        }

        return spec, metadata

    def extract_name(file_name):
        """Given a file name like 'WASP_39_b_3.11466_5077_1.csv' extract 'WASP_39_b_3.11466_5077_1'"""
        return file_name.replace('.csv', '')

    planets = os.listdir(os.path.join(base_directory, data_path))
    for planet in tqdm(planets):
        planet_dir = os.path.join(base_directory, data_path, planet)
        if not os.path.isdir(planet_dir):
            continue  # skip non-directories, just in case

        observations = os.listdir(planet_dir)

        for obs_file in observations:
            obs_name = extract_name(obs_file)
            df = pd.read_csv(os.path.join(base_directory, data_path, planet, obs_file))

            spec, metadata = extract_meaningful_data(df)

            # Create the output directories if they don't exist
            os.makedirs(os.path.join(base_directory, output_dir, planet, obs_name), exist_ok=True)

            # Save the spectrum data
            # spec.to_csv(
            #     os.path.join(output_dir, planet, obs_name, 'spectrum.dat'),
            #     index=False
            # )
            #np.save(spec.to_numpy(), os.path.join(output_dir, planet, obs_name, 'spectrum.npy'))
            np.savetxt(
                os.path.join(base_directory, output_dir, planet, obs_name, "spectrum.dat"),
                spec,
                fmt="%.10e"
            )


            # Save the metadata
            with open(os.path.join(base_directory, output_dir, planet, obs_name, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=4)


# NASA Exoplanet Archive TAP service URL
TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


def get_exoplanet_params_tap(planet_name: str, columns: list | str, table: str = "pscomppars") -> dict | None:
    """
    Query NASA Exoplanet Archive TAP service for one or multiple parameters.

    Parameters:
    - planet_name: Name of the exoplanet (e.g., "Kepler-10 b")
    - columns: List of parameter columns to retrieve or single column name string
    - table: TAP table name, "pscomppars" (default) for composite parameters or "ps" for all parameters

    Returns:
    - Dictionary with requested parameters or None if planet not found
    """
    # Convert single string to list
    if isinstance(columns, str):
        columns = [columns]

    # Build ADQL SELECT clause
    select_clause = ", ".join(columns)

    # Build query
    adql = (
        f"SELECT {select_clause} "
        f"FROM {table} "
        f"WHERE pl_name = '{planet_name}'"
    )

    # Call TAP sync
    params = {
        "query": adql,
        "format": "csv"
    }

    response = requests.get(TAP_SYNC_URL, params=params)
    response.raise_for_status()

    # Parse CSV in memory
    csv_file = io.StringIO(response.text)
    reader = csv.DictReader(csv_file)

    row = next(reader, None)
    if row is None:
        return None

    # Return results as a dictionary
    return {col: row[col] for col in columns}

def find_exoplanets_by_condition(conditions: list[str] | str, return_columns: list[str] | str, table: str = "pscomppars", distinct: bool = True, limit: int | None = None,) -> list[dict]:
    """
    Query NASA Exoplanet Archive TAP service for planets using conditions on parameters.

    Parameters:
    - conditions: either a single WHERE condition string or a list of conditions
                  e.g. "sy_pnum >= 2"
                  or ["sy_pnum >= 2", "pl_bmasse > 5"]
    - return_columns: columns to return
    - table: TAP table name
    - distinct: whether to return unique values only
    - limit: optional row limit

    Returns:
    - List of dictionaries containing the requested columns for each matching row
    """
    # Make sure conditions is a list
    if isinstance(conditions, str):
        conditions = [conditions]

    where_clause = " AND ".join(f"({c})" for c in conditions)

    select_keyword = "SELECT DISTINCT" if distinct else "SELECT"
    top_clause = f"TOP {limit} " if limit is not None else ""
    if isinstance(return_columns, str):
        return_columns = [return_columns]
    
    select_clause = ", ".join(return_columns)

    adql = (
        f"{select_keyword} {top_clause}{select_clause} "
        f"FROM {table} "
        f"WHERE {where_clause}"
    )

    # Send request
    params = {
        "query": adql,
        "format": "csv"
    }

    response = requests.get(TAP_SYNC_URL, params=params)
    response.raise_for_status()

    # Parse CSV response
    csv_file = io.StringIO(response.text)
    reader = csv.DictReader(csv_file)

    # this: row = next(reader, None) only grab the first matching planet, does not work here

    # Return ALL requested columns per row
    results = []
    for row in reader:
        results.append({col: row.get(col) for col in return_columns})

    return results


_ARCHIVE_TABLES = ("pscomppars", "ps")

_PLANET_DEFAULT_COLUMNS = ["pl_radj", "pl_bmassj", "pl_eqt"]

# Parameter display mapping for single-planet results (column -> (label, unit)).
_PARAM_LABELS = {
    'pl_name': ('Planet Name', ''),
    'pl_radj': ('Planet Radius', 'RJup'),
    'pl_rade': ('Planet Radius', 'REarth'),
    'pl_bmassj': ('Planet Mass', 'MJup'),
    'pl_bmasse': ('Planet Mass', 'MEarth'),
    'pl_eqt': ('Equilibrium Temp', 'K'),
    'pl_orbper': ('Orbital Period', 'days'),
    'pl_orbsmax': ('Semi-major Axis', 'AU'),
    'hostname': ('Star Name', ''),
    'st_rad': ('Star Radius', 'Rsun'),
    'st_teff': ('Star Temp', 'K'),
    'st_mass': ('Star Mass', 'Msun'),
    'st_met': ('Stellar Metallicity', 'dex'),
    'st_logg': ('Stellar log(g)', 'log10(cm/s^2)'),
}


def _as_str_list(value) -> list[str]:
    """Coerce LLM-supplied list-ish input ('a', "['a','b']", ['a']) to list[str]."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[0] in "[(" and text[-1] in "])":
            import ast
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return [text]
            if isinstance(parsed, (list, tuple)):
                return [str(v) for v in parsed if str(v).strip()]
        return [text]
    return [str(v) for v in value if str(v).strip()]


def _format_planet_parameters(planet_name: str, result: dict) -> str:
    output = f"Parameters for {planet_name}:\n\n"
    for param, value in result.items():
        label, unit = _PARAM_LABELS.get(param, (param, ''))
        try:
            formatted = f"{float(value):.5g}"
        except (ValueError, TypeError):
            formatted = value
        output += f"  {label}: {formatted} {unit}".rstrip() + "\n"
    return output


class QueryExoplanetArchive(BaseTool):
    """
    The ONE tool for the NASA Exoplanet Archive parameter tables (TAP):
    one named planet's parameters, or a list of planets matching criteria.

    USE WHEN:
      * a named planet's / host star's measured properties — radius, mass,
        Teq, period, a, st_teff, st_logg, st_met... → pass ``planet_name``.
      * every planet satisfying conditions — "sub-Neptunes within 50 pc",
        "Teq > 1000 K in multi-planet systems" → pass ``conditions``.

    NOT FOR: "who observed X", instruments, JWST programs/PIs/cycles — the
    archive parameter tables have no observation logs. Route those to
    `SearchMastJwstObservations` / `GetJwstProgramInfo`. To join JWST
    observations to a planet population, use `CrossmatchJwstToPlanets`.

    Modes (supply exactly one of ``planet_name`` / ``conditions``)
    -------------------------------------------------------------
    * ``planet_name`` — exact archive name ('GJ 9827 d', 'WASP-39 b').
      ``columns`` defaults to ['pl_radj', 'pl_bmassj', 'pl_eqt'].
    * ``conditions`` — ADQL WHERE expressions combined with AND, e.g.
      ["pl_bmassj > 0.3", "pl_eqt > 1000", "discoverymethod = 'Transit'"]
      (quote strings; numbers unquoted). ``columns`` defaults to ['pl_name'];
      ``distinct`` and ``limit`` apply.

    Common columns: pl_name, hostname, pl_orbper [d], pl_orbsmax [AU],
    pl_rade / pl_radj, pl_bmasse / pl_bmassj, pl_eqt [K], st_rad [Rsun],
    st_teff [K], st_mass [Msun], st_met [dex], st_logg, sy_dist [pc],
    sy_pnum. Full list: https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html

    ``table`` accepts ONLY 'pscomppars' (default; composite, may not be
    self-consistent) or 'ps' (all parameter sets). Never a file path.

    Cite the NASA Exoplanet Archive if results are used in research.

    Examples
    --------
        QueryExoplanetArchive(planet_name="WASP-39 b",
                              columns=["pl_radj", "pl_bmassj", "st_teff"])
        QueryExoplanetArchive(conditions=["sy_pnum >= 2"],
                              columns=["pl_name", "pl_orbsmax"])
    """

    planet_name: str | None = OptionalRuntimeField(
        description="Single-planet mode: exact archive name, e.g. 'Kepler-10 b'.",
    )
    conditions: list | str | None = OptionalRuntimeField(
        description="List mode: ADQL WHERE conditions, e.g. ['sy_pnum >= 2', 'pl_eqt > 1000'].",
    )
    columns: list | str | None = OptionalRuntimeField(
        description=(
            "Columns to return. Default ['pl_radj','pl_bmassj','pl_eqt'] "
            "(planet_name mode) or ['pl_name'] (conditions mode)."
        ),
    )
    table: str = RuntimeField(
        default="pscomppars",
        description="TAP table: 'pscomppars' (composite) or 'ps' (all parameter sets).",
    )
    distinct: bool | str = RuntimeField(
        default=True,
        description="conditions mode: return unique rows only.",
    )
    limit: int | str | None = OptionalRuntimeField(
        description="conditions mode: optional max number of rows.",
    )

    def _run(self) -> str:
        planet_name = (self.planet_name or "").strip() or None
        conditions = _as_str_list(self.conditions)
        columns = _as_str_list(self.columns)
        table = (self.table or "pscomppars").strip() or "pscomppars"
        if table not in _ARCHIVE_TABLES:
            raise ValueError(
                f"table must be one of {_ARCHIVE_TABLES}, got {table!r}. "
                "This tool queries archive tables, not files."
            )
        if planet_name and conditions:
            raise ValueError(
                "Supply planet_name OR conditions, not both. To filter around "
                "one planet, add \"pl_name = '<name>'\" to conditions instead."
            )
        if not planet_name and not conditions:
            raise ValueError("Supply planet_name (one planet) or conditions (a list).")

        if planet_name:
            # ADQL string literal: double any single quote so a name cannot
            # break out of the WHERE clause.
            safe_name = planet_name.replace("'", "''")
            result = get_exoplanet_params_tap(
                planet_name=safe_name,
                columns=columns or list(_PLANET_DEFAULT_COLUMNS),
                table=table,
            )
            if result is None:
                return f"No data found for planet '{planet_name}' in table '{table}'"
            return _format_planet_parameters(planet_name, result)

        distinct = self.distinct
        if isinstance(distinct, str):
            distinct = distinct.strip().lower() not in ("false", "0", "no", "")
        limit = self.limit
        if isinstance(limit, str):
            limit = int(limit) if limit.strip() else None
        matches = find_exoplanets_by_condition(
            conditions=conditions,
            return_columns=columns or ["pl_name"],
            table=table,
            distinct=bool(distinct),
            limit=limit,
        )
        if not matches:
            return "No matching results found."
        output = f"Matching results ({len(matches)} rows):\n\n"
        for i, row in enumerate(matches, start=1):
            output += f"{i}. " + ", ".join(f"{k}: {v}" for k, v in row.items()) + "\n"
        return output


class DownloadDataset(BaseTool):
    """
    Download REDUCED transit spectra from the NASA Exoplanet Archive (Firefly).

    USE WHEN: the user has Firefly wget commands / URL and wants ready-to-fit
    transit spectra (wavelength, depth, error) for a retrieval.

    NOT FOR: JWST FITS products (X1DINTS, UNCAL, etc.) — those come from MAST via
    `DownloadMastJwstProducts`. This tool only ingests Exoplanet-Archive wget
    commands and cannot construct URLs itself; the user must supply them.

    This tool supports THREE ways to provide wget commands:

    1. **File path** (wgets_file_path): User saves wget commands to a file, agent reads it
       - User creates 'wgets.txt' in workspace
       - Agent calls: DownloadDataset(wgets_file_path='wgets.txt')

    2. **Direct text** (wget_text): User pastes wget commands directly into chat
       - User: "Here are my wget commands: [pastes text]"
       - Agent calls: DownloadDataset(wget_text="wget -O file.tbl 'http://...'")

    3. **URL to wget page** (wget_url): User provides URL to Firefly wget page (EASIEST!)
       - User pastes: https://exoplanetarchive.ipac.caltech.edu/staging/...
       - Agent calls: DownloadDataset(wget_url="https://...")
       - Tool automatically scrapes wget commands from the page

    Working files are stored in download_dataset_tool/queryNNN/ for debugging.
    Final spectra are organized by planet name in the output_dir (default: spectra/).
    Each download gets a unique query ID (query001, query002, etc.).

    To generate wget commands:
    1. Go to https://exoplanetarchive.ipac.caltech.edu/cgi-bin/atmospheres/nph-firefly
    2. Filter by instrument and planet(s)
    3. Check boxes for desired spectra
    4. Click "Download all checked spectra"
    5. Provide the URL, text, or save to file
    """

    # Only ONE of these three should be provided
    wgets_file_path: str | None = RuntimeField(
        default=None,
        description="Path to existing wgets text file (relative to base_directory). Use if user saved wget commands to a file."
    )
    wget_text: str | None = RuntimeField(
        default=None,
        description="Raw wget commands as text. Use when user pastes wget commands directly into chat."
    )
    wget_url: str | None = RuntimeField(
        default=None,
        description="URL to Firefly wget page. Tool will scrape wget commands from this URL automatically."
    )

    output_dir: str = RuntimeField(
        default="spectra",
        description="Directory where final spectrum.dat files will be organized by planet name (e.g., spectra/WASP_39_b/dataset_id/spectrum.dat)"
    )

    base_directory: str = StateField()

    def _run(self) -> str:
        """Download and process spectra from NASA archive."""
        import tempfile
        import shutil
        from glob import glob

        # Count how many input methods were provided
        inputs_provided = sum([
            self.wgets_file_path is not None,
            self.wget_text is not None,
            self.wget_url is not None
        ])

        if inputs_provided == 0:
            return "Error: Please provide ONE of: wgets_file_path, wget_text, or wget_url"

        if inputs_provided > 1:
            return "Error: Please provide only ONE input method (file_path, text, or url)"

        # Generate unique query ID
        download_tool_dir = os.path.join(self.base_directory, 'download_dataset_tool')
        os.makedirs(download_tool_dir, exist_ok=True)

        # Find next available query number
        existing_queries = glob(os.path.join(download_tool_dir, 'query*'))
        query_nums = [int(os.path.basename(q).replace('query', '')) for q in existing_queries
                      if os.path.basename(q).replace('query', '').isdigit()]
        next_num = max(query_nums) + 1 if query_nums else 1
        query_id = f"query{next_num:03d}"

        # Create working directories
        query_dir = os.path.join(download_tool_dir, query_id)
        raw_path = os.path.join(query_dir, 'raw')
        processed_path = os.path.join(query_dir, 'processed')
        os.makedirs(raw_path, exist_ok=True)
        os.makedirs(processed_path, exist_ok=True)

        # Determine the wget file path
        wget_file_to_use = None
        temp_file_created = False

        if self.wgets_file_path:
            # Method 1: User provided file path
            wget_file_to_use = self.wgets_file_path

        elif self.wget_text:
            # Method 2: User provided text directly - create temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir=self.base_directory)
            temp_file.write(self.wget_text)
            temp_file.close()
            wget_file_to_use = os.path.basename(temp_file.name)
            temp_file_created = True

        elif self.wget_url:
            # Method 3: User provided URL - scrape and create temporary file
            try:
                response = requests.get(self.wget_url)
                response.raise_for_status()
                wget_commands = response.text

                # Create temporary file with scraped content
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir=self.base_directory)
                temp_file.write(wget_commands)
                temp_file.close()
                wget_file_to_use = os.path.basename(temp_file.name)
                temp_file_created = True

            except requests.RequestException as e:
                return f"Error fetching wget commands from URL: {e}"

        try:
            # Process wget file to download data
            process_wgets_file(
                base_directory=self.base_directory,
                wgets_file_path=wget_file_to_use,
                data_path=os.path.relpath(raw_path, self.base_directory)
            )

            # Process downloads to extract spectral data
            process_downloads(
                base_directory=self.base_directory,
                data_path=os.path.relpath(raw_path, self.base_directory),
                output_dir=os.path.relpath(processed_path, self.base_directory)
            )

            # Copy final spectra to output directory organized by planet
            final_output_dir = os.path.join(self.base_directory, self.output_dir)
            planet_dirs = os.listdir(processed_path)
            planets_processed = []

            for planet_dir in planet_dirs:
                planet_path = os.path.join(processed_path, planet_dir)
                if os.path.isdir(planet_path):
                    # Copy to final location
                    dest_path = os.path.join(final_output_dir, planet_dir)
                    os.makedirs(dest_path, exist_ok=True)

                    # Copy all dataset subdirectories
                    for dataset in os.listdir(planet_path):
                        src_dataset = os.path.join(planet_path, dataset)
                        dst_dataset = os.path.join(dest_path, dataset)
                        if os.path.isdir(src_dataset):
                            shutil.copytree(src_dataset, dst_dataset, dirs_exist_ok=True)

                    planets_processed.append(planet_dir)

            # Build response message with detailed spectrum file paths
            result = f"Download complete! Query ID: {query_id}\n\n"
            result += f"Planets processed: {', '.join(planets_processed)}\n\n"
            result += f"Working files: download_dataset_tool/{query_id}/\n"
            result += f"Final spectra saved to: {self.output_dir}/\n\n"
            result += "Spectrum file paths:\n"
            for planet in planets_processed:
                planet_spectra_dir = os.path.join(final_output_dir, planet)
                # List all spectrum.dat files in dataset subdirectories
                if os.path.isdir(planet_spectra_dir):
                    for dataset in os.listdir(planet_spectra_dir):
                        dataset_path = os.path.join(planet_spectra_dir, dataset)
                        spectrum_file = os.path.join(dataset_path, 'spectrum.dat')
                        if os.path.isfile(spectrum_file):
                            # Show relative path from base_directory
                            rel_spectrum_path = os.path.relpath(spectrum_file, self.base_directory)
                            result += f"  - {rel_spectrum_path}\n"

        finally:
            # Clean up temporary file if we created one
            if temp_file_created and wget_file_to_use:
                try:
                    os.remove(os.path.join(self.base_directory, wget_file_to_use))
                except:
                    pass

        return result


if __name__ == "__main__":
    base_directory = 'workspace'
    wgets_file = 'wfc3_wgets.txt'
    raw_data_path = 'raw_data'
    processed_data_path = 'processed_data'

    process_wgets_file(base_directory, wgets_file, raw_data_path)
    process_downloads(base_directory, raw_data_path, processed_data_path)