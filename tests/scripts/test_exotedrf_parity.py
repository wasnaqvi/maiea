"""DownloadMastJwstProducts vs exoTEDRF extra_functions.download_observations.

Both sides run against ONE fake MAST archive (fixture below, modelled on
ERS 1366 / WASP-39 b with decoys and target-acquisition exposures):

* ASTER side: the real mast.py end to end; only the HTTP layer
  (``_mast_query``) and the byte transfer (``download_mast_product``) are
  faked, so query building, instrument aliasing, grating/visit selection and
  product filtering all execute for real. Always runs.
* exoTEDRF side: the REAL ``download_observations`` source with
  ``astroquery.mast.Observations`` replaced by a fake over the same archive.
  Runs when exoTEDRF source is found (``ASTER_EXOTEDRF_REPO`` checkout, an
  importable ``exotedrf``, or ``ASTER_EXOTEDRF_SRC`` pointing at
  ``extra_functions.py``); skipped otherwise.

Assumption baked into the fake astroquery (unverifiable offline): a bare
grating such as ``filters="G395H"`` matches the joined CAOM value
"F290LP;G395H", i.e. exoTEDRF works as its docstring says. ASTER matches
tokens, so it is correct under either representation.
"""
import math

WASP39 = (217.32667, -3.44450)
OTHER = (10.0, 20.0)

OBS = [
    # obsid, obs_id, instrument, filters, program, coords, visit prefixes
    ("101", "jw01366-o003_t001_nirspec_clear-prism-s200a1-sub512",
     "NIRSPEC/SLIT", "CLEAR;PRISM", "1366", WASP39, ["jw01366003001"]),
    ("102", "jw01366-o004_t001_nirspec_f290lp-g395h-s1600a1-sub2048",
     "NIRSPEC/SLIT", "F290LP;G395H", "1366", WASP39, ["jw01366004001", "jw01366004002"]),
    ("103", "jw01366-o001_t001_niriss_clear-gr700xd-substrip256",
     "NIRISS/SOSS", "CLEAR;GR700XD", "1366", WASP39, ["jw01366001001"]),
    ("104", "jw01366-o002_t001_nircam_f322w2-grismr-subgrism256",
     "NIRCAM/GRISM", "F322W2;GRISMR", "1366", WASP39, ["jw01366002001"]),
    ("105", "jw02783-o001_t001_nirspec_f290lp-g395h-s1600a1-sub2048",
     "NIRSPEC/SLIT", "F290LP;G395H", "2783", WASP39, ["jw02783001001"]),
    ("106", "jw01366-o099_t009_nirspec_f290lp-g395h-s1600a1-sub2048",
     "NIRSPEC/SLIT", "F290LP;G395H", "1366", OTHER, ["jw01366099001"]),
]


def obs_rows():
    return [
        {"obsid": o, "obs_id": oid, "instrument_name": inst, "filters": flt,
         "proposal_id": prog, "s_ra": c[0], "s_dec": c[1],
         "obs_collection": "JWST", "dataproduct_type": "timeseries",
         "calib_level": 3, "target_name": "WASP-39" if c == WASP39 else "OTHER"}
        for o, oid, inst, flt, prog, c, _ in OBS
    ]


def products_for(obsid):
    _, oid, inst, *_r, prefixes = next(x for x in OBS if x[0] == obsid)
    dets = ("nis",) if "NIRISS" in inst else ("nrcalong",) if "NIRCAM" in inst else ("nrs1", "nrs2")
    out = []
    for p in prefixes:
        for seg in (1, 2):
            for d in dets:
                for kind, sub in (("uncal", "UNCAL"), ("rateints", "RATEINTS")):
                    name = f"{p}_04101_00001-seg{seg:03d}_{d}_{kind}.fits"
                    out.append({"obsID": obsid, "productFilename": name,
                                "dataURI": f"mast:JWST/product/{name}",
                                "productSubGroupDescription": sub,
                                "productType": "SCIENCE", "dataRights": "PUBLIC",
                                "dataproduct_type": "timeseries", "size": 1})
        # target acquisition exposure in the same visit
        ta = f"{p}_02101_00001_{dets[0]}_uncal.fits"
        out.append({"obsID": obsid, "productFilename": ta,
                    "dataURI": f"mast:JWST/product/{ta}",
                    "productSubGroupDescription": "UNCAL", "productType": "SCIENCE",
                    "dataRights": "PUBLIC", "dataproduct_type": "image", "size": 1})
    x1d = f"{oid}_x1dints.fits"
    out.append({"obsID": obsid, "productFilename": x1d, "dataURI": f"mast:JWST/product/{x1d}",
                "productSubGroupDescription": "X1DINTS", "productType": "SCIENCE",
                "dataRights": "PUBLIC", "dataproduct_type": "timeseries", "size": 1})
    return out


def sep_deg(a, b):
    r1, d1, r2, d2 = map(math.radians, (*a, *b))
    h = math.sin((d2 - d1) / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2) ** 2
    return math.degrees(2 * math.asin(min(1, math.sqrt(h))))


def resolve(name):
    key = name.replace(" ", "").upper()
    if key.startswith("WASP-39"):
        return WASP39
    raise ValueError(name)

import importlib.util
import os
import pathlib
import sys
import tempfile
import types
from unittest.mock import patch

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]


def _exotedrf_source():
    """Path to exoTEDRF's extra_functions.py, or None."""
    candidates = []
    if os.environ.get("ASTER_EXOTEDRF_SRC"):
        candidates.append(pathlib.Path(os.environ["ASTER_EXOTEDRF_SRC"]))
    for root in (os.environ.get("ASTER_EXOTEDRF_REPO"), "~/exoTEDRF",
                 "/Users/wasi/Desktop/exoTEDRF"):
        if root:
            candidates.append(pathlib.Path(root).expanduser() / "exotedrf" / "extra_functions.py")
    spec = importlib.util.find_spec("exotedrf")
    if spec and spec.submodule_search_locations:
        for loc in spec.submodule_search_locations:
            candidates.append(pathlib.Path(loc) / "extra_functions.py")
    return next((c for c in candidates if c.is_file()), None)


ETD_SRC = _exotedrf_source()


# ---------------- exoTEDRF side ----------------

def _table(rows):
    from astropy.table import Table
    cols = sorted({k for r in rows for k in r})
    t = Table({c: [str(r.get(c, "")) for r in rows] for c in cols} if rows
              else {"productFilename": np.array([], dtype=str)}, masked=True)
    return t


class FakeObservations:
    query_log = []

    @staticmethod
    def query_criteria(proposal_id=None, instrument_name=None, filters=None,
                       objectname=None, radius=".2 deg"):
        FakeObservations.query_log.append(dict(
            proposal_id=proposal_id, instrument_name=instrument_name,
            filters=filters, objectname=objectname, radius=radius))
        rad = float(str(radius).split()[0])
        rows = []
        for r in obs_rows():
            if proposal_id is not None and r["proposal_id"] != str(proposal_id):
                continue
            if instrument_name is not None and r["instrument_name"] != instrument_name:
                continue
            # ASSUMPTION (unverifiable offline): treat a bare grating as
            # matching the joined CAOM value, i.e. exoTEDRF works as documented.
            if filters is not None and str(filters).upper() not in r["filters"].split(";"):
                continue
            if objectname is not None and sep_deg(resolve(objectname),
                                                    (r["s_ra"], r["s_dec"])) > rad:
                continue
            rows.append(r)
        return _table(rows)

    @staticmethod
    def get_product_list(obs_table):
        rows = []
        for obsid in obs_table["obsid"]:
            rows += products_for(str(obsid))
        return _table(rows)

    @staticmethod
    def filter_products(products, dataproduct_type=None, extension=None, productType=None):
        keep = []
        for i, row in enumerate(products):
            if dataproduct_type and row["dataproduct_type"] != dataproduct_type:
                continue
            if productType and row["productType"] != productType:
                continue
            if extension and not str(row["productFilename"]).endswith(extension):
                continue
            keep.append(i)
        return products[np.array(keep, dtype=int)]

    @staticmethod
    def download_products(products):
        for row in products:
            d = pathlib.Path("mastDownload/JWST") / str(row["productFilename"]).split("_")[0]
            d.mkdir(parents=True, exist_ok=True)
            (d / str(row["productFilename"])).write_bytes(b"x")


def load_exotedrf():
    from astropy.table import Table  # noqa: F401  (needed by _table)
    astroquery = types.ModuleType("astroquery")
    mastmod = types.ModuleType("astroquery.mast")
    mastmod.Observations = FakeObservations
    pkg = types.ModuleType("exotedrf")
    utils = types.ModuleType("exotedrf.utils")
    utils.fancyprint = lambda *a, **k: None
    pkg.utils = utils
    sys.modules.update({"astroquery": astroquery, "astroquery.mast": mastmod,
                        "exotedrf": pkg, "exotedrf.utils": utils})
    spec = importlib.util.spec_from_file_location("etd_extra", ETD_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_exotedrf(**kwargs):
    etd = load_exotedrf()
    saved = {k: sys.modules.get(k) for k in
             ("astroquery", "astroquery.mast", "exotedrf", "exotedrf.utils")}
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            etd.download_observations(**kwargs)
            return sorted(p.name for p in pathlib.Path("DMS_uncal").iterdir())
        finally:
            os.chdir(cwd)
            for k, v in saved.items():   # never leak the stubs into other tests
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v


# ---------------- ASTER side ----------------

def load_mast():
    spec = importlib.util.spec_from_file_location(
        "mast_parity", REPO / "aster_toolkit" / "data_acquisition" / "mast.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REQUESTS = []


def fake_mast_query(request, **_):
    REQUESTS.append(request)
    svc = request["service"]
    params = request.get("params", {})
    if svc == "Mast.Name.Lookup":
        ra, dec = resolve(params["input"])
        return {"resolvedCoordinate": [{"ra": ra, "decl": dec}]}
    if svc == "Mast.Caom.Products":
        return {"status": "COMPLETE", "data": products_for(str(params["obsid"]))}
    if svc in ("Mast.Caom.Filtered", "Mast.Caom.Filtered.Position"):
        rows = obs_rows()
        for flt in params.get("filters", []):
            vals = flt.get("values") or []
            if vals:  # discrete filters are EXACT match server-side
                rows = [r for r in rows if r.get(flt["paramName"]) in vals
                        or str(r.get(flt["paramName"])) in map(str, vals)]
        if "position" in params:
            ra, dec, rad = (float(x) for x in params["position"].split(","))
            rows = [r for r in rows if sep_deg((ra, dec), (r["s_ra"], r["s_dec"])) <= rad]
        return {"status": "COMPLETE", "data": rows,
                "paging": {"rowsFiltered": len(rows), "rowsTotal": len(rows)}}
    raise AssertionError(f"unexpected service {svc}")


def run_aster(**kwargs):
    mast = load_mast()
    REQUESTS.clear()

    def fake_download(data_uri, out_dir, filename=None, **_):
        p = pathlib.Path(out_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return p

    with tempfile.TemporaryDirectory() as tmp, \
         patch.object(mast, "_mast_query", side_effect=fake_mast_query), \
         patch.object(mast, "download_mast_product", side_effect=fake_download):
        try:
            tool = mast.DownloadMastJwstProducts(base_directory=tmp, **kwargs)
        except TypeError:
            tool = mast.DownloadMastJwstProducts()
            tool.base_directory = tmp
            for k, v in kwargs.items():
                setattr(tool, k, v)
        out = tool._run()
        files = sorted(p.name for p in pathlib.Path(tmp).rglob("*.fits"))
        dirs = sorted({str(p.parent.relative_to(tmp)) for p in pathlib.Path(tmp).rglob("*.fits")})
    return files, dirs, out


CASES = [
    ("exoTEDRF docstring example",
     dict(proposal_id="1366", objectname="WASP-39b", instrument_name="NIRSPEC/SLIT",
          filters="G395H"),
     dict(proposal_id="1366", planet_name="WASP-39b", instruments="NIRSPEC/SLIT",
          filters="G395H", dataproduct_types=["timeseries"], raw_only=True)),
    ("program only (no target)",
     dict(proposal_id="1366", instrument_name="NIRSPEC/SLIT", filters="G395H"),
     dict(proposal_id="1366", instruments="NIRSPEC/SLIT", filters="G395H",
          dataproduct_types=["timeseries"], raw_only=True)),
    ("visit_nos=2",
     dict(proposal_id="1366", objectname="WASP-39b", instrument_name="NIRSPEC/SLIT",
          filters="G395H", visit_nos=2),
     dict(proposal_id="1366", planet_name="WASP-39b", instruments="NIRSPEC/SLIT",
          filters="G395H", dataproduct_types=["timeseries"], raw_only=True, visits=[2])),
    ("whole program, all instruments",
     dict(proposal_id="1366"),
     dict(proposal_id="1366", dataproduct_types=["timeseries"], raw_only=True)),
    ("NIRISS SOSS",
     dict(proposal_id="1366", objectname="WASP-39b", instrument_name="NIRISS/SOSS"),
     dict(proposal_id="1366", planet_name="WASP-39b", instruments="NIRISS/SOSS",
          dataproduct_types=["timeseries"], raw_only=True)),
]



# ---------------- tests ----------------

EXAMPLE_SCIENCE = sorted(
    f"{p}_04101_00001-seg{s:03d}_{d}_uncal.fits"
    for p in ("jw01366004001", "jw01366004002") for s in (1, 2) for d in ("nrs1", "nrs2")
)


def test_exotedrf_docstring_example_aster_side():
    """download_observations(proposal_id='1366', objectname='WASP-39b',
    instrument_name='NIRSPEC/SLIT', filters='G395H')"""
    files, dirs, out = run_aster(**CASES[0][2])
    assert files == EXAMPLE_SCIENCE                       # PRISM, SOSS, NIRCam,
    assert dirs == ["mast/WASP-39b/jw01366-o004_t001_nirspec_f290lp-g395h-s1600a1-sub2048"]
    assert not any("_02101_" in f for f in files)         # no target acquisition
    assert "Files downloaded: 8" in out                   # other program, other star


def test_query_sent_to_mast_is_exact_and_normalized():
    run_aster(proposal_id="01366", planet_name="WASP-39 b", instruments="NIRSpec",
              filters=["G395H"], dataproduct_types=["timeseries"], raw_only=True)
    cone = [r for r in REQUESTS if r["service"] == "Mast.Caom.Filtered.Position"]
    assert len(cone) == 1
    flt = {f["paramName"]: f["values"] for f in cone[0]["params"]["filters"]}
    assert flt["proposal_id"] == ["1366"]                 # MAST is exact-match
    assert "NIRSPEC/SLIT" in flt["instrument_name"]
    # grating filtering happens before product listing: only the G395H obs
    assert [r["params"]["obsid"] for r in REQUESTS
            if r["service"] == "Mast.Caom.Products"] == ["102"]


def test_target_acquisition_dropped_in_every_mode():
    for kw in (dict(obsids=["102"]), dict(proposal_id="1366", filters="G395H")):
        files, _, _ = run_aster(dataproduct_types=["timeseries"], raw_only=True, **kw)
        assert files and not any("_02101_" in f for f in files), kw
    # ...and kept when the caller does not restrict the product type
    files, _, _ = run_aster(obsids=["102"], raw_only=True)
    assert any("_02101_" in f for f in files)


def test_list_only_matches_download_selection():
    _, _, out = run_aster(list_only=True, **CASES[2][2])   # visit 2 of the example
    assert "LIST ONLY" in out and "Visits selected:  ['jw01366004002']" in out
    assert "jw01366004002_04101" in out and "jw01366004001_04101" not in out
    assert "_02101_" not in out


@pytest.mark.skipif(ETD_SRC is None, reason="exoTEDRF source not found "
                    "(set ASTER_EXOTEDRF_REPO or ASTER_EXOTEDRF_SRC)")
@pytest.mark.parametrize("name, etd_kw, aster_kw", CASES, ids=[c[0] for c in CASES])
def test_parity_with_real_exotedrf(name, etd_kw, aster_kw):
    pytest.importorskip("astropy")
    pytest.importorskip("scipy")
    reference = run_exotedrf(**etd_kw)
    files, _, _ = run_aster(**aster_kw)
    assert files == reference, (
        f"only exoTEDRF: {sorted(set(reference) - set(files))}; "
        f"only ASTER: {sorted(set(files) - set(reference))}")


def test_agent_execute_path_with_llm_style_strings():
    """How the agent actually calls it: orchestral execute() with stringified
    lists/bools and empty-string stuffing in unused fields."""
    pytest.importorskip("orchestral")
    mast = load_mast()
    REQUESTS.clear()

    def fake_download(data_uri, out_dir, filename=None, **_):
        p = pathlib.Path(out_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return p

    with tempfile.TemporaryDirectory() as tmp, \
         patch.object(mast, "_mast_query", side_effect=fake_mast_query), \
         patch.object(mast, "download_mast_product", side_effect=fake_download):
        out = mast.DownloadMastJwstProducts(base_directory=tmp).execute(
            proposal_id="1366", planet_name="WASP-39b", instruments="['NIRSPEC/SLIT']",
            filters="G395H", dataproduct_types="['timeseries']", raw_only="True",
            visits="", obsids="", rows_path="", max_products="",
        )
        files = sorted(p.name for p in pathlib.Path(tmp).rglob("*.fits"))
    assert "Missing Required Fields" not in out and "Error" not in out.splitlines()[0]
    assert files == EXAMPLE_SCIENCE
