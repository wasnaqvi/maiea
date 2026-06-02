from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MAST_PATH = REPO_ROOT / "aster_toolkit" / "data_acquisition" / "mast.py"


def load_mast_module():
    spec = importlib.util.spec_from_file_location("mast_under_test", MAST_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeDownloadResponse:
    def __init__(self, chunks, status_code=200):
        self._chunks = chunks
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        return iter(self._chunks)

    def close(self):
        return None


class FakeDownloadSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, stream=False, timeout=None):
        self.calls.append(
            {
                "url": url,
                "params": params,
                "stream": stream,
                "timeout": timeout,
            }
        )
        return FakeDownloadResponse([b"abc", b"", b"def"])


class MastWrapperTests(unittest.TestCase):
    def setUp(self):
        self.mast = load_mast_module()

    def test_build_jwst_observation_filters(self):
        filters = self.mast._build_jwst_observation_filters(
            instruments=["NIRSpec", "NIRCam"],
            dataproduct_types="spectrum",
            calib_levels=[2, 3],
            target_name="WASP-39 b",
            proposal_id=1366,
        )

        self.assertEqual(filters[0], {"paramName": "obs_collection", "values": ["JWST"]})
        instrument_filter = next(f for f in filters if f["paramName"] == "instrument_name")
        self.assertIn("NIRSPEC/SLIT", instrument_filter["values"])
        self.assertIn("NIRCAM/IMAGE", instrument_filter["values"])
        self.assertIn({"paramName": "dataproduct_type", "values": ["spectrum"]}, filters)
        self.assertIn({"paramName": "calib_level", "values": [2, 3]}, filters)
        self.assertIn({"paramName": "proposal_id", "values": ["1366"]}, filters)

    def test_search_jwst_observations_uses_position_service_and_filters(self):
        with (
            patch.object(self.mast, "resolve_target_coordinates", return_value=(322.4167, -45.1234)),
            patch.object(
                self.mast,
                "_mast_query",
                return_value={"data": [{"obsid": "123", "instrument_name": "NIRSpec"}]},
            ) as query,
        ):
            rows = self.mast.search_jwst_observations(
                "WASP-39 b",
                radius_deg=0.03,
                instruments="NIRSpec",
                dataproduct_types=["spectrum", "timeseries"],
                calib_levels=2,
            )

        self.assertEqual(rows, [{"obsid": "123", "instrument_name": "NIRSpec"}])
        request = query.call_args.args[0]
        self.assertEqual(request["service"], "Mast.Caom.Filtered.Position")
        self.assertEqual(request["params"]["position"], "322.4167, -45.1234, 0.03")
        self.assertIn("obsid", request["params"]["columns"])
        self.assertIn(
            {"paramName": "obs_collection", "values": ["JWST"]},
            request["params"]["filters"],
        )
        self.assertIn(
            {
                "paramName": "instrument_name",
                "values": ["NIRSPEC/SLIT", "NIRSPEC/IFU", "NIRSPEC/MSA", "NIRSPEC/IMAGE"],
            },
            request["params"]["filters"],
        )

    def test_get_observation_products_filters_raw_jwst_fits(self):
        products = [
            {
                "productType": "SCIENCE",
                "productSubGroupDescription": "UNCAL",
                "productFilename": "jw01234_uncal.fits",
                "dataURI": "mast:JWST/product/jw01234_uncal.fits",
            },
            {
                "productType": "SCIENCE",
                "productSubGroupDescription": "X1D",
                "productFilename": "jw01234_x1d.fits",
                "dataURI": "mast:JWST/product/jw01234_x1d.fits",
            },
            {
                "productType": "AUXILIARY",
                "productSubGroupDescription": "UNCAL",
                "productFilename": "jw01234_uncal.fits",
                "dataURI": "mast:JWST/product/jw01234_uncal.fits",
            },
        ]

        with patch.object(self.mast, "_mast_query", return_value={"data": products}) as query:
            selected = self.mast.get_observation_products("98765", raw_only=True)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["productSubGroupDescription"], "UNCAL")
        request = query.call_args.args[0]
        self.assertEqual(request["service"], "Mast.Caom.Products")
        self.assertEqual(request["params"], {"obsid": "98765"})

    def test_download_mast_product_writes_streamed_bytes(self):
        session = FakeDownloadSession()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.mast.download_mast_product(
                "mast:JWST/product/jw01234_uncal.fits",
                tmpdir,
                filename="raw.fits",
                session=session,
                timeout=7,
            )

            self.assertEqual(path.read_bytes(), b"abcdef")

        self.assertEqual(session.calls[0]["url"], self.mast.MAST_DOWNLOAD_URL)
        self.assertEqual(
            session.calls[0]["params"],
            {"uri": "mast:JWST/product/jw01234_uncal.fits"},
        )
        self.assertTrue(session.calls[0]["stream"])
        self.assertEqual(session.calls[0]["timeout"], 7)

    def test_list_like_strings_are_coerced_for_tool_calls(self):
        filters = self.mast._build_jwst_observation_filters(
            instruments="['NIRSpec']",
            dataproduct_types="['spectrum', 'timeseries']",
            calib_levels="[3]",
        )

        self.assertIn(
            {
                "paramName": "instrument_name",
                "values": ["NIRSPEC/SLIT", "NIRSPEC/IFU", "NIRSPEC/MSA", "NIRSPEC/IMAGE"],
            },
            filters,
        )
        self.assertIn(
            {"paramName": "dataproduct_type", "values": ["spectrum", "timeseries"]},
            filters,
        )
        self.assertIn({"paramName": "calib_level", "values": [3]}, filters)

    def test_filter_products_accepts_list_like_string_subgroups(self):
        products = [
            {
                "productType": "SCIENCE",
                "productSubGroupDescription": "X1DINTS",
                "productFilename": "jw01234_x1dints.fits",
            }
        ]

        selected = self.mast.filter_products(
            products,
            product_subgroups="['X1DINTS']",
        )

        self.assertEqual(selected, products)

    def test_resolve_target_coordinates_parses_payload(self):
        payload = {
            "resolvedCoordinate": [
                {"ra": 322.4167, "decl": -45.1234, "resolver": "NED"}
            ]
        }
        with patch.object(self.mast, "_mast_query", return_value=payload) as query:
            ra, dec = self.mast.resolve_target_coordinates("WASP-39 b")

        self.assertAlmostEqual(ra, 322.4167)
        self.assertAlmostEqual(dec, -45.1234)
        request = query.call_args.args[0]
        self.assertEqual(request["service"], "Mast.Name.Lookup")
        self.assertEqual(request["params"]["input"], "WASP-39 b")

    def test_resolve_target_coordinates_raises_when_unresolved(self):
        with patch.object(self.mast, "_mast_query", return_value={"resolvedCoordinate": []}):
            with self.assertRaises(ValueError):
                self.mast.resolve_target_coordinates("Nonexistent Object")

    def test_search_with_target_name_filter_adds_target_name_filter(self):
        with (
            patch.object(self.mast, "resolve_target_coordinates", return_value=(1.0, 2.0)),
            patch.object(
                self.mast,
                "_mast_query",
                return_value={"data": []},
            ) as query,
        ):
            self.mast.search_jwst_observations(
                "WASP-39 b",
                target_name_filter=True,
            )

        filters = query.call_args.args[0]["params"]["filters"]
        self.assertIn(
            {"paramName": "target_name", "values": [{"freeText": "WASP-39 b"}]},
            filters,
        )

    def test_search_without_target_name_filter_omits_target_name(self):
        with (
            patch.object(self.mast, "resolve_target_coordinates", return_value=(1.0, 2.0)),
            patch.object(
                self.mast,
                "_mast_query",
                return_value={"data": []},
            ) as query,
        ):
            self.mast.search_jwst_observations("WASP-39 b")

        filters = query.call_args.args[0]["params"]["filters"]
        param_names = {f["paramName"] for f in filters}
        self.assertNotIn("target_name", param_names)

    def test_download_planet_jwst_products_writes_manifest(self):
        observations = [
            {
                "obsid": "111",
                "obs_id": "jw01234001",
                "instrument_name": "NIRSpec",
                "target_name": "WASP-39",
            }
        ]
        products = [
            {
                "productType": "SCIENCE",
                "productSubGroupDescription": "X1DINTS",
                "productFilename": "jw01234_x1dints.fits",
                "dataURI": "mast:JWST/product/jw01234_x1dints.fits",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(self.mast, "search_jwst_observations", return_value=observations),
                patch.object(self.mast, "get_observation_products", return_value=products),
                patch.object(self.mast, "download_mast_product") as fake_download,
            ):
                target_path = pathlib.Path(tmpdir) / "WASP-39_b" / "jw01234001" / "jw01234_x1dints.fits"
                fake_download.return_value = target_path

                manifest = self.mast.download_planet_jwst_products(
                    "WASP-39 b",
                    tmpdir,
                    product_subgroups=["X1DINTS"],
                )

            target_dir = pathlib.Path(tmpdir) / "WASP-39_b"
            self.assertTrue((target_dir / "manifest.json").is_file())

            with (target_dir / "manifest.json").open() as handle:
                saved = json.load(handle)

            self.assertEqual(saved["planet_name"], "WASP-39 b")
            self.assertEqual(len(saved["observations"]), 1)
            self.assertEqual(len(saved["downloaded"]), 1)
            self.assertEqual(saved["downloaded"][0]["path"], str(target_path))
            self.assertEqual(manifest["downloaded"][0]["product"]["productSubGroupDescription"], "X1DINTS")

            fake_download.assert_called_once()
            called_args, called_kwargs = fake_download.call_args
            self.assertEqual(called_args[0], "mast:JWST/product/jw01234_x1dints.fits")
            self.assertEqual(
                os.path.normpath(str(called_args[1])),
                os.path.normpath(str(target_dir / "jw01234001")),
            )

    def test_format_helpers_handle_empty_inputs(self):
        self.assertIn("No JWST observations", self.mast._format_observations_summary([]))
        self.assertIn("No matching products", self.mast._format_products_summary([]))

    # -------- new: filter-only (no-position) search --------

    def test_search_all_jwst_observations_uses_filtered_service(self):
        payload = {
            "data": [
                {"obsid": "1", "instrument_name": "NIRSPEC/SLIT", "proposal_id": "1366"},
                {"obsid": "2", "instrument_name": "NIRCAM/GRISM", "proposal_id": "2734"},
            ]
        }
        with patch.object(self.mast, "_mast_query", return_value=payload) as query:
            rows, filters = self.mast.search_all_jwst_observations(
                instruments=["NIRSpec", "NIRCam"],
                dataproduct_types=["spectrum", "timeseries"],
                calib_levels=[3],
            )

        self.assertEqual(len(rows), 2)
        request = query.call_args.args[0]
        self.assertEqual(request["service"], "Mast.Caom.Filtered")
        self.assertNotIn("position", request["params"])
        self.assertEqual(
            request["params"]["filters"][0],
            {"paramName": "obs_collection", "values": ["JWST"]},
        )
        # filters_used echoed back to caller
        self.assertEqual(filters, request["params"]["filters"])

    def test_format_observations_summary_echoes_filters(self):
        filters = [
            {"paramName": "obs_collection", "values": ["JWST"]},
            {"paramName": "instrument_name", "values": ["NIRSPEC/SLIT"]},
            {"paramName": "calib_level", "values": [3]},
        ]
        text = self.mast._format_observations_summary(
            [{"obsid": "1", "instrument_name": "NIRSPEC/SLIT"}],
            filters=filters,
            query_extra={"mode": "demographics"},
        )
        self.assertIn("Filters used:", text)
        self.assertIn("obs_collection", text)
        self.assertIn("NIRSPEC/SLIT", text)
        self.assertIn("calib_level: [3]", text)
        self.assertIn("mode: demographics", text)

    def _make_tool(self, Tool, **kwargs):
        """
        BaseTool fallback in mast.py is a bare class (orchestral not installed
        during these tests). Instantiate then set attributes.
        """
        tool = Tool()
        for key, value in kwargs.items():
            setattr(tool, key, value)
        return tool

    def test_search_basetool_demographics_mode(self):
        Tool = self.mast.SearchMastJwstObservations
        with patch.object(
            self.mast,
            "search_all_jwst_observations",
            return_value=(
                [{"obsid": "1", "instrument_name": "NIRSPEC/SLIT"}],
                [{"paramName": "obs_collection", "values": ["JWST"]}],
            ),
        ) as call:
            tool = self._make_tool(
                Tool,
                planet_name=None,
                ra=None,
                dec=None,
                instruments=["NIRSpec"],
                dataproduct_types=["spectrum"],
                calib_levels=[3],
                proposal_id=None,
                target_name=None,
            )
            output = tool._run()

        call.assert_called_once()
        self.assertIn("demographics", output)
        self.assertIn("Filters used:", output)

    def test_search_basetool_per_planet_mode_still_works(self):
        Tool = self.mast.SearchMastJwstObservations
        with patch.object(
            self.mast,
            "search_jwst_observations",
            return_value=[{"obsid": "1", "instrument_name": "NIRSPEC/SLIT"}],
        ) as call:
            tool = self._make_tool(
                Tool,
                planet_name="WASP-39 b",
                ra=None,
                dec=None,
                radius_deg=0.02,
                instruments=["NIRSpec"],
                dataproduct_types=None,
                calib_levels=None,
                proposal_id=None,
                target_name=None,
            )
            output = tool._run()

        call.assert_called_once()
        self.assertIn("per-planet cone search", output)
        self.assertIn("planet_name: WASP-39 b", output)

    def test_search_basetool_raises_when_only_ra_supplied(self):
        Tool = self.mast.SearchMastJwstObservations
        tool = self._make_tool(
            Tool,
            planet_name=None,
            ra=10.0,
            dec=None,
            instruments=None,
            dataproduct_types=None,
            calib_levels=None,
            proposal_id=None,
            target_name=None,
        )
        with self.assertRaises(ValueError):
            tool._run()

    # -------- new: batch obsid download --------

    def test_download_observations_products_writes_manifest(self):
        products = [
            {
                "productType": "SCIENCE",
                "productSubGroupDescription": "X1DINTS",
                "productFilename": "jw01_x1dints.fits",
                "dataURI": "mast:JWST/product/jw01_x1dints.fits",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(self.mast, "get_observation_products", return_value=products),
                patch.object(self.mast, "download_mast_product") as fake_download,
            ):
                fake_download.return_value = pathlib.Path(tmpdir) / "label" / "111" / "jw01_x1dints.fits"
                manifest = self.mast.download_observations_products(
                    ["111", "222"],
                    tmpdir,
                    product_subgroups=["X1DINTS"],
                    label="hot_jupiters",
                )

            target_dir = pathlib.Path(tmpdir) / "hot_jupiters"
            self.assertTrue((target_dir / "manifest.json").is_file())
            with (target_dir / "manifest.json").open() as handle:
                saved = json.load(handle)

            self.assertEqual(saved["label"], "hot_jupiters")
            self.assertEqual(saved["obsids"], ["111", "222"])
            # one download per obsid (mocked products list is the same for each)
            self.assertEqual(len(saved["downloaded"]), 2)
            self.assertEqual(manifest["downloaded"][0]["obsid"], "111")

    def test_download_basetool_batch_obsids_mode(self):
        Tool = self.mast.DownloadMastJwstProducts
        with patch.object(
            self.mast,
            "download_observations_products",
            return_value={"label": "hot", "obsids": ["1"], "downloaded": []},
        ) as call:
            tool = self._make_tool(
                Tool,
                planet_name=None,
                obsids=["1", "2"],
                label="hot",
                output_dir="mast",
                ra=None,
                dec=None,
                radius_deg=0.02,
                instruments=None,
                dataproduct_types=None,
                product_subgroups=None,
                raw_only=False,
                max_observations=None,
                max_products=None,
                base_directory="/tmp",
            )
            output = tool._run()

        call.assert_called_once()
        self.assertIn("batch 'hot'", output)

    def test_download_basetool_raises_when_no_planet_and_no_obsids(self):
        Tool = self.mast.DownloadMastJwstProducts
        tool = self._make_tool(
            Tool,
            planet_name=None,
            obsids=None,
            label="aggregate",
            output_dir="mast",
            ra=None,
            dec=None,
            radius_deg=0.02,
            instruments=None,
            dataproduct_types=None,
            product_subgroups=None,
            raw_only=False,
            max_observations=None,
            max_products=None,
            base_directory="/tmp",
        )
        with self.assertRaises(ValueError):
            tool._run()


# ---------------- demographics integration test ----------------


def _haversine_deg(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    a = (
        math.sin((d2 - d1) / 2) ** 2
        + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2) ** 2
    )
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(a))))


def crossmatch_jwst_to_planets(
    observations,
    planets,
    *,
    radius_deg=0.02,
):
    """Cross-match MAST JWST observations to planet rows by RA/Dec cone."""
    rows = []
    for obs in observations:
        try:
            ora = float(obs.get("s_ra"))
            odec = float(obs.get("s_dec"))
        except (TypeError, ValueError):
            continue
        for planet in planets:
            try:
                pra = float(planet.get("ra"))
                pdec = float(planet.get("dec"))
            except (TypeError, ValueError):
                continue
            if _haversine_deg(ora, odec, pra, pdec) <= radius_deg:
                row = {**planet, **{
                    "obsid": obs.get("obsid"),
                    "obs_id": obs.get("obs_id"),
                    "instrument_name": obs.get("instrument_name"),
                    "dataproduct_type": obs.get("dataproduct_type"),
                    "calib_level": obs.get("calib_level"),
                    "proposal_id": obs.get("proposal_id"),
                    "proposal_pi": obs.get("proposal_pi"),
                    "target_name": obs.get("target_name"),
                    "t_min": obs.get("t_min"),
                    "t_max": obs.get("t_max"),
                    "filters": obs.get("filters"),
                }}
                rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        with open(path, "w") as fh:
            fh.write("")
        return
    # union of keys so partial-coverage planets still serialize
    keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class DemographicsTests(unittest.TestCase):
    def setUp(self):
        self.mast = load_mast_module()

    def test_demographics_pipeline_compiles_two_csvs(self):
        # Fake archive populations (hot/warm Jupiters + sub-Neptunes)
        hot_warm_jupiters = [
            {
                "pl_name": "WASP-39 b",
                "hostname": "WASP-39",
                "ra": 217.3267,
                "dec": -3.4444,
                "pl_radj": 1.27,
                "pl_bmassj": 0.28,
                "pl_eqt": 1120.0,
                "pl_orbper": 4.055,
                "st_rad": 0.93,
                "st_teff": 5485,
                "discoverymethod": "Transit",
                "disc_year": 2011,
            },
            {
                "pl_name": "HD 209458 b",
                "hostname": "HD 209458",
                "ra": 330.795,
                "dec": 18.884,
                "pl_radj": 1.36,
                "pl_bmassj": 0.69,
                "pl_eqt": 1450.0,
                "pl_orbper": 3.524,
                "st_rad": 1.20,
                "st_teff": 6065,
                "discoverymethod": "Transit",
                "disc_year": 1999,
            },
        ]
        sub_neptunes = [
            {
                "pl_name": "GJ 1214 b",
                "hostname": "GJ 1214",
                "ra": 258.831,
                "dec": 4.964,
                "pl_rade": 2.74,
                "pl_bmasse": 6.55,
                "pl_eqt": 596.0,
                "pl_orbper": 1.580,
                "st_rad": 0.211,
                "st_teff": 3026,
                "discoverymethod": "Transit",
                "disc_year": 2009,
            }
        ]

        # Fake MAST filtered response, matching one obs per planet/instrument
        observations = [
            {
                "obsid": "10001",
                "obs_id": "jw01366001",
                "target_name": "WASP-39",
                "s_ra": 217.3266,
                "s_dec": -3.4443,
                "instrument_name": "NIRSPEC/SLIT",
                "dataproduct_type": "spectrum",
                "calib_level": 3,
                "proposal_id": "1366",
                "proposal_pi": "Natalie Batalha",
                "filters": "PRISM/CLEAR",
                "t_min": 59800.1,
                "t_max": 59800.4,
            },
            {
                "obsid": "10002",
                "obs_id": "jw02734005",
                "target_name": "WASP-39",
                "s_ra": 217.3266,
                "s_dec": -3.4443,
                "instrument_name": "NIRCAM/GRISM",
                "dataproduct_type": "timeseries",
                "calib_level": 3,
                "proposal_id": "2734",
                "proposal_pi": "Jacob Bean",
                "filters": "F322W2",
                "t_min": 59900.0,
                "t_max": 59900.4,
            },
            {
                "obsid": "10003",
                "obs_id": "jw01633001",
                "target_name": "HD 209458",
                "s_ra": 330.7949,
                "s_dec": 18.8841,
                "instrument_name": "NIRISS/SOSS",
                "dataproduct_type": "timeseries",
                "calib_level": 3,
                "proposal_id": "1633",
                "proposal_pi": "Knicole Colon",
                "filters": "CLEAR",
                "t_min": 60100.0,
                "t_max": 60100.3,
            },
            {
                "obsid": "10004",
                "obs_id": "jw01803002",
                "target_name": "GJ 1214",
                "s_ra": 258.8310,
                "s_dec": 4.9639,
                "instrument_name": "MIRI/LRS",
                "dataproduct_type": "timeseries",
                "calib_level": 3,
                "proposal_id": "1803",
                "proposal_pi": "Eliza Kempton",
                "filters": "P750L",
                "t_min": 59950.0,
                "t_max": 59950.6,
            },
            # non-matching observation, far from any planet
            {
                "obsid": "99999",
                "obs_id": "jw09999",
                "target_name": "M31",
                "s_ra": 10.0,
                "s_dec": 41.0,
                "instrument_name": "NIRCAM/IMAGE",
                "dataproduct_type": "image",
                "calib_level": 3,
                "proposal_id": "9999",
                "proposal_pi": "Someone Else",
                "filters": "F200W",
                "t_min": 60200.0,
                "t_max": 60200.1,
            },
        ]

        # Drive the demographics MAST query via the tool surface (verifies
        # the no-position path is what an agent would actually hit).
        with patch.object(
            self.mast,
            "_mast_query",
            return_value={"data": observations},
        ) as query:
            obs_rows, filters_used = self.mast.search_all_jwst_observations(
                instruments=["NIRSpec", "NIRCam", "MIRI", "NIRISS"],
                dataproduct_types=["spectrum", "timeseries"],
                calib_levels=[3],
            )

        # Filter-only service, no position
        request = query.call_args.args[0]
        self.assertEqual(request["service"], "Mast.Caom.Filtered")
        self.assertNotIn("position", request["params"])
        # filters_used surfaces what the agent should also see in the summary
        param_names = {f["paramName"] for f in filters_used}
        self.assertIn("instrument_name", param_names)
        self.assertIn("dataproduct_type", param_names)
        self.assertIn("calib_level", param_names)

        jupiter_rows = crossmatch_jwst_to_planets(obs_rows, hot_warm_jupiters)
        sub_neptune_rows = crossmatch_jwst_to_planets(obs_rows, sub_neptunes)

        # Hot/warm Jupiter rows: WASP-39 b NIRSpec, WASP-39 b NIRCam, HD 209458 b NIRISS
        self.assertEqual(len(jupiter_rows), 3)
        wasp_instruments = sorted(
            r["instrument_name"] for r in jupiter_rows if r["pl_name"] == "WASP-39 b"
        )
        self.assertEqual(wasp_instruments, ["NIRCAM/GRISM", "NIRSPEC/SLIT"])
        # Proposal info preserved
        wasp_nirspec = next(
            r for r in jupiter_rows
            if r["pl_name"] == "WASP-39 b" and r["instrument_name"] == "NIRSPEC/SLIT"
        )
        self.assertEqual(wasp_nirspec["proposal_id"], "1366")
        self.assertEqual(wasp_nirspec["proposal_pi"], "Natalie Batalha")
        # Archive params preserved alongside
        self.assertAlmostEqual(wasp_nirspec["pl_radj"], 1.27)
        self.assertAlmostEqual(wasp_nirspec["pl_eqt"], 1120.0)
        # M31 row should not match any planet
        self.assertFalse(any(r["obsid"] == "99999" for r in jupiter_rows))
        self.assertFalse(any(r["obsid"] == "99999" for r in sub_neptune_rows))

        # Sub-Neptune rows: GJ 1214 b MIRI/LRS
        self.assertEqual(len(sub_neptune_rows), 1)
        self.assertEqual(sub_neptune_rows[0]["pl_name"], "GJ 1214 b")
        self.assertEqual(sub_neptune_rows[0]["instrument_name"], "MIRI/LRS")
        self.assertEqual(sub_neptune_rows[0]["proposal_pi"], "Eliza Kempton")

        # Persist as CSV and verify round-trip
        with tempfile.TemporaryDirectory() as tmpdir:
            jp = pathlib.Path(tmpdir) / "warm_hot_jupiters_jwst.csv"
            sn = pathlib.Path(tmpdir) / "sub_neptunes_jwst.csv"
            write_csv(jp, jupiter_rows)
            write_csv(sn, sub_neptune_rows)

            with jp.open() as fh:
                read_back = list(csv.DictReader(fh))
            self.assertEqual(len(read_back), 3)
            self.assertIn("pl_eqt", read_back[0])
            self.assertIn("proposal_pi", read_back[0])
            self.assertIn("instrument_name", read_back[0])


# ---------------- crossmatch + aggregate tool tests ----------------


class CrossmatchAndAggregateTests(unittest.TestCase):
    def setUp(self):
        self.mast = load_mast_module()

    def _make_tool(self, Tool, **kwargs):
        tool = Tool()
        for key, value in kwargs.items():
            setattr(tool, key, value)
        return tool

    PLANETS = [
        {
            "pl_name": "WASP-39 b", "hostname": "WASP-39",
            "ra": 217.3267, "dec": -3.4444,
            "pl_radj": 1.27, "pl_bmassj": 0.28, "pl_eqt": 1120.0,
        },
        {
            "pl_name": "HD 209458 b", "hostname": "HD 209458",
            "ra": 330.795, "dec": 18.884,
            "pl_radj": 1.36, "pl_bmassj": 0.69, "pl_eqt": 1450.0,
        },
    ]

    OBSERVATIONS = [
        {  # WASP-39 NIRSpec
            "obsid": "10001", "obs_id": "jw01366001",
            "s_ra": 217.3266, "s_dec": -3.4443,
            "instrument_name": "NIRSPEC/SLIT",
            "dataproduct_type": "spectrum", "calib_level": 3,
            "proposal_id": "1366", "proposal_pi": "Natalie Batalha",
            "filters": "PRISM/CLEAR", "target_name": "WASP-39",
        },
        {  # WASP-39 NIRCam (multi-instrument!)
            "obsid": "10002", "obs_id": "jw02734005",
            "s_ra": 217.3266, "s_dec": -3.4443,
            "instrument_name": "NIRCAM/GRISM",
            "dataproduct_type": "timeseries", "calib_level": 3,
            "proposal_id": "2734", "proposal_pi": "Jacob Bean",
            "filters": "F322W2", "target_name": "WASP-39",
        },
        {  # HD 209458 NIRISS
            "obsid": "10003", "obs_id": "jw01633001",
            "s_ra": 330.7949, "s_dec": 18.8841,
            "instrument_name": "NIRISS/SOSS",
            "dataproduct_type": "timeseries", "calib_level": 3,
            "proposal_id": "1633", "proposal_pi": "Knicole Colon",
            "filters": "CLEAR", "target_name": "HD 209458",
        },
        {  # M31 — should not match any planet
            "obsid": "99999", "obs_id": "jw09999",
            "s_ra": 10.0, "s_dec": 41.0,
            "instrument_name": "NIRCAM/IMAGE",
            "dataproduct_type": "image", "calib_level": 3,
            "proposal_id": "9999", "proposal_pi": "Someone Else",
            "filters": "F200W", "target_name": "M31",
        },
    ]

    # ---- pure helpers ----

    def test_crossmatch_observations_to_planets(self):
        rows = self.mast.crossmatch_observations_to_planets(
            self.OBSERVATIONS, self.PLANETS, radius_deg=0.02,
        )
        # 2 WASP-39 + 1 HD 209458 = 3
        self.assertEqual(len(rows), 3)
        pl_names = sorted(r["pl_name"] for r in rows)
        self.assertEqual(pl_names, ["HD 209458 b", "WASP-39 b", "WASP-39 b"])
        # Planet attributes preserved
        wasp = next(r for r in rows if r["pl_name"] == "WASP-39 b" and r["instrument_name"] == "NIRSPEC/SLIT")
        self.assertAlmostEqual(wasp["pl_eqt"], 1120.0)
        self.assertEqual(wasp["proposal_pi"], "Natalie Batalha")
        # M31 obs not paired
        self.assertFalse(any(r["obsid"] == "99999" for r in rows))

    def test_aggregate_observations_per_instrument(self):
        groups = self.mast.aggregate_observations(
            self.OBSERVATIONS, group_by=["instrument_name"],
        )
        counts = {g["instrument_name"]: g["count"] for g in groups}
        self.assertEqual(counts["NIRSPEC/SLIT"], 1)
        self.assertEqual(counts["NIRCAM/GRISM"], 1)
        self.assertEqual(counts["NIRISS/SOSS"], 1)
        self.assertEqual(counts["NIRCAM/IMAGE"], 1)

    def test_aggregate_observations_per_instrument_per_filter(self):
        groups = self.mast.aggregate_observations(
            self.OBSERVATIONS, group_by=["instrument_name", "filters"],
        )
        keys = {(g["instrument_name"], g["filters"]): g["count"] for g in groups}
        self.assertEqual(keys[("NIRSPEC/SLIT", "PRISM/CLEAR")], 1)
        self.assertEqual(keys[("NIRCAM/GRISM", "F322W2")], 1)

    def test_aggregate_multi_instrument_per_planet(self):
        crossmatched = self.mast.crossmatch_observations_to_planets(
            self.OBSERVATIONS, self.PLANETS, radius_deg=0.02,
        )
        groups = self.mast.aggregate_observations(
            crossmatched,
            group_by=["pl_name"],
            distinct_fields=["instrument_name", "proposal_id"],
        )
        per_planet = {g["pl_name"]: g for g in groups}
        # WASP-39 b: NIRSpec + NIRCam → 2 distinct instruments
        self.assertEqual(per_planet["WASP-39 b"]["instrument_name_distinct"], 2)
        self.assertEqual(
            sorted(per_planet["WASP-39 b"]["instrument_name_values"]),
            ["NIRCAM/GRISM", "NIRSPEC/SLIT"],
        )
        # HD 209458 b: only NIRISS
        self.assertEqual(per_planet["HD 209458 b"]["instrument_name_distinct"], 1)
        # Multi-instrument planets = those with distinct > 1
        multi = [g["pl_name"] for g in groups if g["instrument_name_distinct"] > 1]
        self.assertEqual(multi, ["WASP-39 b"])

    def test_aggregate_observations_rejects_empty_group_by(self):
        with self.assertRaises(ValueError):
            self.mast.aggregate_observations(self.OBSERVATIONS, group_by=[])

    # ---- BaseTool wiring ----

    def test_crossmatch_basetool_runs_archive_and_mast_then_writes_csv(self):
        Tool = self.mast.CrossmatchJwstToPlanets
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(self.mast, "archive_tap_query", return_value=self.PLANETS) as archive_call,
                patch.object(
                    self.mast,
                    "search_all_jwst_observations",
                    return_value=(self.OBSERVATIONS, [{"paramName": "obs_collection", "values": ["JWST"]}]),
                ) as mast_call,
            ):
                tool = self._make_tool(
                    Tool,
                    archive_conditions=["pl_bmassj > 0.3", "pl_eqt > 500"],
                    archive_columns=None,
                    archive_table="pscomppars",
                    archive_limit=None,
                    instruments=["NIRSpec", "NIRCam", "MIRI", "NIRISS"],
                    dataproduct_types=["spectrum", "timeseries"],
                    calib_levels=[3],
                    proposal_id=None,
                    radius_deg=0.02,
                    output_csv="demographics/test_xmatch.csv",
                    base_directory=tmpdir,
                )
                output = tool._run()

            archive_call.assert_called_once()
            mast_call.assert_called_once()
            self.assertIn("Cross-matched 3", output)
            self.assertIn("Population planets considered: 2", output)
            self.assertIn("JWST observations considered: 4", output)

            csv_path = pathlib.Path(tmpdir) / "demographics" / "test_xmatch.csv"
            self.assertTrue(csv_path.is_file())
            with csv_path.open() as fh:
                read_back = list(csv.DictReader(fh))
            self.assertEqual(len(read_back), 3)
            # Planet + obs columns both present
            self.assertIn("pl_name", read_back[0])
            self.assertIn("instrument_name", read_back[0])
            self.assertIn("proposal_pi", read_back[0])

    def test_aggregate_basetool_rows_path_mode(self):
        Tool = self.mast.AggregateJwstObservations
        # Build a CSV the agent could have written from CrossmatchJwstToPlanets
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = self.mast.crossmatch_observations_to_planets(
                self.OBSERVATIONS, self.PLANETS,
            )
            csv_path = pathlib.Path(tmpdir) / "rows.csv"
            self.mast._write_rows_csv(csv_path, rows)

            tool = self._make_tool(
                Tool,
                group_by=["pl_name"],
                distinct_fields=["instrument_name"],
                rows_path="rows.csv",
                instruments=None,
                dataproduct_types=None,
                calib_levels=None,
                proposal_id=None,
                target_name=None,
                output_csv="grouped.csv",
                base_directory=tmpdir,
            )
            output = tool._run()

            self.assertIn("Aggregated 3 row(s)", output)
            self.assertIn("instrument_name_distinct", output)
            grouped_csv = pathlib.Path(tmpdir) / "grouped.csv"
            self.assertTrue(grouped_csv.is_file())
            with grouped_csv.open() as fh:
                groups = list(csv.DictReader(fh))
            per_planet = {g["pl_name"]: g for g in groups}
            self.assertEqual(per_planet["WASP-39 b"]["instrument_name_distinct"], "2")
            self.assertIn("NIRCAM/GRISM", per_planet["WASP-39 b"]["instrument_name_values"])

    def test_aggregate_basetool_demographics_mode(self):
        Tool = self.mast.AggregateJwstObservations
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                self.mast,
                "search_all_jwst_observations",
                return_value=(self.OBSERVATIONS, []),
            ) as call:
                tool = self._make_tool(
                    Tool,
                    group_by=["instrument_name", "filters"],
                    distinct_fields=None,
                    rows_path=None,
                    instruments=["NIRSpec", "NIRCam", "MIRI", "NIRISS"],
                    dataproduct_types=["spectrum", "timeseries"],
                    calib_levels=[3],
                    proposal_id=None,
                    target_name=None,
                    output_csv=None,
                    base_directory=tmpdir,
                )
                output = tool._run()

            call.assert_called_once()
            self.assertIn("Aggregated 4 row(s)", output)
            self.assertIn("NIRSPEC/SLIT", output)
            self.assertIn("filters=", output)

    # ---- DownloadDemographicJwstProducts ----

    def test_download_demographic_products_groups_by_planet(self):
        # Mocked crossmatch rows: WASP-39 b has 2 obsids, HD 209458 b has 1
        rows = [
            {"pl_name": "WASP-39 b", "obsid": "10001"},
            {"pl_name": "WASP-39 b", "obsid": "10002"},
            {"pl_name": "WASP-39 b", "obsid": "10001"},   # duplicate — should dedupe
            {"pl_name": "HD 209458 b", "obsid": "10003"},
            {"pl_name": "",          "obsid": "99"},      # missing planet — skipped
            {"pl_name": "X",         "obsid": ""},        # missing obsid — skipped
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                self.mast,
                "download_observations_products",
                side_effect=lambda obsids, root, label, **kw: {
                    "label": label,
                    "obsids": list(obsids),
                    "downloaded": [{"obsid": o, "product": {}, "path": f"/x/{o}"} for o in obsids],
                },
            ) as call:
                manifest = self.mast.download_demographic_products(
                    rows,
                    tmpdir,
                    label="warm_hot_jupiters",
                    product_subgroups=["X1DINTS"],
                )

            # 2 planets processed, 3 unique (planet, obsid) files
            self.assertEqual(manifest["planet_count"], 2)
            self.assertEqual(manifest["total_files_downloaded"], 3)
            self.assertEqual(call.call_count, 2)

            # Per-planet calls used pl_name as label and the deduped obsid list
            wasp_call = next(c for c in call.call_args_list if c.kwargs["label"] == "WASP-39 b")
            self.assertEqual(wasp_call.args[0], ["10001", "10002"])
            self.assertEqual(wasp_call.kwargs["product_subgroups"], ["X1DINTS"])

            # Top-level demographic manifest written under {output_dir}/{label}/
            root = pathlib.Path(tmpdir) / "warm_hot_jupiters"
            self.assertTrue((root / "demographic_manifest.json").is_file())
            with (root / "demographic_manifest.json").open() as fh:
                saved = json.load(fh)
            self.assertEqual(saved["label"], "warm_hot_jupiters")
            pl_names = sorted(e["pl_name"] for e in saved["per_planet"])
            self.assertEqual(pl_names, ["HD 209458 b", "WASP-39 b"])

    def test_download_demographic_products_respects_caps(self):
        rows = [
            {"pl_name": "A", "obsid": "1"},
            {"pl_name": "A", "obsid": "2"},
            {"pl_name": "A", "obsid": "3"},
            {"pl_name": "B", "obsid": "4"},
            {"pl_name": "C", "obsid": "5"},
        ]
        recorded: list[tuple[str, list[str]]] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            def fake_download(obsids, root, label, **kw):
                recorded.append((label, list(obsids)))
                return {"label": label, "obsids": list(obsids), "downloaded": []}

            with patch.object(self.mast, "download_observations_products", side_effect=fake_download):
                self.mast.download_demographic_products(
                    rows,
                    tmpdir,
                    label="cap_test",
                    max_planets=2,
                    max_obs_per_planet=2,
                )

        # max_planets=2 keeps A,B (insertion order)
        self.assertEqual([r[0] for r in recorded], ["A", "B"])
        # A had 3 obsids, capped at 2
        a_call = next(r for r in recorded if r[0] == "A")
        self.assertEqual(a_call[1], ["1", "2"])

    def test_download_demographic_basetool_reads_csv_and_runs(self):
        Tool = self.mast.DownloadDemographicJwstProducts
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = pathlib.Path(tmpdir) / "xmatch.csv"
            self.mast._write_rows_csv(
                csv_path,
                [
                    {"pl_name": "WASP-39 b", "obsid": "10001"},
                    {"pl_name": "WASP-39 b", "obsid": "10002"},
                    {"pl_name": "HD 209458 b", "obsid": "10003"},
                ],
            )

            with patch.object(
                self.mast,
                "download_observations_products",
                side_effect=lambda obsids, root, label, **kw: {
                    "label": label,
                    "obsids": list(obsids),
                    "downloaded": [{"obsid": o, "product": {}, "path": f"/x/{o}"} for o in obsids],
                },
            ):
                tool = self._make_tool(
                    Tool,
                    rows_path="xmatch.csv",
                    output_dir="mast/raw",
                    label="warm_hot_jupiters",
                    product_subgroups=["X1DINTS"],
                    raw_only=False,
                    max_planets=None,
                    max_obs_per_planet=None,
                    max_products_per_obs=None,
                    base_directory=tmpdir,
                )
                output = tool._run()

            self.assertIn("warm_hot_jupiters", output)
            self.assertIn("Planets processed: 2", output)
            self.assertIn("Total files downloaded: 3", output)
            self.assertIn("WASP-39 b", output)

            root = pathlib.Path(tmpdir) / "mast" / "raw" / "warm_hot_jupiters"
            self.assertTrue((root / "demographic_manifest.json").is_file())

    def test_aggregate_basetool_rejects_empty_group_by(self):
        Tool = self.mast.AggregateJwstObservations
        tool = self._make_tool(
            Tool,
            group_by=[],
            distinct_fields=None,
            rows_path=None,
            instruments=None,
            dataproduct_types=None,
            calib_levels=None,
            proposal_id=None,
            target_name=None,
            output_csv=None,
            base_directory="/tmp",
        )
        with self.assertRaises(ValueError):
            tool._run()

    # ---- auto-CSV behaviour (regression: agent must never have to fabricate
    # downstream CSVs from the truncated preview text) ----

    def test_slugify_for_filename_handles_special_chars_and_caps(self):
        slug = self.mast._slugify_for_filename("WASP-39 b / 'NIRSpec' & NIRCam!")
        self.assertRegex(slug, r"^[a-z0-9_]+$")
        self.assertNotIn("__", slug.strip("_"))
        self.assertEqual(
            self.mast._slugify_for_filename(""), "query",
            "empty input must fall back to 'query'",
        )
        long_slug = self.mast._slugify_for_filename("x" * 200, max_len=40)
        self.assertLessEqual(len(long_slug), 40)

    def test_autoname_csv_path_structure(self):
        path = self.mast._autoname_csv_path(
            "/tmp/base",
            kind="crossmatch",
            hint_parts=[["pl_radj < 0.4"], ["NIRSpec"], [3]],
        )
        self.assertTrue(str(path).endswith(".csv"))
        self.assertTrue(path.name.startswith("crossmatch_"))
        self.assertEqual(path.parent.as_posix(), "/tmp/base/mast/demographics")

    def test_autoname_csv_path_distinguishes_different_hints(self):
        a = self.mast._autoname_csv_path(
            "/tmp/base", kind="crossmatch",
            hint_parts=[["pl_radj < 0.4"], ["NIRSpec"]],
        )
        b = self.mast._autoname_csv_path(
            "/tmp/base", kind="crossmatch",
            hint_parts=[["pl_bmassj > 0.3"], ["NIRSpec"]],
        )
        # Different inputs → different 6-char hash suffix → different file names.
        # (Timestamp may coincide if invoked in the same UTC second; the hash
        # disambiguates regardless.)
        a_hash = a.stem.rsplit("_", 1)[-1]
        b_hash = b.stem.rsplit("_", 1)[-1]
        self.assertNotEqual(a_hash, b_hash)

    def test_format_crossmatch_summary_leads_with_csv_directive(self):
        rows = [{"pl_name": "X b", "instrument_name": "NIRISS/SOSS",
                 "filters": "CLEAR;GR700XD", "obsid": 1,
                 "proposal_id": 111, "proposal_pi": "PI1"}]
        text = self.mast._format_crossmatch_summary(
            rows, planet_count=5, obs_count=9, radius_deg=0.02,
            csv_path=pathlib.Path("/tmp/x.csv"),
        )
        self.assertIn("FULL RESULTS", text)
        self.assertIn("/tmp/x.csv", text)
        self.assertIn("do NOT reconstruct", text)
        self.assertIn("Preview", text)

    def test_format_aggregate_summary_leads_with_csv_directive(self):
        groups = [{"instrument_name": "NIRISS/SOSS", "count": 4}]
        text = self.mast._format_aggregate_summary(
            groups, group_by=["instrument_name"], distinct_fields=[],
            total_rows=4, csv_path=pathlib.Path("/tmp/g.csv"),
        )
        self.assertIn("FULL RESULTS", text)
        self.assertIn("/tmp/g.csv", text)
        self.assertIn("do NOT reconstruct", text)

    def test_crossmatch_auto_writes_full_csv_when_output_csv_omitted(self):
        """
        Regression for the 214-row sub-Neptune bug: when an agent forgets to
        set output_csv, the tool must still persist EVERY matched row to disk
        (not just the ~20 shown in the preview), and the returned text must
        point the agent at the file.
        """
        # Synthesize 50 (planet, observation) pairs. Each planet is placed
        # 1 deg apart in RA so it is the ONLY planet within the cone of its
        # paired observation — guarantees a 1-to-1 match, not a 50*50 cross.
        planets = [
            {
                "pl_name": f"FAKE-{i:03d} b", "hostname": f"FAKE-{i:03d}",
                "ra": 100.0 + 1.0 * i, "dec": -10.0,
                "pl_radj": 0.2, "pl_bmassj": 0.05, "pl_eqt": 600.0,
            }
            for i in range(50)
        ]
        observations = [
            {
                "obsid": f"{20000 + i}", "obs_id": f"jw0{20000 + i}",
                "s_ra": p["ra"] + 0.0001, "s_dec": p["dec"] + 0.0001,
                "instrument_name": "NIRISS/SOSS",
                "dataproduct_type": "timeseries", "calib_level": 3,
                "proposal_id": "2589", "proposal_pi": "Lim, Olivia",
                "filters": "CLEAR;GR700XD", "target_name": p["hostname"],
            }
            for i, p in enumerate(planets)
        ]

        Tool = self.mast.CrossmatchJwstToPlanets
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(self.mast, "archive_tap_query", return_value=planets),
                patch.object(
                    self.mast, "search_all_jwst_observations",
                    return_value=(observations, []),
                ),
            ):
                tool = self._make_tool(
                    Tool,
                    archive_conditions=["pl_radj < 0.4"],
                    archive_columns=None,
                    archive_table="pscomppars",
                    archive_limit=None,
                    instruments=["NIRSpec", "NIRCam", "MIRI", "NIRISS"],
                    dataproduct_types=["spectrum", "timeseries"],
                    calib_levels=[3],
                    proposal_id=None,
                    radius_deg=0.02,
                    output_csv=None,                # <-- the case that broke
                    base_directory=tmpdir,
                )
                output = tool._run()

            self.assertIn("FULL RESULTS (50 rows)", output)
            self.assertIn("do NOT reconstruct", output)

            demographics_dir = pathlib.Path(tmpdir) / "mast" / "demographics"
            self.assertTrue(demographics_dir.is_dir())
            csvs = list(demographics_dir.glob("crossmatch_*.csv"))
            self.assertEqual(
                len(csvs), 1, f"expected one auto-named CSV, got {csvs}",
            )

            with csvs[0].open() as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(
                len(rows), 50,
                "auto CSV must contain ALL matched rows, not just the preview",
            )
            pl_names = {r["pl_name"] for r in rows}
            self.assertEqual(len(pl_names), 50)
            # Verify the auto-name path is the one announced in the summary
            self.assertIn(str(csvs[0]), output)

    def test_aggregate_auto_writes_csv_when_output_csv_omitted(self):
        Tool = self.mast.AggregateJwstObservations
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                self.mast, "search_all_jwst_observations",
                return_value=(self.OBSERVATIONS, []),
            ):
                tool = self._make_tool(
                    Tool,
                    group_by=["instrument_name"],
                    distinct_fields=None,
                    rows_path=None,
                    instruments=["NIRSpec", "NIRCam", "MIRI", "NIRISS"],
                    dataproduct_types=["spectrum", "timeseries"],
                    calib_levels=[3],
                    proposal_id=None,
                    target_name=None,
                    output_csv=None,
                    base_directory=tmpdir,
                )
                output = tool._run()

            self.assertIn("FULL RESULTS", output)
            demographics_dir = pathlib.Path(tmpdir) / "mast" / "demographics"
            csvs = list(demographics_dir.glob("aggregate_*.csv"))
            self.assertEqual(len(csvs), 1)
            with csvs[0].open() as fh:
                groups = list(csv.DictReader(fh))
            self.assertEqual(
                {g["instrument_name"] for g in groups},
                {"NIRSPEC/SLIT", "NIRCAM/GRISM", "NIRISS/SOSS", "NIRCAM/IMAGE"},
            )


# ---------------- optional live demographics compilation ----------------
# Enabled by ASTER_LIVE_MAST=1. Hits live NASA Exoplanet Archive + MAST.

@unittest.skipUnless(
    os.environ.get("ASTER_LIVE_MAST") == "1",
    "Set ASTER_LIVE_MAST=1 to run live demographics compilation.",
)
class LiveDemographicsCompile(unittest.TestCase):
    """
    Live end-to-end: compile JWST demographics CSVs for warm+hot Jupiters
    and sub-Neptunes using the actual MAST + Exoplanet Archive endpoints.

    Output: ``tests/scripts/_artifacts/{warm_hot_jupiters_jwst.csv,
    sub_neptunes_jwst.csv}``.
    """

    ARCHIVE_COLUMNS = [
        "pl_name", "hostname", "ra", "dec",
        "pl_radj", "pl_rade", "pl_bmassj", "pl_bmasse",
        "pl_orbper", "pl_orbsmax", "pl_eqt", "pl_dens", "pl_insol",
        "pl_orbeccen", "pl_orbincl", "pl_trandep", "pl_imppar",
        "st_rad", "st_teff", "st_mass", "st_logg", "st_met", "st_age",
        "sy_dist", "sy_vmag", "sy_kmag",
        "discoverymethod", "disc_year",
    ]

    def setUp(self):
        self.mast = load_mast_module()

    def _archive_query(self, conditions):
        import io
        import requests

        select_clause = ", ".join(self.ARCHIVE_COLUMNS)
        where_clause = " AND ".join(f"({c})" for c in conditions)
        adql = f"SELECT {select_clause} FROM pscomppars WHERE {where_clause}"
        response = requests.get(
            "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
            params={"query": adql, "format": "csv"},
            timeout=120,
        )
        response.raise_for_status()
        return list(csv.DictReader(io.StringIO(response.text)))

    def test_compile_warm_hot_jupiters_and_sub_neptunes(self):
        artifacts = pathlib.Path(__file__).parent / "_artifacts"
        artifacts.mkdir(exist_ok=True)

        hot_warm = self._archive_query([
            "pl_bmassj > 0.3",
            "pl_eqt > 500",
            "ra is not null",
            "dec is not null",
        ])
        sub_neptunes = self._archive_query([
            "pl_rade between 1.5 and 4",
            "pl_bmasse < 20",
            "ra is not null",
            "dec is not null",
        ])
        self.assertGreater(len(hot_warm), 0)
        self.assertGreater(len(sub_neptunes), 0)

        observations, filters_used = self.mast.search_all_jwst_observations(
            instruments=["NIRSpec", "NIRCam", "MIRI", "NIRISS"],
            dataproduct_types=["spectrum", "timeseries"],
            calib_levels=[3],
        )
        self.assertGreater(len(observations), 0)

        jupiter_rows = crossmatch_jwst_to_planets(observations, hot_warm)
        sub_neptune_rows = crossmatch_jwst_to_planets(observations, sub_neptunes)

        write_csv(artifacts / "warm_hot_jupiters_jwst.csv", jupiter_rows)
        write_csv(artifacts / "sub_neptunes_jwst.csv", sub_neptune_rows)
        with (artifacts / "filters_used.json").open("w") as fh:
            json.dump(filters_used, fh, indent=2)

        # Also persist instrument-grouped counts for quick inspection
        def by_instrument(rows):
            counts = {}
            for r in rows:
                inst = r.get("instrument_name", "?")
                counts[inst] = counts.get(inst, 0) + 1
            return counts

        with (artifacts / "by_instrument.json").open("w") as fh:
            json.dump(
                {
                    "warm_hot_jupiters": by_instrument(jupiter_rows),
                    "sub_neptunes": by_instrument(sub_neptune_rows),
                },
                fh,
                indent=2,
            )

        print(
            f"\nWarm+hot Jupiter JWST rows: {len(jupiter_rows)}  "
            f"Sub-Neptune JWST rows: {len(sub_neptune_rows)}"
        )
        print(f"Artifacts written to {artifacts}")


if __name__ == "__main__":
    unittest.main()
