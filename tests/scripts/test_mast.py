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


def _construct_tool(Tool, **kwargs):
    """Build a tool instance in either environment.

    With orchestral installed, tools are pydantic models: fields (including
    the base_directory StateField) must be passed to the constructor. Without
    orchestral, mast.py's fallback BaseTool is a bare class that takes no
    constructor args, so we instantiate empty and setattr.
    """
    try:
        return Tool(**kwargs)
    except TypeError:
        tool = Tool()
        for key, value in kwargs.items():
            setattr(tool, key, value)
        return tool


class FakeDownloadResponse:
    def __init__(self, chunks, status_code=200, headers=None):
        self._chunks = chunks
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        return iter(self._chunks)

    def close(self):
        return None


class FakeDownloadSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, stream=False, timeout=None, headers=None):
        self.calls.append(
            {
                "url": url,
                "params": params,
                "stream": stream,
                "timeout": timeout,
                "headers": headers,
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
        self.assertIn("NIRSPEC/BOTS", instrument_filter["values"])
        self.assertIn("NIRCAM/IMAGE", instrument_filter["values"])
        self.assertIn("NIRCAM/WFSS", instrument_filter["values"])
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
                "values": [
                    "NIRSPEC/BOTS", "NIRSPEC/SLIT", "NIRSPEC/IFU",
                    "NIRSPEC/MSA", "NIRSPEC/IMAGE",
                ],
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
                "values": [
                    "NIRSPEC/BOTS", "NIRSPEC/SLIT", "NIRSPEC/IFU",
                    "NIRSPEC/MSA", "NIRSPEC/IMAGE",
                ],
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
            {
                "paramName": "target_name",
                "values": [],
                "freeText": "%WASP-39 b%",
            },
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

    def test_search_all_dedupes_and_recovers_pagination_gaps(self):
        """MAST paged reads have no stable ordering: pages can overlap AND
        drop rows (observed live 2026-07-10: identical queries differed by
        41 / 5 observations). The reader must dedupe by obsid, notice the
        shortfall via the Mashup paging totals, and refetch single-page."""
        r = lambda i: {"obsid": str(i), "instrument_name": "NIRSPEC/SLIT"}
        responses = [
            # page 1 (full): A B C     — server says 5 rows total
            {"data": [r(1), r(2), r(3)], "paging": {"rowsFiltered": 5}},
            # page 2 (short): C D      — overlap C, row E lost at boundary
            {"data": [r(3), r(4)], "paging": {"rowsFiltered": 5}},
            # single-page retry: everything
            {"data": [r(1), r(2), r(3), r(4), r(5)],
             "paging": {"rowsFiltered": 5}},
        ]
        with patch.object(self.mast, "_mast_query", side_effect=responses) as q:
            rows, _filters = self.mast.search_all_jwst_observations(
                calib_levels=[3], pagesize=3,
            )
        self.assertEqual([row["obsid"] for row in rows], ["1", "2", "3", "4", "5"])
        # Retry request: page 1, pagesize sized for the whole result set.
        retry = q.call_args_list[-1].args[0]
        self.assertEqual(retry["page"], 1)
        self.assertGreaterEqual(retry["pagesize"], 5)

    def test_search_all_raises_when_still_incomplete_after_retry(self):
        r = lambda i: {"obsid": str(i)}
        responses = [
            {"data": [r(1), r(2)], "paging": {"rowsFiltered": 4}},
            {"data": [r(2)], "paging": {"rowsFiltered": 4}},
            {"data": [r(1), r(2), r(3)], "paging": {"rowsFiltered": 4}},  # retry still short
        ]
        with patch.object(self.mast, "_mast_query", side_effect=responses):
            with self.assertRaisesRegex(RuntimeError, "incomplete result set"):
                self.mast.search_all_jwst_observations(pagesize=2)

    def test_search_all_no_retry_when_totals_match(self):
        responses = [
            {"data": [{"obsid": "1"}, {"obsid": "2"}],
             "paging": {"rowsFiltered": 2}},
        ]
        with patch.object(self.mast, "_mast_query", side_effect=responses) as q:
            rows, _ = self.mast.search_all_jwst_observations(pagesize=50000)
        self.assertEqual(len(rows), 2)
        self.assertEqual(q.call_count, 1)

    def test_crossmatch_emits_each_planet_obsid_pair_once(self):
        planets = [{"pl_name": "K2-18 b", "ra": 172.56, "dec": 7.5878}]
        obs = {
            "obsid": "42", "obs_id": "jw02372-o001",
            "s_ra": 172.56, "s_dec": 7.5878,
        }
        rows = self.mast.crossmatch_observations_to_planets(
            [obs, dict(obs)], planets,  # duplicated observation row
        )
        self.assertEqual(len(rows), 1)

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
        return _construct_tool(Tool, **kwargs)

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

    # -------- MAST vocabulary + robustness regressions --------

    def test_nirspec_alias_includes_bots_and_miri_slitless(self):
        """NIRSPEC/BOTS (all NIRSpec TSO) and MIRI/SLITLESS (LRS TSO) must be
        in the expansions; MIRI/MRS and MIRI/LRS are not real CAOM values."""
        filters = self.mast._build_jwst_observation_filters(
            instruments=["NIRSpec", "MIRI"]
        )
        values = next(
            f for f in filters if f["paramName"] == "instrument_name"
        )["values"]
        self.assertIn("NIRSPEC/BOTS", values)
        self.assertIn("MIRI/SLITLESS", values)
        self.assertIn("MIRI/SLIT", values)
        self.assertNotIn("MIRI/MRS", values)
        self.assertNotIn("MIRI/LRS", values)

    def test_mode_shorthand_aliases(self):
        filters = self.mast._build_jwst_observation_filters(
            instruments=["BOTS", "LRS"]
        )
        values = next(
            f for f in filters if f["paramName"] == "instrument_name"
        )["values"]
        self.assertEqual(values, ["NIRSPEC/BOTS", "MIRI/SLIT", "MIRI/SLITLESS"])

    def test_mast_query_raises_on_error_payload(self):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"status": "ERROR", "msg": "Bad column name"}

        class FakeSession:
            def post(self, *args, **kwargs):
                return FakeResponse()

        with self.assertRaises(RuntimeError):
            self.mast._mast_query(
                {"service": "Mast.Caom.Filtered"}, session=FakeSession()
            )

    def test_mast_query_polls_while_executing(self):
        payloads = [
            {"status": "EXECUTING", "data": []},
            {"status": "COMPLETE", "data": [{"obsid": "1"}]},
        ]

        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        class FakeSession:
            def __init__(self):
                self.calls = 0

            def post(self, *args, **kwargs):
                payload = payloads[min(self.calls, len(payloads) - 1)]
                self.calls += 1
                return FakeResponse(payload)

        session = FakeSession()
        payload = self.mast._mast_query(
            {"service": "Mast.Caom.Filtered"}, session=session, poll_wait=0.0
        )
        self.assertEqual(session.calls, 2)
        self.assertEqual(payload["data"], [{"obsid": "1"}])

    def test_instrument_matches_prefix_and_exact(self):
        self.assertTrue(self.mast._instrument_matches("NIRSPEC/BOTS", ["NIRSpec"]))
        self.assertTrue(self.mast._instrument_matches("NIRSPEC/BOTS", ["NIRSPEC/BOTS"]))
        self.assertFalse(self.mast._instrument_matches("NIRSPEC/BOTS", ["NIRISS"]))
        self.assertFalse(self.mast._instrument_matches("NIRSPEC/BOTS", ["NIRSPEC/IFU"]))

    def test_search_basetool_normalizes_llm_args_and_recovers_on_empty(self):
        """Reproduces the real failing call: planet_name='', string-typed
        coords, calib_levels='[]'. Strict query returns nothing; the relaxed
        cone + client-side matching must recover the NIRSPEC/BOTS row."""
        Tool = self.mast.SearchMastJwstObservations
        relaxed_result = [
            {
                "obsid": "1",
                "instrument_name": "NIRSPEC/BOTS",
                "dataproduct_type": "timeseries",
                "calib_level": 3,
                "proposal_id": "4098",
            }
        ]
        calls = []

        def fake_search(planet_name, **kwargs):
            calls.append((planet_name, kwargs))
            if kwargs.get("instruments") is None:
                return relaxed_result  # relaxed diagnostic cone
            return []  # strict query

        with patch.object(
            self.mast, "search_jwst_observations", side_effect=fake_search
        ):
            tool = self._make_tool(
                Tool,
                planet_name="",
                ra="359.8386",
                dec="-60.8506",
                radius_deg="0.5",
                instruments="['NIRSpec']",
                dataproduct_types="['image', 'cube', 'spectrum', 'timeseries']",
                calib_levels="[]",
                proposal_id="4098",
                target_name="",
            )
            output = tool._run()

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1]["ra"], 359.8386)
        self.assertEqual(calls[0][1]["radius_deg"], 0.5)
        self.assertIn("NIRSPEC/BOTS", output)
        self.assertIn("recovered", output.lower())

    def test_search_basetool_empty_cone_reports_proposal_location(self):
        Tool = self.mast.SearchMastJwstObservations
        with (
            patch.object(self.mast, "search_jwst_observations", return_value=[]),
            patch.object(
                self.mast,
                "search_all_jwst_observations",
                return_value=(
                    [
                        {
                            "obsid": "9",
                            "target_name": "LTT 9779",
                            "instrument_name": "NIRSPEC/BOTS",
                        }
                    ],
                    [],
                ),
            ),
        ):
            tool = self._make_tool(
                Tool,
                planet_name="LTT 9779 b",
                ra=10.0,
                dec=-10.0,
                radius_deg=0.1,
                instruments=None,
                dataproduct_types=None,
                calib_levels=None,
                proposal_id="4098",
                target_name=None,
            )
            output = tool._run()

        self.assertIn("no JWST observations of ANY kind", output)
        self.assertIn("Proposal 4098 DOES exist", output)

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
        return _construct_tool(Tool, **kwargs)

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
                    add_cycle_number=False,  # no STScI lookups in unit tests
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
                    add_cycle_number=False,  # no STScI lookups in unit tests
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


def _orchestral_available() -> bool:
    try:
        import orchestral.tools.base.tool  # noqa: F401
        return True
    except Exception:
        return False


class LlmArgCoercionTests(unittest.TestCase):
    """Regression tests for the 'Missing Required Fields' bug: LLM callers
    send booleans/ints as strings and the framework passes them through."""

    @classmethod
    def setUpClass(cls):
        cls.mast = load_mast_module()

    def _make_tool(self, Tool, **kwargs):
        return _construct_tool(Tool, **kwargs)

    def test_as_bool_coercion(self):
        _as_bool = self.mast._as_bool
        self.assertTrue(_as_bool("True", "raw_only"))
        self.assertTrue(_as_bool("true", "raw_only"))
        self.assertTrue(_as_bool("1", "raw_only"))
        self.assertFalse(_as_bool("False", "raw_only"))  # bool('False') footgun
        self.assertFalse(_as_bool("false", "raw_only"))
        self.assertFalse(_as_bool("0", "raw_only"))
        self.assertFalse(_as_bool("", "raw_only"))
        self.assertFalse(_as_bool(None, "raw_only"))
        self.assertTrue(_as_bool(None, "raw_only", default=True))
        self.assertTrue(_as_bool(True, "raw_only"))
        with self.assertRaises(ValueError):
            _as_bool("maybe", "raw_only")

    def test_as_int_or_none_coercion(self):
        _as_int_or_none = self.mast._as_int_or_none
        self.assertEqual(_as_int_or_none("10", "max_observations"), 10)
        self.assertEqual(_as_int_or_none("50.0", "max_products"), 50)
        self.assertEqual(_as_int_or_none(50, "max_products"), 50)
        self.assertIsNone(_as_int_or_none("", "max_products"))
        self.assertIsNone(_as_int_or_none(None, "max_products"))
        with self.assertRaises(ValueError):
            _as_int_or_none("many", "max_products")

    def test_download_tool_normalizes_llm_strings(self):
        """Replays the failing downloadmastjwstproducts call: obsids as a
        repr-string, raw_only='True', max_products='50'. The downstream
        download function must receive real Python types."""
        Tool = self.mast.DownloadMastJwstProducts
        seen = {}

        def fake_download(obsids, out_dir, *, product_subgroups=None,
                          raw_only=False, max_products_per_obs=None,
                          label="aggregate"):
            seen.update(
                obsids=obsids, out_dir=out_dir, raw_only=raw_only,
                max_products=max_products_per_obs, label=label,
            )
            return {"observations": obsids, "files": [], "output_dir": out_dir}

        with patch.object(
            self.mast, "download_observations_products", side_effect=fake_download
        ):
            tool = self._make_tool(
                Tool,
                base_directory="/tmp/aster_ws",
                planet_name="K2-18 b",
                obsids="['233644595', '130902800', '266469228']",
                label="aggregate",
                output_dir="mast/jwst_raw",
                ra=None,
                dec=None,
                radius_deg=0.02,
                instruments="['NIRSpec']",
                dataproduct_types=None,
                product_subgroups="['UNCAL']",
                raw_only="True",
                max_observations="10",
                max_products="50",
            )
            tool._run()

        self.assertIs(seen["raw_only"], True)
        self.assertEqual(seen["max_products"], 50)
        self.assertEqual(seen["obsids"], ["233644595", "130902800", "266469228"])
        self.assertEqual(seen["out_dir"], os.path.join("/tmp/aster_ws", "mast/jwst_raw"))
        self.assertEqual(seen["label"], "aggregate")

    def test_download_tool_requires_planet_or_obsids(self):
        Tool = self.mast.DownloadMastJwstProducts
        tool = self._make_tool(
            Tool, base_directory="/tmp/aster_ws", planet_name="",
            obsids="", label="aggregate", output_dir="mast",
            ra=None, dec=None, radius_deg=0.02, instruments=None,
            dataproduct_types=None, product_subgroups=None,
            raw_only=False, max_observations=None, max_products=None,
        )
        with self.assertRaisesRegex(ValueError, "planet_name.*obsids|obsids.*planet_name"):
            tool._run()


class FakeProgramInfoResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeProgramInfoSession:
    """Serves canned STScI program-info HTML keyed by proposal id."""

    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def get(self, url, timeout=None, **kwargs):
        self.calls.append(url)
        for pid, html in self.pages.items():
            if f"program={pid}" in url:
                return FakeProgramInfoResponse(html)
        return FakeProgramInfoResponse("<html>Program not found</html>", 404)


# Mimics the real page: labels wrapped in tags, value outside them.
_PROGRAM_2372_HTML = """
<html><body>
<h1><a href="/jwst-program-info/program-help/?program=2372#types">GO</a> 2372</h1>
<p><b>Principal Investigator:</b> Renyu Hu<br>
<b>Title:</b> Deep Characterization of the Atmosphere of a Temperate Sub-Neptune<br>
<b>Cycle:</b> 1<br>
<b>Exclusive Access Period:</b> 12 months</p>
<p><b>Program Status:</b> <a href="#completed">Program has been Completed</a></p>
</body></html>
"""


class JwstProgramInfoTests(unittest.TestCase):
    """Cycle comes from STScI program info — never from proposal_id math."""

    def setUp(self):
        self.mast = load_mast_module()

    def test_parses_cycle_and_metadata_then_caches(self):
        session = FakeProgramInfoSession({"2372": _PROGRAM_2372_HTML})
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = pathlib.Path(tmpdir) / "cache.json"

            info = self.mast.get_jwst_program_info(
                2372, cache_path=cache, session=session, request_pause=0
            )
            self.assertEqual(info["cycle"], 1)  # GO 2372 is Cycle 1, not 2372//1000+1
            self.assertEqual(info["proposal_type"], "GO")
            self.assertEqual(info["pi"], "Renyu Hu")
            self.assertIn("Temperate Sub-Neptune", info["title"])
            self.assertIn("Completed", info["status"])

            # Disk cache written and honored: second call, zero new fetches.
            self.assertTrue(cache.is_file())
            self.mast._PROGRAM_INFO_MEMO.clear()  # force the disk-cache path
            again = self.mast.get_jwst_program_info(
                "2372", cache_path=cache, session=session, request_pause=0
            )
            self.assertEqual(again["cycle"], 1)
            self.assertEqual(len(session.calls), 1)

    def test_missing_cycle_field_raises_instead_of_guessing(self):
        session = FakeProgramInfoSession({"9999": "<html>layout changed</html>"})
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = pathlib.Path(tmpdir) / "cache.json"
            with self.assertRaisesRegex(ValueError, "no 'Cycle:' field"):
                self.mast.get_jwst_program_info(
                    9999, cache_path=cache, session=session, request_pause=0
                )
            # Nothing bogus cached.
            self.assertEqual(self.mast._load_program_info_cache(cache), {})

    def test_rejects_non_numeric_proposal_id(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.mast.get_jwst_program_info("GO-2372", request_pause=0)

    def test_annotate_rows_one_lookup_per_proposal_and_warns(self):
        rows = [
            {"obsid": "1", "proposal_id": "2372"},
            {"obsid": "2", "proposal_id": "2372"},   # duplicate pid: no 2nd lookup
            {"obsid": "3", "proposal_id": "9999"},   # lookup fails
            {"obsid": "4"},                          # no pid at all
        ]
        calls = []

        def fake_info(pid, **kwargs):
            calls.append(str(pid))
            if str(pid) == "2372":
                return {"proposal_id": "2372", "cycle": 1}
            raise ValueError("boom")

        with patch.object(self.mast, "get_jwst_program_info", side_effect=fake_info):
            warnings = self.mast.annotate_rows_with_cycles(rows)

        self.assertEqual([r["cycle_number"] for r in rows], [1, 1, None, None])
        self.assertEqual(calls, ["2372", "9999"])  # memoized per pid
        self.assertEqual(len(warnings), 1)
        self.assertIn("9999", warnings[0])

    def test_crossmatch_tool_writes_cycle_number_column(self):
        planets = [{"pl_name": "K2-18 b", "hostname": "K2-18",
                    "ra": 172.56, "dec": 7.5878,
                    "pl_tranmid": 2460000.5, "pl_orbper": 32.94,
                    "pl_trandur": 2.7, "pl_orbeccen": 0.0}]
        observations = [{
            "obsid": "233595537", "obs_id": "jw02372-o007",
            "s_ra": 172.56, "s_dec": 7.5878,
            "instrument_name": "NIRSPEC/SLIT", "dataproduct_type": "timeseries",
            "calib_level": 3, "proposal_id": "2372", "proposal_pi": "Hu, Renyu",
            "filters": "F290LP;G395H", "target_name": "K2-18",
            "t_min": 59999.95, "t_max": 60000.25,  # covers the transit center
        }]
        Tool = self.mast.CrossmatchJwstToPlanets
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(self.mast, "archive_tap_query", return_value=planets),
                patch.object(
                    self.mast, "search_all_jwst_observations",
                    return_value=(observations, []),
                ),
                patch.object(
                    self.mast, "get_jwst_program_info",
                    return_value={"proposal_id": "2372", "cycle": 1},
                ),
            ):
                tool = _construct_tool(
                    Tool,
                    archive_conditions=["pl_rade >= 1.75", "pl_rade <= 4.0"],
                    output_csv="xmatch_cycle.csv",
                    base_directory=tmpdir,
                )
                output = tool._run()

            self.assertIn("cycle=1", output)
            self.assertIn("event=transit", output)
            self.assertNotIn("WARNING — cycle_number", output)
            with (pathlib.Path(tmpdir) / "xmatch_cycle.csv").open() as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(rows[0]["cycle_number"], "1")
            self.assertEqual(rows[0]["obs_type"], "transit")
            self.assertEqual(rows[0]["n_transits_in_window"], "1")

    def test_classifier_transit_eclipse_phase_curve_baseline(self):
        # Synthetic circular ephemeris: P = 2 d, transit centers at MJD 60000,
        # 60002, ...; eclipse centers at 60001, 60003, ...
        kw = dict(tranmid_bjd=2460000.5, period_days=2.0, duration_hours=2.4)

        transit = self.mast.classify_jwst_observation_event(59999.9, 60000.2, **kw)
        self.assertEqual(transit["obs_type"], "transit")
        self.assertEqual(transit["n_transits_in_window"], 1)
        self.assertEqual(transit["n_eclipses_in_window"], 0)

        eclipse = self.mast.classify_jwst_observation_event(60000.9, 60001.15, **kw)
        self.assertEqual(eclipse["obs_type"], "eclipse")

        # Spans 1.7 orbits -> phase curve regardless of event counts.
        pc = self.mast.classify_jwst_observation_event(60000.0, 60003.4, **kw)
        self.assertEqual(pc["obs_type"], "phase_curve")

        # Covers a transit AND an eclipse without a full orbit -> phase curve.
        pc2 = self.mast.classify_jwst_observation_event(59999.95, 60001.05, **kw)
        self.assertEqual(pc2["obs_type"], "phase_curve")

        # Short window between events (e.g. MIRI background) -> baseline.
        base = self.mast.classify_jwst_observation_event(60000.35, 60000.6, **kw)
        self.assertEqual(base["obs_type"], "baseline")

        # String inputs (CSV round-trip) coerce fine.
        s = self.mast.classify_jwst_observation_event(
            "59999.9", "60000.2",
            tranmid_bjd="2460000.5", period_days="2.0", duration_hours="2.4",
        )
        self.assertEqual(s["obs_type"], "transit")

    def test_classifier_padding_counts_partial_transit_coverage(self):
        # Window ends just before the center but inside T14/2 -> still a transit.
        kw = dict(tranmid_bjd=2460000.5, period_days=2.0, duration_hours=4.8)
        partial = self.mast.classify_jwst_observation_event(59999.7, 59999.95, **kw)
        self.assertEqual(partial["obs_type"], "transit")

    def test_classifier_unknown_and_eccentricity_note(self):
        missing = self.mast.classify_jwst_observation_event(
            60000.0, 60000.2, tranmid_bjd=None, period_days=2.0,
        )
        self.assertEqual(missing["obs_type"], "unknown")
        self.assertIn("pl_tranmid", missing["obs_type_note"])

        weird_epoch = self.mast.classify_jwst_observation_event(
            60000.0, 60000.2, tranmid_bjd=9000.0, period_days=2.0,
        )
        self.assertEqual(weird_epoch["obs_type"], "unknown")

        ecc = self.mast.classify_jwst_observation_event(
            60000.9, 60001.15,
            tranmid_bjd=2460000.5, period_days=2.0,
            duration_hours=2.4, eccentricity=0.47,
        )
        self.assertEqual(ecc["obs_type"], "eclipse")
        self.assertIn("approximate", ecc["obs_type_note"])

    def test_classifier_real_gj1214_miri_phase_curve(self):
        # GO 1803 (Bean): the GJ 1214 b MIRI LRS phase curve. Window from the
        # Q2 crossmatch CSV; ephemeris from pscomppars. Span 1.73 d ~ 1.1 P.
        result = self.mast.classify_jwst_observation_event(
            59780.61943254629, 59782.347708402776,
            tranmid_bjd=2455701.413328, period_days=1.58040433,
            duration_hours=0.8703,
        )
        self.assertEqual(result["obs_type"], "phase_curve")

    def test_annotate_rows_with_event_types_per_planet(self):
        # One observation, two planets of the same host: a transit of planet b
        # is NOT an event for planet c (different ephemeris).
        rows = [
            {   # planet b: transit centered in window
                "t_min": 59999.9, "t_max": 60000.2,
                "pl_tranmid": 2460000.5, "pl_orbper": 2.0, "pl_trandur": 2.4,
            },
            {   # planet c: nothing in window
                "t_min": 59999.9, "t_max": 60000.2,
                "pl_tranmid": 2460000.9, "pl_orbper": 9.0, "pl_trandur": 3.0,
            },
            {   # non-transiting RV planet: no ephemeris
                "t_min": 59999.9, "t_max": 60000.2,
                "pl_tranmid": None, "pl_orbper": 9.0,
            },
        ]
        classified = self.mast.annotate_rows_with_event_types(rows)
        self.assertEqual(classified, 2)
        self.assertEqual(rows[0]["obs_type"], "transit")
        self.assertEqual(rows[1]["obs_type"], "baseline")
        self.assertEqual(rows[2]["obs_type"], "unknown")

    def _write_typed_fixture_csv(self, tmpdir):
        rows = [
            {   # transit of a circular planet — no note
                "pl_name": "b", "obsid": "1",
                "t_min": 59999.9, "t_max": 60000.2,
                "pl_tranmid": 2460000.5, "pl_orbper": 2.0,
                "pl_trandur": 2.4, "pl_orbeccen": 0.0,
            },
            {   # eccentric eclipse — approximate-timing note
                "pl_name": "e", "obsid": "2",
                "t_min": 60000.9, "t_max": 60001.15,
                "pl_tranmid": 2460000.5, "pl_orbper": 2.0,
                "pl_trandur": 2.4, "pl_orbeccen": 0.47,
            },
            {   # RV planet, no transit epoch — unknown + note
                "pl_name": "c", "obsid": "3",
                "t_min": 59999.9, "t_max": 60000.2,
                "pl_tranmid": "", "pl_orbper": 9.0,
            },
        ]
        in_path = pathlib.Path(tmpdir) / "xmatch.csv"
        self.mast._write_rows_csv(in_path, rows)
        return in_path

    def test_aggregate_auto_classifies_when_grouping_by_obs_type(self):
        # No add_obs_type flag: requesting obs_type on a CSV that lacks the
        # column must derive it on the fly (and not write a *_typed.csv).
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_typed_fixture_csv(tmpdir)
            tool = _construct_tool(
                self.mast.AggregateJwstObservations,
                group_by=["obs_type"],
                rows_path="xmatch.csv",
                output_csv="grouped.csv",
                base_directory=tmpdir,
            )
            output = tool._run()

            self.assertIn("Event classification: 2/3 rows typed", output)
            self.assertIn("obs_type_note breakdown", output)
            self.assertIn("approximate", output)
            self.assertFalse(
                (pathlib.Path(tmpdir) / "xmatch_typed.csv").exists(),
                "auto mode must not write a typed CSV",
            )
            with (pathlib.Path(tmpdir) / "grouped.csv").open() as fh:
                groups = {r["obs_type"]: r["count"] for r in csv.DictReader(fh)}
            self.assertEqual(
                groups, {"transit": "1", "eclipse": "1", "unknown": "1"},
            )

    def test_aggregate_add_obs_type_persists_typed_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = self._write_typed_fixture_csv(tmpdir)
            tool = _construct_tool(
                self.mast.AggregateJwstObservations,
                group_by=["pl_name"],          # obs_type not even requested
                rows_path="xmatch.csv",
                add_obs_type=True,
                output_csv="grouped.csv",
                base_directory=tmpdir,
            )
            output = tool._run()

            typed = pathlib.Path(tmpdir) / "xmatch_typed.csv"
            self.assertIn(str(typed), output)
            self.assertTrue(typed.is_file())
            with typed.open() as fh:
                read_back = {r["obsid"]: r for r in csv.DictReader(fh)}
            self.assertEqual(read_back["1"]["obs_type"], "transit")
            self.assertEqual(read_back["1"]["obs_type_note"], "")
            self.assertEqual(read_back["2"]["obs_type"], "eclipse")
            self.assertIn("approximate", read_back["2"]["obs_type_note"])
            self.assertEqual(read_back["3"]["obs_type"], "unknown")
            self.assertIn("pl_tranmid", read_back["3"]["obs_type_note"])
            # Input file untouched (no obs_type column added in place).
            with in_path.open() as fh:
                original = list(csv.DictReader(fh))
            self.assertNotIn("obs_type", original[0])

    def test_aggregate_skips_classification_when_column_already_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [{"pl_name": "b", "obs_type": "transit"},
                    {"pl_name": "b", "obs_type": "eclipse"}]
            self.mast._write_rows_csv(pathlib.Path(tmpdir) / "typed.csv", rows)
            with patch.object(
                self.mast, "annotate_rows_with_event_types"
            ) as annotate:
                tool = _construct_tool(
                    self.mast.AggregateJwstObservations,
                    group_by=["obs_type"],
                    rows_path="typed.csv",
                    output_csv="grouped.csv",
                    base_directory=tmpdir,
                )
                output = tool._run()
            annotate.assert_not_called()
            self.assertIn("Groups: 2", output)

    def test_get_jwst_program_info_tool_formats_and_reports_failures(self):
        def fake_info(pid, **kwargs):
            if str(pid) == "2372":
                return {
                    "proposal_id": "2372", "cycle": 1, "proposal_type": "GO",
                    "pi": "Renyu Hu", "title": "Deep Characterization",
                    "status": "Program has been Completed",
                }
            raise ValueError("no such program")

        with patch.object(self.mast, "get_jwst_program_info", side_effect=fake_info):
            tool = _construct_tool(
                self.mast.GetJwstProgramInfo,
                proposal_ids="[2372, 9999]",
                base_directory="/tmp/aster_ws",
            )
            output = tool._run()

        self.assertIn("2372: Cycle 1 | GO | PI Renyu Hu", output)
        self.assertIn("Lookups FAILED", output)
        self.assertIn("9999", output)


@unittest.skipUnless(
    _orchestral_available(),
    "orchestral-ai not importable; schema-optionality contract verified only "
    "where the real framework is installed",
)
class OrchestralSchemaContractTests(unittest.TestCase):
    """The root cause of 'Missing Required Fields': Orchestral's
    SchemaGenerator marks any runtime field whose default is None as REQUIRED
    (only a non-None default or a default_factory makes it optional), and
    BaseTool.execute() enforces that list against the kwargs the LLM sent.
    OptionalRuntimeField (pydantic Field + runtime marker + default_factory)
    is the fix; these tests pin the contract end-to-end."""

    @classmethod
    def setUpClass(cls):
        cls.mast = load_mast_module()

    def test_only_genuinely_required_fields_are_required(self):
        expected = {
            "SearchMastJwstObservations": [],
            "GetMastObservationProducts": ["obsid"],
            "DownloadMastJwstProducts": [],
            "CrossmatchJwstToPlanets": [],
            "AggregateJwstObservations": ["group_by"],
            "DownloadDemographicJwstProducts": ["rows_path"],
            "GetJwstProgramInfo": ["proposal_ids"],
        }
        for name, required in expected.items():
            tool_cls = getattr(self.mast, name)
            self.assertEqual(
                tool_cls._get_required_fields(), required,
                f"{name}: unexpected required-field list",
            )

    def test_execute_replays_failing_download_call(self):
        """The exact call that previously died with
        Missing: ['ra', 'dec', 'dataproduct_types']."""
        seen = {}

        def fake_download(obsids, out_dir, *, product_subgroups=None,
                          raw_only=False, max_products_per_obs=None,
                          label="aggregate"):
            seen.update(obsids=obsids, raw_only=raw_only,
                        max_products=max_products_per_obs)
            return {"observations": obsids, "files": [], "output_dir": out_dir}

        with patch.object(
            self.mast, "download_observations_products", side_effect=fake_download
        ):
            tool = self.mast.DownloadMastJwstProducts(base_directory="/tmp/aster_ws")
            output = tool.execute(
                planet_name="K2-18 b",
                instruments="['NIRSpec']",
                product_subgroups="['UNCAL']",
                raw_only="True",
                max_observations="10",
                max_products="50",
                output_dir="mast/jwst_raw",
                obsids=(
                    "['233644595', '130902800', '266469228', '233595537', "
                    "'365789525', '236587044', '233647234']"
                ),
            )

        self.assertNotIn("Missing Required Fields", output)
        self.assertNotIn("Error:", output.splitlines()[0])
        self.assertIs(seen["raw_only"], True)
        self.assertEqual(seen["max_products"], 50)
        self.assertEqual(len(seen["obsids"]), 7)

    def test_execute_tolerates_empty_string_numerics(self):
        """LLMs stuff '' into unused numeric fields; the str-widened
        annotations must let them through pydantic so _run can null them."""
        with patch.object(
            self.mast, "download_planet_jwst_products",
            return_value={"observations": [], "files": [], "output_dir": "x"},
        ):
            tool = self.mast.DownloadMastJwstProducts(base_directory="/tmp/aster_ws")
            output = tool.execute(
                planet_name="K2-18 b", ra="", dec="", radius_deg="",
                raw_only="", max_observations="", max_products="",
            )
        self.assertNotIn("Validation Error", output)
        self.assertNotIn("Missing Required Fields", output)

    def test_execute_missing_required_field_still_errors(self):
        tool = self.mast.GetMastObservationProducts(base_directory="/tmp/aster_ws")
        output = tool.execute()
        self.assertIn("Missing Required Fields", output)
        self.assertIn("obsid", output)


if __name__ == "__main__":
    unittest.main()
