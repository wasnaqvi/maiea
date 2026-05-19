from __future__ import annotations

import importlib.util
import json
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
    def __init__(self, chunks):
        self._chunks = chunks

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        return iter(self._chunks)


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
        self.assertIn(
            {"paramName": "instrument_name", "values": ["NIRSpec", "NIRCam"]},
            filters,
        )
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
            {"paramName": "instrument_name", "values": ["NIRSpec"]},
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


if __name__ == "__main__":
    unittest.main()
