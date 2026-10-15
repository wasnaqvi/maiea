"""Offline tests for ASTER Alert: MAST + Exoplanet Archive are mocked."""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from aster_toolkit.data_acquisition import alert, mast  # noqa: E402

NOW = 61300.0  # fixed "today" in MJD

PLANETS = [
    {"pl_name": "TOI-270 d", "hostname": "TOI-270", "ra": "68.4", "dec": "-51.95"},
    {"pl_name": "GJ 1214 b", "hostname": "GJ 1214", "ra": "258.83", "dec": "4.96"},
]


def obs(obsid, ra, dec, release, rights, filters="F290LP;G395H", pid="5959"):
    return {
        "obsid": obsid, "obs_id": f"jw0{pid}-o{obsid[-3:]}", "s_ra": ra, "s_dec": dec,
        "instrument_name": "NIRSPEC/BOTS", "dataproduct_type": "timeseries",
        "calib_level": 3, "filters": filters, "proposal_id": pid,
        "proposal_pi": "Luque, Rafael", "t_min": release - 365, "t_max": release - 364.9,
        "t_obs_release": release, "dataRights": rights, "target_name": "X",
    }


class AlertTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = pathlib.Path(self.tmp.name) / "state.json"
        self.config = alert.AlertConfig(enrich_program_info=False)
        self.sent: list[tuple[str, str]] = []
        self.observations = [
            obs("001", 68.4, -51.95, NOW + 40, "EXCLUSIVE_ACCESS"),         # TOI-270 d
            obs("002", 258.83, 4.96, NOW - 200, "PUBLIC", "F444W", "1803"),  # GJ 1214 b
            obs("003", 10.0, 10.0, NOW + 5, "EXCLUSIVE_ACCESS"),             # no planet
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, now=NOW):
        with patch.object(mast, "search_all_jwst_observations",
                          return_value=(list(self.observations), [])), \
             patch.object(mast, "archive_tap_query", return_value=PLANETS), \
             patch.object(alert, "now_mjd", return_value=now):
            return alert.run_alert(
                self.config, state_path=self.state,
                notifiers=[lambda s, b: self.sent.append((s, b))],
            )

    def kinds(self, events):
        return sorted((e.kind, e.record["pl_name"]) for e in events)

    def test_bootstrap_is_quiet_digest_of_upcoming_only(self):
        events = self._run()
        self.assertEqual(self.kinds(events), [("UPCOMING", "TOI-270 d")])
        self.assertTrue(self.state.exists())
        self.assertIn("★ **TOI-270 d**", self.sent[0][1])

    def test_no_changes_no_notification(self):
        self._run()
        self.sent.clear()
        self.assertEqual(self._run(), [])
        self.assertEqual(self.sent, [])

    def test_soon_fires_once_per_threshold(self):
        self._run()
        self.assertEqual(self.kinds(self._run(now=NOW + 11)), [("SOON", "TOI-270 d")])
        self.assertEqual(self._run(now=NOW + 12), [])           # still inside 30 d
        self.assertEqual(self.kinds(self._run(now=NOW + 34)), [("SOON", "TOI-270 d")])

    def test_skipped_thresholds_collapse_to_one_alert(self):
        self._run()
        events = self._run(now=NOW + 39.5)                     # crosses 30, 7, 1
        self.assertEqual(len(events), 1)
        self.assertIn("<= 1 d", events[0].detail)

    def test_public_transition_and_download_hint(self):
        self._run()
        self.observations[0]["dataRights"] = "PUBLIC"
        events = self._run(now=NOW + 41)
        self.assertIn(("PUBLIC", "TOI-270 d"), self.kinds(events))
        self.assertIn('DownloadMastJwstProducts(obsids="001"', self.sent[-1][1])

    def test_new_observation_and_date_change(self):
        self._run()
        self.observations.append(obs("004", 258.83, 4.96, NOW + 300, "EXCLUSIVE_ACCESS"))
        self.observations[0]["t_obs_release"] = NOW + 90
        kinds = self.kinds(self._run())
        self.assertIn(("NEW", "GJ 1214 b"), kinds)
        self.assertIn(("DATE_CHANGED", "TOI-270 d"), kinds)

    def test_vanished_observation_is_not_reannounced(self):
        self._run()
        held = self.observations.pop(0)
        self._run()
        self.observations.insert(0, held)
        self.assertEqual(self._run(), [])

    def test_planet_list_overrides_preset(self):
        cfg = alert.AlertConfig(planets=["TOI-270 d", "O'Hare b"])
        conds = cfg.planet_conditions()
        self.assertEqual(conds, ["pl_name IN ('TOI-270 d', 'O''Hare b')"])

    def test_preset_adds_transiting_flag(self):
        self.assertIn("tran_flag = 1", alert.AlertConfig().planet_conditions())

    def test_mjd_roundtrip(self):
        self.assertEqual(alert.mjd_to_iso(60676.0), "2025-01-01")


if __name__ == "__main__":
    unittest.main()
