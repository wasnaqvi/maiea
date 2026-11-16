"""Offline tests for ASTER Alert's helpers (no network)."""
from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from aster_toolkit.data_acquisition import alert  # noqa: E402


def test_planet_list_overrides_preset():
    cfg = alert.AlertConfig(planets=["TOI-270 d", "O'Hare b"])
    assert cfg.planet_conditions() == ["pl_name IN ('TOI-270 d', 'O''Hare b')"]


def test_preset_adds_transiting_flag():
    assert "tran_flag = 1" in alert.AlertConfig().planet_conditions()


def test_mjd_roundtrip():
    assert alert.mjd_to_iso(60676.0) == "2025-01-01"


def test_there_is_no_digest_or_check_mode():
    """ASTER Alert is lookup + subscribe + an email on the release date. The
    daily population digest (state diff, Slack/macOS notifiers, `check`) is
    gone on purpose."""
    for name in ("run_alert", "diff_snapshot", "render_digest", "slack_notifier"):
        assert not hasattr(alert, name)
    assert "check" not in alert._ACTIONS
