"""Offline tests for the shared NASA Exoplanet Archive TAP helper."""
from __future__ import annotations

import pathlib
import sys
from unittest.mock import MagicMock

import pytest
import requests

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from aster_toolkit.data_acquisition import archive_tap, exoarchive, mast  # noqa: E402

CSV_OK = "pl_name,pl_orbper\nWASP-39 b,4.0552842\n"
VOTABLE_ERROR = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<VOTABLE version="1.4">\n'
    '<RESOURCE type="results">\n<INFO name="QUERY_STATUS" value="ERROR">\n'
    "ORA-00904: 'NOSUCHCOL': invalid identifier\n</INFO>\n</RESOURCE>\n</VOTABLE>\n"
)


def fake_session(text, status=200):
    response = MagicMock()
    response.text = text
    response.status_code = status
    response.ok = status < 400
    response.reason = "Bad Request" if status >= 400 else "OK"
    session = MagicMock()
    session.get.return_value = response
    return session


def sent_adql(session):
    return session.get.call_args.kwargs["params"]["query"]


def test_build_adql_variants():
    assert archive_tap.build_adql(["pl_name"]) == "SELECT pl_name FROM pscomppars"
    assert archive_tap.build_adql(
        ["pl_name", "ra"], ["sy_pnum >= 2", "pl_eqt > 1000"],
        table="ps", distinct=True, limit=5,
    ) == ("SELECT DISTINCT TOP 5 pl_name, ra FROM ps "
          "WHERE (sy_pnum >= 2) AND (pl_eqt > 1000)")
    assert archive_tap.build_adql(["x"], "a = 1") == "SELECT x FROM pscomppars WHERE (a = 1)"
    with pytest.raises(ValueError):
        archive_tap.build_adql([])


def test_run_adql_parses_csv_and_sends_timeout():
    session = fake_session(CSV_OK)
    rows = archive_tap.run_adql("SELECT 1", session=session, timeout=7)
    assert rows == [{"pl_name": "WASP-39 b", "pl_orbper": "4.0552842"}]
    call = session.get.call_args
    assert call.args[0] == archive_tap.TAP_SYNC_URL
    assert call.kwargs["params"]["format"] == "csv"
    assert call.kwargs["timeout"] == 7


@pytest.mark.parametrize("status", [400, 200])
def test_rejected_query_raises_with_archive_reason(status):
    # The archive answers a bad query with a VOTable, normally HTTP 400; a
    # 200 VOTable must not be parsed as a one-column CSV either.
    with pytest.raises(archive_tap.ArchiveTapError) as exc:
        archive_tap.run_adql("SELECT nosuchcol FROM pscomppars",
                             session=fake_session(VOTABLE_ERROR, status))
    assert "ORA-00904" in str(exc.value)
    assert "SELECT nosuchcol" in str(exc.value)
    assert isinstance(exc.value, requests.RequestException)


def test_http_error_without_votable():
    with pytest.raises(archive_tap.ArchiveTapError, match="HTTP 503"):
        archive_tap.run_adql("SELECT 1", session=fake_session("", 503))


def test_mast_archive_tap_query_delegates():
    session = fake_session(CSV_OK)
    rows = mast.archive_tap_query(["pl_name = 'WASP-39 b'"],
                                  columns=["pl_name", "pl_orbper"], limit=0,
                                  session=session, timeout=9)
    assert rows[0]["pl_name"] == "WASP-39 b"
    assert sent_adql(session) == (
        "SELECT pl_name, pl_orbper FROM pscomppars WHERE (pl_name = 'WASP-39 b')")
    assert session.get.call_args.kwargs["timeout"] == 9


def test_mast_archive_tap_query_default_columns_and_limit():
    session = fake_session("pl_name\nX\n")
    mast.archive_tap_query("sy_pnum >= 2", limit=3, session=session)
    assert sent_adql(session).startswith(
        f"SELECT TOP 3 {', '.join(mast.DEFAULT_ARCHIVE_COLUMNS)} FROM pscomppars")


def test_exoarchive_helpers_route_through_shared_request(monkeypatch):
    session = fake_session(CSV_OK)
    monkeypatch.setattr(archive_tap, "requests", session)  # session.get stands in

    assert exoarchive.get_exoplanet_params_tap("WASP-39 b", "pl_orbper") == {
        "pl_orbper": "4.0552842"}
    assert sent_adql(session) == (
        "SELECT pl_orbper FROM pscomppars WHERE (pl_name = 'WASP-39 b')")
    assert session.get.call_args.kwargs["timeout"] == archive_tap.DEFAULT_TIMEOUT

    rows = exoarchive.find_exoplanets_by_condition(
        ["sy_pnum >= 2"], ["pl_name", "missing_col"], limit=2)
    assert rows == [{"pl_name": "WASP-39 b", "missing_col": None}]
    assert sent_adql(session) == (
        "SELECT DISTINCT TOP 2 pl_name, missing_col FROM pscomppars WHERE (sy_pnum >= 2)")

    session.get.return_value.text = "pl_orbper\n"
    assert exoarchive.get_exoplanet_params_tap("Nope b", ["pl_orbper"]) is None
