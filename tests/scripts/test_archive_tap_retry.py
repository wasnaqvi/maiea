import pytest
import requests

from aster_toolkit.data_acquisition import archive_tap

OUTAGE = ('<INFO name="QUERY_STATUS" value="ERROR">Failed to connect to cx_Oracle</INFO>')
BAD = ('<INFO name="QUERY_STATUS" value="ERROR">ORA-00904: invalid identifier</INFO>')


class FakeSession:
    def __init__(self, bodies):
        self.bodies, self.calls = list(bodies), 0

    def get(self, url, **kw):
        self.calls += 1
        r = requests.Response()
        r.status_code, r._content = 200, self.bodies.pop(0).encode()
        return r


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(archive_tap.time, "sleep", lambda s: None)


def test_outage_is_retried_then_succeeds():
    s = FakeSession([OUTAGE, OUTAGE, "pl_name\nWASP-39 b\n"])
    assert archive_tap.run_adql("SELECT 1", session=s) == [{"pl_name": "WASP-39 b"}]
    assert s.calls == 3


def test_persistent_outage_raises_after_retries():
    s = FakeSession([OUTAGE] * 4)
    with pytest.raises(archive_tap.ArchiveTapError):
        archive_tap.run_adql("SELECT 1", session=s)
    assert s.calls == 4


def test_bad_query_is_not_retried():
    s = FakeSession([BAD])
    with pytest.raises(archive_tap.ArchiveTapError):
        archive_tap.run_adql("SELECT 1", session=s)
    assert s.calls == 1
