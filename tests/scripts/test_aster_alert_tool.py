"""ASTER Alert inside the GUI: the AsterAlert tool, release-day email
subscriptions, and the GUI layer (aster_toolkit/gui.py). Offline: MAST and
the Exoplanet Archive are patched; email goes to a local fake SMTP server.
"""
import json
import pathlib
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fake_smtp  # noqa: E402

from aster_toolkit.data_acquisition import alert, mast  # noqa: E402

NOW = 61329.0          # 2026-10-16
HD12572 = (30.9050061, 21.2813274)


def caom(obsid, release, rights, *, filters="F290LP;G395H", pid="8004",
         t_min=None, obs_id=None):
    return {"obsid": obsid, "obs_id": obs_id or f"jw0{pid}-o{obsid[-3:]}_t002_nirspec",
            "target_name": "HD-12572b", "s_ra": HD12572[0], "s_dec": HD12572[1],
            "instrument_name": "NIRSPEC/SLIT", "filters": filters, "proposal_id": pid,
            "proposal_pi": "Batalha, Natasha", "t_min": t_min or release - 365,
            "t_obs_release": release, "dataRights": rights}


PLANETS = [{"pl_name": "HIP 9618 b", "hostname": "HIP 9618", "ra": HD12572[0],
            "dec": HD12572[1], "pl_rade": "3.83", "pl_eqt": "641"},
           {"pl_name": "HIP 9618 c", "hostname": "HIP 9618", "ra": HD12572[0],
            "dec": HD12572[1], "pl_rade": "3.43", "pl_eqt": "470"}]


@pytest.fixture
def fake_mast(monkeypatch):
    """HD 12572: one released visit, one locked until NOW + 89 d."""
    rows = [caom("343406345", NOW - 36, "PUBLIC"),
            caom("376915403", NOW + 89, "EXCLUSIVE_ACCESS")]
    tried = []

    def resolve(name, **_):
        tried.append(name)
        if name in ("HD-12572b", "HD 12572", "HIP 9618 b"):
            return HD12572
        raise ValueError(f"MAST could not resolve target name '{name}'.")

    monkeypatch.setattr(mast, "resolve_target_coordinates", resolve)
    monkeypatch.setattr(mast, "search_jwst_observations", lambda *a, **k: list(rows))
    def archive(conditions, **_):
        text = " ".join(conditions) if isinstance(conditions, list) else str(conditions)
        if "pl_name =" in text:      # name fallback: only real names resolve
            return [p for p in PLANETS if f"'{p['pl_name']}'" in text]
        return list(PLANETS)         # the RA/Dec box around HD 12572

    monkeypatch.setattr(mast, "archive_tap_query", archive)
    monkeypatch.setattr(mast, "get_jwst_program_info", lambda pid, **k: {
        "pi": "Natasha Batalha", "title": "Cliff Hangers", "cycle": 4})
    monkeypatch.setattr(alert, "now_mjd", lambda: NOW)
    return {"rows": rows, "tried": tried}


# ---------------- names ----------------

@pytest.mark.parametrize("name, expected", [
    ("HD-12572b", ["HD-12572b", "HD-12572 b", "HD 12572 b", "HD-12572", "HD 12572"]),
    ("HIP 9618 b", ["HIP 9618 b", "HIP 9618"]),
    ("TOI-836.01", ["TOI-836.01", "TOI 836.01"]),
])
def test_name_variants(name, expected):
    assert alert._name_variants(name) == expected


def test_resolve_falls_through_unresolvable_spellings(fake_mast):
    ra, dec, used = alert.resolve_target("HD-12572 B")   # case/space variant
    assert (ra, dec) == HD12572 and used == "HD 12572"
    assert fake_mast["tried"][0] == "HD-12572 B"


# ---------------- lookup ----------------

def test_lookup_reports_next_release_and_planets(fake_mast):
    text = alert.AsterAlert(base_directory="/tmp").execute(target="HD-12572b")
    head = text.splitlines()[0]
    assert head == ("ASTER Alert · HD-12572b: next release 2027-01-13 (in 89 days); "
                    "1 of 2 JWST time-series observations still locked")
    assert "HIP 9618 b (3.83 R⊕, 641 K)" in text
    assert "PUBLIC since 2026-09-10" in text and "LOCKED until 2027-01-13" in text
    assert 'DownloadMastJwstProducts(obsids="343406345"' in text
    assert "GO 8004 (Cycle 4): Cliff Hangers" in text      # title once, in the footer
    assert text.count("Cliff Hangers") == 1


def test_lookup_all_public_and_nothing_observed(fake_mast):
    fake_mast["rows"][1]["dataRights"] = "PUBLIC"
    fake_mast["rows"][1]["t_obs_release"] = NOW - 1
    text = alert.AsterAlert(base_directory="/tmp").execute(target="HD-12572b")
    assert "all 2 JWST time-series observations are already public" in text.splitlines()[0]
    fake_mast["rows"].clear()
    text = alert.AsterAlert(base_directory="/tmp").execute(target="HD-12572b")
    assert "no executed JWST time-series observations" in text.splitlines()[0]
    assert "not in MAST" in text


def test_unresolvable_target_is_an_error_not_a_crash(fake_mast):
    out = alert.AsterAlert(base_directory="/tmp").execute(target="Nowhere-1 b")
    assert out.startswith("Error") and "Could not resolve" in out


# ---------------- watchlist ----------------

def test_watchlist_splits_recent_upcoming_later_and_flags_first_data(monkeypatch):
    def rec(pl, obsid, release, public):
        return alert.observation_record(
            {"pl_name": pl, "obsid": obsid, "t_obs_release": release,
             "dataRights": "PUBLIC" if public else "EXCLUSIVE_ACCESS",
             "filters": "F290LP;G395H", "proposal_id": "1"}, NOW)
    snap = {
        "A|1": rec("A", "1", NOW - 10, True),      # recent, A's first public data
        "B|2": rec("B", "2", NOW - 400, True),     # old release of B
        "B|3": rec("B", "3", NOW - 5, True),       # recent, but B had data before
        "C|4": rec("C", "4", NOW + 30, False),     # upcoming, C has nothing public
        "B|5": rec("B", "5", NOW + 60, False),     # upcoming, B has public data
        "D|6": rec("D", "6", NOW + 400, False),    # later
    }
    monkeypatch.setattr(alert, "fetch_snapshot", lambda *a, **k: snap)
    monkeypatch.setattr(mast, "get_jwst_program_info", lambda *a, **k: {})
    rep = alert.watchlist_report(alert.AlertConfig(), days_back=60, days_ahead=120, t_now=NOW)
    assert [r["obsid"] for r in rep["released"]] == ["3", "1"]
    assert [r["obsid"] for r in rep["upcoming"]] == ["4", "5"]
    assert [r["obsid"] for r in rep["later"]] == ["6"]
    flags = {r["obsid"]: r["first_public_data"] for r in rep["released"] + rep["upcoming"]}
    assert flags == {"1": True, "3": False, "4": True, "5": False}
    text = alert.render_watchlist(rep)
    assert text.splitlines()[0] == ("ASTER Alert · 2 planet(s) go public in the next 120 "
                                    "days; 2 went public in the last 60 days")
    assert "Later: 1 more locked observation(s)" in text


# ---------------- subscriptions ----------------

def test_subscribe_needs_an_email(fake_mast, tmp_path, monkeypatch):
    monkeypatch.delenv("ASTER_ALERT_EMAIL_TO", raising=False)
    out = alert.AsterAlert(base_directory=str(tmp_path)).execute(
        action="subscribe", target="HD-12572b")
    assert out.startswith("Error") and "Which email address" in out
    out = alert.AsterAlert(base_directory=str(tmp_path)).execute(
        action="subscribe", target="HD-12572b", email="not-an-address")
    assert "does not look like an email" in out


def test_subscribe_locked_only_dedupes_and_warns_without_smtp(fake_mast, tmp_path, monkeypatch):
    monkeypatch.delenv("ASTER_ALERT_SMTP_HOST", raising=False)
    monkeypatch.setenv("ASTER_ALERT_EMAIL_TO", "me@example.org")
    tool = alert.AsterAlert(base_directory=str(tmp_path))
    out = tool.execute(action="subscribe", target="HD-12572b")
    assert out.splitlines()[0] == ("ASTER Alert · subscribed me@example.org: 1 release(s) "
                                   "for HD-12572b, the first on 2027-01-13")
    assert "email is not configured" in out
    subs = alert.load_subscriptions(alert.subscriptions_path(tmp_path))
    assert [(s["obsid"], s["status"], s["planets"]) for s in subs] == [
        ("376915403", "pending", ["HIP 9618 b", "HIP 9618 c"])]
    again = tool.execute(action="subscribe", target="HD-12572b")
    assert "(already subscribed)" in again
    assert len(alert.load_subscriptions(alert.subscriptions_path(tmp_path))) == 1
    listing = tool.execute(action="subscriptions")
    assert listing.startswith("ASTER Alert · 1 email(s) waiting, 0 sent")
    gone = tool.execute(action="unsubscribe", target="hd-12572b")
    assert gone.startswith("ASTER Alert · removed 1 subscription(s)")
    assert alert.load_subscriptions(alert.subscriptions_path(tmp_path)) == []


def test_subscribe_when_everything_is_public(fake_mast, tmp_path):
    fake_mast["rows"].pop()
    out = alert.AsterAlert(base_directory=str(tmp_path)).execute(
        action="subscribe", target="HD-12572b", email="me@example.org")
    assert "nothing to subscribe to" in out
    assert not alert.subscriptions_path(tmp_path).exists()


def _pending(tmp_path, release=NOW - 0.5):
    path = alert.subscriptions_path(tmp_path)
    rec = alert.observation_record(caom("376915403", release, "EXCLUSIVE_ACCESS"), NOW - 10)
    alert.add_subscriptions(path, email="me@example.org", target="HD-12572b",
                            planets=["HIP 9618 b"], records=[rec])
    return path


def test_deliver_sends_once_when_mast_says_public(tmp_path, monkeypatch):
    path = _pending(tmp_path)
    monkeypatch.setattr(alert, "fetch_observations_by_obsid", lambda ids, **k: {
        "376915403": alert.observation_record(caom("376915403", NOW - 0.5, "PUBLIC"), NOW)})
    sent = []
    log = alert.deliver_due_subscriptions(path, now=NOW, send=lambda *m: sent.append(m))
    assert len(sent) == 1 and sent[0][0] == "me@example.org"
    assert "HIP 9618 b: JWST data is public - GO 8004" in sent[0][1]
    assert 'DownloadMastJwstProducts(obsids="376915403"' in sent[0][2]
    assert log and log[0].startswith("emailed me@example.org")
    assert alert.load_subscriptions(path)[0]["status"] == "sent"
    assert alert.deliver_due_subscriptions(path, now=NOW + 1, send=lambda *m: sent.append(m)) == []
    assert len(sent) == 1


def test_deliver_follows_a_moved_release_date(tmp_path, monkeypatch):
    path = _pending(tmp_path, release=NOW + 5)
    monkeypatch.setattr(alert, "fetch_observations_by_obsid", lambda ids, **k: {
        "376915403": alert.observation_record(caom("376915403", NOW + 40, "EXCLUSIVE_ACCESS"), NOW)})
    sent = []
    log = alert.deliver_due_subscriptions(path, now=NOW, send=lambda *m: sent.append(m))
    assert not sent and "release moved" in log[0]
    sub = alert.load_subscriptions(path)[0]
    assert sub["status"] == "pending" and sub["t_obs_release"] == NOW + 40
    assert sub["rescheduled_from"] == alert.mjd_to_iso(NOW + 5)


def test_deliver_retries_failed_sends_and_mast_outages(tmp_path, monkeypatch):
    path = _pending(tmp_path)
    monkeypatch.setattr(alert, "fetch_observations_by_obsid", lambda ids, **k: {
        "376915403": alert.observation_record(caom("376915403", NOW - 0.5, "PUBLIC"), NOW)})

    def boom(*_):
        raise OSError("connection refused")

    log = alert.deliver_due_subscriptions(path, now=NOW, send=boom)
    sub = alert.load_subscriptions(path)[0]
    assert sub["status"] == "pending" and "connection refused" in sub["last_error"]
    assert "will retry" in log[0]

    def down(*a, **k):
        raise ConnectionError("MAST down")

    monkeypatch.setattr(alert, "fetch_observations_by_obsid", down)
    assert "MAST unreachable" in alert.deliver_due_subscriptions(path, now=NOW, send=boom)[0]


def test_deliver_with_no_subscriptions_touches_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(alert, "fetch_observations_by_obsid",
                        lambda *a, **k: pytest.fail("no network without subscriptions"))
    assert alert.deliver_due_subscriptions(alert.subscriptions_path(tmp_path)) == []


# ---------------- real SMTP (local fake) ----------------

def test_send_email_over_plain_smtp(monkeypatch):
    smtp = fake_smtp.start()
    try:
        for k, v in smtp.env().items():
            monkeypatch.setenv(k, v)
        alert.send_email("me@example.org", "hello", "body text")
        assert smtp.received.wait(5)
        msg = smtp.messages[0]
        assert msg["Subject"] == "[ASTER Alert] hello" and msg["To"] == "me@example.org"
        assert msg["From"] == "aster-alert@test.local"
        assert "body text" in msg.get_content()
    finally:
        smtp.shutdown()


def test_test_email_action_and_unconfigured_error(monkeypatch, tmp_path):
    monkeypatch.delenv("ASTER_ALERT_SMTP_HOST", raising=False)
    out = alert.AsterAlert(base_directory=str(tmp_path)).execute(
        action="test_email", email="me@example.org")
    assert out.startswith("Error") and "ASTER_ALERT_SMTP_HOST" in out
    smtp = fake_smtp.start()
    try:
        for k, v in smtp.env().items():
            monkeypatch.setenv(k, v)
        out = alert.AsterAlert(base_directory=str(tmp_path)).execute(
            action="test_email", email="me@example.org")
        assert out == "ASTER Alert · test email sent to me@example.org"
        assert smtp.received.wait(5)
    finally:
        smtp.shutdown()


def test_unknown_action_is_reported():
    out = alert.AsterAlert(base_directory="/tmp").execute(action="explode")
    assert out.startswith("Error") and "Unknown action" in out


# ---------------- GUI layer ----------------

def test_inject_assets_goes_first_in_head_and_is_idempotent():
    from aster_toolkit import gui
    page = "<html><head>\n<title>x</title><script src='/static/app.js'></script></head></html>"
    once = gui.inject_assets(page)
    assert once.index("aster_alert.js") < once.index("/static/app.js")
    assert gui.inject_assets(once) == once


def test_install_serves_page_assets_and_subscriptions(tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    import types
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from fastapi.testclient import TestClient
    from aster_toolkit import gui

    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<html><head><title>Orchestral</title></head></html>")
    app = FastAPI()

    @app.get("/")
    async def original():
        return FileResponse(static / "index.html")

    server = types.SimpleNamespace(app=app, STATIC_DIR=static)
    monkeypatch.setenv("ASTER_ALERT_CHECK_SECONDS", "0")      # no notifier thread
    _pending(tmp_path / "ws")
    gui.install(server, base_directory=str(tmp_path / "ws"))
    gui.install(server, base_directory=str(tmp_path / "ws"))   # second call: no-op

    client = TestClient(app)
    page = client.get("/").text
    assert page.count("/aster-static/aster_alert.js") == 1
    assert client.get("/aster-static/aster_alert.js").status_code == 200
    assert client.get("/aster-static/aster_alert.css").status_code == 200
    data = client.get("/api/aster-alert/subscriptions").json()
    assert data["pending"][0]["target"] == "HD-12572b" and data["sent"] == 0


def test_notifier_thread_delivers_and_stops(tmp_path, monkeypatch):
    path = _pending(tmp_path)
    monkeypatch.setattr(alert, "fetch_observations_by_obsid", lambda ids, **k: {
        "376915403": alert.observation_record(caom("376915403", NOW - 0.5, "PUBLIC"), NOW)})
    smtp = fake_smtp.start()
    try:
        for k, v in smtp.env().items():
            monkeypatch.setenv(k, v)
        logs = []
        stop = alert.start_subscription_notifier(tmp_path, interval_seconds=0.2, log=logs.append)
        assert smtp.received.wait(10)
        import time
        deadline = time.time() + 5
        while not any("emailed" in line for line in logs) and time.time() < deadline:
            time.sleep(0.05)
        stop.set()
        assert "HIP 9618 b: JWST data is public - GO 8004" in smtp.messages[0]["Subject"]
        assert any("emailed me@example.org" in line for line in logs)
        assert alert.load_subscriptions(path)[0]["status"] == "sent"
    finally:
        smtp.shutdown()


def test_host_without_login_is_not_configured(monkeypatch, fake_mast, tmp_path):
    """A Gmail host with the login left blank (as .env ships) must not claim
    email works: the warning names exactly what is still missing."""
    for k, v in {"ASTER_ALERT_SMTP_HOST": "smtp.gmail.com", "ASTER_ALERT_SMTP_PORT": "587",
                 "ASTER_ALERT_SMTP_USER": "", "ASTER_ALERT_SMTP_PASSWORD": ""}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("ASTER_ALERT_SMTP_TLS", raising=False)
    assert alert.smtp_missing() == ["ASTER_ALERT_SMTP_USER", "ASTER_ALERT_SMTP_PASSWORD"]
    assert not alert.smtp_configured()
    with pytest.raises(RuntimeError, match="ASTER_ALERT_SMTP_USER, ASTER_ALERT_SMTP_PASSWORD"):
        alert.send_email("me@example.org", "x", "y")
    out = alert.AsterAlert(base_directory=str(tmp_path)).execute(
        action="subscribe", target="HD-12572b", email="me@example.org")
    assert "Set ASTER_ALERT_SMTP_USER, ASTER_ALERT_SMTP_PASSWORD in .env" in out
    monkeypatch.setenv("ASTER_ALERT_SMTP_TLS", "none")      # a local relay needs no login
    assert alert.smtp_configured()
