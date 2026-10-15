"""End-to-end: ASTER Alert inside the GUI.

The real ``run_aster.py`` GUI with the stand-in model, the fake MAST (which
also serves the Exoplanet Archive's TAP endpoint) and a local SMTP server:

* the page carries the ASTER Alert card (script + stylesheet injected);
* "When is HD-12572b going public?" and "which sub-Neptunes ... go public"
  route to AsterAlert and answer from MAST's release dates;
* "email me ... when HD-12572b goes public" subscribes, and the GUI's own
  background notifier emails on release day — here a release a few seconds
  away, so the whole loop runs in the test.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

pytest.importorskip("orchestral")
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
pytest.importorskip("websockets")
pytest.importorskip("taurex")

import aster_gui  # noqa: E402
import fake_mast_archive  # noqa: E402
import fake_mast_server  # noqa: E402
import fake_openai_server  # noqa: E402
import fake_smtp  # noqa: E402

PROGRAMS = {
    "8004": {"proposal_id": "8004", "cycle": 4, "pi": "Natasha Batalha",
             "title": "Cliff Hangers: Testing for Atmosphere-Mantle Interactions"},
    "1366": {"proposal_id": "1366", "cycle": 1, "pi": "Natalie Batalha",
             "title": "The Transiting Exoplanet Community ERS Program"},
}


@pytest.fixture(scope="module")
def stack():
    mast = fake_mast_server.start()
    model = fake_openai_server.start()
    smtp = fake_smtp.start()
    env = {
        "OPENAI_API_KEY": "sk-test-not-a-real-key",
        "OPENAI_BASE_URL": model.url,
        "ASTER_MAST_BASE_URL": mast.url,
        "ASTER_ARCHIVE_TAP_URL": mast.url + "/TAP/sync",
        "ASTER_ALERT_CHECK_SECONDS": "1",
        **smtp.env(),
    }
    with aster_gui.launch_gui(env) as gui:
        # Program titles come from STScI's pages; seed the cache so nothing
        # leaves the machine.
        cache = gui.workspace / "mast" / "jwst_program_info_cache.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(PROGRAMS))
        yield gui, model, mast, smtp
    for server in (mast, model, smtp):
        server.shutdown()


def _get(url: str) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read().decode()


def _alert_calls(model, since: int):
    return [d for d in model.decisions[since:] if d.get("camel") == "AsterAlert"]


def _complete(result):
    assert result["status"] == "complete", result["error"]
    assert "SCHEMA_VIOLATION" not in result["text"], result["text"]


def test_gui_page_carries_the_aster_alert_card(stack):
    gui, model, *_ = stack
    status, page = _get(gui.url + "/")
    assert status == 200
    head = page[:page.index("</head>")]
    assert head.index("/aster-static/aster_alert.js") < page.index("/static/app.js")
    assert _get(gui.url + "/aster-static/aster_alert.js")[1].count("asteralert(") >= 1
    assert _get(gui.url + "/aster-static/aster_alert.css")[0] == 200
    assert json.loads(_get(gui.url + "/api/aster-alert/subscriptions")[1])["pending"] == []


def test_nl_when_is_a_target_going_public(stack):
    gui, model, *_ = stack
    since = len(model.decisions)
    r = gui.chat("When is HD-12572b going public?")
    _complete(r)
    calls = _alert_calls(model, since)
    assert [c["arguments"] for c in calls] == [{"target": "HD-12572b"}]
    assert "ASTER Alert · HD-12572b: next release" in r["text"]
    assert "HIP 9618 b" in r["text"] and "LOCKED until" in r["text"]


def test_nl_which_subneptunes_go_public(stack):
    gui, model, *_ = stack
    gui.clear()
    since = len(model.decisions)
    r = gui.chat("Use ASTER Alert: which new sub-Neptunes went public recently "
                 "or are going public soon?")
    _complete(r)
    calls = _alert_calls(model, since)
    assert [c["arguments"] for c in calls] == [{"action": "watchlist"}]
    assert "go public in the next 120 days" in r["text"]
    assert "HIP 9618 b" in r["text"] and "WASP-39 b" not in r["text"]   # not a sub-Neptune


def test_nl_email_me_on_release_day_is_delivered_by_the_gui(stack):
    gui, model, mast, smtp = stack
    gui.clear()
    saved = fake_mast_archive.LOCKED_RELEASE_MJD
    fake_mast_archive.LOCKED_RELEASE_MJD = fake_mast_archive.now_mjd() + 12 / 86400
    try:
        since = len(model.decisions)
        r = gui.chat("Email me at astro@example.org when HD-12572b goes public")
        _complete(r)
        assert [c["arguments"] for c in _alert_calls(model, since)] == [
            {"target": "HD-12572b", "action": "subscribe", "email": "astro@example.org"}]
        assert "subscribed astro@example.org: 1 release(s) for HD-12572b" in r["text"]

        subs = json.loads((gui.workspace / "mast/alert/subscriptions.json").read_text())
        assert [s["status"] for s in subs["subscriptions"]] == ["pending"]

        # The GUI's notifier (checking every second here) emails once MAST
        # reports the observation public, about 12 s from now.
        assert smtp.received.wait(60), "\n".join(gui.log[-30:])
        msg = smtp.messages[-1]
        assert msg["To"] == "astro@example.org"
        assert msg["Subject"] == "[ASTER Alert] HIP 9618 b, HIP 9618 c: JWST data is public - GO 8004"
        assert 'DownloadMastJwstProducts(obsids="202"' in msg.get_content()

        deadline = time.time() + 10
        while time.time() < deadline:
            summary = json.loads(_get(gui.url + "/api/aster-alert/subscriptions")[1])
            if summary["sent"] == 1:
                break
            time.sleep(0.2)
        assert summary == {"pending": [], "sent": 1, "email_configured": True}
        assert any("[aster-alert] emailed astro@example.org" in line for line in gui.log)
    finally:
        fake_mast_archive.LOCKED_RELEASE_MJD = saved
