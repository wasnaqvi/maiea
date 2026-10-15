"""End-to-end: natural-language requests typed into ASTER's GUI search MAST
and download JWST data.

What runs for real: ``python run_aster.py`` (unmodified), orchestral's web
GUI and agent loop, the GPT provider and ``openai`` client, ASTER's system
prompt, tool schemas and tools, HTTP to MAST, and files written to disk.

What is stood in: the model (tests/e2e/fake_openai_server.py, a
deterministic natural-language interpreter that also checks every call
against the schemas ASTER sends) and MAST itself
(tests/e2e/fake_mast_server.py, serving the ERS 1366 fixture over HTTP).
Real-model routing is measured separately by scripts/eval_aster_nl.py.

Run:  pytest tests/e2e -v        (about 20 s; needs fastapi, uvicorn,
                                  websockets, rich and ASTER's own deps)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

pytest.importorskip("orchestral")
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
pytest.importorskip("websockets")
pytest.importorskip("taurex")   # run_aster.py imports every ASTER tool

import aster_gui  # noqa: E402
import fake_mast_server  # noqa: E402
import fake_openai_server  # noqa: E402

G395H_DIR = "mast/WASP-39_b/jw01366-o004_t001_nirspec_f290lp-g395h-s1600a1-sub2048"
G395H_UNCALS = sorted(
    f"{G395H_DIR}/{p}_04101_00001-seg{s:03d}_{d}_uncal.fits"
    for p in ("jw01366004001", "jw01366004002") for s in (1, 2) for d in ("nrs1", "nrs2")
)


@pytest.fixture(scope="module")
def stack():
    mast = fake_mast_server.start()
    model = fake_openai_server.start()
    env = {
        "OPENAI_API_KEY": "sk-test-not-a-real-key",
        "OPENAI_BASE_URL": model.url,
        "ASTER_MAST_BASE_URL": mast.url,
    }
    with aster_gui.launch_gui(env) as gui:
        yield gui, model, mast
    mast.shutdown()
    model.shutdown()


def _agent_calls(model, since: int):
    return [d for d in model.decisions[since:] if "tool" in d]


def _assert_clean_turn(result, model, since):
    assert result["status"] == "complete", result["error"]
    violations = [d["text"] for d in model.decisions[since:]
                  if d.get("text", "").startswith("SCHEMA_VIOLATION")]
    assert not violations, violations


def _rel(gui, paths):
    return sorted(str(p.relative_to(gui.workspace)) for p in paths)


def _mast(mast, since, kind=None, service=None):
    out = mast.log[since:]
    if kind:
        out = [e for e in out if e["kind"] == kind]
    if service:
        out = [e for e in out if e.get("request", {}).get("service") == service]
    return out


def test_gui_serves_aster_with_the_consolidated_tools(stack):
    gui, model, _ = stack
    start = len(model.decisions)
    result = gui.chat("Which JWST observations of WASP-39 b exist in program 1366?")
    _assert_clean_turn(result, model, start)
    offered = {t["function"]["name"] for t in model.requests[start]["tools"]}
    for camel in ("SearchMastJwstObservations", "DownloadMastJwstProducts",
                  "AggregateJwstObservations", "CrossmatchJwstToPlanets",
                  "QueryExoplanetArchive", "GetJwstProgramInfo"):
        assert fake_openai_server.resolve_tool(camel, model.requests[start]["tools"]), camel
    for retired in ("getmastobservationproducts", "downloaddemographicjwstproducts",
                    "getexoplanetparameters", "findexoplanetsbycondition"):
        assert retired not in offered
    system = model.requests[start]["messages"][0]["content"]
    assert "Tool Routing" in system                     # ASTER's prompt reached the model


def test_nl_session_search_then_download_then_list(stack):
    gui, model, mast = stack
    gui.clear()

    # --- 1. search, in plain English -------------------------------------
    d0, m0 = len(model.decisions), len(mast.log)
    r = gui.chat("Which JWST observations of WASP-39 b used NIRSpec G395H in program 1366?")
    _assert_clean_turn(r, model, d0)
    (call,) = _agent_calls(model, d0)
    assert call["camel"] == "SearchMastJwstObservations"
    cone = _mast(mast, m0, service="Mast.Caom.Filtered.Position")
    assert cone, "the search never reached MAST"
    sent = {f["paramName"]: f["values"] for f in cone[0]["request"]["params"]["filters"]}
    assert sent["proposal_id"] == ["1366"] and "NIRSPEC/SLIT" in sent["instrument_name"]
    assert "obsid=102" in r["text"] and "F290LP;G395H" in r["text"]   # shown in the GUI
    assert "CLEAR;PRISM" not in r["text"]                             # grating honoured
    assert not _mast(mast, m0, kind="download") and not gui.files()

    # --- 2. follow-up: "download those" ------------------------------------
    d1, m1 = len(model.decisions), len(mast.log)
    r = gui.chat("Great, download the raw uncal files for those.")
    _assert_clean_turn(r, model, d1)
    (call,) = _agent_calls(model, d1)
    assert call["camel"] == "DownloadMastJwstProducts"
    assert call["arguments"]["planet_name"] == "WASP-39 b"            # carried over
    assert call["arguments"]["filters"] == ["G395H"]
    assert _rel(gui, gui.files()) == G395H_UNCALS
    assert len(_mast(mast, m1, kind="download")) == 8
    assert "Files downloaded: 8" in r["text"]

    from astropy.io import fits
    for path in gui.files():
        hdr = fits.getheader(path)                    # real, readable FITS
        assert hdr["DETECTOR"] in ("NRS1", "NRS2") and hdr["EXSEGTOT"] == 2
    assert not any("_02101_" in p.name for p in gui.files())   # no target acquisition
    manifest = gui.workspace / "mast" / "WASP-39_b" / "manifest.json"
    assert manifest.is_file()

    # --- 3. "list visit 2, don't download" -----------------------------------
    d2, m2 = len(model.decisions), len(mast.log)
    r = gui.chat("For visit 2 only, list the uncal files but don't download anything.")
    _assert_clean_turn(r, model, d2)
    (call,) = _agent_calls(model, d2)
    assert call["arguments"]["list_only"] is True and call["arguments"]["visits"] == [2]
    assert not _mast(mast, m2, kind="download")
    assert "LIST ONLY" in r["text"] and "jw01366004002" in r["text"]
    assert "Visits selected:  ['jw01366004002']" in r["text"]
    assert _rel(gui, gui.files()) == G395H_UNCALS          # nothing new on disk


def test_nl_program_only_download_in_a_new_conversation(stack):
    gui, model, mast = stack
    gui.clear()
    d0, m0 = len(model.decisions), len(mast.log)
    r = gui.chat("Download the raw uncal time series from NIRISS in program 1366.")
    _assert_clean_turn(r, model, d0)
    (call,) = _agent_calls(model, d0)
    assert call["camel"] == "DownloadMastJwstProducts"
    assert "planet_name" not in call["arguments"]          # new conversation: no carry-over
    program = gui.workspace / "mast" / "program_1366"
    got = sorted(p.name for p in program.rglob("*.fits"))
    assert got == [f"jw01366001001_04101_00001-seg00{s}_nis_uncal.fits" for s in (1, 2)]
    assert not _mast(mast, m0, service="Mast.Caom.Filtered.Position")   # no cone: program query


def test_unknown_target_is_reported_not_crashed(stack):
    gui, model, _ = stack
    gui.clear()
    d0 = len(model.decisions)
    r = gui.chat("Which JWST observations of K2-18 b used NIRSpec?")
    _assert_clean_turn(r, model, d0)
    assert "could not resolve" in r["text"].lower() or "no jwst observations" in r["text"].lower()
    assert "Traceback" not in r["text"]


def test_real_model_eval_grader_detects_misrouting():
    """scripts/eval_aster_nl.py grades a REAL model. Run it against the
    stand-in, whose gaps are known (no "second visit" parsing, no archive
    routing), to prove the grader fails what it should and passes the rest."""
    import os
    import subprocess

    model = fake_openai_server.start()
    try:
        proc = subprocess.run(
            [sys.executable, str(aster_gui.REPO / "scripts" / "eval_aster_nl.py"),
             "--base-url", model.url],
            env=dict(os.environ, OPENAI_API_KEY="sk-test-not-a-real-key"),
            capture_output=True, text=True, timeout=600,
        )
    finally:
        model.shutdown()
    out = proc.stdout
    assert proc.returncode == 1, out + proc.stderr
    for name in ("search: program-level", "search: grating, informal",
                 "download: follow-up 'that one'", "download: program-only",
                 "alert: when does a target go public", "alert: what is going public soon",
                 "alert: email on release day"):
        assert f"PASS  [1] {name}" in out, out
    assert "FAIL  [1] list: one visit" in out and "visit 1 files listed too" in out
    assert "FAIL  [1] routing: parameters are not MAST" in out
    assert "7/9 scenario runs passed" in out
