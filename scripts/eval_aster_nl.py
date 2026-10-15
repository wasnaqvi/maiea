#!/usr/bin/env python
"""Does ASTER, with YOUR model, turn plain-English requests into the right
MAST searches and downloads? Measured through the real GUI.

This launches ``run_aster.py`` exactly as you do, talks to it over the GUI's
websocket like the browser, and grades OUTCOMES — what the agent asked MAST
for, which files landed on disk, what the GUI showed — rather than exact
tool-call wording, so any reasonable phrasing a model chooses can pass.

Modes
-----
  default        fake MAST (tests/e2e/fake_mast_server.py, ERS 1366 fixture):
                 deterministic, offline-safe, every scenario incl. downloads.
  --live-mast    real mast.stsci.edu; runs only the search / list-only
                 scenarios (no multi-GB downloads) with looser checks.

Needs your provider credentials as for the GUI (.env next to run_aster.py:
OPENAI_API_KEY, optional ASTER_LLM_MODEL, or ASTER_LLM_PROVIDER=groq ...).

    python scripts/eval_aster_nl.py                  # all scenarios, fake MAST
    python scripts/eval_aster_nl.py --repeat 3       # routing is stochastic
    python scripts/eval_aster_nl.py --live-mast -k search
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests" / "e2e"))
sys.path.insert(0, str(REPO / "tests" / "scripts"))
sys.path.insert(0, str(REPO))

import aster_gui  # noqa: E402

TOOL_CALL_RE = re.compile(r"tool call:[\s\u2502\u2503|]*([a-z_0-9]+)\s*\(", re.I)
G395H_PREFIXES = ("jw01366004001", "jw01366004002")


@dataclass
class Ctx:
    gui: "aster_gui.AsterGui"
    mast: object | None          # fake MAST server, or None in live mode
    text: str = ""
    calls: list[str] = field(default_factory=list)
    mast_since: int = 0
    files_before: set = field(default_factory=set)

    def mast_log(self, kind=None, service=None):
        if self.mast is None:
            return []
        out = self.mast.log[self.mast_since:]
        if kind:
            out = [e for e in out if e["kind"] == kind]
        if service:
            out = [e for e in out if e.get("request", {}).get("service") == service]
        return out

    def new_files(self):
        return sorted(set(map(str, self.gui.files())) - self.files_before)


@dataclass
class Scenario:
    name: str
    turns: list[str]                      # sent in order in ONE conversation
    check: Callable[[Ctx], list[str]]     # returns failure reasons ([] = pass)
    live_ok: bool = False                 # safe against real MAST (no downloads)


def _searched_mast(c: Ctx):
    if c.mast is None:
        return [] if "searchmastjwstobservations" in c.calls else ["no MAST search tool call"]
    hits = c.mast_log(service="Mast.Caom.Filtered.Position") + c.mast_log(service="Mast.Caom.Filtered")
    return [] if hits else ["agent never queried MAST observations"]


def _no_downloads(c: Ctx):
    if c.mast is None:
        return ["files were downloaded"] if c.new_files() else []
    n = len(c.mast_log(kind="download"))
    return [f"{n} files downloaded but none were asked for"] if n else []


def chk_search_ers(c):
    errs = _searched_mast(c) + _no_downloads(c)
    if c.mast is not None and "jw01366" not in c.text:
        errs.append("program 1366 observations not shown to the user")
    return errs


def chk_search_grating(c):
    errs = _searched_mast(c) + _no_downloads(c)
    if not re.search(r"jw01366-o004|obsid=102|G395H", c.text):
        errs.append("G395H observation (jw01366-o004) not reported")
    return errs


def chk_download_followup(c):
    files = [Path(f).name for f in c.new_files()]
    errs = []
    if not files:
        errs.append("nothing downloaded")
    if any("_02101_" in f for f in files):
        errs.append("target-acquisition uncals downloaded (auxiliary-exposure filter in get_observation_products failed)")
    stray = [f for f in files if not f.startswith(G395H_PREFIXES)]
    if stray:
        errs.append(f"downloaded files outside the G395H observation: {stray[:4]}")
    if files and not all(f.endswith("_uncal.fits") for f in files):
        errs.append("non-uncal products downloaded")
    expected = {f"{p}_04101_00001-seg{s:03d}_{d}_uncal.fits"
                for p in G395H_PREFIXES for s in (1, 2) for d in ("nrs1", "nrs2")}
    if files and set(files) != expected:
        errs.append(f"got {len(files)} of the 8 G395H science uncals")
    return errs


def chk_list_visit2(c):
    errs = _no_downloads(c)
    if "jw01366004002" not in c.text:
        errs.append("visit 2 (jw01366004002) files not listed")
    if c.mast is not None and re.search(r"file=jw01366004001_", c.text):
        errs.append("visit 1 files listed too")
    return errs


def chk_program_niriss(c):
    files = [Path(f).name for f in c.new_files()]
    errs = []
    want = {f"jw01366001001_04101_00001-seg00{s}_nis_uncal.fits" for s in (1, 2)}
    if set(files) != want:
        errs.append(f"expected the 2 NIRISS SOSS uncals, got {files[:6]}")
    return errs


def chk_archive_not_mast(c):
    errs = []
    if c.mast_log(kind="invoke"):
        errs.append("a planet-parameter question was routed to MAST")
    if "queryexoplanetarchive" not in c.calls:
        errs.append(f"QueryExoplanetArchive not called (calls: {c.calls})")
    return errs


def _alert_called(c):
    return [] if "asteralert" in c.calls else [f"AsterAlert not called (calls: {c.calls})"]


def chk_alert_when_public(c):
    errs = _alert_called(c) + _no_downloads(c)
    if c.mast is not None:
        import fake_mast_archive as fx
        from aster_toolkit.data_acquisition.alert import mjd_to_iso
        if mjd_to_iso(fx.LOCKED_RELEASE_MJD) not in c.text:
            errs.append("the locked visit's release date was not reported")
    elif not re.search(r"20\d\d-\d\d-\d\d", c.text):
        errs.append("no release date in the answer")
    return errs


def chk_alert_watchlist(c):
    errs = _alert_called(c) + _no_downloads(c)
    if c.mast is not None and "HIP 9618" not in c.text:
        errs.append("HIP 9618 (locked GO 8004 visit) not mentioned")
    return errs


def chk_alert_subscribe(c):
    errs = _alert_called(c)
    path = c.gui.workspace / "mast" / "alert" / "subscriptions.json"
    subs = json.loads(path.read_text())["subscriptions"] if path.exists() else []
    if not any(s["email"] == "astro@example.org" and s["status"] == "pending" for s in subs):
        errs.append("no pending subscription for astro@example.org was saved")
    return errs


SCENARIOS = [
    Scenario("search: program-level",
             ["What JWST data exists for WASP-39b from the ERS program 1366?"],
             chk_search_ers, live_ok=True),
    Scenario("search: grating, informal",
             ["Did anyone observe WASP-39 b with NIRSpec's G395H grating?"],
             chk_search_grating, live_ok=True),
    Scenario("download: follow-up 'that one'",
             ["Did anyone observe WASP-39 b with NIRSpec's G395H grating in program 1366?",
              "Grab the uncalibrated ramps for that observation."],
             chk_download_followup),
    Scenario("list: one visit, no download",
             ["I only need the second G395H visit of WASP-39 b in program 1366 - show me "
              "which uncal files that is, but don't download anything yet."],
             chk_list_visit2, live_ok=True),
    Scenario("download: program-only, instrument mode",
             ["Pull down the NIRISS SOSS raw time-series data from program 1366."],
             chk_program_niriss),
    Scenario("routing: parameters are not MAST",
             ["What's the equilibrium temperature of WASP-39 b?"],
             chk_archive_not_mast, live_ok=True),
    Scenario("alert: when does a target go public",
             ["When is HD-12572b going public?"],
             chk_alert_when_public, live_ok=True),
    Scenario("alert: what is going public soon",
             ["Which sub-Neptunes are about to have their JWST data released?"],
             chk_alert_watchlist, live_ok=True),
    Scenario("alert: email on release day",
             ["Email me at astro@example.org when HD-12572b's JWST data goes public."],
             chk_alert_subscribe, live_ok=True),
]


def run(args) -> int:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")

    mast = None
    env = {k: os.environ[k] for k in os.environ if k.startswith(("OPENAI", "GROQ", "ASTER_", "OLLAMA"))}
    env.pop("OPENAI_BASE_URL", None) if not args.base_url else env.update(OPENAI_BASE_URL=args.base_url)
    if not args.live_mast:
        import fake_mast_server
        mast = fake_mast_server.start()
        env["ASTER_MAST_BASE_URL"] = mast.url
        env["ASTER_ARCHIVE_TAP_URL"] = mast.url + "/TAP/sync"
    else:
        env.pop("ASTER_MAST_BASE_URL", None)
        env.pop("ASTER_ARCHIVE_TAP_URL", None)
    # Never let an eval run send a real release-day email.
    env["ASTER_ALERT_CHECK_SECONDS"] = "0"
    if args.model:
        env["ASTER_LLM_MODEL"] = args.model

    chosen = [s for s in SCENARIOS
              if (not args.live_mast or s.live_ok)
              and (not args.k or args.k.lower() in s.name.lower())]
    results = []
    with aster_gui.launch_gui(env, keep_workdir=args.keep) as gui:
        print(f"ASTER GUI at {gui.url}  workspace: {gui.workspace}")
        print(f"MAST: {'LIVE mast.stsci.edu' if args.live_mast else mast.url}\n")
        for rep in range(args.repeat):
            for sc in chosen:
                gui.clear()
                ctx = Ctx(gui, mast, mast_since=len(mast.log) if mast else 0,
                          files_before=set(map(str, gui.files())))
                t0 = time.time()
                try:
                    for turn in sc.turns:
                        r = gui.chat(turn, timeout=args.timeout)
                        if r["status"] != "complete":
                            raise RuntimeError(r["error"] or r["status"])
                    ctx.text = r["text"]
                    ctx.calls = [m.lower() for m in TOOL_CALL_RE.findall(ctx.text)]
                    errs = sc.check(ctx)
                except Exception as exc:          # noqa: BLE001
                    errs = [f"{type(exc).__name__}: {exc}"]
                ok = not errs
                results.append(ok)
                print(f"{'PASS' if ok else 'FAIL'}  [{rep + 1}] {sc.name}  "
                      f"({time.time() - t0:.0f}s; tools: {', '.join(ctx.calls) or '-'})")
                for e in errs:
                    print(f"        - {e}")
    passed = sum(results)
    print(f"\n{passed}/{len(results)} scenario runs passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-mast", action="store_true",
                    help="use real MAST; search/list scenarios only")
    ap.add_argument("--model", help="override ASTER_LLM_MODEL for this run")
    ap.add_argument("--repeat", type=int, default=1, help="runs per scenario")
    ap.add_argument("-k", help="only scenarios whose name contains this")
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds per turn")
    ap.add_argument("--keep", action="store_true", help="keep the scratch workspace")
    ap.add_argument("--base-url", help=argparse.SUPPRESS)   # self-test hook
    sys.exit(run(ap.parse_args()))
