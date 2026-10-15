"""
ASTER Alert — know when JWST data for your planets is about to go public.

How it works
------------
MAST's CAOM catalog lists every *executed* JWST observation, including ones
still under exclusive access. Each row carries ``t_obs_release`` (MJD of the
public release) and ``dataRights`` (``EXCLUSIVE_ACCESS`` / ``PUBLIC``). So the
archive already knows, months ahead, when e.g. a Luque or Batalha sub-Neptune
visit will unlock. ASTER Alert:

  1. pulls every JWST time-series observation from MAST (reusing mast.py's
     pagination-safe ``search_all_jwst_observations``),
  2. cone-matches them to a planet population from the NASA Exoplanet Archive
     (``subneptune`` preset by default, or an explicit planet list),
  3. diffs the result against a JSON state file from the previous run, and
  4. notifies (console / macOS / Slack / email / markdown digest) on:

       NEW          an observation of a watched planet appeared in MAST
       SOON         a proprietary observation will be public within N days
                    (one alert per lead-time threshold, default 30/7/1 d)
       PUBLIC       an observation became public since the last run
       DATE_CHANGED t_obs_release moved by more than half a day

Rows whose CAOM ``filters`` contain the tag grating (default ``G395H``) are
flagged ``patchwork_ready`` so G395H visits stand out from the rest.

What it cannot see: observations that are approved but not yet executed are
not in CAOM. Those live only on STScI's program/visit-status pages.

Run it
------
    python -m aster_toolkit.data_acquisition.alert                 # cron mode
    python -m aster_toolkit.data_acquisition.alert --upcoming      # stateless list
    python -m aster_toolkit.data_acquisition.alert --planets "TOI-270 d,LHS 1140 b"

The first run bootstraps silently (records state, prints one digest of what's
upcoming) so you are not flooded with NEW alerts for the whole archive.
"""
from __future__ import annotations

import argparse
import json
import os
import smtplib
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

from . import mast

MJD_UNIX_EPOCH = 40587.0
DEFAULT_STATE_PATH = "workspace/mast/alert/state.json"
DEFAULT_DIGEST_DIR = "workspace/mast/alert/digests"
ALERT_COLUMNS = list(mast.JWST_OBSERVATION_COLUMNS) + ["t_obs_release", "dataRights"]
PLANET_COLUMNS = ["pl_name", "hostname", "ra", "dec", "pl_rade", "pl_eqt"]
STATE_VERSION = 1


# --------------------------------------------------------------------------
# time helpers
# --------------------------------------------------------------------------
def now_mjd() -> float:
    return time.time() / 86400.0 + MJD_UNIX_EPOCH


def mjd_to_iso(mjd: float | None) -> str:
    if mjd is None:
        return "unknown"
    unix = (mjd - MJD_UNIX_EPOCH) * 86400.0
    return datetime.fromtimestamp(unix, tz=timezone.utc).strftime("%Y-%m-%d")


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # drop NaN


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclass
class AlertConfig:
    population_preset: str | None = "subneptune"
    planets: list[str] = field(default_factory=list)       # overrides preset
    extra_conditions: list[str] = field(default_factory=list)
    transiting_only: bool = True
    instruments: list[str] | None = None                   # None = every TSO mode
    calib_levels: list[int] = field(default_factory=lambda: [3])
    tag_grating: str = "G395H"
    lead_days: tuple[int, ...] = (30, 7, 1)
    match_radius_deg: float = 0.02
    enrich_program_info: bool = True
    program_cache_path: str | None = None

    def planet_conditions(self) -> list[str]:
        if self.planets:
            quoted = ", ".join("'" + p.replace("'", "''").strip() + "'"
                               for p in self.planets if p.strip())
            conds = [f"pl_name IN ({quoted})"]
        else:
            conds = mast._resolve_population_preset(self.population_preset)
            if self.transiting_only:
                conds.append("tran_flag = 1")
        return conds + list(self.extra_conditions)


# --------------------------------------------------------------------------
# snapshot: what MAST says right now
# --------------------------------------------------------------------------
def fetch_snapshot(
    config: AlertConfig,
    *,
    session: requests.Session | None = None,
    reference_mjd: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Return {key: record} for every (planet, observation) pair being watched.

    The key is ``"<pl_name>|<obsid>"`` so a multi-planet system observed in one
    visit raises one alert per planet (each planet's catalog cares).
    """
    t_now = now_mjd() if reference_mjd is None else reference_mjd
    observations, _ = mast.search_all_jwst_observations(
        instruments=config.instruments,
        dataproduct_types=["timeseries"],
        calib_levels=config.calib_levels,
        columns=ALERT_COLUMNS,
        session=session,
    )
    planets = mast.archive_tap_query(
        config.planet_conditions(), columns=PLANET_COLUMNS, session=session
    )
    pairs = mast.crossmatch_observations_to_planets(
        observations, planets, radius_deg=config.match_radius_deg,
        obs_keep_keys=tuple(ALERT_COLUMNS),
    )

    tag = (config.tag_grating or "").upper()
    snapshot: dict[str, dict[str, Any]] = {}
    for row in pairs:
        release = _as_float(row.get("t_obs_release"))
        rights = str(row.get("dataRights") or "").upper() or "UNKNOWN"
        public = rights == "PUBLIC" or (release is not None and release <= t_now)
        key = f"{row.get('pl_name')}|{row.get('obsid')}"
        snapshot[key] = {
            "pl_name": row.get("pl_name"),
            "obsid": str(row.get("obsid")),
            "obs_id": row.get("obs_id"),
            "proposal_id": str(row.get("proposal_id") or ""),
            "proposal_pi": row.get("proposal_pi"),
            "instrument_name": row.get("instrument_name"),
            "filters": row.get("filters"),
            "t_min": _as_float(row.get("t_min")),
            "t_obs_release": release,
            "release_date": mjd_to_iso(release),
            "dataRights": rights,
            "is_public": bool(public),
            "days_until_public": (
                None if public or release is None else round(release - t_now, 1)
            ),
            "patchwork_ready": bool(tag) and tag in str(row.get("filters") or "").upper(),
        }
    return snapshot


# --------------------------------------------------------------------------
# state + diff
# --------------------------------------------------------------------------
def load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as fh:
        state = json.load(fh)
    if state.get("version") != STATE_VERSION:
        raise RuntimeError(
            f"{path} has state version {state.get('version')}, expected "
            f"{STATE_VERSION}. Move it aside to re-bootstrap."
        )
    return state


def save_state(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as fh:
        json.dump({
            "version": STATE_VERSION,
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "records": records,
        }, fh, indent=1, sort_keys=True)
    tmp.replace(path)  # atomic: a crash never leaves half a state file


@dataclass
class AlertEvent:
    kind: str                     # NEW | SOON | PUBLIC | DATE_CHANGED | UPCOMING
    record: dict[str, Any]
    detail: str = ""


def diff_snapshot(
    previous: dict[str, Any] | None,
    snapshot: dict[str, dict[str, Any]],
    lead_days: Iterable[int],
) -> tuple[list[AlertEvent], dict[str, dict[str, Any]]]:
    """Compare a fresh snapshot to the stored state.

    Returns (events, new_records). Records carry bookkeeping
    (``first_seen``, ``alerted_leads``) so each alert fires exactly once.
    On bootstrap (``previous is None``) only an UPCOMING digest is emitted.
    """
    leads = sorted({int(d) for d in lead_days}, reverse=True)
    old = (previous or {}).get("records", {})
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    events: list[AlertEvent] = []
    records: dict[str, dict[str, Any]] = {}

    for key, rec in snapshot.items():
        prior = old.get(key)
        new = dict(rec)
        new["first_seen"] = prior.get("first_seen", today) if prior else today
        alerted = set(prior.get("alerted_leads", [])) if prior else set()
        days = rec["days_until_public"]

        if previous is None:
            if not rec["is_public"]:
                events.append(AlertEvent("UPCOMING", rec))
            # mark leads already inside the window as sent: no backlog spam
            if days is not None:
                alerted |= {d for d in leads if days <= d}
        else:
            if prior is None:
                events.append(AlertEvent(
                    "NEW", rec,
                    "public now" if rec["is_public"]
                    else f"proprietary until {rec['release_date']}",
                ))
                if days is not None:
                    alerted |= {d for d in leads if days <= d}
            else:
                if rec["is_public"] and not prior.get("is_public"):
                    events.append(AlertEvent("PUBLIC", rec))
                old_rel, new_rel = prior.get("t_obs_release"), rec["t_obs_release"]
                if (old_rel is not None and new_rel is not None
                        and abs(old_rel - new_rel) > 0.5):
                    events.append(AlertEvent(
                        "DATE_CHANGED", rec,
                        f"{mjd_to_iso(old_rel)} -> {rec['release_date']}",
                    ))
                    alerted = set()  # re-arm countdown on a moved date
                if days is not None:
                    crossed = [d for d in leads if days <= d and d not in alerted]
                    if crossed:
                        tightest = min(crossed)
                        events.append(AlertEvent(
                            "SOON", rec, f"public in {days:g} d (<= {tightest} d)"
                        ))
                        alerted |= set(crossed)
        new["alerted_leads"] = sorted(alerted)
        records[key] = new

    # observations that vanished from MAST (withdrawn / re-keyed): keep them so
    # a reappearance is not re-announced as NEW, but flag as missing.
    for key, prior in old.items():
        if key not in records:
            records[key] = dict(prior, missing_since=prior.get("missing_since", today))
    return events, records


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
_KIND_ORDER = {"PUBLIC": 0, "SOON": 1, "NEW": 2, "DATE_CHANGED": 3, "UPCOMING": 4}
_KIND_TITLE = {
    "PUBLIC": "Now public — ready to download",
    "SOON": "Going public soon",
    "NEW": "New observations of watched planets",
    "DATE_CHANGED": "Release date changed",
    "UPCOMING": "Proprietary observations on the watchlist",
}


def enrich_with_programs(events: list[AlertEvent], cache_path: str | None) -> None:
    """Attach program title / PI / EAP from STScI (cached). Never fatal."""
    seen: dict[str, dict[str, Any] | None] = {}
    for ev in events:
        pid = ev.record.get("proposal_id") or ""
        if not pid.isdigit():
            continue
        if pid not in seen:
            try:
                seen[pid] = mast.get_jwst_program_info(pid, cache_path=cache_path)
            except Exception:  # network / layout change: alert anyway
                seen[pid] = None
        if seen[pid]:
            ev.record["program_title"] = seen[pid].get("title")
            ev.record["program_pi"] = seen[pid].get("pi")
            ev.record["cycle"] = seen[pid].get("cycle")


def render_digest(events: list[AlertEvent], config: AlertConfig) -> str:
    if not events:
        return ""
    watch = (", ".join(config.planets) if config.planets
             else f"population '{config.population_preset}'")
    lines = [f"# ASTER Alert — {datetime.now(timezone.utc):%Y-%m-%d}",
             f"Watching {watch}; tagging {config.tag_grating or '(none)'}.", ""]
    for kind in sorted({e.kind for e in events}, key=_KIND_ORDER.get):
        group = sorted((e for e in events if e.kind == kind),
                       key=lambda e: (e.record.get("t_obs_release") or 9e9,
                                      e.record["pl_name"]))
        lines.append(f"## {_KIND_TITLE[kind]} ({len(group)})")
        for ev in group:
            r = ev.record
            star = "★ " if r.get("patchwork_ready") else ""
            prog = f"GO {r['proposal_id']}" if r.get("proposal_id") else "program ?"
            pi = r.get("program_pi") or r.get("proposal_pi") or "?"
            title = f" — {r['program_title']}" if r.get("program_title") else ""
            when = (f"public {r['release_date']}" if not r["is_public"]
                    else f"released {r['release_date']}")
            detail = f" [{ev.detail}]" if ev.detail else ""
            lines.append(
                f"- {star}**{r['pl_name']}** · {r.get('instrument_name')} "
                f"{r.get('filters') or ''} · {prog} (PI {pi}){title} · "
                f"{when}{detail} · `{r.get('obs_id')}`"
            )
            if kind == "PUBLIC":
                lines.append(
                    f"  - fetch: `DownloadMastJwstProducts(obsids=\"{r['obsid']}\", "
                    "raw_only=True, list_only=True)`"
                )
        lines.append("")
    lines.append("★ = Patchwork-ready (tag grating in CAOM filters)")
    return "\n".join(lines)


def short_summary(events: list[AlertEvent]) -> str:
    counts: dict[str, int] = {}
    for e in events:
        counts[e.kind] = counts.get(e.kind, 0) + 1
    return ", ".join(f"{n} {k.lower().replace('_', ' ')}"
                     for k, n in sorted(counts.items(), key=lambda kv: _KIND_ORDER[kv[0]]))


# --------------------------------------------------------------------------
# notifiers — each takes (subject, markdown body); failures are reported,
# never fatal, so one dead channel does not swallow the others.
# --------------------------------------------------------------------------
Notifier = Callable[[str, str], None]


def console_notifier(subject: str, body: str) -> None:
    print(body)


def macos_notifier(subject: str, body: str) -> None:
    if sys.platform != "darwin":
        return
    script = (f'display notification {json.dumps(subject)} '
              f'with title "ASTER Alert" sound name "Glass"')
    subprocess.run(["osascript", "-e", script], check=False)


def slack_notifier(webhook_url: str) -> Notifier:
    def _send(subject: str, body: str) -> None:
        text = body.replace("**", "*")  # markdown -> Slack mrkdwn bold
        text = "\n".join(
            f"*{ln[3:]}*" if ln.startswith("## ") else
            f"*{ln[2:]}*" if ln.startswith("# ") else ln
            for ln in text.splitlines()
        )
        resp = requests.post(webhook_url, json={"text": text[:39000]}, timeout=30)
        resp.raise_for_status()
    return _send


def email_notifier(to_addr: str) -> Notifier:
    """SMTP via env: ASTER_ALERT_SMTP_HOST/PORT/USER/PASSWORD/FROM."""
    def _send(subject: str, body: str) -> None:
        host = os.environ["ASTER_ALERT_SMTP_HOST"]
        port = int(os.environ.get("ASTER_ALERT_SMTP_PORT", "587"))
        user = os.environ.get("ASTER_ALERT_SMTP_USER")
        msg = EmailMessage()
        msg["Subject"] = f"[ASTER Alert] {subject}"
        msg["From"] = os.environ.get("ASTER_ALERT_SMTP_FROM", user or "aster-alert")
        msg["To"] = to_addr
        msg.set_content(body)
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.starttls()
            if user:
                smtp.login(user, os.environ["ASTER_ALERT_SMTP_PASSWORD"])
            smtp.send_message(msg)
    return _send


def file_notifier(digest_dir: Path) -> Notifier:
    def _send(subject: str, body: str) -> None:
        digest_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
        (digest_dir / f"aster_alert_{stamp}.md").write_text(body + "\n")
    return _send


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------
def run_alert(
    config: AlertConfig,
    *,
    state_path: Path,
    notifiers: list[Notifier],
    dry_run: bool = False,
    session: requests.Session | None = None,
) -> list[AlertEvent]:
    previous = load_state(state_path)
    snapshot = fetch_snapshot(config, session=session)
    events, records = diff_snapshot(previous, snapshot, config.lead_days)
    if events and config.enrich_program_info:
        enrich_with_programs(events, config.program_cache_path)
    if events:
        subject = short_summary(events)
        body = render_digest(events, config)
        for notify in notifiers:
            try:
                notify(subject, body)
            except Exception as exc:
                print(f"[aster-alert] notifier {getattr(notify, '__name__', notify)} "
                      f"failed: {exc}", file=sys.stderr)
    else:
        print(f"[aster-alert] no changes across {len(snapshot)} watched "
              "(planet, observation) pairs.")
    if not dry_run:
        save_state(state_path, records)
    return events


def list_upcoming(config: AlertConfig, session: requests.Session | None = None) -> str:
    """Stateless: every watched observation not yet public, soonest first."""
    snapshot = fetch_snapshot(config, session=session)
    events = [AlertEvent("UPCOMING", r) for r in snapshot.values() if not r["is_public"]]
    if config.enrich_program_info:
        enrich_with_programs(events, config.program_cache_path)
    return render_digest(events, config) or "Nothing on the watchlist is proprietary."


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="aster-alert", description=__doc__.split("\n\n")[0])
    p.add_argument("--base-dir", default=".", help="ASTER base directory.")
    p.add_argument("--state", default=DEFAULT_STATE_PATH)
    p.add_argument("--preset", default="subneptune",
                   help=f"Population preset: {', '.join(sorted(mast.POPULATION_PRESETS))}.")
    p.add_argument("--planets", default="",
                   help="Comma-separated pl_name list; overrides --preset.")
    p.add_argument("--where", action="append", default=[],
                   help="Extra Exoplanet Archive ADQL condition (repeatable).")
    p.add_argument("--instruments", default="",
                   help="e.g. 'BOTS' or 'NIRSpec,MIRI'. Default: all TSO modes.")
    p.add_argument("--tag-grating", default="G395H")
    p.add_argument("--lead-days", default="30,7,1")
    p.add_argument("--upcoming", action="store_true",
                   help="Print everything still proprietary and exit (no state).")
    p.add_argument("--dry-run", action="store_true", help="Do not write state.")
    p.add_argument("--no-program-info", action="store_true")
    p.add_argument("--macos", action="store_true", help="macOS notification banner.")
    p.add_argument("--slack-webhook", default=os.environ.get("ASTER_ALERT_SLACK_WEBHOOK"))
    p.add_argument("--email-to", default=os.environ.get("ASTER_ALERT_EMAIL_TO"))
    p.add_argument("--quiet", action="store_true", help="No console digest.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    base = Path(args.base_dir).expanduser()
    config = AlertConfig(
        population_preset=args.preset,
        planets=[s.strip() for s in args.planets.split(",") if s.strip()],
        extra_conditions=args.where,
        instruments=[s.strip() for s in args.instruments.split(",") if s.strip()] or None,
        tag_grating=args.tag_grating,
        lead_days=tuple(int(d) for d in args.lead_days.split(",") if d.strip()),
        enrich_program_info=not args.no_program_info,
        program_cache_path=str(base / "workspace" / mast.JWST_PROGRAM_INFO_CACHE),
    )
    if args.upcoming:
        print(list_upcoming(config))
        return 0

    notifiers: list[Notifier] = [] if args.quiet else [console_notifier]
    notifiers.append(file_notifier(base / DEFAULT_DIGEST_DIR))
    if args.macos:
        notifiers.append(macos_notifier)
    if args.slack_webhook:
        notifiers.append(slack_notifier(args.slack_webhook))
    if args.email_to:
        notifiers.append(email_notifier(args.email_to))

    state_path = Path(args.state)
    if not state_path.is_absolute():
        state_path = base / state_path
    run_alert(config, state_path=state_path, notifiers=notifiers, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
