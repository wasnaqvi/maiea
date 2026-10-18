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
import math
import os
import re
import smtplib
import subprocess
import sys
import threading
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
ALERT_COLUMNS = list(dict.fromkeys(
    list(mast.JWST_OBSERVATION_COLUMNS) + ["t_obs_release", "dataRights"]))
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

    snapshot: dict[str, dict[str, Any]] = {}
    for row in pairs:
        key = f"{row.get('pl_name')}|{row.get('obsid')}"
        snapshot[key] = observation_record(row, t_now, config.tag_grating)
    return snapshot


def observation_record(row: dict[str, Any], t_now: float,
                       tag_grating: str | None = "G395H") -> dict[str, Any]:
    """One CAOM observation row (optionally joined to a planet) -> the record
    every ASTER Alert path uses: snapshot, lookup, watchlist, subscriptions."""
    release = _as_float(row.get("t_obs_release"))
    rights = str(row.get("dataRights") or "").upper() or "UNKNOWN"
    public = rights == "PUBLIC" or (release is not None and release <= t_now)
    tag = (tag_grating or "").upper()
    return {
        "pl_name": row.get("pl_name"),
        "pl_rade": _as_float(row.get("pl_rade")),
        "pl_eqt": _as_float(row.get("pl_eqt")),
        "target_name": row.get("target_name"),
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


def _smtp_tls() -> str:
    port = int(os.environ.get("ASTER_ALERT_SMTP_PORT") or "587")
    return (os.environ.get("ASTER_ALERT_SMTP_TLS") or ("ssl" if port == 465 else "starttls")).lower()


def smtp_missing() -> list[str]:
    """Settings still needed before an email can be sent ([] = ready). A
    plain local relay (TLS 'none') needs only the host; anything else, such
    as Gmail, also needs a login."""
    def unset(name: str) -> bool:
        return not os.environ.get(name, "").strip()
    missing = [n for n in ("ASTER_ALERT_SMTP_HOST",) if unset(n)]
    if _smtp_tls() != "none":
        missing += [n for n in ("ASTER_ALERT_SMTP_USER", "ASTER_ALERT_SMTP_PASSWORD") if unset(n)]
    return missing


def smtp_configured() -> bool:
    return not smtp_missing()


def send_email(to_addr: str, subject: str, body: str) -> None:
    """Send one message over SMTP configured by environment variables:

    ASTER_ALERT_SMTP_HOST (required), _PORT (587), _USER, _PASSWORD, _FROM,
    and _TLS: ``starttls`` (default), ``ssl`` (default when the port is 465)
    or ``none`` (a local relay only — the password would cross in clear).
    """
    missing = smtp_missing()
    if missing:
        raise RuntimeError(f"Email is not configured: set {', '.join(missing)} in .env "
                           "and restart the ASTER GUI.")
    host = os.environ["ASTER_ALERT_SMTP_HOST"].strip()
    port = int(os.environ.get("ASTER_ALERT_SMTP_PORT") or "587")
    tls = _smtp_tls()
    user = os.environ.get("ASTER_ALERT_SMTP_USER", "").strip() or None
    msg = EmailMessage()
    msg["Subject"] = f"[ASTER Alert] {subject}"
    msg["From"] = os.environ.get("ASTER_ALERT_SMTP_FROM") or user or "aster-alert@localhost"
    msg["To"] = to_addr
    msg.set_content(body)
    smtp_cls = smtplib.SMTP_SSL if tls == "ssl" else smtplib.SMTP
    with smtp_cls(host, port, timeout=30) as smtp:
        if tls == "starttls":
            smtp.starttls()
        if user:
            smtp.login(user, os.environ.get("ASTER_ALERT_SMTP_PASSWORD", ""))
        smtp.send_message(msg)


def email_notifier(to_addr: str) -> Notifier:
    """SMTP via env; see ``send_email``."""
    def _send(subject: str, body: str) -> None:
        send_email(to_addr, subject, body)
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


# --------------------------------------------------------------------------
# one target: "when is HD-12572b going public?"
# --------------------------------------------------------------------------
_PLANET_SUFFIX_RE = re.compile(r"^(.*?\d)[\s_-]*([b-i])$", re.IGNORECASE)


def _name_variants(name: str) -> list[str]:
    """'HD-12572b' -> ['HD-12572b', 'HD-12572 b', 'HD 12572 b', 'HD-12572',
    'HD 12572']. MAST's resolver knows some spellings and not others (it
    resolves 'HD-12572b' and 'HD 12572' but not 'HD-12572'), so try several."""
    base = " ".join(str(name or "").split())
    out = [base]
    m = _PLANET_SUFFIX_RE.match(base)
    if m:
        host = m[1].rstrip(" -_")
        letter = m[2].lower()
        out += [f"{host} {letter}", f"{host.replace('-', ' ')} {letter}",
                host, host.replace("-", " ")]
    else:
        out.append(base.replace("-", " "))
    return list(dict.fromkeys(v for v in out if v))


def resolve_target(name: str, *, session: requests.Session | None = None
                   ) -> tuple[float, float, str]:
    """(ra, dec, the spelling that resolved). MAST's name resolver first,
    then the Exoplanet Archive by planet or host name."""
    variants = _name_variants(name)
    if not variants:
        raise ValueError("A target name is required.")
    for v in variants:
        try:
            ra, dec = mast.resolve_target_coordinates(v, session=session)
            return ra, dec, v
        except Exception:
            continue
    for v in variants:
        quoted = v.replace("'", "''")
        try:
            rows = mast.archive_tap_query(
                [f"pl_name = '{quoted}' OR hostname = '{quoted}'"],
                columns=["pl_name", "ra", "dec"], session=session)
        except Exception:
            continue
        for row in rows:
            ra, dec = _as_float(row.get("ra")), _as_float(row.get("dec"))
            if ra is not None and dec is not None:
                return ra, dec, v
    raise ValueError(f"Could not resolve {name!r} to sky coordinates "
                     f"(tried {', '.join(repr(v) for v in variants)}).")


def planets_near(ra: float, dec: float, radius_deg: float = 0.02, *,
                 session: requests.Session | None = None) -> list[dict[str, Any]]:
    """Exoplanet Archive planets within a small RA/Dec box (the archive's TAP
    service refuses CONTAINS/CIRCLE with HTTP 403). Never fatal."""
    dra = radius_deg / max(math.cos(math.radians(dec)), 1e-6)
    conds = [f"ra between {ra - dra:.6f} and {ra + dra:.6f}",
             f"dec between {dec - radius_deg:.6f} and {dec + radius_deg:.6f}"]
    try:
        return mast.archive_tap_query(conds, columns=PLANET_COLUMNS, session=session)
    except Exception:
        return []


def lookup_target(name: str, *, radius_deg: float = 0.02, tag_grating: str = "G395H",
                  session: requests.Session | None = None,
                  t_now: float | None = None) -> dict[str, Any]:
    """Every executed JWST time-series observation at a target, public or not."""
    t_now = now_mjd() if t_now is None else t_now
    ra, dec, resolved_as = resolve_target(name, session=session)
    rows = mast.search_jwst_observations(
        None, ra=ra, dec=dec, radius_deg=radius_deg,
        dataproduct_types=["timeseries"], calib_levels=[3],
        columns=ALERT_COLUMNS, session=session,
    )
    planets = planets_near(ra, dec, radius_deg, session=session)
    seen: set[str] = set()
    observations = []
    for row in rows:
        rec = observation_record(row, t_now, tag_grating)
        if rec["obsid"] in seen:
            continue
        seen.add(rec["obsid"])
        observations.append(rec)
    observations.sort(key=lambda r: (r["t_obs_release"] or 9e9, r["obs_id"] or ""))
    return {"target": name, "resolved_as": resolved_as, "ra": ra, "dec": dec,
            "planets": planets, "observations": observations, "t_now": t_now}


def fetch_observations_by_obsid(obsids: Iterable[str], *, t_now: float | None = None,
                                session: requests.Session | None = None
                                ) -> dict[str, dict[str, Any]]:
    """Current CAOM record (release date, access) for specific obsids."""
    t_now = now_mjd() if t_now is None else t_now
    ids = list(dict.fromkeys(str(o) for o in obsids if str(o).strip()))
    out: dict[str, dict[str, Any]] = {}
    for i in range(0, len(ids), 100):
        request = {
            "service": "Mast.Caom.Filtered", "format": "json",
            "pagesize": 500, "page": 1,
            "params": {"columns": ",".join(ALERT_COLUMNS),
                       "filters": [{"paramName": "obsid", "values": ids[i:i + 100]}]},
        }
        for row in mast._extract_rows(mast._mast_query(request, session=session)):
            out[str(row.get("obsid"))] = observation_record(row, t_now)
    return out


def _program_infos(pids: Iterable[str], cache_path: str | os.PathLike[str] | None
                   ) -> dict[str, dict[str, Any]]:
    infos: dict[str, dict[str, Any]] = {}
    for pid in dict.fromkeys(p for p in pids if p and str(p).isdigit()):
        try:
            infos[pid] = mast.get_jwst_program_info(pid, cache_path=cache_path)
        except Exception:  # network / page layout: report without it
            continue
    return infos


def _days_phrase(days: float) -> str:
    d = int(round(days))
    return "today" if d <= 0 else "tomorrow" if d == 1 else f"in {d} days"


def _obs_line(r: dict[str, Any], infos: dict[str, dict[str, Any]]) -> str:
    star = "★ " if r.get("patchwork_ready") else ""
    if r["is_public"]:
        when = f"PUBLIC since {r['release_date']}"
    elif r["t_obs_release"] is None:
        when = "LOCKED, release date unknown"
    else:
        when = f"LOCKED until {r['release_date']} ({_days_phrase(r['days_until_public'])})"
    info = infos.get(r.get("proposal_id") or "", {})
    pi = info.get("pi") or r.get("proposal_pi") or "?"
    observed = f" · observed {mjd_to_iso(r['t_min'])}" if r.get("t_min") else ""
    return (f"  {star}{when} · GO {r.get('proposal_id') or '?'} (PI {pi}) · "
            f"{r.get('instrument_name')} {r.get('filters') or ''}{observed} · "
            f"obsid {r['obsid']} · {r.get('obs_id')}")


def _programs_footer(records: list[dict[str, Any]],
                     infos: dict[str, dict[str, Any]]) -> list[str]:
    """Each program's title once, rather than on every observation line."""
    pids = list(dict.fromkeys(r.get("proposal_id") for r in records
                              if infos.get(r.get("proposal_id") or "", {}).get("title")))
    if not pids:
        return []
    return ["", "Programs:"] + [
        f"  GO {pid} (Cycle {infos[pid].get('cycle') or '?'}): {infos[pid]['title']}"
        for pid in pids]


def _planet_label(p: dict[str, Any]) -> str:
    bits = []
    rade, eqt = _as_float(p.get("pl_rade")), _as_float(p.get("pl_eqt"))
    if rade is not None:
        bits.append(f"{rade:.2f} R⊕")
    if eqt is not None:
        bits.append(f"{eqt:.0f} K")
    return f"{p.get('pl_name')}" + (f" ({', '.join(bits)})" if bits else "")


def render_lookup(result: dict[str, Any], cache_path: str | None = None) -> str:
    obs = result["observations"]
    target = result["target"]
    locked = [r for r in obs if not r["is_public"]]
    if locked:
        nxt = locked[0]
        head = (f"ASTER Alert · {target}: next release {nxt['release_date']} "
                f"({_days_phrase(nxt['days_until_public'] or 0)}); "
                f"{len(locked)} of {len(obs)} JWST time-series observations still locked")
    elif obs:
        last = max((r for r in obs if r["t_obs_release"] is not None),
                   key=lambda r: r["t_obs_release"], default=obs[-1])
        head = (f"ASTER Alert · {target}: all {len(obs)} JWST time-series observations "
                f"are already public (latest released {last['release_date']})")
    else:
        head = (f"ASTER Alert · {target}: no executed JWST time-series observations "
                "in MAST")
    lines = [head, ""]
    lines.append(f"Resolved '{result['resolved_as']}' to RA {result['ra']:.5f}, "
                 f"Dec {result['dec']:+.5f}.")
    if result["planets"]:
        lines.append("Planets at this position (NASA Exoplanet Archive): "
                     + ", ".join(_planet_label(p) for p in result["planets"]))
    infos = _program_infos((r.get("proposal_id") for r in obs), cache_path)
    if obs:
        lines += ["", "JWST time-series observations, by release date:"]
        lines += [_obs_line(r, infos) for r in obs]
        lines += _programs_footer(obs, infos)
    else:
        lines.append("Approved visits that have not been executed yet are not in "
                     "MAST, so ASTER Alert cannot date them.")
    lines.append("")
    if locked:
        lines.append(f'Email on release day: AsterAlert(action="subscribe", '
                     f'target="{target}", email="you@example.com")')
    public = [r for r in obs if r["is_public"]]
    if public:
        lines.append(f'Fetch the latest public visit: DownloadMastJwstProducts('
                     f'obsids="{public[-1]["obsid"]}", raw_only=True, list_only=True)')
    if any(r.get("patchwork_ready") for r in obs):
        lines.append("★ = NIRSpec G395H (Patchwork-ready)")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# a population: "what sub-Neptunes went / are going public recently?"
# --------------------------------------------------------------------------
def watchlist_report(config: AlertConfig, *, days_back: float = 60, days_ahead: float = 120,
                     session: requests.Session | None = None,
                     t_now: float | None = None) -> dict[str, Any]:
    t_now = now_mjd() if t_now is None else t_now
    snap = fetch_snapshot(config, session=session, reference_mjd=t_now)
    by_planet: dict[str, list[dict[str, Any]]] = {}
    for r in snap.values():
        by_planet.setdefault(r["pl_name"], []).append(r)

    def released_before(planet: str, mjd: float) -> bool:
        return any(o["is_public"] and o["t_obs_release"] is not None
                   and o["t_obs_release"] < mjd for o in by_planet[planet])

    released, upcoming, later = [], [], []
    for r in snap.values():
        rel = r["t_obs_release"]
        if rel is None:
            continue
        if r["is_public"] and t_now - days_back <= rel <= t_now:
            r["first_public_data"] = not released_before(r["pl_name"], t_now - days_back)
            released.append(r)
        elif not r["is_public"] and rel <= t_now + days_ahead:
            r["first_public_data"] = not any(o["is_public"] for o in by_planet[r["pl_name"]])
            upcoming.append(r)
        elif not r["is_public"]:
            later.append(r)
    released.sort(key=lambda r: (-r["t_obs_release"], r["pl_name"]))
    upcoming.sort(key=lambda r: (r["t_obs_release"], r["pl_name"]))
    return {"released": released, "upcoming": upcoming, "later": later,
            "days_back": days_back, "days_ahead": days_ahead, "t_now": t_now,
            "watching": (", ".join(config.planets) if config.planets
                         else f"the '{config.population_preset}' population"),
            "n_watched": len(snap)}


def _grouped(records: list[dict[str, Any]], infos: dict[str, dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    order: list[str] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        if r["pl_name"] not in groups:
            order.append(r["pl_name"])
        groups.setdefault(r["pl_name"], []).append(r)
    for planet in order:
        rs = groups[planet]
        first = " ✦ first public JWST time-series data" if rs[0].get("first_public_data") else ""
        lines.append(f"- {_planet_label(rs[0])}{first}")
        lines += ["  " + _obs_line(r, infos) for r in rs]
    return lines


def render_watchlist(report: dict[str, Any], cache_path: str | None = None) -> str:
    rel, up, later = report["released"], report["upcoming"], report["later"]
    n_rel_p = len({r["pl_name"] for r in rel})
    n_up_p = len({r["pl_name"] for r in up})
    head = (f"ASTER Alert · {n_up_p} planet(s) go public in the next "
            f"{report['days_ahead']:g} days; {n_rel_p} went public in the last "
            f"{report['days_back']:g} days")
    infos = _program_infos((r.get("proposal_id") for r in rel + up), cache_path)
    lines = [head, "", f"Watching {report['watching']} "
             f"({report['n_watched']} planet-observation pairs in MAST).", ""]
    lines.append(f"## Going public in the next {report['days_ahead']:g} days ({len(up)})")
    lines += _grouped(up, infos) or ["- none"]
    lines += ["", f"## Went public in the last {report['days_back']:g} days ({len(rel)})"]
    lines += _grouped(rel, infos) or ["- none"]
    lines += _programs_footer(up + rel, infos)
    if later:
        last = max(r["t_obs_release"] for r in later)
        lines += ["", f"Later: {len(later)} more locked observation(s), the last on "
                      f"{mjd_to_iso(last)} (raise days_ahead to list them)."]
    lines += ["", "★ = NIRSpec G395H (Patchwork-ready) · ✦ = the planet's first public "
                  "JWST time-series data",
              "Only executed visits are in MAST; approved-but-unobserved visits are not."]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# email on release day
# --------------------------------------------------------------------------
SUBSCRIPTIONS_RELPATH = "mast/alert/subscriptions.json"
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SUBS_LOCK = threading.RLock()   # tool calls and the GUI notifier thread


def subscriptions_path(base_directory: str | os.PathLike[str] | None) -> Path:
    return Path(base_directory or ".") / SUBSCRIPTIONS_RELPATH


def load_subscriptions(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    with Path(path).open() as fh:
        return list(json.load(fh).get("subscriptions", []))


def save_subscriptions(path: Path, subs: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as fh:
        json.dump({"version": 1, "subscriptions": subs}, fh, indent=1)
    tmp.replace(path)


def add_subscriptions(path: Path, *, email: str, target: str, planets: list[str],
                      records: list[dict[str, Any]],
                      infos: dict[str, dict[str, Any]] | None = None
                      ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Subscribe ``email`` to each record's release. Returns (added, already)."""
    infos = infos or {}
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _SUBS_LOCK:
        subs = load_subscriptions(path)
        by_id = {s["id"]: s for s in subs}
        added, already = [], []
        for r in records:
            sid = f"{r['obsid']}:{email.lower()}"
            if sid in by_id and by_id[sid].get("status") == "pending":
                already.append(by_id[sid])
                continue
            info = infos.get(r.get("proposal_id") or "", {})
            sub = {
                "id": sid, "email": email, "target": target, "planets": planets,
                "obsid": r["obsid"], "obs_id": r.get("obs_id"),
                "proposal_id": r.get("proposal_id"),
                "pi": info.get("pi") or r.get("proposal_pi"),
                "program_title": info.get("title"),
                "instrument_name": r.get("instrument_name"), "filters": r.get("filters"),
                "patchwork_ready": r.get("patchwork_ready", False),
                "t_obs_release": r.get("t_obs_release"),
                "release_date": r.get("release_date"),
                "status": "pending", "created_utc": stamp,
            }
            subs = [s for s in subs if s["id"] != sid] + [sub]
            added.append(sub)
        save_subscriptions(path, subs)
    return added, already


def last_subscription_email(path: Path) -> str | None:
    """The address on the most recent subscription, if any."""
    subs = load_subscriptions(path)
    latest = max(subs, key=lambda s: s.get("created_utc") or "", default=None)
    return latest.get("email") if latest else None


def remove_subscriptions(path: Path, *, subscription_id: str | None = None,
                         target: str | None = None) -> list[dict[str, Any]]:
    with _SUBS_LOCK:
        subs = load_subscriptions(path)

        def hit(s: dict[str, Any]) -> bool:
            if subscription_id:
                return s["id"] == subscription_id
            return bool(target) and str(s.get("target", "")).lower() == target.lower()

        removed = [s for s in subs if hit(s)]
        save_subscriptions(path, [s for s in subs if not hit(s)])
    return removed


def _release_email(sub: dict[str, Any]) -> tuple[str, str]:
    who = ", ".join(sub.get("planets") or []) or sub.get("target") or "your target"
    star = " (NIRSpec G395H, Patchwork-ready)" if sub.get("patchwork_ready") else ""
    # ASCII only: a non-ASCII subject is RFC 2047-encoded, and folding the
    # encoded words left a double space in some clients.
    subject = f"{who}: JWST data is public - GO {sub.get('proposal_id')}"
    body = "\n".join([
        f"JWST time-series data for {who} became public on {sub.get('release_date')}.",
        "",
        f"Target you asked about: {sub.get('target')}",
        f"Program: GO {sub.get('proposal_id')} (PI {sub.get('pi') or '?'})"
        + (f" — {sub['program_title']}" if sub.get("program_title") else ""),
        f"Instrument: {sub.get('instrument_name')} {sub.get('filters') or ''}{star}",
        f"Observation: {sub.get('obs_id')} (MAST obsid {sub.get('obsid')})",
        "",
        "To fetch it, ask ASTER, or run in the ASTER GUI:",
        f'  DownloadMastJwstProducts(obsids="{sub.get("obsid")}", raw_only=True, list_only=True)',
        "",
        "— ASTER Alert. You subscribed in the ASTER GUI; reply to nobody, "
        f'unsubscribe there with AsterAlert(action="unsubscribe", subscription_id="{sub["id"]}").',
    ])
    return subject, body


def deliver_due_subscriptions(path: Path, *, now: float | None = None,
                              send: Callable[[str, str, str], None] | None = None,
                              session: requests.Session | None = None) -> list[str]:
    """Email every pending subscription whose observation MAST now reports
    public, and follow release dates that moved. Returns log lines.

    Every pending observation is re-read from MAST on each pass (one request
    per 100 obsids), so a date moved EARLIER is caught too, not only one
    that slipped. A failed send stays pending and is retried next pass.
    """
    now = now_mjd() if now is None else now
    send = send or send_email
    log: list[str] = []
    with _SUBS_LOCK:
        subs = load_subscriptions(path)
        pending = [s for s in subs if s.get("status") == "pending"]
        if not pending:
            return log
        try:
            current = fetch_observations_by_obsid([s["obsid"] for s in pending],
                                                  t_now=now, session=session)
        except Exception as exc:
            return [f"MAST unreachable ({exc}); will retry."]
        changed = False
        for s in pending:
            rec = current.get(str(s["obsid"]))
            if rec is None:
                continue
            if not rec["is_public"]:
                new_rel = rec["t_obs_release"]
                if new_rel is not None and (s.get("t_obs_release") is None
                                            or abs(new_rel - s["t_obs_release"]) > 0.5):
                    log.append(f"{s['obs_id']}: release moved {s.get('release_date')} "
                               f"-> {rec['release_date']}")
                    s["rescheduled_from"] = s.get("release_date")
                    s["t_obs_release"], s["release_date"] = new_rel, rec["release_date"]
                    changed = True
                continue
            if rec["t_obs_release"] is not None:
                s["t_obs_release"], s["release_date"] = rec["t_obs_release"], rec["release_date"]
            subject, body = _release_email(s)
            try:
                send(s["email"], subject, body)
            except Exception as exc:
                s["last_error"] = str(exc)
                log.append(f"{s['obs_id']}: email to {s['email']} failed ({exc}); will retry.")
                changed = True
                continue
            s["status"] = "sent"
            s["sent_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            s.pop("last_error", None)
            log.append(f"emailed {s['email']}: {subject}")
            changed = True
        if changed:
            save_subscriptions(path, subs)
    return log


def start_subscription_notifier(base_directory: str | os.PathLike[str], *,
                                interval_seconds: float = 3600.0,
                                log: Callable[[str], None] = print) -> threading.Event:
    """Background thread that delivers release-day emails while the process
    (the ASTER GUI) runs. Checks at start-up, so anything that went public
    while the GUI was closed goes out on the next launch. Set the returned
    Event to stop it."""
    stop = threading.Event()
    path = subscriptions_path(base_directory)

    def loop() -> None:
        while not stop.is_set():
            try:
                for line in deliver_due_subscriptions(path):
                    log(f"[aster-alert] {line}")
            except Exception as exc:  # never let the thread die
                log(f"[aster-alert] notifier error: {exc}")
            stop.wait(interval_seconds)

    threading.Thread(target=loop, name="aster-alert-notifier", daemon=True).start()
    return stop


# --------------------------------------------------------------------------
# the agent tool
# --------------------------------------------------------------------------
_ACTIONS = ("lookup", "watchlist", "check", "subscribe", "unsubscribe",
            "subscriptions", "test_email")


def _as_int(value: Any, default: int) -> int:
    try:
        return int(float(value)) if value not in (None, "") else default
    except (TypeError, ValueError):
        raise ValueError(f"Expected a number of days, got {value!r}.") from None


class AsterAlert(mast.BaseTool):
    """
    ASTER Alert: when does a planet's JWST data go public, what was just
    released, and email me on release day.

    USE WHEN the user says "ASTER Alert", asks when a target's JWST data
    becomes public, what was released recently or is about to be, or wants
    an email when it unlocks. MAST lists every EXECUTED observation with its
    release date, proprietary ones included.

    action (default: lookup if `target` is given, else watchlist):
      lookup        every JWST time-series observation of `target` (any name
                    MAST resolves: 'HD-12572b', 'HIP 9618 b', 'TOI-1801'),
                    public or locked, with release dates and programs.
      watchlist     a population (sub-Neptune preset, or `planets`): what
                    went public in the last `days_back` days and what goes
                    public in the next `days_ahead`.
      check         what changed since the previous check (NEW / SOON /
                    PUBLIC / DATE_CHANGED).
      subscribe     email `email` on the day each locked observation of
                    `target` (or one `obsid`) goes public.
      unsubscribe   by `subscription_id`, or every subscription for `target`.
      subscriptions list email subscriptions.
      test_email    send a test email to check the SMTP settings.

    Cannot see approved visits that have not been executed (not in MAST).
    Planet parameters: QueryExoplanetArchive. Files: DownloadMastJwstProducts.
    """

    action: str | None = mast.OptionalRuntimeField(
        description="lookup | watchlist | check | subscribe | unsubscribe | "
                    "subscriptions | test_email.")
    target: str | None = mast.OptionalRuntimeField(
        description="Planet or host star, any MAST-resolvable name, e.g. "
                    "'HD-12572b', 'HIP 9618 b', 'TOI-1801 b'.")
    planets: list | str | None = mast.OptionalRuntimeField(
        description="watchlist/check: Exoplanet Archive pl_names to watch "
                    "instead of the preset population.")
    preset: str | None = mast.OptionalRuntimeField(
        description="watchlist/check population preset (default 'subneptune').")
    days_back: int | str | None = mast.OptionalRuntimeField(
        description="watchlist: releases in the last N days (default 60).")
    days_ahead: int | str | None = mast.OptionalRuntimeField(
        description="watchlist: releases in the next N days (default 120).")
    email: str | None = mast.OptionalRuntimeField(
        description="subscribe/test_email address, any provider. Omit to reuse the "
                    "last address the user gave.")
    obsid: str | None = mast.OptionalRuntimeField(
        description="subscribe: a single MAST obsid instead of all of target's.")
    subscription_id: str | None = mast.OptionalRuntimeField(
        description="unsubscribe: an id from action='subscriptions'.")
    base_directory: str = mast.StateField()

    # ------------------------------------------------------------------
    def _base(self) -> Path:
        return Path(self.base_directory or ".")

    def _cache(self) -> str:
        return str(self._base() / mast.JWST_PROGRAM_INFO_CACHE)

    def _config(self) -> AlertConfig:
        planets = [p.strip() for p in (mast._as_list(self.planets) or []) if str(p).strip()]
        return AlertConfig(population_preset=(mast._clean_str(self.preset) or "subneptune"),
                           planets=planets, program_cache_path=self._cache())

    def _email(self) -> str:
        """The address named in this call; else the last one used for a
        subscription in this workspace (so the GUI asks only once); else
        ASTER_ALERT_EMAIL_TO from .env."""
        email = (mast._clean_str(self.email)
                 or last_subscription_email(subscriptions_path(self._base()))
                 or os.environ.get("ASTER_ALERT_EMAIL_TO", "").strip())
        if not email:
            raise ValueError("Which email address should ASTER Alert use? Ask the user, "
                             "then call again with email='...'.")
        if not _EMAIL_RE.match(email):
            raise ValueError(f"{email!r} does not look like an email address.")
        return email

    def _run(self) -> str:
        target = mast._clean_str(self.target)
        action = (mast._clean_str(self.action) or ("lookup" if target else "watchlist")).lower()
        if action not in _ACTIONS:
            raise ValueError(f"Unknown action {action!r}; use one of {', '.join(_ACTIONS)}.")
        return getattr(self, f"_do_{action}")(target)

    def _do_lookup(self, target: str | None) -> str:
        if not target:
            raise ValueError("lookup needs a target, e.g. target='HD-12572b'.")
        return render_lookup(lookup_target(target), self._cache())

    def _do_watchlist(self, target: str | None) -> str:
        report = watchlist_report(self._config(),
                                  days_back=_as_int(self.days_back, 60),
                                  days_ahead=_as_int(self.days_ahead, 120))
        return render_watchlist(report, self._cache())

    def _do_check(self, target: str | None) -> str:
        captured: dict[str, str] = {}

        def capture(subject: str, body: str) -> None:
            captured.update(subject=subject, body=body)

        state = self._base() / "mast/alert/state.json"
        first = not state.exists()
        events = run_alert(self._config(), state_path=state, notifiers=[
            capture, file_notifier(self._base() / "mast/alert/digests")])
        if first:
            locked = sum(e.kind == "UPCOMING" for e in events)
            return (f"ASTER Alert · first check: recorded the watchlist; {locked} "
                    "observation(s) still locked. Later checks report what changed.\n\n"
                    + captured.get("body", ""))
        if not events:
            return "ASTER Alert · no changes since the last check."
        return f"ASTER Alert · {short_summary(events)}\n\n{captured.get('body', '')}"

    def _do_subscribe(self, target: str | None) -> str:
        email = self._email()
        obsid = mast._clean_str(self.obsid)
        if not target and not obsid:
            raise ValueError("subscribe needs a target (or an obsid).")
        if target:
            result = lookup_target(target)
            planets = [p["pl_name"] for p in result["planets"]]
            records = [r for r in result["observations"]
                       if not r["is_public"] and (not obsid or r["obsid"] == obsid)]
            n_all = len(result["observations"])
        else:
            records = [r for r in fetch_observations_by_obsid([obsid]).values()
                       if not r["is_public"]]
            planets, target, n_all = [], f"obsid {obsid}", 1
        if not records:
            return (f"ASTER Alert · nothing to subscribe to: all {n_all} JWST time-series "
                    f"observation(s) of {target} are already public (or none exist). "
                    "Look them up with action='lookup'.")
        infos = _program_infos((r.get("proposal_id") for r in records), self._cache())
        added, already = add_subscriptions(
            subscriptions_path(self._base()), email=email, target=target,
            planets=planets, records=records, infos=infos)
        subs = sorted(added + already, key=lambda s: s.get("t_obs_release") or 9e9)
        head = (f"ASTER Alert · subscribed {email}: {len(subs)} release(s) for {target}, "
                f"the first on {subs[0]['release_date']}")
        lines = [head, ""]
        for s in subs:
            mark = "" if s in added else " (already subscribed)"
            lines.append(f"  - {s['release_date']} · GO {s['proposal_id']} · "
                         f"{s['instrument_name']} {s.get('filters') or ''} · {s['obs_id']}{mark}")
        lines += ["", "Emails go out from the ASTER GUI's background notifier, which "
                      "checks MAST every hour while the GUI runs (and at start-up, so a "
                      "release while it was closed is emailed on the next launch). If a "
                      "release date moves, the email follows it."]
        if not smtp_configured():
            lines += ["", "WARNING: email is not configured yet, so nothing can be sent. "
                          f"Set {', '.join(smtp_missing())} in .env and restart the GUI; "
                          "then try action='test_email'."]
        return "\n".join(lines)

    def _do_unsubscribe(self, target: str | None) -> str:
        sid = mast._clean_str(self.subscription_id)
        if not sid and not target:
            raise ValueError("unsubscribe needs a subscription_id or a target.")
        removed = remove_subscriptions(subscriptions_path(self._base()),
                                       subscription_id=sid, target=target)
        if not removed:
            return "ASTER Alert · no matching subscription."
        return (f"ASTER Alert · removed {len(removed)} subscription(s)\n\n"
                + "\n".join(f"  - {s['id']} ({s['target']}, {s['release_date']})"
                            for s in removed))

    def _do_subscriptions(self, target: str | None) -> str:
        subs = load_subscriptions(subscriptions_path(self._base()))
        pending = [s for s in subs if s.get("status") == "pending"]
        sent = [s for s in subs if s.get("status") == "sent"]
        if not subs:
            return "ASTER Alert · no email subscriptions yet."
        lines = [f"ASTER Alert · {len(pending)} email(s) waiting, {len(sent)} sent", ""]
        for s in sorted(pending, key=lambda s: s.get("t_obs_release") or 9e9):
            err = f" · last error: {s['last_error']}" if s.get("last_error") else ""
            lines.append(f"  - waiting: {s['release_date']} · {s['target']} · GO "
                         f"{s['proposal_id']} · {s['email']} · id {s['id']}{err}")
        for s in sent:
            lines.append(f"  - sent {s.get('sent_utc', '')[:10]}: {s['target']} · "
                         f"GO {s['proposal_id']} · {s['email']}")
        if not smtp_configured():
            lines += ["", f"WARNING: email is not configured ({', '.join(smtp_missing())} unset)."]
        return "\n".join(lines)

    def _do_test_email(self, target: str | None) -> str:
        email = self._email()
        send_email(email, "test email",
                   "This is a test from ASTER Alert. Release-day emails will look "
                   "like this one's sender and subject prefix.")
        return f"ASTER Alert · test email sent to {email}"


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
    p.add_argument("--tolerate-outage", action="store_true",
                   help="If MAST or the Exoplanet Archive is down after retries, "
                        "emit a ::warning:: and exit 0 (state is left untouched).")
    p.add_argument("--deliver", action="store_true",
                   help="Send due release-day emails for GUI subscriptions and exit "
                        "(for cron, when the GUI is not running).")
    p.add_argument("--no-program-info", action="store_true")
    p.add_argument("--macos", action="store_true", help="macOS notification banner.")
    p.add_argument("--slack-webhook", default=os.environ.get("ASTER_ALERT_SLACK_WEBHOOK"))
    p.add_argument("--email-to", default=os.environ.get("ASTER_ALERT_EMAIL_TO"))
    p.add_argument("--quiet", action="store_true", help="No console digest.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    base = Path(args.base_dir).expanduser()
    try:  # same settings the GUI reads (run_aster.py loads .env too)
        from dotenv import load_dotenv
        load_dotenv(base / ".env")
    except ImportError:  # e.g. the GitHub workflow, which uses secrets instead
        pass
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
    if args.deliver:
        path = subscriptions_path(base / "workspace")
        for line in deliver_due_subscriptions(path):
            print(f"[aster-alert] {line}")
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
    try:
        run_alert(config, state_path=state_path, notifiers=notifiers, dry_run=args.dry_run)
    except requests.RequestException as exc:
        if not args.tolerate_outage:
            raise
        first = str(exc).splitlines()[0]
        print(f"::warning title=ASTER Alert skipped::upstream outage, no state written: {first}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
