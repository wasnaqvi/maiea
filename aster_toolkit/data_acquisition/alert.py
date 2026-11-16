"""
ASTER Alert — an email on the day a JWST observation goes public.

It does something only when the user asks in the ASTER GUI:

  1. "When does HD-12572b go public?"  One MAST query: every *executed* JWST
     time-series observation of the target carries ``t_obs_release`` (MJD of
     the public release) and ``dataRights``, proprietary ones included.
  2. "Email me when it does."  The release date and the user's address are
     stored in ``workspace/mast/alert/subscriptions.json``.
  3. On that date the email is sent over SMTP (``ASTER_ALERT_SMTP_*`` in
     ``.env``) and the subscription is marked sent. MAST is not asked again.

The GUI sends due emails from a background thread while it runs, and at
start-up for any date that passed while it was closed.
``python -m aster_toolkit.data_acquisition.alert`` sends them without the GUI.

Nothing here runs on a schedule of its own, and nothing is sent to anyone
who did not subscribe. Approved visits that have not been executed are not in
MAST, so they have no release date to subscribe to.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import smtplib
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
ALERT_COLUMNS = list(dict.fromkeys(
    list(mast.JWST_OBSERVATION_COLUMNS) + ["t_obs_release", "dataRights"]))
PLANET_COLUMNS = ["pl_name", "hostname", "ra", "dec", "pl_rade", "pl_eqt"]


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
    match_radius_deg: float = 0.02
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
# email
# --------------------------------------------------------------------------
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
        f"JWST time-series data for {who} was due to become public on "
        f"{sub.get('release_date')}, the release date MAST gave when you subscribed.",
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
                              send: Callable[[str, str, str], None] | None = None
                              ) -> list[str]:
    """Email every pending subscription whose release date has arrived.
    Returns log lines.

    The date is the ``t_obs_release`` MAST gave when the user subscribed; MAST
    is not asked again. A failed send stays pending and is retried next pass.
    """
    now = now_mjd() if now is None else now
    send = send or send_email
    log: list[str] = []
    with _SUBS_LOCK:
        subs = load_subscriptions(path)
        due = [s for s in subs if s.get("status") == "pending"
               and s.get("t_obs_release") is not None and s["t_obs_release"] <= now]
        for s in due:
            subject, body = _release_email(s)
            try:
                send(s["email"], subject, body)
            except Exception as exc:
                s["last_error"] = str(exc)
                log.append(f"{s['obs_id']}: email to {s['email']} failed ({exc}); will retry.")
                continue
            s["status"] = "sent"
            s["sent_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            s.pop("last_error", None)
            log.append(f"emailed {s['email']}: {subject}")
        if due:
            save_subscriptions(path, subs)
    return log


def start_subscription_notifier(base_directory: str | os.PathLike[str], *,
                                interval_seconds: float = 3600.0,
                                log: Callable[[str], None] = print) -> threading.Event:
    """Background thread that sends release-day emails while the process (the
    ASTER GUI) runs. It only reads the subscriptions file. Checks at start-up,
    so a date that passed while the GUI was closed is emailed on the next
    launch. Set the returned Event to stop it."""
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
_ACTIONS = ("lookup", "watchlist", "subscribe", "unsubscribe",
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
      subscribe     email `email` on the day each locked observation of
                    `target` (or one `obsid`) goes public.
      unsubscribe   by `subscription_id`, or every subscription for `target`.
      subscriptions list email subscriptions.
      test_email    send a test email to check the SMTP settings.

    Cannot see approved visits that have not been executed (not in MAST).
    Planet parameters: QueryExoplanetArchive. Files: DownloadMastJwstProducts.
    """

    action: str | None = mast.OptionalRuntimeField(
        description="lookup | watchlist | subscribe | unsubscribe | "
                    "subscriptions | test_email.")
    target: str | None = mast.OptionalRuntimeField(
        description="Planet or host star, any MAST-resolvable name, e.g. "
                    "'HD-12572b', 'HIP 9618 b', 'TOI-1801 b'.")
    planets: list | str | None = mast.OptionalRuntimeField(
        description="watchlist: Exoplanet Archive pl_names to watch "
                    "instead of the preset population.")
    preset: str | None = mast.OptionalRuntimeField(
        description="watchlist population preset (default 'subneptune').")
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

    def _do_subscribe(self, target: str | None) -> str:
        email = self._email()
        obsid = mast._clean_str(self.obsid)
        if not target and not obsid:
            raise ValueError("subscribe needs a target (or an obsid).")
        if target:
            result = lookup_target(target)
            planets = [p["pl_name"] for p in result["planets"]]
            locked = [r for r in result["observations"]
                      if not r["is_public"] and (not obsid or r["obsid"] == obsid)]
            n_all = len(result["observations"])
        else:
            locked = [r for r in fetch_observations_by_obsid([obsid]).values()
                      if not r["is_public"]]
            planets, target, n_all = [], f"obsid {obsid}", 1
        records = [r for r in locked if r["t_obs_release"] is not None]
        if locked and not records:
            return (f"ASTER Alert · cannot subscribe: MAST gives no release date for the "
                    f"{len(locked)} locked observation(s) of {target}, so there is no day "
                    "to email on.")
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
        if len(records) < len(locked):
            lines.append(f"  ({len(locked) - len(records)} locked observation(s) skipped: "
                         "MAST gives no release date.)")
        lines += ["", "The email is sent on that date by the ASTER GUI, which looks for "
                      "due emails every hour while it runs and at start-up (so if it is "
                      "closed that day, the email goes out on the next launch). The date "
                      "is the one MAST gives today; it is not re-checked."]
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


def main(argv: list[str] | None = None) -> int:
    """Send the release-day emails that are due and exit: the GUI notifier's
    one pass, for when the GUI is not running."""
    p = argparse.ArgumentParser(prog="aster-alert", description=main.__doc__)
    p.add_argument("--base-dir", default=".", help="ASTER base directory (holds .env "
                                                   "and workspace/).")
    args = p.parse_args(argv)
    base = Path(args.base_dir).expanduser()
    try:  # same settings the GUI reads (run_aster.py loads .env too)
        from dotenv import load_dotenv
        load_dotenv(base / ".env")
    except ImportError:
        pass
    lines = deliver_due_subscriptions(subscriptions_path(base / "workspace"))
    for line in lines:
        print(f"[aster-alert] {line}")
    if not lines:
        print("[aster-alert] no release-day emails due.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
