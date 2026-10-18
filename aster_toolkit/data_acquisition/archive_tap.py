"""
The one place that talks to the NASA Exoplanet Archive TAP service.

Builds the ADQL, sends it, and turns the archive's failures into readable
errors. ``mast.archive_tap_query`` (Patchwork, ASTER Alert, the JWST
crossmatch) and the ``QueryExoplanetArchive`` helpers in ``exoarchive.py``
are thin wrappers over :func:`query_archive`.

Depends only on ``requests`` and the standard library, so it imports
anywhere ``mast.py`` does — including without orchestral.
"""
from __future__ import annotations

import csv
import io
import os
import re
import time
from typing import Any, Iterable

import requests

# ASTER_ARCHIVE_TAP_URL exists for end-to-end tests (tests/e2e) that point
# the real GUI at a local fake archive, like ASTER_MAST_BASE_URL; leave it
# unset in normal use.
TAP_SYNC_URL = os.environ.get(
    "ASTER_ARCHIVE_TAP_URL", "https://exoplanetarchive.ipac.caltech.edu/TAP/sync")
DEFAULT_TIMEOUT = 120.0

# On a bad query the archive answers with a VOTable instead of CSV, carrying
# the reason (e.g. "ORA-00904: 'NOSUCHCOL': invalid identifier") here.
_QUERY_ERROR_RE = re.compile(
    r'<INFO[^>]*name="QUERY_STATUS"[^>]*value="ERROR"[^>]*>(.*?)</INFO>',
    re.DOTALL | re.IGNORECASE,
)


# The archive's backing database drops out now and then ("Failed to connect to
# cx_Oracle", HTTP 5xx). Those are retried; a query it rejects is not.
RETRY_DELAYS = (15.0, 45.0, 120.0)
_TRANSIENT_RE = re.compile(r"failed to connect|connection (?:refused|reset)|timed? ?out|unavailable",
                           re.IGNORECASE)


class ArchiveTapError(requests.HTTPError):
    """The archive rejected a query. Subclasses HTTPError, so existing
    ``except requests.RequestException`` handlers still catch it."""


def build_adql(
    columns: Iterable[str],
    conditions: Iterable[str] | str | None = None,
    *,
    table: str = "pscomppars",
    distinct: bool = False,
    limit: int | None = None,
) -> str:
    """SELECT [DISTINCT] [TOP n] <columns> FROM <table> [WHERE (c1) AND (c2)...].

    ``conditions`` are raw ADQL expressions; quoting string literals inside
    them is the caller's job.
    """
    columns = list(columns)
    if not columns:
        raise ValueError("build_adql needs at least one column.")
    if isinstance(conditions, str):
        conditions = [conditions]
    conditions = [c for c in (conditions or []) if c and c.strip()]

    select = "SELECT DISTINCT" if distinct else "SELECT"
    top = f"TOP {int(limit)} " if limit is not None else ""
    adql = f"{select} {top}{', '.join(columns)} FROM {table}"
    if conditions:
        adql += " WHERE " + " AND ".join(f"({c})" for c in conditions)
    return adql


def run_adql(
    adql: str,
    *,
    session: requests.Session | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    retry_delays: Iterable[float] | None = None,
) -> list[dict[str, Any]]:
    """:func:`_run_adql_once`, retried with backoff on transient archive outages."""
    delays = list(RETRY_DELAYS if retry_delays is None else retry_delays)
    for attempt in range(len(delays) + 1):
        try:
            return _run_adql_once(adql, session=session, timeout=timeout)
        except ArchiveTapError as exc:
            status = getattr(exc.response, "status_code", 0) or 0
            transient = status >= 500 or bool(_TRANSIENT_RE.search(str(exc).split("\nADQL:")[0]))
            if not transient or attempt == len(delays):
                raise
        except (requests.ConnectionError, requests.Timeout):
            if attempt == len(delays):
                raise
        time.sleep(delays[attempt])
    raise AssertionError("unreachable")


def _run_adql_once(
    adql: str,
    *,
    session: requests.Session | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[dict[str, Any]]:
    """Send one ADQL query to the archive's sync endpoint; return CSV rows.

    Raises :class:`ArchiveTapError` (with the archive's reason and the ADQL)
    when the archive rejects the query. Connection errors and timeouts
    propagate as the usual ``requests`` exceptions.
    """
    client = session or requests
    response = client.get(
        TAP_SYNC_URL,
        params={"query": adql, "format": "csv"},
        timeout=timeout,
    )
    text = response.text or ""
    match = _QUERY_ERROR_RE.search(text)
    if match or not response.ok:
        reason = (
            " ".join(match.group(1).split()) if match
            else f"HTTP {response.status_code} {response.reason or ''}".strip()
        )
        raise ArchiveTapError(
            f"NASA Exoplanet Archive rejected the query: {reason}\nADQL: {adql}",
            response=response,
        )
    return list(csv.DictReader(io.StringIO(text)))


def query_archive(
    columns: Iterable[str],
    conditions: Iterable[str] | str | None = None,
    *,
    table: str = "pscomppars",
    distinct: bool = False,
    limit: int | None = None,
    session: requests.Session | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[dict[str, Any]]:
    """Build and run an archive query; every row as a ``{column: str}`` dict."""
    adql = build_adql(columns, conditions, table=table, distinct=distinct, limit=limit)
    return run_adql(adql, session=session, timeout=timeout)
