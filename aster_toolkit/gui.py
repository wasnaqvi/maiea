"""ASTER's additions to orchestral's web GUI.

``run_aster.py`` calls :func:`install` before starting the server. Nothing in
the orchestral package is modified; everything is added to its FastAPI app:

* ``/aster-static/``  ASTER's own page assets (``aster_toolkit/gui_static``).
* ``/``               orchestral's index.html, re-served with ASTER's
                      stylesheet and script injected at the TOP of ``<head>``.
                      The script must run before orchestral's app.js opens the
                      ``/ws`` websocket, because it wraps ``WebSocket`` to watch
                      for AsterAlert tool calls and animate the ASTER Alert card.
* ``GET /api/aster-alert/subscriptions``  pending release-day emails, for the
                      card's reminder bell.
* ASTER Alert's release-day email notifier, a background thread that checks
  MAST every ``ASTER_ALERT_CHECK_SECONDS`` (default 3600; 0 disables it).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from aster_toolkit.data_acquisition import alert

STATIC_DIR = Path(__file__).resolve().parent / "gui_static"
_INJECT = ('<link rel="stylesheet" href="/aster-static/aster_alert.css">\n'
           '    <script src="/aster-static/aster_alert.js"></script>')


def inject_assets(html: str) -> str:
    """Insert ASTER's assets first in <head> (idempotent)."""
    if "aster_alert.js" in html:
        return html
    return html.replace("<head>", "<head>\n    " + _INJECT, 1)


def subscription_summary(base_directory: str | os.PathLike[str]) -> dict[str, Any]:
    subs = alert.load_subscriptions(alert.subscriptions_path(base_directory))
    pending = sorted((s for s in subs if s.get("status") == "pending"),
                     key=lambda s: s.get("t_obs_release") or 9e9)
    return {
        "pending": [{"target": s.get("target"), "release_date": s.get("release_date"),
                     "proposal_id": s.get("proposal_id"),
                     "instrument": f"{s.get('instrument_name') or ''} {s.get('filters') or ''}".strip()}
                    for s in pending],
        "sent": sum(s.get("status") == "sent" for s in subs),
        "email_configured": alert.smtp_configured(),
        "missing": alert.smtp_missing(),
    }


def install(server_module: Any, *, base_directory: str) -> None:
    """Attach ASTER's GUI additions to orchestral's app server module."""
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    app = server_module.app
    if getattr(app.state, "aster_gui_installed", False):
        return
    index_path = Path(server_module.STATIC_DIR) / "index.html"

    # Replace orchestral's GET / with the injected page.
    app.router.routes[:] = [
        r for r in app.router.routes
        if not (getattr(r, "path", None) == "/" and "GET" in (getattr(r, "methods", None) or ()))
    ]
    app.mount("/aster-static", StaticFiles(directory=str(STATIC_DIR)), name="aster-static")

    @app.get("/", response_class=HTMLResponse)
    async def aster_index() -> HTMLResponse:
        return HTMLResponse(inject_assets(index_path.read_text()))

    @app.get("/api/aster-alert/subscriptions")
    async def aster_alert_subscriptions() -> JSONResponse:
        return JSONResponse(subscription_summary(base_directory))

    app.state.aster_gui_installed = True

    interval = float(os.environ.get("ASTER_ALERT_CHECK_SECONDS", "3600"))
    if interval > 0:
        alert.start_subscription_notifier(base_directory, interval_seconds=interval)
