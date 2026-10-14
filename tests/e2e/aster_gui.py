"""Launch the real ASTER GUI (``python run_aster.py``) and talk to it the way
the browser does — over the ``/ws`` websocket.

Used by tests/e2e/test_gui_e2e.py (stand-in model, fake MAST) and by
scripts/eval_aster_nl.py (your real model, fake or live MAST).

The GUI runs in a scratch directory holding a copy of run_aster.py, the
system prompt and the aster_toolkit package, with an empty ``workspace/`` —
so a test never touches your real workspace, and everything the agent
downloads can be inspected afterwards under ``gui.workspace``.
"""
from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_URL_RE = re.compile(r"Open in browser: (http://[\w.]+:\d+)")
_TAG_RE = re.compile(r"<[^>]+>")


def html_to_text(content: str) -> str:
    return html.unescape(_TAG_RE.sub(" ", content or ""))


class AsterGui:
    def __init__(self, url: str, workdir: Path, proc: subprocess.Popen, log: list[str]):
        self.url = url
        self.workdir = workdir
        self.workspace = workdir / "workspace"
        self.proc = proc
        self.log = log

    def chat(self, message: str, timeout: float = 120.0) -> dict:
        """Send one chat message on a fresh browser connection and wait for
        the turn to finish. Conversation history persists server-side, so
        follow-ups ("download those") see earlier turns, as in the browser."""
        from websockets.sync.client import connect

        events: list[dict] = []
        ws_url = self.url.replace("http://", "ws://") + "/ws"
        deadline = time.time() + timeout
        with connect(ws_url, open_timeout=30, max_size=None) as ws:
            ws.send(json.dumps({"type": "chat", "message": message}))
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(f"GUI turn did not complete in {timeout}s: {message!r}")
                event = json.loads(ws.recv(timeout=remaining))
                events.append(event)
                if event.get("type") in ("complete", "error", "interrupted"):
                    break
        updates = [e for e in events if e.get("type") == "agent_update"]
        final_html = updates[-1].get("content", "") if updates else ""
        return {
            "events": events,
            "status": events[-1].get("type"),
            "error": events[-1].get("message") if events[-1].get("type") == "error" else None,
            # agent_update carries the whole rendered conversation so far
            "text": html_to_text(final_html),
        }

    def clear(self) -> None:
        """Start a new conversation (the GUI's "clear" message)."""
        from websockets.sync.client import connect

        ws_url = self.url.replace("http://", "ws://") + "/ws"
        with connect(ws_url, open_timeout=30, max_size=None) as ws:
            ws.send(json.dumps({"type": "clear"}))
            deadline = time.time() + 30
            while time.time() < deadline:
                try:
                    event = json.loads(ws.recv(timeout=2))
                except TimeoutError:
                    break
                if event.get("type") in ("info", "error"):
                    break

    def files(self, pattern: str = "*.fits") -> list[Path]:
        return sorted(self.workspace.rglob(pattern))


@contextmanager
def launch_gui(env_overrides: dict[str, str], *, startup_timeout: float = 120.0,
               keep_workdir: bool = False):
    workdir = Path(tempfile.mkdtemp(prefix="aster_gui_"))
    for name in ("run_aster.py", "aster_system_prompt.md"):
        shutil.copy(REPO / name, workdir / name)
    shutil.copytree(REPO / "aster_toolkit", workdir / "aster_toolkit",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (workdir / "workspace").mkdir()

    env = dict(os.environ)
    env.update({
        "PYTHONUNBUFFERED": "1",
        "BROWSER": "true",          # webbrowser.open() becomes a no-op
        "ASTER_LLM_PROVIDER": "openai",
        "MPLCONFIGDIR": str(workdir / ".mpl"),
    })
    env.update(env_overrides)

    proc = subprocess.Popen(
        [sys.executable, "run_aster.py"], cwd=workdir, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    log: list[str] = []
    found: dict = {}

    def pump():
        for line in proc.stdout:
            log.append(line.rstrip("\n"))
            if (m := _URL_RE.search(line)) and "url" not in found:
                found["url"] = m[1]

    threading.Thread(target=pump, daemon=True).start()
    try:
        deadline = time.time() + startup_timeout
        while "url" not in found:
            if proc.poll() is not None:
                raise RuntimeError("run_aster.py exited during startup:\n" + "\n".join(log[-40:]))
            if time.time() > deadline:
                raise TimeoutError("GUI did not start:\n" + "\n".join(log[-40:]))
            time.sleep(0.2)
        _wait_healthy(found["url"], deadline)
        yield AsterGui(found["url"], workdir, proc, log)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def _wait_healthy(url: str, deadline: float) -> None:
    import urllib.request

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"GUI at {url} never became healthy")
