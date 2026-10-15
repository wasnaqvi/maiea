"""An OpenAI-compatible stand-in model for driving ASTER's GUI end to end.

ASTER's default provider is orchestral's ``GPT`` class, which talks to
``/v1/chat/completions`` through the official ``openai`` client — and that
client honours ``OPENAI_BASE_URL``. Pointing it here lets the REAL
``run_aster.py`` GUI, agent loop, tool schemas and tools run unmodified,
with only the model swapped out.

What this stand-in is, and is not
---------------------------------
It is a deterministic, rule-based interpreter of a small natural-language
vocabulary (target names, program ids, instruments, gratings, visits, and
search / list / download intents), carrying entities across turns the way a
model resolves "download those". It is NOT a language model: it proves the
GUI -> agent -> tool -> MAST -> disk path works and that ASTER's tool
schemas accept the calls a model would make. Whether a REAL model routes
free-form requests correctly is measured by scripts/eval_aster_nl.py, which
runs the same scenarios against your configured provider.

Contract checks it enforces on every call (violations are returned as the
assistant's text, prefixed SCHEMA_VIOLATION, so tests fail loudly):
  * the tool it picks is among the ``tools`` ASTER actually sent;
  * every argument is a property in that tool's JSON schema;
  * every ``required`` property is present.
It also only adds ``dataproduct_types=["timeseries"]`` to raw downloads when
ASTER's system prompt tells it to, so removing that guidance breaks the test.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PLANET_RE = re.compile(
    r"\b((?:WASP|HAT-P|HD|GJ|K2|TOI|TRAPPIST|LHS|LTT|KELT|XO|TrES|Kepler|LP)"
    r"[- ]?\d+(?:-\d+)?)\s?([b-h])\b", re.I)
PROGRAM_RE = re.compile(r"\b(?:program|proposal|pid|go|ers|gto|dd)\s*#?\s*(\d{3,5})\b", re.I)
INSTRUMENTS = {"nirspec": "NIRSpec", "niriss": "NIRISS", "miri": "MIRI", "nircam": "NIRCam"}
GRATING_RE = re.compile(r"\b(G\d{3}[HM]|PRISM|GR700XD|F\d{3}W2?)\b", re.I)
VISIT_RE = re.compile(r"\bvisit\s*#?\s*(\d+)\b", re.I)
DOWNLOAD_RE = re.compile(r"\b(download|fetch|grab|pull down|save)\b", re.I)
LIST_RE = re.compile(r"\b(list|don'?t download|without downloading|which files|what files)\b", re.I)
RAW_RE = re.compile(r"\b(uncal\w*|raw|uncalibrated|ramps?)\b", re.I)
TIMESERIES_GUIDANCE = 'dataproduct_types=["timeseries"]'
ALERT_RE = re.compile(r"\baster alert\b|\bgo(?:es|ing)? public\b|\bwhen\b.*\bpublic\b|"
                      r"\breleased?\b|\brelease date\b|\bunlock", re.I)
REMIND_RE = re.compile(r"\b(email|remind|notify)\b", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
ALERT_GUIDANCE = "AsterAlert"


def _text(content) -> str:
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def entities(user_messages: list[str]) -> dict:
    """Latest mention wins; earlier turns fill gaps ("download those")."""
    found: dict = {}
    for msg in user_messages:          # oldest -> newest, later overwrite
        if m := PLANET_RE.search(msg):
            found["planet_name"] = f"{m[1].upper()} {m[2].lower()}"
        if m := PROGRAM_RE.search(msg):
            found["proposal_id"] = m[1]
        insts = [v for k, v in INSTRUMENTS.items() if re.search(rf"\b{k}\b", msg, re.I)]
        if insts:
            found["instruments"] = insts
        grat = [g.upper() for g in GRATING_RE.findall(msg)]
        if grat:
            found["filters"] = grat
        if m := VISIT_RE.search(msg):
            found["visits"] = [int(m[1])]
    return found


def _offered(tools: list[dict]) -> dict[str, str]:
    """normalized name -> name as offered. orchestral sends tool names
    lower-cased with any 'Tool' suffix dropped ('searchmastjwstobservations'),
    while ASTER's prompt routes by CamelCase; a model maps one to the other."""
    names = [t.get("function", {}).get("name", "") for t in tools]
    return {re.sub(r"tool$", "", n.lower()): n for n in names}


def resolve_tool(camel: str, tools: list[dict]) -> str | None:
    return _offered(tools).get(re.sub(r"tool$", "", camel.lower()))


def plan(messages: list[dict], tools: list[dict]) -> dict:
    """Return {"tool": name, "arguments": {...}} or {"text": "..."}."""
    if not resolve_tool("SearchMastJwstObservations", tools):
        # Not an ASTER agent turn (e.g. orchestral's conversation-naming call).
        return {"text": "JWST data request"}
    if messages and messages[-1].get("role") == "tool":
        result = _text(messages[-1].get("content"))
        head = "\n".join(result.splitlines()[:30])
        return {"text": f"Done. The tool reported:\n\n{head}"}

    system = " ".join(_text(m.get("content")) for m in messages if m.get("role") == "system")
    users = [_text(m.get("content")) for m in messages if m.get("role") == "user"]
    latest = users[-1] if users else ""
    ent = entities(users)

    # Release-date questions go to ASTER Alert -- but only when ASTER's system
    # prompt routes them there, so deleting that guidance fails the test.
    if (ALERT_GUIDANCE in system and resolve_tool("AsterAlert", tools)
            and (ALERT_RE.search(latest) or REMIND_RE.search(latest))):
        args: dict = {}
        if m := PLANET_RE.search(latest):
            args["target"] = m[0]              # as typed: 'HD-12572b'
        if REMIND_RE.search(latest) and (em := EMAIL_RE.search(latest)):
            args.update(action="subscribe", email=em[0])
        elif "target" not in args:
            args["action"] = "watchlist"
        return {"tool": "AsterAlert", "arguments": args, "tools": tools}

    if LIST_RE.search(latest):
        tool, args = "DownloadMastJwstProducts", {**ent, "list_only": True}
    elif DOWNLOAD_RE.search(latest):
        tool, args = "DownloadMastJwstProducts", dict(ent)
    else:
        tool = "SearchMastJwstObservations"
        args = {k: v for k, v in ent.items() if k != "visits"}

    if tool == "DownloadMastJwstProducts" and (RAW_RE.search(latest) or
                                               any(RAW_RE.search(u) for u in users)):
        args["raw_only"] = True
        if TIMESERIES_GUIDANCE in system:
            args["dataproduct_types"] = ["timeseries"]
    if tool == "DownloadMastJwstProducts" and "list_only" not in args:
        args.setdefault("max_observations", 5)   # what the prompt asks for UNCAL
    return {"tool": tool, "arguments": args, "tools": tools}


def validate(tool: str, args: dict, tools: list[dict]) -> str | None:
    schema = next((t["function"] for t in tools
                   if t.get("function", {}).get("name") == tool), None)
    if schema is None:
        offered = sorted(t.get("function", {}).get("name") for t in tools)
        return f"tool {tool!r} not offered by ASTER (offered: {offered})"
    props = schema.get("parameters", {}).get("properties", {})
    unknown = sorted(set(args) - set(props))
    if unknown:
        return f"{tool}: arguments {unknown} not in schema {sorted(props)}"
    missing = sorted(set(schema.get("parameters", {}).get("required", [])) - set(args))
    if missing:
        return f"{tool}: schema marks {missing} required but a model would omit them"
    return None


class FakeOpenAI(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr):
        super().__init__(addr, _Handler)
        self.requests: list[dict] = []     # every chat.completions request body
        self.decisions: list[dict] = []    # what the stand-in chose each time
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/v1"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        tools = body.get("tools") or []
        decision = plan(body.get("messages", []), tools)
        if "tool" in decision:
            decision["camel"] = decision["tool"]
            decision["tool"] = resolve_tool(decision["tool"], tools) or decision["tool"]
            problem = validate(decision["tool"], decision["arguments"], tools)
            if problem:
                decision = {"text": f"SCHEMA_VIOLATION: {problem}"}
        decision.pop("tools", None)
        with self.server._lock:
            self.server.requests.append(body)
            self.server.decisions.append(decision)

        model = body.get("model", "stand-in")
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        # The *_details blocks are what the real API sends; orchestral 1.4.x
        # dereferences completion_tokens_details unconditionally.
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
                 "completion_tokens_details": {
                     "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0,
                     "audio_tokens": 0, "reasoning_tokens": 0},
                 "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0}}
        call = None
        if "tool" in decision:
            call = {"id": f"call_{uuid.uuid4().hex[:10]}", "type": "function",
                    "function": {"name": decision["tool"],
                                 "arguments": json.dumps(decision["arguments"])}}

        if not body.get("stream"):
            message = {"role": "assistant", "content": decision.get("text"),
                       "tool_calls": [call] if call else None}
            payload = {"id": cid, "object": "chat.completion", "created": int(time.time()),
                       "model": model, "usage": usage,
                       "choices": [{"index": 0, "message": message,
                                    "finish_reason": "tool_calls" if call else "stop"}]}
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        def chunk(delta, finish=None, with_usage=False):
            c = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                 "model": model,
                 "choices": [] if with_usage else
                 [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if with_usage:
                c["usage"] = usage
            return f"data: {json.dumps(c)}\n\n".encode()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        out = [chunk({"role": "assistant", "content": ""})]
        if call:
            out.append(chunk({"tool_calls": [{"index": 0, **call}]}))
            out.append(chunk({}, finish="tool_calls"))
        else:
            for piece in re.findall(r".{1,40}", decision["text"], re.S):
                out.append(chunk({"content": piece}))
            out.append(chunk({}, finish="stop"))
        out.append(chunk(None, with_usage=True))
        out.append(b"data: [DONE]\n\n")
        for piece in out:
            self.wfile.write(piece)
            self.wfile.flush()


def start(port: int = 0) -> FakeOpenAI:
    server = FakeOpenAI(("127.0.0.1", port))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
