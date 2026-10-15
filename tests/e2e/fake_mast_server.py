"""A local HTTP stand-in for the two MAST endpoints ASTER calls.

    POST /api/v0/invoke              Mashup: Name.Lookup, Caom.Filtered(.Position), Caom.Products
    GET  /api/v0.1/Download/file     ?uri=mast:JWST/product/<file>  (Range-aware)
    GET  /TAP/sync                   the NASA Exoplanet Archive's TAP endpoint, as CSV
                                     (ASTER reaches it through ASTER_ARCHIVE_TAP_URL)

Serves the shared fixture in tests/scripts/fake_mast_archive.py. Discrete
filters are EXACT-match, like the real service (that is what the instrument
alias and grating-token logic exist to cope with). Downloads are tiny but
valid FITS files whose primary header carries the keys ASTER's segment
inventory reads. Every request is appended to ``server.log`` so tests can
assert what the agent actually asked MAST for.

Run standalone:  python tests/e2e/fake_mast_server.py --port 8765
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fake_mast_archive import obs_rows, products_for, resolve, sep_deg, tap_rows  # noqa: E402

_UNCAL_RE = re.compile(
    r"^(?P<prefix>jw\d{11})_\d{5}_\d{5}(?:-seg(?P<seg>\d{3}))?_(?P<det>[a-z0-9]+)_"
)


def fits_bytes(filename: str) -> bytes:
    from astropy.io import fits

    hdr = fits.Header()
    m = _UNCAL_RE.match(filename)
    if m:
        hdr["DETECTOR"] = m["det"].upper()
        hdr["EXSEGNUM"] = int(m["seg"] or 1)
        hdr["EXSEGTOT"] = 2 if m["seg"] else 1
        hdr["NINTS"] = 10
        hdr["NGROUPS"] = 5
        hdr["SUBARRAY"] = "SUB2048"
        hdr["READPATT"] = "NRSRAPID"
        hdr["TARGPROP"] = "WASP-39"
        hdr["DATE-OBS"] = "2022-07-30"
    buf = io.BytesIO()
    fits.PrimaryHDU(header=hdr).writeto(buf)
    return buf.getvalue()


def mashup(request: dict) -> dict:
    svc = request.get("service")
    params = request.get("params", {})
    if svc == "Mast.Name.Lookup":
        try:
            ra, dec = resolve(params["input"])
        except ValueError:
            return {"resolvedCoordinate": []}
        return {"resolvedCoordinate": [{"ra": ra, "decl": dec}]}
    if svc == "Mast.Caom.Products":
        return {"status": "COMPLETE", "data": products_for(str(params["obsid"]))}
    if svc in ("Mast.Caom.Filtered", "Mast.Caom.Filtered.Position"):
        rows = obs_rows()
        for flt in params.get("filters", []):
            values = flt.get("values") or []
            if values:
                rows = [r for r in rows if str(r.get(flt["paramName"])) in map(str, values)]
            elif flt.get("freeText"):
                needle = flt["freeText"].strip("%").upper()
                rows = [r for r in rows if needle in str(r.get(flt["paramName"], "")).upper()]
        if "position" in params:
            ra, dec, rad = (float(x) for x in str(params["position"]).split(","))
            rows = [r for r in rows if sep_deg((ra, dec), (r["s_ra"], r["s_dec"])) <= rad]
        return {"status": "COMPLETE", "data": rows,
                "paging": {"rowsFiltered": len(rows), "rowsTotal": len(rows)}}
    return {"status": "ERROR", "msg": f"fake MAST: unsupported service {svc!r}"}


def tap_csv(adql: str) -> str:
    import csv
    m = re.search(r"SELECT\s+(?:DISTINCT\s+)?(?:TOP\s+\d+\s+)?(.*?)\s+FROM", adql, re.I | re.S)
    cols = [c.strip() for c in (m[1] if m else "*").split(",")]
    rows = tap_rows(adql)
    if cols == ["*"]:
        cols = list(rows[0]) if rows else ["pl_name"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({c: r.get(c, "") for c in cols})
    return buf.getvalue()


class FakeMast(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr):
        super().__init__(addr, _Handler)
        self.log: list[dict] = []
        self._lock = threading.Lock()

    def record(self, entry: dict) -> None:
        with self._lock:
            self.log.append(entry)

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep test output quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if urlparse(self.path).path != "/api/v0/invoke":
            return self._send(404, b"not found", "text/plain")
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
        request = json.loads(unquote(parse_qs(raw, keep_blank_values=True)["request"][0]))
        self.server.record({"kind": "invoke", "request": request})
        self._send(200, json.dumps(mashup(request)).encode(), "application/json")

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/health":
            return self._send(200, b"ok", "text/plain")
        if url.path == "/TAP/sync":
            adql = parse_qs(url.query).get("query", [""])[0]
            self.server.record({"kind": "tap", "adql": adql})
            return self._send(200, tap_csv(adql).encode(), "text/csv")
        if url.path != "/api/v0.1/Download/file":
            return self._send(404, b"not found", "text/plain")
        uri = parse_qs(url.query).get("uri", [""])[0]
        name = uri.rsplit("/", 1)[-1]
        self.server.record({"kind": "download", "uri": uri})
        body = fits_bytes(name)
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(body):
                return self._send(416, b"", "application/octet-stream",
                                  {"Content-Range": f"bytes */{len(body)}"})
            part = body[start:]
            return self._send(206, part, "application/octet-stream",
                              {"Content-Range": f"bytes {start}-{len(body) - 1}/{len(body)}"})
        self._send(200, body, "application/octet-stream")


def start(port: int = 0) -> FakeMast:
    server = FakeMast(("127.0.0.1", port))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    srv = start(ap.parse_args().port)
    print(f"fake MAST at {srv.url}  (ASTER_MAST_BASE_URL={srv.url})", flush=True)
    threading.Event().wait()
