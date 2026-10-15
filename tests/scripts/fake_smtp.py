"""A minimal local SMTP server that records messages instead of sending them.

Enough of RFC 5321 for smtplib's plain (no TLS, no AUTH) path: EHLO/HELO,
MAIL, RCPT, DATA, RSET, NOOP, QUIT. Used by test_aster_alert_tool.py and the
GUI end-to-end test with ASTER_ALERT_SMTP_TLS=none.
"""
from __future__ import annotations

import socketserver
import threading
from email import message_from_bytes, policy


class FakeSmtp(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr=("127.0.0.1", 0)):
        super().__init__(addr, _Handler)
        self.messages: list = []          # email.message.EmailMessage
        self.received = threading.Event()

    @property
    def port(self) -> int:
        return self.server_address[1]

    def env(self) -> dict[str, str]:
        return {"ASTER_ALERT_SMTP_HOST": "127.0.0.1",
                "ASTER_ALERT_SMTP_PORT": str(self.port),
                "ASTER_ALERT_SMTP_TLS": "none",
                "ASTER_ALERT_SMTP_FROM": "aster-alert@test.local"}


class _Handler(socketserver.StreamRequestHandler):
    def _reply(self, line: str) -> None:
        self.wfile.write((line + "\r\n").encode())

    def handle(self) -> None:
        self._reply("220 fake-smtp ready")
        while True:
            raw = self.rfile.readline()
            if not raw:
                return
            cmd = raw.decode(errors="replace").strip().upper()
            if cmd.startswith(("EHLO", "HELO")):
                self._reply("250 fake-smtp")
            elif cmd.startswith(("MAIL", "RCPT", "RSET", "NOOP")):
                self._reply("250 OK")
            elif cmd == "DATA":
                self._reply("354 end with <CRLF>.<CRLF>")
                lines = []
                while True:
                    line = self.rfile.readline()
                    if not line or line in (b".\r\n", b".\n"):
                        break
                    lines.append(line[1:] if line.startswith(b"..") else line)
                self.server.messages.append(
                    message_from_bytes(b"".join(lines), policy=policy.default))
                self.server.received.set()
                self._reply("250 queued")
            elif cmd == "QUIT":
                self._reply("221 bye")
                return
            else:
                self._reply("502 not implemented")


def start() -> FakeSmtp:
    server = FakeSmtp()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
