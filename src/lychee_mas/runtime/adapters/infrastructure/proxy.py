"""Host-to-container proxy relay used by tool sandboxes."""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from urllib.parse import urlparse


class ProxyRelay:
    """Expose a host-loopback proxy on the Docker bridge for one run process."""

    def __init__(self, upstream_url: str, *, listen_host: str, listen_port: int) -> None:
        parsed = urlparse(upstream_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or not parsed.port:
            raise ValueError("proxy relay upstream must be an HTTP(S) URL with an explicit port")
        self.upstream_host = parsed.hostname
        self.upstream_port = parsed.port
        self.listen_host = listen_host
        self.listen_port = int(listen_port)
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> "ProxyRelay":
        self.start()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.stop()

    def start(self) -> None:
        self._require_connectable(self.upstream_host, self.upstream_port, "upstream proxy")
        if self._connectable(self.listen_host, self.listen_port):
            return
        socat = shutil.which("socat")
        if not socat:
            raise RuntimeError("socat is required to expose the experiment proxy to Docker")
        self.process = subprocess.Popen(
            [
                socat,
                f"TCP-LISTEN:{self.listen_port},bind={self.listen_host},reuseaddr,fork",
                f"TCP:{self.upstream_host}:{self.upstream_port}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                error = (self.process.stderr.read() if self.process.stderr else b"").decode(
                    "utf-8", errors="replace"
                )
                raise RuntimeError(f"proxy relay exited during startup: {error.strip()}")
            if self._connectable(self.listen_host, self.listen_port):
                print(
                    "[network] Docker proxy relay ready "
                    f"{self.listen_host}:{self.listen_port} -> "
                    f"{self.upstream_host}:{self.upstream_port}",
                    flush=True,
                )
                return
            time.sleep(0.1)
        self.stop()
        raise RuntimeError("proxy relay did not become reachable within 5 seconds")

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)
        self.process = None

    @staticmethod
    def _connectable(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            return False

    @classmethod
    def _require_connectable(cls, host: str, port: int, label: str) -> None:
        if not cls._connectable(host, port):
            raise RuntimeError(f"{label} is unreachable at {host}:{port}")
