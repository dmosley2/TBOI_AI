"""TCP link to the Isaac AI Bridge mod.

Python is the server; the game connects to us. The protocol is newline-
delimited JSON in strict alternation:

    game -> python : state      (once per game update, 30 Hz)
    python -> game : action/cmd (exactly one reply per state)

Because the game waits for every reply, `step()` is the only way time moves
forward. Think of it like `env.step` at the lowest level.
"""

from __future__ import annotations

import json
import socket
import sys

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 28771


class GameDisconnected(RuntimeError):
    pass


def as_list(x):
    """Isaac's json.lua may encode an empty Lua table as {} instead of []."""
    return x if isinstance(x, list) else []


class IsaacBridge:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if sys.platform != "win32":  # on Windows REUSEADDR allows double-binding
            self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(1)
        self._conn: socket.socket | None = None
        self._rfile = None
        self.state: dict | None = None
        self.port = port

    def wait_for_game(self, quiet: bool = False) -> dict:
        """Block until the game connects, return its first state."""
        if not quiet:
            print(f"[bridge] waiting for the game on port {self.port} "
                  "(launch with --luadebug and start a run)...", flush=True)
        conn, _ = self._srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._conn = conn
        self._rfile = conn.makefile("r", encoding="utf-8", newline="\n")
        self.state = self._read()
        if not quiet:
            print("[bridge] game connected", flush=True)
        return self.state

    def _read(self) -> dict:
        line = self._rfile.readline()
        if not line:
            raise GameDisconnected("game closed the connection")
        return json.loads(line)

    def step(self, msg: dict) -> dict:
        """Send one message, advance one game update, return the new state."""
        if self._conn is None:
            raise GameDisconnected("not connected")
        try:
            self._conn.sendall((json.dumps(msg) + "\n").encode("utf-8"))
            self.state = self._read()
        except (ConnectionError, OSError) as e:
            raise GameDisconnected(str(e)) from e
        return self.state

    def close(self):
        # Closing the socket makes the mod release control on its side.
        for s in (self._rfile, self._conn, self._srv):
            try:
                if s is not None:
                    s.close()
            except OSError:
                pass
        self._conn = self._rfile = None
