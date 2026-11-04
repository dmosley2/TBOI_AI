"""A crude fake of the game side of the protocol, for testing Python code
without launching Isaac. It speaks the same JSON as main.lua; the "physics"
(chasing enemies, straight bullets, axis tears) are only rough stand-ins.

It is NOT a training simulator; real gameplay differs a lot.
"""

from __future__ import annotations

import json
import math
import random
import socket
import time

W, H = 15, 9                     # 1x1 room grid incl. walls
TL, BR = (60.0, 140.0), (580.0, 420.0)
ROCKS = {2 * W + 4, 6 * W + 10}


def grid_string():
    cells = []
    for i in range(W * H):
        r, c = divmod(i, W)
        cells.append("4" if r in (0, H - 1) or c in (0, W - 1) else "2" if i in ROCKS else "0")
    return "".join(cells)


def grid_index(x, y):
    c = int((x - TL[0]) // 40) + 1
    r = int((y - TL[1]) // 40) + 1
    return r * W + c


class MockIsaac:
    def __init__(self, port=28771, seed=0):
        self.port = port
        self.rng = random.Random(seed)
        self.frame = 0
        self.room_index = 84
        self.px, self.py = 320.0, 280.0
        self.hearts, self.max_hearts = 6, 6
        self.enemies: list[dict] = []
        self.projectiles: list[dict] = []
        self.tears: list[dict] = []
        self.control = False
        self.arena = False
        self.move, self.shoot, self.cooldown = (0, 0), 0, 0
        self.iframes = 0
        self.grid = grid_string()
        self._reset_events()

    def _reset_events(self):
        self.ev = {"hits": 0, "damage_taken": 0, "damage_dealt": 0.0, "kills": 0, "lethal": False}

    # -- protocol -------------------------------------------------------
    def run(self, max_updates=None):
        sock = None
        for _ in range(200):
            try:
                sock = socket.create_connection(("127.0.0.1", self.port), timeout=10)
                break
            except OSError:
                time.sleep(0.05)
        if sock is None:
            raise RuntimeError("mock game could not connect")
        rfile = sock.makefile("r", encoding="utf-8")
        try:
            while max_updates is None or self.frame < max_updates:
                sock.sendall((json.dumps(self.snapshot()) + "\n").encode())
                self._reset_events()
                line = rfile.readline()
                if not line:
                    return
                self.apply(json.loads(line))
                self.update()
        except OSError:
            return
        finally:
            sock.close()

    def snapshot(self):
        return {
            "t": "state", "frame": self.frame, "stage": 1, "stage_type": 0,
            "control": self.control, "arena": self.arena, "events": self.ev,
            "room": {"index": self.room_index, "type": 1, "shape": 1,
                     "clear": not self.enemies, "tl": list(TL), "br": list(BR),
                     "w": W, "h": H, "grid": self.grid},
            "player": {"pos": [self.px, self.py], "vel": [0, 0],
                       "grid": grid_index(self.px, self.py), "player_type": 0,
                       "hearts": self.hearts, "max_hearts": self.max_hearts, "soul": 0,
                       "damage": 3.5, "fire_delay": 10, "speed": 1.0, "range": 260,
                       "shot_speed": 1.0, "coins": 0, "bombs": 1, "keys": 0, "dead": False},
            "enemies": [{"id": i, "type": e["type"], "var": 0, "sub": 0,
                         "pos": [e["x"], e["y"]], "vel": [0, 0], "hp": e["hp"],
                         "max_hp": e["max_hp"], "size": 13, "boss": False, "vuln": True}
                        for i, e in enumerate(self.enemies)],
            "projectiles": [{"pos": [p["x"], p["y"]], "vel": [p["vx"], p["vy"]], "size": 7}
                            for p in self.projectiles],
        }

    def apply(self, msg):
        if "control" in msg:
            self.control = bool(msg["control"])
        if self.control:
            self.move = tuple(msg.get("move", (0, 0)))
            self.shoot = int(msg.get("shoot", 0))
        if msg.get("cmd") == "reset_arena":
            self.enemies.clear(); self.projectiles.clear(); self.tears.clear()
            self.hearts = self.max_hearts
            self.px, self.py = 320.0, 280.0
            for spec in msg.get("enemies", []):
                while True:
                    x = self.rng.uniform(TL[0] + 20, BR[0] - 20)
                    y = self.rng.uniform(TL[1] + 20, BR[1] - 20)
                    if math.hypot(x - self.px, y - self.py) > 160:
                        break
                self.enemies.append({"type": spec[0], "x": x, "y": y, "hp": 10.0,
                                     "max_hp": 10.0, "t": self.rng.randint(0, 60)})
            self.arena = True
        elif msg.get("cmd") == "release":
            self.control = self.arena = False

    # -- toy physics ----------------------------------------------------
    def update(self):
        self.frame += 1
        dx, dy = self.move
        n = math.hypot(dx, dy) or 1
        self.px = min(max(self.px + 4.5 * dx / n, TL[0] + 10), BR[0] - 10)
        self.py = min(max(self.py + 4.5 * dy / n, TL[1] + 10), BR[1] - 10)

        self.cooldown = max(0, self.cooldown - 1)
        if self.shoot and self.cooldown == 0:
            vx, vy = [(-10, 0), (10, 0), (0, -10), (0, 10)][self.shoot - 1]
            self.tears.append({"x": self.px, "y": self.py, "vx": vx, "vy": vy, "life": 26})
            self.cooldown = 10

        for t in self.tears:
            t["x"] += t["vx"]; t["y"] += t["vy"]; t["life"] -= 1
            for e in self.enemies:
                if t["life"] > 0 and math.hypot(t["x"] - e["x"], t["y"] - e["y"]) < 20:
                    dealt = min(3.5, e["hp"])
                    e["hp"] -= dealt
                    self.ev["damage_dealt"] += dealt
                    t["life"] = 0
        self.tears = [t for t in self.tears if t["life"] > 0]
        before = len(self.enemies)
        self.enemies = [e for e in self.enemies if e["hp"] > 0]
        self.ev["kills"] += before - len(self.enemies)

        for e in self.enemies:
            ex, ey = self.px - e["x"], self.py - e["y"]
            d = math.hypot(ex, ey) or 1
            e["x"] += 1.2 * ex / d; e["y"] += 1.2 * ey / d
            e["t"] += 1
            if e["t"] % 60 == 0:
                self.projectiles.append({"x": e["x"], "y": e["y"], "vx": 5 * ex / d,
                                         "vy": 5 * ey / d, "life": 90})
            if d < 22:
                self._hurt()

        for p in self.projectiles:
            p["x"] += p["vx"]; p["y"] += p["vy"]; p["life"] -= 1
            if math.hypot(p["x"] - self.px, p["y"] - self.py) < 15:
                p["life"] = 0
                self._hurt()
        self.projectiles = [p for p in self.projectiles if p["life"] > 0]
        self.iframes = max(0, self.iframes - 1)

    def _hurt(self):
        if self.iframes:
            return
        self.ev["hits"] += 1
        self.ev["damage_taken"] += 1
        if self.arena and self.hearts <= 1:
            self.ev["lethal"] = True
            return
        self.hearts -= 1
        self.iframes = 30
