"""Turn a raw game snapshot into a fixed-size vector for the neural network.

Neural networks need the same input size every step, but a room can hold any
number of enemies and bullets. The usual trick: keep the K *nearest* of each,
pad the rest with zeros, and add a "present" flag so the network can tell a
real entry from padding.

Positions are made RELATIVE to the player (enemy at +100 px means 100 px to
the right). That way "dodge the bullet just left of me" looks the same
wherever you stand, which makes learning much faster.

Everything is scaled to roughly [-1, 1]. Networks train badly on raw pixel
values in the hundreds.

Enemy TYPE is a one-hot vector: all zeros except a 1 in that enemy's slot (see
enemies.py). Why not one number (Gaper = 10, Horf = 12)? Because the network
would read that as an amount: Horf would count as "a bit more than a Gaper".
One-hot keeps every type separate. The first layer's weights for each slot
effectively become a learned description of that enemy, which is the same
thing an embedding layer would learn.

OBS_DIM depends on the vocabulary size. Change the catalog and old saved
models no longer fit; SB3 refuses to load them with a shape error.
"""

from __future__ import annotations

import math

import numpy as np

from .bridge import as_list
from .enemies import VOCAB_SIZE, type_index

K_ENEMIES = 8
K_PROJECTILES = 12
GRID_WINDOW = 7  # 7x7 cells of terrain around the player (1 cell = 40 px)

PLAYER_FEATURES = 11
ENEMY_BASE_FEATURES = 9
ENEMY_FEATURES = ENEMY_BASE_FEATURES + VOCAB_SIZE
PROJECTILE_FEATURES = 6
MISC_FEATURES = 3

OBS_DIM = (PLAYER_FEATURES + K_ENEMIES * ENEMY_FEATURES + K_PROJECTILES * PROJECTILE_FEATURES
           + GRID_WINDOW * GRID_WINDOW + MISC_FEATURES)

POS_SCALE = 400.0
VEL_SCALE = 10.0


def _nearest(items, px, py, k):
    def d2(it):
        x, y = it["pos"]
        return (x - px) ** 2 + (y - py) ** 2
    return sorted(items, key=d2)[:k]


def encode(state: dict, time_frac: float = 0.0) -> np.ndarray:
    p = state["player"]
    room = state["room"]
    px, py = p["pos"]
    tl, br = room["tl"], room["br"]
    cx, cy = (tl[0] + br[0]) / 2, (tl[1] + br[1]) / 2
    hw, hh = max(1.0, (br[0] - tl[0]) / 2), max(1.0, (br[1] - tl[1]) / 2)

    out: list[float] = [
        (px - cx) / hw, (py - cy) / hh,              # where in the room am I
        p["vel"][0] / VEL_SCALE, p["vel"][1] / VEL_SCALE,
        p["hearts"] / 24, p["max_hearts"] / 24, p["soul"] / 24,
        p["damage"] / 10, p["fire_delay"] / 20, p["speed"] / 2, p["shot_speed"] / 2,
    ]

    enemies = as_list(state.get("enemies"))
    for e in _pad(_nearest(enemies, px, py, K_ENEMIES), K_ENEMIES):
        if e is None:
            out += [0.0] * ENEMY_FEATURES
            continue
        dx, dy = e["pos"][0] - px, e["pos"][1] - py
        out += [
            1.0,
            dx / POS_SCALE, dy / POS_SCALE, math.hypot(dx, dy) / POS_SCALE,
            e["vel"][0] / VEL_SCALE, e["vel"][1] / VEL_SCALE,
            e["hp"] / max(1.0, e["max_hp"]),
            e["size"] / 40,
            1.0 if e.get("boss") else 0.0,
        ]
        one_hot = [0.0] * VOCAB_SIZE
        one_hot[type_index(e["type"], e.get("var", 0))] = 1.0
        out += one_hot

    projectiles = as_list(state.get("projectiles"))
    for pr in _pad(_nearest(projectiles, px, py, K_PROJECTILES), K_PROJECTILES):
        if pr is None:
            out += [0.0] * PROJECTILE_FEATURES
            continue
        dx, dy = pr["pos"][0] - px, pr["pos"][1] - py
        out += [
            1.0,
            dx / POS_SCALE, dy / POS_SCALE, math.hypot(dx, dy) / POS_SCALE,
            pr["vel"][0] / VEL_SCALE, pr["vel"][1] / VEL_SCALE,
        ]

    out += _grid_window(room, p.get("grid", -1))
    out += [len(enemies) / 10, 1.0 if room.get("clear") else 0.0, time_frac]

    arr = np.asarray(out, dtype=np.float32)
    assert arr.shape == (OBS_DIM,), arr.shape
    return np.clip(arr, -5.0, 5.0)


def _pad(items, k):
    return list(items) + [None] * (k - len(items))


def _grid_window(room: dict, player_idx: int) -> list[float]:
    """Terrain around the player: 0 free, 0.5 pit, 1 blocked (or off the map)."""
    w, h, grid = room["w"], room["h"], room["grid"]
    half = GRID_WINDOW // 2
    if player_idx < 0 or w <= 0:
        return [1.0] * (GRID_WINDOW * GRID_WINDOW)
    pc, pr = player_idx % w, player_idx // w
    vals = []
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            r, c = pr + dr, pc + dc
            if not (0 <= r < h and 0 <= c < w) or r * w + c >= len(grid):
                vals.append(1.0)
                continue
            ch = grid[r * w + c]
            vals.append(0.0 if ch == "0" else 0.5 if ch == "1" else 1.0)
    return vals
