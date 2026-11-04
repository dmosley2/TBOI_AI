"""The enemies the AI knows by name.

Two jobs:
  1. CATALOG maps a readable name to the IDs the game uses to spawn it,
     (EntityType, Variant, SubType). These are the same numbers as the console
     `spawn` command.
  2. The type VOCABULARY gives every known enemy a slot in a one-hot vector,
     so the network can tell a Horf (shoots) from a Gaper (chases) without
     having to infer it from how they move.

To teach the AI a new enemy: add it to CATALOG, then put its name in a pool in
curriculum.py. Adding an entry changes the observation size, so models
trained before the change can't be loaded afterwards (see features.py).
"""

from __future__ import annotations

CATALOG: dict[str, tuple[int, int, int]] = {
    # --- Basement regulars ---
    "fly":            (13, 0, 0),
    "attack_fly":     (18, 0, 0),
    "gaper":          (10, 0, 0),
    "frowning_gaper": (10, 1, 0),
    "gusher":         (11, 0, 0),   # also appears when a Gaper loses its head
    "pooter":         (14, 0, 0),
    "horf":           (12, 0, 0),
    "clotty":         (15, 0, 0),
    "hopper":         (29, 0, 0),
    "charger":        (23, 0, 0),
    "maw":            (26, 0, 0),
    "maggot":         (21, 0, 0),
    "spider":         (85, 0, 0),
    "boom_fly":       (25, 0, 0),
    # --- First-floor bosses ---
    "monstro":        (20, 0, 0),
    "duke_of_flies":  (67, 0, 0),
    "gemini":         (79, 0, 0),
}

# Index 0 is reserved for "an enemy we don't have a name for" (anything a
# boss spawns, enemies on later floors, ...). The network still sees its
# position and HP; it just can't tell what kind of enemy it is.
VOCAB: list[str] = ["other"] + list(CATALOG)
VOCAB_SIZE = len(VOCAB)

_BY_TYPE_VAR = {(t, v): i for i, (t, v, _s) in enumerate(CATALOG.values(), start=1)}
_BY_TYPE = {}
for (t, _v), i in _BY_TYPE_VAR.items():
    _BY_TYPE.setdefault(t, i)  # first variant listed wins, e.g. 10 -> gaper


def type_index(etype: int, variant: int) -> int:
    """Vocabulary slot for an enemy seen in the game.

    Exact (type, variant) first, then any variant of the same type (an
    unknown Gaper variant still behaves a lot like a Gaper), else "other".
    """
    return _BY_TYPE_VAR.get((etype, variant)) or _BY_TYPE.get(etype, 0)


def spawn_spec(name: str) -> list[int]:
    return list(CATALOG[name])
