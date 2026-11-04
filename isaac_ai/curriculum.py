"""Which enemies to spawn in each training episode.

A curriculum means starting easy and making it harder once the agent copes.
An agent dropped straight into a Monstro fight gets hit constantly, earns
almost nothing but penalties, and never works out what shooting is for.

Two ideas combine here:

1. LEVELS: the room gets harder in steps: one easy enemy, then one enemy of
   any kind, then mixed groups, then bosses.

2. MIXED TYPES, ADAPTIVE WEIGHTS: within a level the enemy type is picked at
   random EVERY episode, rather than training on one type for hours and then
   the next. Training on a single type for hours makes the network forget the
   earlier ones ("catastrophic forgetting"), because nothing in the data
   reminds it of them.

   To still focus where it's needed, the sampler keeps a running skill score
   per enemy and spawns the enemies the agent is BAD at more often. It never
   drops a mastered enemy completely (`floor`), so those stay in practice.

Show the learned skill table:
    python -m isaac_ai.curriculum runs/ppo/skills.json
"""

from __future__ import annotations

import json
import os
import sys

from .enemies import CATALOG, spawn_spec

EASY = ["fly", "attack_fly", "gaper"]
BASEMENT = ["fly", "attack_fly", "gaper", "frowning_gaper", "pooter", "horf", "clotty",
            "hopper", "charger", "maw", "maggot", "spider", "boom_fly"]
BOSSES = ["monstro", "duke_of_flies", "gemini"]

LEVELS: dict[int, tuple[list[str], tuple[int, int]]] = {
    # level: (pool, (min enemies, max enemies))
    1: (EASY, (1, 1)),        # one easy enemy: learn to aim and that shooting pays
    2: (BASEMENT, (1, 1)),    # one of anything: learn each enemy's behaviour
    3: (BASEMENT, (2, 3)),    # small mixed groups: juggle threats
    4: (BASEMENT, (3, 6)),    # full rooms
    5: (BOSSES, (1, 1)),      # first-floor bosses
}

for _pool, _ in LEVELS.values():
    for _name in _pool:
        assert _name in CATALOG, f"curriculum names unknown enemy {_name!r}"


def episode_score(outcome: str | None, hits: int) -> float:
    """How well an episode went, from 0 to 1: a clean clear = 1, each hit halves it, no clear = 0."""
    return (0.5 ** hits) if outcome == "cleared" else 0.0


class Curriculum:
    def __init__(self, adaptive: bool = False, path: str | None = None,
                 alpha: float = 0.05, floor: float = 0.15):
        """
        adaptive: if False, all enemies in a pool are equally likely and
                  nothing is recorded (used by eval and the scripted bot).
        path:     JSON file the skill table is saved to and loaded from, so it
                  survives restarts and --resume.
        alpha:    how fast skill scores move. Each episode moves the score 5%
                  of the way toward that episode's result, so it reflects
                  roughly the last 20 episodes with that enemy.
        floor:    minimum weight, so even mastered enemies keep showing up.
        """
        self.adaptive = adaptive
        self.path = path
        self.alpha = alpha
        self.floor = floor
        self.skill: dict[str, float] = {}
        self.episodes: dict[str, int] = {}
        if path and os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            self.skill = data.get("skill", {})
            self.episodes = data.get("episodes", {})

    def weight(self, name: str) -> float:
        # Unseen enemies count as skill 0, so they get the highest weight.
        return max(self.floor, 1.0 - self.skill.get(name, 0.0))

    def sample(self, level: int, rng) -> list[str]:
        pool, (lo, hi) = LEVELS[level]
        n = int(rng.integers(lo, hi + 1))
        if self.adaptive:
            w = [self.weight(name) for name in pool]
            total = sum(w)
            probs = [x / total for x in w]
        else:
            probs = None  # uniform
        return [pool[int(i)] for i in rng.choice(len(pool), size=n, p=probs)]

    def record(self, names: list[str], score: float):
        """Update skill for every enemy type that was in the episode.

        With several enemy types in one room we can't tell which one caused a
        hit, so they all share the result. On average this is fair, and
        levels 1-2 (single enemy) give each type clean data.
        """
        if not self.adaptive:
            return
        for name in set(names):
            old = self.skill.get(name, 0.0)
            self.skill[name] = old + self.alpha * (score - old)
            self.episodes[name] = self.episodes.get(name, 0) + 1
        self.save()

    def save(self):
        if not self.path:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"skill": self.skill, "episodes": self.episodes}, f, indent=1)
        os.replace(tmp, self.path)  # atomic: a crash never leaves a half-written file

    def table(self) -> str:
        rows = sorted(self.skill, key=lambda n: self.skill[n])
        lines = [f"{'enemy':16} {'skill':>6} {'weight':>7} {'episodes':>9}"]
        for n in rows:
            lines.append(f"{n:16} {self.skill[n]:6.2f} {self.weight(n):7.2f} {self.episodes.get(n, 0):9}")
        return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m isaac_ai.curriculum runs/ppo/skills.json")
    print(Curriculum(adaptive=True, path=sys.argv[1]).table())


__all__ = ["Curriculum", "LEVELS", "episode_score", "spawn_spec"]
