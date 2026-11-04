"""Step 2: a hand-written bot running in the training arena.

Why write a dumb bot before the AI?
  1. It proves the whole pipeline works (reset, spawning, control, rewards)
     before you spend hours training.
  2. It gives you a BASELINE score. If PPO can't beat this after training,
     something is wrong (rewards, features, bugs), not just "needs more time".

The strategy: line up with the nearest enemy, shoot along that axis, keep a
comfortable distance, and sidestep bullets heading our way.

    python scripts/scripted_bot.py --level 2 --episodes 20
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaac_ai.bridge import as_list  # noqa: E402
from isaac_ai.env import MOVES, IsaacArenaEnv  # noqa: E402


def to_move_index(v) -> int:
    dx = int(np.sign(v[0])) if abs(v[0]) > 0.3 else 0
    dy = int(np.sign(v[1])) if abs(v[1]) > 0.3 else 0
    return MOVES.index((dx, dy))


def scripted_action(state: dict):
    p = np.array(state["player"]["pos"], dtype=float)
    enemies = as_list(state.get("enemies"))
    if not enemies:
        return [0, 0]

    target = min(enemies, key=lambda e: np.linalg.norm(np.array(e["pos"]) - p))
    d = np.array(target["pos"]) - p
    dist = max(1e-6, float(np.linalg.norm(d)))

    # Shoot along the axis where the enemy is farther away; move to line up on the other.
    desired = np.zeros(2)
    if abs(d[0]) >= abs(d[1]):
        shoot = 2 if d[0] > 0 else 1
        desired[1] += np.sign(d[1]) if abs(d[1]) > 15 else 0
    else:
        shoot = 4 if d[1] > 0 else 3
        desired[0] += np.sign(d[0]) if abs(d[0]) > 15 else 0

    # Keep a comfortable distance.
    if dist < 140:
        desired -= d / dist * 1.5
    elif dist > 260:
        desired += d / dist

    # Sidestep bullets that are close and coming toward us.
    for pr in as_list(state.get("projectiles")):
        rel = p - np.array(pr["pos"])
        v = np.array(pr["vel"])
        if np.linalg.norm(rel) < 110 and np.dot(rel, v) > 0 and np.linalg.norm(v) > 0.1:
            perp = np.array([-v[1], v[0]]) / np.linalg.norm(v)
            if np.dot(perp, rel) < 0:
                perp = -perp
            desired += perp * 2

    # Stay away from walls.
    tl, br = state["room"]["tl"], state["room"]["br"]
    margin = 50
    desired[0] += float(p[0] < tl[0] + margin) - float(p[0] > br[0] - margin)
    desired[1] += float(p[1] < tl[1] + margin) - float(p[1] > br[1] - margin)

    return [to_move_index(desired), shoot]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--port", type=int, default=28771)
    args = ap.parse_args()

    env = IsaacArenaEnv(level=args.level, port=args.port)
    outcomes, total_hits, returns = {}, 0, []
    try:
        for ep in range(args.episodes):
            env.reset(seed=ep)
            done, ep_return, ep_hits, info = False, 0.0, 0, {}
            while not done:
                _, r, term, trunc, info = env.step(scripted_action(env.state))
                ep_return += r
                ep_hits += info["hits"]
                done = term or trunc
            outcomes[info["outcome"]] = outcomes.get(info["outcome"], 0) + 1
            total_hits += ep_hits
            returns.append(ep_return)
            print(f"episode {ep:>3}: {info['outcome']:9} return {ep_return:6.2f} hits {ep_hits}")
    finally:
        env.close()

    n = len(returns)
    print(f"\nlevel {args.level}, {n} episodes: mean return {np.mean(returns):.2f}, "
          f"hits/episode {total_hits / max(1, n):.2f}, outcomes {outcomes}")


if __name__ == "__main__":
    main()
