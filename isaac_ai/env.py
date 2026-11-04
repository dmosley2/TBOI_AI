"""Gymnasium environment: clear a room of spawned enemies without getting hit.

This is the standard interface RL libraries expect:

    obs, info = env.reset()                 # start an episode
    obs, reward, terminated, truncated, info = env.step(action)

One episode = one "arena" fight in whatever room the player is standing in
(use the Basement 1 starting room). The mod wipes the room, heals the player,
and spawns enemies from the curriculum.

ACTION: two choices made at the same time (MultiDiscrete([9, 5])):
    move  : 0 none, 1 L, 2 R, 3 U, 4 D, 5 UL, 6 UR, 7 DL, 8 DR
    shoot : 0 none, 1 L, 2 R, 3 U, 4 D

REWARD: what we want, turned into numbers. The agent only ever maximises
this, so it IS the goal from the agent's point of view:
    + damage dealt, scaled so killing everything adds up to +1
    + 1 bonus for clearing the room
    - 0.5 per hit taken
    - 1 for "dying" (the mod cancels the killing blow) or leaving the room
    - a small cost per step, so finishing faster is better than stalling
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .bridge import DEFAULT_PORT, IsaacBridge, as_list
from .curriculum import Curriculum, episode_score, spawn_spec
from .features import OBS_DIM, encode

MOVES = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]

REWARD_CLEAR = 1.0
REWARD_HIT = 0.5
REWARD_DEATH = 1.0
TIME_PENALTY = 0.002


class IsaacArenaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, level: int = 1, frame_skip: int = 2, max_steps: int = 1500,
                 port: int = DEFAULT_PORT, bridge: IsaacBridge | None = None,
                 curriculum: Curriculum | None = None):
        super().__init__()
        self.level = level
        # Decides which enemies each episode gets. Training passes an adaptive
        # one; the default picks uniformly and records nothing.
        self.curriculum = curriculum or Curriculum()
        # frame_skip: repeat each action for N game updates. The game runs at
        # 30 updates/s; deciding 15 times a second is plenty and halves the
        # number of decisions the agent has to learn from.
        self.frame_skip = frame_skip
        self.max_steps = max_steps

        self.observation_space = gym.spaces.Box(-5.0, 5.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([len(MOVES), 5])

        self.bridge = bridge or IsaacBridge(port=port)
        if self.bridge.state is None:
            self.bridge.wait_for_game()

        self.state: dict = self.bridge.state
        self._steps = 0
        self._total_hp = 1.0
        self._room_index = None
        self._names: list[str] = []
        self._ep_hits = 0

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # seeds self.np_random
        options = options or {}
        # options["enemies"] = ["horf", ...] forces a specific fight (used by
        # eval_enemies.py); otherwise the curriculum picks.
        names = options.get("enemies") or self.curriculum.sample(
            options.get("level", self.level), self.np_random)
        self._names = list(names)
        self._ep_hits = 0

        self.state = self.bridge.step({
            "control": True, "move": [0, 0], "shoot": 0,
            "cmd": "reset_arena", "enemies": [spawn_spec(n) for n in names],
            "seed": int(self.np_random.integers(1, 2**31 - 1)),
        })
        # Give the spawns a few updates to show up if needed.
        for _ in range(10):
            if as_list(self.state.get("enemies")):
                break
            self.state = self.bridge.step(self._msg(0, 0))

        self._steps = 0
        self._room_index = self.state["room"]["index"]
        self._total_hp = max(1.0, sum(e["max_hp"] for e in as_list(self.state.get("enemies"))))
        return encode(self.state, 0.0), {"enemies": self._names}

    def step(self, action):
        move, shoot = int(action[0]), int(action[1])
        reward = -TIME_PENALTY
        hits = 0
        terminated = False
        outcome = None

        for _ in range(self.frame_skip):
            s = self.state = self.bridge.step(self._msg(move, shoot))
            ev = s.get("events") or {}
            reward += ev.get("damage_dealt", 0.0) / self._total_hp
            hits += ev.get("hits", 0)
            reward -= REWARD_HIT * ev.get("hits", 0)

            if ev.get("lethal") or s["player"]["dead"]:
                reward -= REWARD_DEATH
                terminated, outcome = True, "died"
            elif s["room"]["index"] != self._room_index:
                reward -= REWARD_DEATH
                terminated, outcome = True, "left_room"
            elif not as_list(s.get("enemies")):
                reward += REWARD_CLEAR
                terminated, outcome = True, "cleared"
            if terminated:
                break

        self._steps += 1
        truncated = not terminated and self._steps >= self.max_steps
        if truncated:
            outcome = "timeout"

        self._ep_hits += hits
        obs = encode(self.state, self._steps / self.max_steps)
        info = {"hits": hits}
        if outcome:
            info["outcome"] = outcome
            info["episode_hits"] = self._ep_hits
            info["enemies"] = self._names
            self.curriculum.record(self._names, episode_score(outcome, self._ep_hits))
        return obs, float(reward), terminated, truncated, info

    def close(self):
        self.bridge.close()

    # ------------------------------------------------------------------
    @staticmethod
    def _msg(move: int, shoot: int) -> dict:
        dx, dy = MOVES[move]
        return {"control": True, "move": [dx, dy], "shoot": shoot}
