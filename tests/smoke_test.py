"""End-to-end check of the Python side against the mock game.

    python tests/smoke_test.py          # env + scripted bot
    python tests/smoke_test.py --ppo    # also a tiny PPO run (needs stable-baselines3)
"""

import argparse
import os
import sys
import tempfile
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_enemies import evaluate  # noqa: E402
from mock_game import MockIsaac  # noqa: E402
from scripted_bot import scripted_action  # noqa: E402

from isaac_ai.bridge import IsaacBridge  # noqa: E402
from isaac_ai.curriculum import Curriculum  # noqa: E402
from isaac_ai.enemies import VOCAB, type_index  # noqa: E402
from isaac_ai.env import IsaacArenaEnv  # noqa: E402
from isaac_ai.features import OBS_DIM  # noqa: E402


def run_episodes(env, policy, n):
    results = []
    for ep in range(n):
        obs, _ = env.reset(seed=ep)
        assert obs.shape == (OBS_DIM,) and env.observation_space.contains(obs)
        done, ret, info = False, 0.0, {}
        while not done:
            obs, r, term, trunc, info = env.step(policy())
            ret += r
            done = term or trunc
        results.append((info["outcome"], round(ret, 2)))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo", action="store_true")
    args = ap.parse_args()

    # Type lookup: exact match, same-type fallback, unknown -> "other" (slot 0).
    assert VOCAB[type_index(10, 1)] == "frowning_gaper"
    assert VOCAB[type_index(10, 7)] == "gaper"
    assert type_index(999, 0) == 0

    port = 28799
    bridge = IsaacBridge(port=port)
    threading.Thread(target=MockIsaac(port=port).run, daemon=True).start()
    bridge.wait_for_game(quiet=True)

    with tempfile.TemporaryDirectory() as tmp:
        skills = os.path.join(tmp, "skills.json")
        curriculum = Curriculum(adaptive=True, path=skills)
        env = IsaacArenaEnv(level=3, max_steps=400, bridge=bridge, curriculum=curriculum)

        print("random  :", run_episodes(env, env.action_space.sample, 3))
        print("scripted:", run_episodes(env, lambda: scripted_action(env.state), 5))

        # The adaptive curriculum recorded those 8 episodes and saved them.
        assert curriculum.skill and os.path.exists(skills)
        reloaded = Curriculum(adaptive=True, path=skills)
        assert reloaded.skill == curriculum.skill
        print("skills  :\n" + reloaded.table())

        # Forced single-enemy fights, as eval_enemies.py does.
        report = evaluate(env, lambda obs: scripted_action(env.state), ["horf", "gaper"], 2)
        assert set(report) == {"horf", "gaper"}

        if args.ppo:
            from stable_baselines3 import PPO
            from stable_baselines3.common.env_checker import check_env
            check_env(env, warn=True)
            model = PPO("MlpPolicy", env, n_steps=256, batch_size=64, device="cpu", verbose=0)
            model.learn(total_timesteps=512)
            print("ppo     : learned 512 steps OK")

        env.close()
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
