"""Per-enemy report card: fight each enemy alone N times and grade the agent.

Training reward is one number averaged over everything, so it hides weak
spots: "mean return 1.6" might mean "perfect on 12 enemies, hopeless against
Horfs". This shows it enemy by enemy.

It differs from the skill table in skills.json in two ways:
  - It's a controlled test: every enemy alone, the same number of tries each.
  - It uses deterministic actions (the policy's single best choice) instead
    of the random sampling used during training, so it shows what the agent
    has actually learned, without exploration noise.

    python scripts/eval_enemies.py --scripted
    python scripts/eval_enemies.py --model runs/ppo/latest.zip --episodes 5
    python scripts/eval_enemies.py --model runs/ppo/latest.zip --enemies horf clotty
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaac_ai.curriculum import BASEMENT, BOSSES, episode_score  # noqa: E402
from isaac_ai.env import IsaacArenaEnv  # noqa: E402


def make_policy(args, env):
    if args.scripted:
        from scripted_bot import scripted_action
        return lambda obs: scripted_action(env.state)
    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")
    return lambda obs: model.predict(obs, deterministic=True)[0]


def evaluate(env, policy, names, episodes):
    """Returns {name: dict(clear=, hits=, score=, ret=)} averaged over episodes."""
    report = {}
    for name in names:
        clears = hits = score = ret = 0.0
        for ep in range(episodes):
            obs, _ = env.reset(seed=ep, options={"enemies": [name]})
            done, ep_ret, info = False, 0.0, {}
            while not done:
                obs, r, term, trunc, info = env.step(policy(obs))
                ep_ret += r
                done = term or trunc
            clears += info["outcome"] == "cleared"
            hits += info["episode_hits"]
            score += episode_score(info["outcome"], info["episode_hits"])
            ret += ep_ret
        report[name] = {k: v / episodes for k, v in
                        (("clear", clears), ("hits", hits), ("score", score), ("ret", ret))}
        r = report[name]
        print(f"  {name:16} clear {r['clear']:4.0%}  hits {r['hits']:4.1f}  score {r['score']:4.2f}",
              flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--model", help="saved PPO .zip")
    who.add_argument("--scripted", action="store_true", help="grade the scripted bot")
    ap.add_argument("--episodes", type=int, default=5, help="tries per enemy")
    ap.add_argument("--enemies", nargs="+", help="names from isaac_ai/enemies.py")
    ap.add_argument("--bosses", action="store_true", help="include first-floor bosses")
    ap.add_argument("--port", type=int, default=28771)
    args = ap.parse_args()

    names = args.enemies or (BASEMENT + (BOSSES if args.bosses else []))
    env = IsaacArenaEnv(port=args.port)  # default curriculum: records nothing
    try:
        print(f"grading {len(names)} enemies x {args.episodes} episodes")
        report = evaluate(env, make_policy(args, env), names, args.episodes)
    finally:
        env.close()

    print("\nworst first:")
    for name, r in sorted(report.items(), key=lambda kv: kv[1]["score"]):
        bar = "#" * round(r["score"] * 20)
        print(f"  {name:16} {r['score']:4.2f} {bar}")
    mean = sum(r["score"] for r in report.values()) / len(report)
    print(f"\noverall score {mean:.2f}  (1.00 = every enemy cleared without a hit)")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for scripted_bot
    main()
