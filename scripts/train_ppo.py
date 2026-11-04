"""Step 3: train a PPO agent in the arena.

PPO (Proximal Policy Optimization) in one paragraph:
  The policy is a neural network: observation in, probabilities over actions
  out. PPO plays for `n_steps` steps, then looks back and asks "which actions
  turned out better than expected?" (the *advantage*, estimated by a second
  "value" network). It nudges the policy toward those actions, but CLIPS each
  update so the policy can't change too much at once. That clipping is the
  "proximal" part, and it is why PPO is stable enough to be the default choice.

Training runs on the CPU on purpose: with small MLPs, a GPU's copy overhead
costs more than it saves, and the game (30 updates/s) is the bottleneck anyway.

    python scripts/train_ppo.py --level 1 --steps 300000
    python scripts/train_ppo.py --level 2 --resume runs/ppo/latest.zip
    python scripts/train_ppo.py --play runs/ppo/latest.zip --level 2
    tensorboard --logdir runs/ppo/tb        # watch learning curves
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from isaac_ai.curriculum import Curriculum  # noqa: E402
from isaac_ai.env import IsaacArenaEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--out", default="runs/ppo")
    ap.add_argument("--resume", help="continue from a saved model .zip")
    ap.add_argument("--play", help="just run a saved model, no training")
    ap.add_argument("--port", type=int, default=28771)
    ap.add_argument("--n-steps", type=int, default=2048)
    ap.add_argument("--no-adaptive", action="store_true",
                    help="pick enemies uniformly instead of favouring weak spots")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # The skill table lives next to the model, so --resume continues both.
    # While just playing (--play) we don't record, so evaluation can't skew it.
    curriculum = Curriculum(adaptive=not (args.no_adaptive or args.play),
                            path=os.path.join(args.out, "skills.json"))
    # Monitor records episode return/length so SB3 can log them.
    env = Monitor(IsaacArenaEnv(level=args.level, port=args.port, curriculum=curriculum),
                  info_keywords=("outcome",))

    try:
        if args.play:
            play(PPO.load(args.play, device="cpu"), env)
            return

        if args.resume:
            model = PPO.load(args.resume, env=env, device="cpu")
        else:
            model = PPO(
                "MlpPolicy", env,
                n_steps=args.n_steps,   # steps collected before each update (~4.5 min of play)
                batch_size=256,
                n_epochs=10,            # passes over the collected data per update
                gamma=0.99,             # how much future reward counts (0.99 ~ 7 s horizon at 15 Hz)
                gae_lambda=0.95,
                learning_rate=3e-4,
                ent_coef=0.01,          # bonus for staying random early on = exploration
                clip_range=0.2,
                policy_kwargs=dict(net_arch=[256, 256]),
                tensorboard_log=os.path.join(args.out, "tb"),
                device="cpu",
                verbose=1,
            )

        ckpt = CheckpointCallback(save_freq=20_000, save_path=os.path.join(args.out, "checkpoints"),
                                  name_prefix=f"level{args.level}")
        model.learn(total_timesteps=args.steps, callback=ckpt,
                    reset_num_timesteps=not args.resume, tb_log_name=f"level{args.level}")
    except KeyboardInterrupt:
        print("interrupted, saving...")
    finally:
        if not args.play and "model" in locals():
            path = os.path.join(args.out, "latest.zip")
            model.save(path)
            print(f"saved {path}")
        env.close()


def play(model, env, episodes: int = 10):
    for ep in range(episodes):
        obs, _ = env.reset()
        done, ret, info = False, 0.0, {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ret += r
            done = term or trunc
        print(f"episode {ep}: {info.get('outcome')} return {ret:.2f}")


if __name__ == "__main__":
    main()
