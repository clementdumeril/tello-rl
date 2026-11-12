"""SAC on the Tello model (tello_env.py), then the tests of evaluate.py and the actor exported for numpy.

python train.py --steps 300000 --seed 0 --out runs/seed0
python train.py --task track ...   (circles and figure-eights: tello_track_env.py, tests of evaluate_track.py)

Writes into --out: model.zip (Stable-Baselines3), actor.npz (weights for the ROS node, no torch needed),
progress.csv (episodes), evaluation.json (evaluate.py on the policy and on the driver's hold controller).
Actor 64 x 64 as in Learning to Fly in Seconds; critics 256 x 256 (Stable-Baselines3 default).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from tello_env import GAMMA, TelloEnv
from evaluate import evaluate_all


class Progress(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["steps", "episodes", "mean_return", "mean_length", "mean_final_error_m", "seconds"])
        self.returns, self.lengths, self.errors, self.ep = [], [], [], 0
        self.t0 = time.time()

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "episode" in info:
                self.ep += 1
                self.returns.append(info["episode"]["r"])
                self.lengths.append(info["episode"]["l"])
                self.errors.append(info["error"])
        if self.num_timesteps % 10000 == 0 and self.returns:
            self.writer.writerow([self.num_timesteps, self.ep, np.mean(self.returns[-50:]),
                                  np.mean(self.lengths[-50:]), np.mean(self.errors[-50:]),
                                  round(time.time() - self.t0, 1)])
            self.file.flush()
        return True


def export_actor(model, path):
    """The deterministic actor as numpy arrays: relu(W0 x + b0) -> relu(W1 h + b1) -> tanh(Wmu h + bmu)."""
    actor = model.policy.actor
    layers = [m for m in actor.latent_pi if isinstance(m, torch.nn.Linear)]
    arrays = {}
    for i, layer in enumerate(layers):
        arrays[f"W{i}"] = layer.weight.detach().cpu().numpy()
        arrays[f"b{i}"] = layer.bias.detach().cpu().numpy()
    arrays["Wmu"] = actor.mu.weight.detach().cpu().numpy()
    arrays["bmu"] = actor.mu.bias.detach().cpu().numpy()
    np.savez(path, **arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--out", required=True)
    # updates per step of the 4 environments: 1 = one update per 4 transitions (first run, 125k updates in 500k
    # steps, under-trained), 4 = one per transition
    ap.add_argument("--gradient-steps", type=int, default=1)
    ap.add_argument("--bonus", type=float, default=0.0, help="closeness bonus weight (tello_env.TelloEnv)")
    ap.add_argument("--bonus-scale", type=float, default=0.15, help="closeness bonus width, m")
    ap.add_argument("--task", choices=["hold", "track"], default="hold")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    from stable_baselines3.common.monitor import Monitor
    if args.task == "track":
        from tello_track_env import TrackEnv as Env
        from evaluate_track import evaluate_all_track as evaluate_run
    else:
        Env, evaluate_run = TelloEnv, evaluate_all
    env = DummyVecEnv([lambda i=i: Monitor(Env(seed=args.seed * 100 + i, bonus_weight=args.bonus,
                                               bonus_scale=args.bonus_scale)) for i in range(args.n_envs)])
    model = SAC("MlpPolicy", env, gamma=GAMMA, learning_rate=3e-4, batch_size=256, buffer_size=args.steps,
                learning_starts=5000, train_freq=1, gradient_steps=args.gradient_steps, seed=args.seed, verbose=0,
                policy_kwargs={"net_arch": {"pi": [64, 64], "qf": [256, 256]}})
    config = {"steps": args.steps, "seed": args.seed, "n_envs": args.n_envs, "gamma": GAMMA, "batch": 256,
              "actor": [64, 64], "critic": [256, 256], "algorithm": "SAC (stable-baselines3)",
              "gradient_steps": args.gradient_steps, "bonus_weight": args.bonus, "bonus_scale": args.bonus_scale,
              "task": args.task}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=Progress(out / "progress.csv"))
    minutes = (time.time() - t0) / 60
    model.save(out / "model.zip")
    export_actor(model, out / "actor.npz")
    result = evaluate_run(out / "actor.npz")
    result["training_minutes"] = round(minutes, 1)
    (out / "evaluation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
