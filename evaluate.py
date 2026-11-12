"""Tests of a Tello policy (actor.npz) and of the driver's hold controller on the same flights.

python evaluate.py runs/seed0/actor.npz

Two sets of 100 flights (seed 12345, the same for every policy): "nominal" (the identified model exactly, no drift,
no noise) and "randomized" (the training distribution: K, tau, delay, drift and noise drawn per flight). Each flight
starts with the target up to 2 m away and lasts 10 s. Per set:
  success      share of flights that stay within 3 m of the target
  hold_rmse_m  RMS distance to the target over the last 5 s (completed flights)
  reach_s      median time to come within 10 cm of the target and stay there (flights that do)
  overshoot_m  median largest distance past the target along the start direction
  command_change  mean |du| per step (smoothness, in [-1, 1] units)
"""
import json
import sys

import numpy as np

from tello_env import TelloEnv, hold_controller


class NumpyActor:
    def __init__(self, path):
        w = np.load(path)
        self.layers = [(w[f"W{i}"], w[f"b{i}"]) for i in range(10) if f"W{i}" in w]
        self.mu = (w["Wmu"], w["bmu"])

    def __call__(self, obs):
        h = np.asarray(obs, dtype=np.float64)
        for W, b in self.layers:
            h = np.maximum(W @ h + b, 0.0)
        return np.tanh(self.mu[0] @ h + self.mu[1])


def fly(env, controller, seed):
    obs, _ = env.reset(seed=seed)
    start_dir = env.target / (np.linalg.norm(env.target) + 1e-9)
    state, errors, along, changes, last = {}, [], [], [], np.zeros(3)
    done = False
    while not done:
        a = controller(obs, state) if controller is hold_controller else controller(obs)
        changes.append(np.linalg.norm(np.clip(a, -1, 1) - last))
        last = np.clip(a, -1, 1)
        obs, _, terminated, truncated, info = env.step(a)
        errors.append(info["error"])
        along.append(float((env.p - env.target) @ start_dir))
        done = terminated or truncated
    errors = np.array(errors)
    reach = None
    inside = errors < 0.10
    for k in range(len(errors)):
        if inside[k:].all():
            reach = (k + 1) * 0.05
            break
    return {"ok": not terminated, "hold": float(np.sqrt(np.mean(errors[len(errors) // 2:] ** 2))),
            "reach": reach, "overshoot": max(0.0, max(along)), "change": float(np.mean(changes))}


def evaluate(controller, randomize, n=100):
    env = TelloEnv(randomize=randomize)
    flights = [fly(env, controller, 12345 + k) for k in range(n)]
    ok = [f for f in flights if f["ok"]]
    reached = [f["reach"] for f in ok if f["reach"] is not None]
    return {"success": len(ok) / n,
            "hold_rmse_m": round(float(np.mean([f["hold"] for f in ok])), 4) if ok else None,
            "reached_10cm": len(reached) / n,
            "reach_s": round(float(np.median(reached)), 2) if reached else None,
            "overshoot_m": round(float(np.median([f["overshoot"] for f in ok])), 3) if ok else None,
            "command_change": round(float(np.mean([f["change"] for f in flights])), 4)}


def evaluate_all(actor_path):
    actor = NumpyActor(actor_path)
    return {name: {"policy": evaluate(actor, randomize), "hold_controller": evaluate(hold_controller, randomize)}
            for name, randomize in (("nominal", False), ("randomized", True))}


if __name__ == "__main__":
    print(json.dumps(evaluate_all(sys.argv[1]), indent=2))
