"""Tests of a Tello tracking policy (actor.npz, 41 inputs) and of our model-based tracker on the same flights.

python evaluate_track.py runs/<run>/actor.npz

Three paths, the sizes the flight node uses in the free zone: a circle of radius 0.5 m and a figure-eight 1.0 m long
(0.5 m wide), one lap in 10 s, and a fixed point. 30 flights each on the randomized model (K, tau, delay, drift, EKF
noise drawn per flight, seed 12345) and 10 on the identified model, 15 s each. Per path:
  success    share of flights that stay within 2 m of the point
  rmse_m     RMS distance to the point after the first 3 s (completed flights)
  max_m      median of the largest distance after the first 3 s
  command_change  mean |du| per step
"""
import json
import sys

import numpy as np

from evaluate import NumpyActor
from tello_track_env import RefPath, TrackEnv, track_controller

PATHS = {"circle": RefPath("circle", size=0.5, period=10.0),
         "eight": RefPath("eight", size=0.5, period=10.0),
         "hold": RefPath("hold")}


def fly(env, controller, seed):
    obs, _ = env.reset(seed=seed)
    state, errors, changes, last, done = {}, [], [], np.zeros(3), False
    while not done:
        a = controller(env, obs, state) if controller is track_controller else controller(obs)
        a = np.clip(a, -1, 1)
        changes.append(float(np.linalg.norm(a - last)))
        last = a
        obs, _, terminated, truncated, info = env.step(a)
        errors.append(info["error"])
        done = terminated or truncated
    late = np.array(errors[60:]) if len(errors) > 60 else np.array(errors)
    return {"ok": not terminated, "rmse": float(np.sqrt(np.mean(late ** 2))), "max": float(late.max()),
            "change": float(np.mean(changes))}


def evaluate(controller, path, randomize, n):
    env = TrackEnv(randomize=randomize, paths=[path])
    flights = [fly(env, controller, 12345 + k) for k in range(n)]
    ok = [f for f in flights if f["ok"]]
    return {"success": len(ok) / n,
            "rmse_m": round(float(np.mean([f["rmse"] for f in ok])), 4) if ok else None,
            "max_m": round(float(np.median([f["max"] for f in ok])), 3) if ok else None,
            "command_change": round(float(np.mean([f["change"] for f in flights])), 4)}


def evaluate_all_track(actor_path):
    actor = NumpyActor(actor_path)
    out = {}
    for name, path in PATHS.items():
        out[name] = {}
        for model, randomize, n in (("randomized", True, 30), ("nominal", False, 10)):
            out[name][model] = {"policy": evaluate(actor, path, randomize, n),
                                "tracker": evaluate(track_controller, path, randomize, n)}
    return out


if __name__ == "__main__":
    print(json.dumps(evaluate_all_track(sys.argv[1]), indent=2))
