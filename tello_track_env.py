"""The Tello following a point that moves on a circle or a figure-eight (or stays still), on the identified model.

Same drone model, randomization, commands and reward form as tello_env.py (the point-hold task); what changes:
  - each episode draws a path: a fixed point (1 in 4), a circle (radius 0.3-0.7 m) or a figure-eight (0.6-1.2 m long,
    half as wide), one lap in 8-16 s, any orientation and direction; the drone starts within 0.5 m of where the point is
  - the policy also sees the point's velocity and where the point will be 0.5 s later (the Tello needs 0.3 s before a
    command acts: without seeing ahead, a policy follows a turning point with a lag, as the airplane did)
    observation: error to the point (3), velocity (3), drift estimate (2), error on a 10 cm scale (3), the point's
    velocity (3), error to the point 0.5 s ahead (3), last 8 commands (24)   -> 41 values
  - the velocity cost is on the velocity error to the point's velocity (0 for a fixed point, as before)
  - an episode lasts 15 s and ends early beyond 2 m from the point
The paths themselves are in rl_paths.py, a copy of the flight node's, so that the drone
flies exactly the observation it was trained on.
"""
import importlib.util
import math
from pathlib import Path

import gymnasium as gym
import numpy as np

from tello_env import DT, GAMMA, HISTORY, TelloEnv

# rl_paths.py: next to this file (this repository), else in the ROS package of tello-balloon-hunter
_here = Path(__file__).resolve().parent
_file = next(p for p in (_here / "rl_paths.py", _here.parent / "src" / "tello_driver" / "tello_driver" / "rl_paths.py")
             if p.exists())
_spec = importlib.util.spec_from_file_location("rl_paths", _file)
rl_paths = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rl_paths)
PREVIEW_S, RefPath, track_observation = rl_paths.PREVIEW_S, rl_paths.Path, rl_paths.track_observation

TRACK_STEPS = 300  # 15 s


class TrackEnv(TelloEnv):
    def __init__(self, randomize=True, seed=None, bonus_weight=0.0, bonus_scale=0.15, paths=None):
        super().__init__(randomize=randomize, seed=seed, bonus_weight=bonus_weight, bonus_scale=bonus_scale)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (17 + 3 * HISTORY,), np.float32)
        self.fixed_paths = paths  # for the tests: a list of RefPath to draw from instead of random ones

    def _draw_path(self):
        r = self.rng
        if self.fixed_paths is not None:
            return self.fixed_paths[r.integers(len(self.fixed_paths))]
        kind = r.choice(["hold", "circle", "eight"], p=[0.25, 0.35, 0.40])
        size = r.uniform(0.3, 0.7) if kind == "circle" else r.uniform(0.3, 0.6)
        return RefPath(kind, size=size, period=r.uniform(8, 16), angle=r.uniform(0, 2 * math.pi),
                       direction=r.choice([-1, 1]), phase=r.uniform(0, 1))

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._draw_model()
        r = self.rng
        self.path = self._draw_path()
        self.t = 0.0
        start, _, _ = self.path.at(0.0)
        if self.path.kind == "hold":  # a fixed point up to 1.5 m away, as in the hold task
            angle, dist = r.uniform(0, 2 * math.pi), 1.5 * math.sqrt(r.uniform())
            self.p = start - np.array([dist * math.cos(angle), dist * math.sin(angle), r.uniform(-0.4, 0.4)])
        else:  # near the point's start
            self.p = start - np.clip(r.normal(0, 0.25, 3), -0.5, 0.5) * np.array([1, 1, 0.5])
        self.v = np.clip(r.normal(0, 0.15, 3), -0.3, 0.3) * np.array([1, 1, 0.5])
        self.queue = np.zeros((9, 3))
        self.history = np.zeros((HISTORY, 3))
        self.drift_estimate = self.drift[:2] + r.normal(0, self.noise_d, 2)
        self.steps = 0
        return self._observe(), {}

    def _observe(self):
        r = self.rng
        p_seen = self.p + r.normal(0, self.noise_p, 3)
        v_seen = self.v + r.normal(0, self.noise_v, 3)
        return track_observation(self.path, self.t, p_seen, v_seen, self.drift_estimate, self.history).astype(np.float32)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        du = self._advance(a)
        self.t += DT
        ref, v_ref, _ = self.path.at(self.t)
        e = float(np.linalg.norm(ref - self.p))
        terminated = bool(e > 2.0)
        if terminated:
            reward = -2.0 / (1 - GAMMA)
        else:
            cost = 2.0 * e + 0.1 * float(np.linalg.norm(self.v - v_ref)) + 0.5 * float(np.linalg.norm(du)) \
                   + 0.1 * float(a @ a)
            reward = max(2.0 - cost, -2.0) + self.bonus_weight * math.exp(-(e / self.bonus_scale) ** 2)
        truncated = self.steps >= TRACK_STEPS
        return self._observe(), float(reward), terminated, truncated, {"error": e}


def track_controller(env, obs, state):
    """Our model-based tracker on the same flight (strike.py track(): feed-forward of the path's velocity and
    acceleration taken L_u = 0.30 s ahead, inverted with K = 1.15 and tau = 0.55 s, plus P and a small integral on
    the EKF position; z: velocity feed-forward plus P). Uses the path itself, as strike.py uses its trajectory."""
    K, tau, L_u, kp, kp_z, ki = 1.15, 0.55, 0.30, 0.8, 0.8, 0.3
    e = obs[0:3].astype(float)  # error to the point now, from the EKF position
    _, vff, aff = env.path.at(env.t + L_u)
    integ = state.setdefault("i", np.zeros(2))
    if abs(e[0]) < 0.3 and abs(e[1]) < 0.3:
        state["i"] = np.clip(integ + ki * e[:2] * DT, -0.15, 0.15)
    uxy = (vff[:2] + tau * aff[:2]) / K + kp * e[:2] + state["i"]
    n = float(np.linalg.norm(uxy))
    if n > 0.6:
        uxy *= 0.6 / n
    uz = float(np.clip(vff[2] + kp_z * e[2], -0.4, 0.4))
    return np.array([uxy[0] / 0.6, uxy[1] / 0.6, uz / 0.4])


__all__ = ["TrackEnv", "track_controller", "TRACK_STEPS", "PREVIEW_S"]
