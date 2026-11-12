"""Tello flying to a point and holding it, at the level the Tello accepts: velocity commands (rc) at 20 Hz.

The Tello's firmware is closed, so unlike Learning to Fly in Seconds (motor commands) the policy commands velocities,
and the simulator is the response identified in flight (tello_step_fit, step_20261111_010806/model.yaml): per axis
    tau dv/dt = K u(t - L) - v
x: K 1.25, tau 0.55 s; y: K 1.09, tau 0.55 s (L 0.48-0.52 s from command to marker pose, of which 0.30 s from command
to motion, docs/SYSTEM.md); z: K 0.78, tau 0.45 s. Each episode draws its own K, tau, command delay and drift around
these values (domain randomization), so that the policy works on the real drone whatever the exact values.

The policy sees what the EKF gives (/ekf/pose, /ekf/velocity, /ekf/drift), with noise, in the map frame:
    error to the target (3), velocity (3), drift estimate (2), the error on a 10 cm scale tanh(e / 0.1) (3),
    its last 8 commands (24)   -> 35 values
and commands a map-frame velocity, scaled to [-1, 1] (0.6 m/s horizontally, 0.4 m/s vertically). The deployment
node rotates it into the body frame with the EKF yaw, as the driver's position hold does.

Reward, in the form of RLtools' L2F reward (norms of the errors, not squares, so that a centimeter closer pays the same
near the target as far from it): 2 - (2 |e| + 0.1 |v| + 0.5 |du| + 0.1 |u|^2), clipped at -2; leaving 3 m around the
target ends the episode with -2 / (1 - gamma), never better than staying at the floor.
"""
import math

import gymnasium as gym
import numpy as np

DT = 0.05                      # s, the driver sends rc at 20 Hz
LIN_GAIN = 0.38 / 30           # m/s per rc point (driver lin_gain): commands are quantized to it
U_MAX = np.array([0.6, 0.6, 0.4])   # m/s, as the driver's hold clamps (0.6 horizontal, 0.3-0.4 vertical)
HISTORY = 8                    # last commands in the observation (0.4 s, longer than the 0.30 s command delay)
EPISODE_STEPS = 200            # 10 s
GAMMA = 0.99


class TelloEnv(gym.Env):
    """randomize=False: the identified values exactly (x and y with their own K), no drift, no noise.
    bonus_weight > 0: a closeness bonus bonus_weight * exp(-(|e| / bonus_scale)^2) added to the reward, as on the
    airplane (Fixed-Wing, results/fine_px4plane): it pays only within a few bonus_scale of the target."""

    def __init__(self, randomize=True, seed=None, start_radius=2.0, bonus_weight=0.0, bonus_scale=0.15):
        self.randomize = randomize
        self.start_radius = start_radius
        self.bonus_weight, self.bonus_scale = bonus_weight, bonus_scale
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (11 + 3 * HISTORY,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.rng = np.random.default_rng(seed)

    # ----- model of the episode -----
    def _draw_model(self):
        r = self.rng
        if self.randomize:
            self.K = np.array([r.uniform(0.9, 1.4), r.uniform(0.9, 1.4), r.uniform(0.6, 1.0)])
            self.tau = np.array([r.uniform(0.4, 0.8), r.uniform(0.4, 0.8), r.uniform(0.3, 0.7)])
            self.delay = np.array([r.integers(4, 9), r.integers(4, 9), r.integers(0, 7)])  # steps: 0.2-0.4 s, z 0-0.3 s
            angle, speed = r.uniform(0, 2 * math.pi), 0.10 * math.sqrt(r.uniform())
            self.drift = np.array([speed * math.cos(angle), speed * math.sin(angle), 0.0])
            self.noise_p, self.noise_v, self.noise_d = 0.02, 0.05, 0.02
        else:
            self.K, self.tau = np.array([1.25, 1.09, 0.78]), np.array([0.55, 0.55, 0.45])
            self.delay = np.array([6, 6, 0])
            self.drift = np.zeros(3)
            self.noise_p = self.noise_v = self.noise_d = 0.0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._draw_model()
        r = self.rng
        # the target up to start_radius away horizontally, +-0.5 m vertically; the drone may already be moving
        angle, dist = r.uniform(0, 2 * math.pi), self.start_radius * math.sqrt(r.uniform())
        self.p = np.zeros(3)
        self.target = np.array([dist * math.cos(angle), dist * math.sin(angle), r.uniform(-0.5, 0.5)])
        self.v = np.clip(r.normal(0, 0.2, 3), -0.4, 0.4) * np.array([1, 1, 0.5])
        self.queue = np.zeros((9, 3))          # commands sent, not yet acting (longest delay 8 steps)
        self.history = np.zeros((HISTORY, 3))  # last commands, scaled to [-1, 1]
        self.drift_estimate = self.drift[:2] + r.normal(0, self.noise_d, 2)
        self.steps = 0
        return self._observe(), {}

    def _observe(self):
        r = self.rng
        e = self.target - (self.p + r.normal(0, self.noise_p, 3))
        v = self.v + r.normal(0, self.noise_v, 3)
        obs = np.concatenate([np.clip(e, -3, 3), v / 0.5, self.drift_estimate / 0.1, np.tanh(e / 0.1), self.history.ravel()])
        return obs.astype(np.float32)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        du = self._advance(a)
        e = np.linalg.norm(self.target - self.p)
        terminated = bool(e > 3.0)
        if terminated:
            reward = -2.0 / (1 - GAMMA)
        else:
            cost = 2.0 * e + 0.1 * np.linalg.norm(self.v) + 0.5 * np.linalg.norm(du) + 0.1 * float(a @ a)
            reward = max(2.0 - cost, -2.0) + self.bonus_weight * math.exp(-(e / self.bonus_scale) ** 2)
        truncated = self.steps >= EPISODE_STEPS
        return self._observe(), float(reward), terminated, truncated, {"error": e}

    def _advance(self, a):
        """One 50 ms step of the drone under the command a in [-1, 1]; returns the command change."""
        u = np.round(a * U_MAX / LIN_GAIN) * LIN_GAIN          # what the driver really sends (integer rc)
        self.queue = np.roll(self.queue, 1, axis=0)
        self.queue[0] = u
        acting = np.array([self.queue[self.delay[i], i] for i in range(3)])
        # exact discretization of tau dv/dt = K u - v over one step, plus the drift
        decay = np.exp(-DT / self.tau)
        v_cmd = self.v - self.drift
        v_cmd = decay * v_cmd + (1 - decay) * self.K * acting
        if self.randomize:  # the drift wanders slowly (EKF: random walk of 0.01 m/s per sqrt(s))
            self.drift[:2] += self.rng.normal(0, 0.01 * math.sqrt(DT), 2)
            self.drift[:2] = np.clip(self.drift[:2], -0.12, 0.12)
            self.drift_estimate += 0.05 * (self.drift[:2] + self.rng.normal(0, self.noise_d, 2) - self.drift_estimate)
        v_new = v_cmd + self.drift
        self.p = self.p + 0.5 * (self.v + v_new) * DT
        self.v = v_new
        du = a - self.history[0]
        self.history = np.roll(self.history, 1, axis=0)
        self.history[0] = a
        self.steps += 1
        return du


def hold_controller(obs, state):
    """The driver's position hold (driver.py hold_axis, gains of Nov 11) on the same observation as the policy, for
    comparison: position predicted L_u = 0.30 s ahead, desired speed kp * error (at most 0.4 m/s), model inversion with
    K = 1.15 (z 0.9), velocity term kv, drift feed-forward, small integral near the target. Returns an action in [-1, 1]."""
    kp, kv, ki = np.array([0.8, 0.8, 1.0]), 0.4, 0.15
    K = np.array([1.15, 1.15, 0.9])
    v = obs[3:6] * 0.5
    e = obs[0:3] - v * 0.30
    vd = np.clip(kp * e, -0.4, 0.4)
    i = state.get("i", np.zeros(3))
    state["i"] = np.where(np.abs(e) < 0.3, np.clip(i + ki * e * DT, -0.1, 0.1), i)
    drift = np.array([obs[6] * 0.1, obs[7] * 0.1, 0.0])
    u = (vd - drift + kv * (vd - v)) / K + state["i"]
    u = np.clip(u, [-0.6, -0.6, -0.3], [0.6, 0.6, 0.3])
    return u / U_MAX
