"""Paths for the RL tracking policy and its observation, shared by the training (tello_track_env.py) and the flight
node (tello_driver/rl_paths.py in tello-balloon-hunter, a copy of this file). No ROS here: the training imports it on any machine.

A path is centred on the origin (the node adds the centre of the free zone): a fixed point, a circle of radius
`size`, or a figure-eight (Lissajous 1:2) of half-length `size` and half-width size / 2; one lap in `period` s,
rotated by `angle` in the horizontal plane, flown in `direction` (+1 / -1), starting at `phase` (fraction of a lap).
"""
import math

import numpy as np

PREVIEW_S = 0.5          # the point is also seen this far ahead (the Tello acts 0.3 s after a command)
U_MAX = np.array([0.6, 0.6, 0.4])


class Path:
    def __init__(self, kind="hold", size=0.5, period=10.0, angle=0.0, direction=1, phase=0.0, centre=(0.0, 0.0, 0.0)):
        assert kind in ("hold", "circle", "eight")
        self.kind, self.size, self.period = kind, float(size), float(period)
        self.angle, self.direction, self.phase = float(angle), int(direction), float(phase)
        self.centre = np.asarray(centre, dtype=float)

    def at(self, t):
        """Position, velocity and acceleration of the point at time t (s)."""
        if self.kind == "hold":
            return self.centre.copy(), np.zeros(3), np.zeros(3)
        w = self.direction * 2 * math.pi / self.period
        s = w * t + 2 * math.pi * self.phase
        if self.kind == "circle":
            r = self.size
            p = np.array([r * math.cos(s), r * math.sin(s)])
            v = np.array([-r * w * math.sin(s), r * w * math.cos(s)])
            a = np.array([-r * w * w * math.cos(s), -r * w * w * math.sin(s)])
        else:  # figure-eight: x = A sin s, y = A/2 sin 2s
            A = self.size
            p = np.array([A * math.sin(s), 0.5 * A * math.sin(2 * s)])
            v = np.array([A * w * math.cos(s), A * w * math.cos(2 * s)])
            a = np.array([-A * w * w * math.sin(s), -2 * A * w * w * math.sin(2 * s)])
        c, sn = math.cos(self.angle), math.sin(self.angle)
        rot = np.array([[c, -sn], [sn, c]])
        return (self.centre + np.append(rot @ p, 0.0), np.append(rot @ v, 0.0), np.append(rot @ a, 0.0))

    def points(self, n=200):
        """Points along one lap (to check the path against the free space before flying it)."""
        return np.array([self.at(self.period * k / n)[0] for k in range(n)])


def track_observation(path, t, p, v, drift_xy, history):
    """The 41 values the policy sees: error to the point (3), velocity / 0.5 (3), drift / 0.1 (2), tanh(error / 0.1)
    (3), the point's velocity / 0.5 (3), error to the point PREVIEW_S ahead (3), last 8 commands in [-1, 1] (24).
    Positions and velocities in the map frame (EKF)."""
    ref, v_ref, _ = path.at(t)
    ahead, _, _ = path.at(t + PREVIEW_S)
    e = ref - p
    return np.concatenate([np.clip(e, -3, 3), np.asarray(v) / 0.5, np.asarray(drift_xy) / 0.1, np.tanh(e / 0.1),
                           v_ref / 0.5, np.clip(ahead - p, -3, 3), np.asarray(history).ravel()])
