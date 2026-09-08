"""A synthetic driving scene generator, so the repo runs with zero downloads.

nuScenes is ~4 GB for the mini split and needs an account.  Every test, the
Phase 1 overfit gate and ``make demo`` run against this module instead, which
emits the same four streams at the same rates with the same dropout
behaviour.  Swapping in :mod:`daystorm.data.nuscenes_src` changes the source
of the ``Stream`` objects and nothing downstream.

The kinematics are deliberately simple but not arbitrary: speeds, decelerations
and following distances sit in ranges an urban driving log actually produces,
because a generator that emits 60 m/s in a city block trains a model to accept
nonsense.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from .align import DEFAULT_SPECS, Stream, align_window

__all__ = ["CAN_CHANNELS", "RADAR_CHANNELS", "Scene", "make_scene"]

CAN_CHANNELS = (
    "steering_angle_deg",
    "steering_rate_dps",
    "throttle",
    "brake",
    "speed_mps",
    "accel_x_mps2",
    "accel_y_mps2",
    "yaw_rate_dps",
    "wheel_speed_mps",
)
RADAR_CHANNELS = (
    "lead_range_m",
    "lead_range_rate_mps",
    "nearest_range_m",
    "nearest_range_rate_mps",
    "n_tracks",
    "ttc_s",
)

EVENTS = ("brake", "turn", "cruise")


@dataclass
class Scene:
    """One synthetic log: four streams plus the event that generated them."""

    streams: dict[str, Stream]
    event: str
    event_t: float
    duration_s: float
    seed: int

    def describe(self) -> str:
        if self.event == "brake":
            return (
                f"hard deceleration beginning {self.event_t:.1f} s into the log, "
                "triggered by a closing lead vehicle"
            )
        if self.event == "turn":
            return f"a signalled left turn beginning {self.event_t:.1f} s into the log"
        return "steady-state cruising with no notable event"


def _ramp(t: np.ndarray, start: float, width: float) -> np.ndarray:
    """Smooth 0->1 transition over [start, start+width] (raised cosine)."""
    x = np.clip((t - start) / max(width, 1e-6), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def _apply_dropout(
    stream: Stream, rng: np.random.Generator, n_gaps: int, gap_s: tuple[float, float]
) -> Stream:
    """Delete random spans of samples, the way a real bus or link does."""
    if len(stream) == 0 or n_gaps <= 0:
        return stream
    keep = np.ones(len(stream), dtype=bool)
    span = stream.t[-1] - stream.t[0]
    for _ in range(n_gaps):
        width = rng.uniform(*gap_s)
        start = stream.t[0] + rng.uniform(0.0, max(span - width, 1e-3))
        keep &= ~((stream.t >= start) & (stream.t < start + width))
    return Stream(stream.t[keep], stream.v[keep], stream.name)


def make_scene(
    seed: int = 0,
    duration_s: float = 20.0,
    event: str | None = None,
    dropout: bool = True,
) -> Scene:
    """Generate one scene with realistic rates, ranges and dropouts."""
    rng = np.random.default_rng(seed)
    event = event or EVENTS[seed % len(EVENTS)]
    event_t = float(rng.uniform(6.0, max(duration_s - 6.0, 7.0)))

    # --- CAN bus at 100 Hz -------------------------------------------------
    t_can = np.arange(0.0, duration_s, 1 / 100.0)
    base_speed = float(rng.uniform(9.0, 13.0))
    speed = np.full_like(t_can, base_speed)
    steer = rng.normal(0.0, 0.8, t_can.shape)
    throttle = np.full_like(t_can, 0.22) + rng.normal(0, 0.01, t_can.shape)
    brake = np.zeros_like(t_can)
    yaw = rng.normal(0.0, 0.4, t_can.shape)

    if event == "brake":
        r = _ramp(t_can, event_t, 3.0)
        speed = base_speed - r * (base_speed - 4.0)
        brake = _ramp(t_can, event_t, 0.6) * 0.8 * (1 - _ramp(t_can, event_t + 3.0, 1.0))
        throttle *= 1 - _ramp(t_can, event_t, 0.4)
    elif event == "turn":
        r = _ramp(t_can, event_t, 1.5) * (1 - _ramp(t_can, event_t + 3.0, 1.5))
        steer = steer + r * 90.0
        yaw = yaw + r * 18.0
        speed = base_speed - r * 3.0

    accel_x = np.gradient(speed, t_can)
    accel_y = np.deg2rad(yaw) * speed
    steer_rate = np.gradient(steer, t_can)
    can = np.stack(
        [steer, steer_rate, throttle, brake, speed, accel_x, accel_y, yaw,
         speed * float(rng.uniform(0.99, 1.01))],
        axis=1,
    )

    # --- radar tracks at 13 Hz --------------------------------------------
    t_radar = np.arange(0.0, duration_s, 1 / 13.0)
    if event == "brake":
        lead = 30.0 - 16.0 * _ramp(t_radar, event_t - 1.5, 3.0)
    else:
        lead = 30.0 + rng.normal(0, 1.2, t_radar.shape)
    lead = np.maximum(lead, 5.0)
    lead_rate = np.gradient(lead, t_radar)
    closing = lead_rate < -0.1
    ttc = np.where(closing, np.clip(lead / np.where(closing, -lead_rate, 1.0), 0.0, 60.0), 60.0)
    nearest = lead - np.abs(rng.normal(0, 2.0, t_radar.shape))
    radar = np.stack(
        [lead, lead_rate, np.maximum(nearest, 1.0), lead_rate,
         np.round(rng.uniform(3, 9, t_radar.shape)), ttc],
        axis=1,
    )

    # --- reference streams: indices into asset tables ----------------------
    t_cam = np.arange(0.0, duration_s, 1 / 12.0)
    cam = np.arange(len(t_cam), dtype=np.float32)[:, None]
    t_aud = np.arange(0.0, duration_s, 1 / 50.0)
    aud = np.arange(len(t_aud), dtype=np.float32)[:, None]

    streams = {
        "can": Stream(t_can, can, "can"),
        "radar": Stream(t_radar, radar, "radar"),
        "camera": Stream(t_cam, cam, "camera"),
        "audio": Stream(t_aud, aud, "audio"),
    }
    if dropout:
        streams["can"] = _apply_dropout(streams["can"], rng, 2, (0.02, 0.08))
        streams["radar"] = _apply_dropout(streams["radar"], rng, 1, (0.05, 0.20))
        streams["camera"] = _apply_dropout(streams["camera"], rng, 1, (0.05, 0.15))

    return Scene(streams, event, event_t, duration_s, seed)


def _demo() -> None:
    ap = argparse.ArgumentParser(description="Print one aligned window from a synthetic scene.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    scene = make_scene(seed=args.seed, event="brake")
    print(f"scene: {scene.describe()}")
    for name, s in scene.streams.items():
        print(f"  {name:<7} {s!r}")

    w = align_window(scene.streams, DEFAULT_SPECS, t0=scene.event_t - 0.5)
    print(f"\nwindow t0={w.t0:.2f}s  grid={np.round(w.grid, 2)}")
    print(f"coverage: { {k: round(v, 2) for k, v in w.coverage().items()} }  usable={w.is_usable()}")
    speed_i = CAN_CHANNELS.index("speed_mps")
    brake_i = CAN_CHANNELS.index("brake")
    print(f"  speed_mps  {np.round(w.values['can'][:, speed_i], 2)}")
    print(f"  brake      {np.round(w.values['can'][:, brake_i], 2)}")
    print(f"  lead_range {np.round(w.values['radar'][:, 0], 2)}")
    print(f"  can valid  {w.valid['can']}  max age {w.age['can'].max():.3f}s")


if __name__ == "__main__":
    _demo()
