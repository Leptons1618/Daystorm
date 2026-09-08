"""Turn scenes into training samples: aligned windows plus grounded text.

Supervision here is *derived from the aligned tensors themselves*, not from a
parallel description of what the generator intended. If the alignment layer
has a bug, the label describes the buggy tensor and the model still learns a
consistent mapping - so the tests in ``tests/test_align.py`` are what keep
this honest, and the labels quote numbers a reviewer can check against the
window they came from.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .align import DEFAULT_SPECS, AlignedWindow, ChannelSpec, align_window
from .synthetic import CAN_CHANNELS, RADAR_CHANNELS, Scene

__all__ = ["Sample", "build_samples", "describe_window", "split_by_scene"]

_SPEED = CAN_CHANNELS.index("speed_mps")
_BRAKE = CAN_CHANNELS.index("brake")
_STEER = CAN_CHANNELS.index("steering_angle_deg")
_YAW = CAN_CHANNELS.index("yaw_rate_dps")
_LEAD = RADAR_CHANNELS.index("lead_range_m")
_TTC = RADAR_CHANNELS.index("ttc_s")


@dataclass
class Sample:
    """One training example."""

    window: AlignedWindow
    question: str
    answer: str
    scene_seed: int
    event: str

    @property
    def manoeuvre(self) -> str:
        return self.event


def _fresh(w: AlignedWindow, name: str, col: int) -> np.ndarray:
    """Values at valid grid points only, so labels never quote a masked zero."""
    return w.values[name][w.valid[name], col]


def describe_window(w: AlignedWindow) -> tuple[str, str, str]:
    """Read the window and write a question, an answer, and a manoeuvre tag.

    Returns ``("", "", "")`` when the window is too sparse to describe, which
    ``build_samples`` treats as a reason to drop it rather than to guess.
    """
    if not w.valid["can"].any() or not w.valid["radar"].any():
        return "", "", ""

    speed = _fresh(w, "can", _SPEED)
    brake = _fresh(w, "can", _BRAKE)
    steer = _fresh(w, "can", _STEER)
    yaw = _fresh(w, "can", _YAW)
    lead = _fresh(w, "radar", _LEAD)
    ttc = _fresh(w, "radar", _TTC)

    dv = float(speed[-1] - speed[0])
    span = float(w.grid[-1] - w.grid[0])
    peak_steer = float(steer[np.argmax(np.abs(steer))])

    if brake.max() > 0.25 and dv < -1.0:
        tag = "brake"
        q = "Why did the vehicle decelerate?"
        a = (
            f"Deceleration from {speed[0]:.1f} to {speed[-1]:.1f} m/s over {span:.1f} s "
            f"({dv / span:.1f} m/s2) with brake demand peaking at {brake.max():.2f}. "
            f"The lead vehicle closed from {lead[0]:.0f} m to {lead[-1]:.0f} m, "
            f"minimum time-to-collision {ttc.min():.1f} s."
        )
    elif abs(peak_steer) > 25.0:
        tag = "turn"
        side = "left" if peak_steer > 0 else "right"
        a_yaw = float(yaw[np.argmax(np.abs(yaw))])
        q = "What manoeuvre is the vehicle performing?"
        a = (
            f"A {side} turn: steering angle reaches {peak_steer:.0f} deg with yaw rate "
            f"{a_yaw:.0f} deg/s at {speed.mean():.1f} m/s. "
            f"Nearest tracked vehicle holds at {lead.mean():.0f} m."
        )
    else:
        tag = "cruise"
        q = "What is the vehicle doing?"
        a = (
            f"Steady cruising at {speed.mean():.1f} m/s, steering within "
            f"{np.abs(steer).max():.0f} deg and no brake demand. "
            f"Lead vehicle steady near {lead.mean():.0f} m."
        )

    stale = [n for n, m in w.valid.items() if not m.all()]
    if stale:
        a += f" Sensor coverage incomplete this window: {', '.join(sorted(stale))}."
    return q, a, tag


def build_samples(
    scenes: Iterable[Scene],
    specs: Sequence[ChannelSpec] = DEFAULT_SPECS,
    stride_s: float = 1.0,
    duration_s: float = 2.0,
    rate_hz: float = 2.0,
    min_coverage: float = 0.6,
) -> list[Sample]:
    """Slide a window across each scene and keep the ones worth training on."""
    samples: list[Sample] = []
    for scene in scenes:
        last_t0 = scene.duration_s - duration_s
        for t0 in np.arange(0.0, last_t0, stride_s):
            w = align_window(scene.streams, specs, float(t0), duration_s, rate_hz)
            if not w.is_usable(min_coverage):
                continue
            q, a, tag = describe_window(w)
            if not q:
                continue
            samples.append(Sample(w, q, a, scene.seed, tag))
    return samples


def split_by_scene(
    samples: Sequence[Sample], val_fraction: float = 0.2, seed: int = 0
) -> tuple[list[Sample], list[Sample]]:
    """Split on scene identity, never on window index.

    Consecutive windows overlap by design, so a random per-window split puts
    near-duplicates on both sides and inflates every metric. Scenes are the
    only safe unit.
    """
    scene_ids = sorted({s.scene_seed for s in samples})
    rng = np.random.default_rng(seed)
    rng.shuffle(scene_ids)
    n_val = max(1, round(len(scene_ids) * val_fraction))
    val_ids = set(scene_ids[:n_val])
    train = [s for s in samples if s.scene_seed not in val_ids]
    val = [s for s in samples if s.scene_seed in val_ids]
    return train, val
