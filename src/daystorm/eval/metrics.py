"""Metrics that grade grounding, not fluency.

A driving copilot that writes a beautifully phrased answer with the wrong
closing distance is worse than useless, so string-similarity scores (BLEU,
ROUGE) are actively misleading here: the reference answers are ~97% shared
boilerplate and every one of them scores well against every other one.

What matters is whether the numbers are right and whether the model invented
any. So:

``exact_match``        the strictest reading - did it reproduce the answer.
``manoeuvre_accuracy`` did it identify what the vehicle was doing at all.
``numeric_mae``        how far off were the quantities it did produce.
``hallucination_rate`` what fraction of its numbers are not supported by
                       anything in the window. This is the metric to watch:
                       a model can score well on the others while quietly
                       fabricating a time-to-collision.
``coverage_honesty``   when sensors were masked, did it say so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from daystorm.data.align import AlignedWindow
from daystorm.data.synthetic import CAN_CHANNELS, RADAR_CHANNELS

__all__ = [
    "Prediction",
    "classify_manoeuvre",
    "evaluate",
    "numbers_in",
    "supportable_values",
]

# A number must not be glued to a preceding word character: "m/s2" is a unit,
# not a quantity, and letting its 2 through inflates both the hallucination
# count and the numeric error with a value nobody claimed.
_NUM = re.compile(r"(?<![\w/.])-?\d+(?:\.\d+)?")

# A quoted value counts as supported if it lands within either tolerance.
_REL_TOL = 0.05
_ABS_TOL = 0.55


@dataclass
class Prediction:
    """One decoded answer alongside the window it was supposed to describe."""

    window: AlignedWindow
    question: str
    reference: str
    prediction: str
    event: str
    ablated: tuple[str, ...] = field(default_factory=tuple)


def numbers_in(text: str) -> list[float]:
    return [float(m) for m in _NUM.findall(text)]


def supportable_values(w: AlignedWindow) -> np.ndarray:
    """Every quantity a truthful answer could quote about this window.

    Raw channel readings at valid grid points, plus the handful of derived
    quantities the reference answers are built from (deltas, means, extremes).
    A predicted number outside this set, within tolerance, was invented.
    """
    vals: list[float] = [float(w.grid[-1] - w.grid[0])]  # the window span itself

    can_ok, radar_ok = w.valid["can"], w.valid["radar"]
    if can_ok.any():
        can = w.values["can"][can_ok]
        vals.extend(can.ravel().tolist())
        speed = can[:, CAN_CHANNELS.index("speed_mps")]
        steer = can[:, CAN_CHANNELS.index("steering_angle_deg")]
        span = max(float(w.grid[-1] - w.grid[0]), 1e-6)
        vals += [
            float(speed.mean()), float(speed[0] - speed[-1]), float(speed[-1] - speed[0]),
            float((speed[-1] - speed[0]) / span), float(np.abs(steer).max()),
            float(can[:, CAN_CHANNELS.index("brake")].max()),
            float(np.abs(can[:, CAN_CHANNELS.index("yaw_rate_dps")]).max()),
        ]
    if radar_ok.any():
        radar = w.values["radar"][radar_ok]
        vals.extend(radar.ravel().tolist())
        vals += [
            float(radar[:, RADAR_CHANNELS.index("lead_range_m")].mean()),
            float(radar[:, RADAR_CHANNELS.index("ttc_s")].min()),
        ]
    return np.asarray(vals, dtype=np.float64)


def _supported(value: float, pool: np.ndarray) -> bool:
    if pool.size == 0:
        return False
    diff = np.abs(pool - value)
    return bool((diff <= np.maximum(_ABS_TOL, np.abs(pool) * _REL_TOL)).any())


def classify_manoeuvre(text: str) -> str:
    """Which manoeuvre does this answer describe?

    Matching on bare "brak" is wrong: a cruising answer ends with "no brake
    demand", which would classify every quiet window as a braking event and
    silently inflate the score. The markers below are the ones that only
    appear when the manoeuvre is actually being asserted.
    """
    t = text.lower()
    if "decelerat" in t:
        return "brake"
    if "turn" in t:
        return "turn"
    if "cruis" in t or "steady" in t:
        return "cruise"
    if "brake demand peaking" in t:
        return "brake"
    return "cruise"


def evaluate(preds: list[Prediction]) -> dict[str, float]:
    """Aggregate metrics over a set of predictions."""
    if not preds:
        return {}

    exact = 0
    manoeuvre = 0
    abs_errors: list[float] = []
    invented = 0
    quoted = 0
    honesty_cases = 0
    honesty_hits = 0

    for p in preds:
        if p.prediction.strip() == p.reference.strip():
            exact += 1
        if classify_manoeuvre(p.prediction) == classify_manoeuvre(p.reference):
            manoeuvre += 1

        pool = supportable_values(p.window)
        pred_nums = numbers_in(p.prediction)
        ref_nums = numbers_in(p.reference)
        quoted += len(pred_nums)
        invented += sum(1 for v in pred_nums if not _supported(v, pool))

        # Compare positionally: the answer templates are fixed, so the k-th
        # number in the prediction is meant to be the k-th in the reference.
        for a, b in zip(pred_nums, ref_nums):
            abs_errors.append(abs(a - b))

        incomplete = any(not m.all() for m in p.window.valid.values())
        if incomplete:
            honesty_cases += 1
            if "coverage incomplete" in p.prediction.lower():
                honesty_hits += 1

    n = len(preds)
    return {
        "n": float(n),
        "exact_match": exact / n,
        "manoeuvre_accuracy": manoeuvre / n,
        "numeric_mae": float(np.mean(abs_errors)) if abs_errors else float("nan"),
        "hallucination_rate": invented / quoted if quoted else 0.0,
        "numbers_per_answer": quoted / n,
        "coverage_honesty": honesty_hits / honesty_cases if honesty_cases else float("nan"),
    }
