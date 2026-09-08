"""Temporal alignment of asynchronous vehicle sensor streams.

Daystorm ingests four streams sampled at unrelated rates, each of which is
allowed to drop out independently:

    camera        12 Hz     nuScenes CAM_* keyframes and sweeps
    radar         13 Hz     nuScenes RADAR_* tracks
    CAN steering  100 Hz    nuScenes CAN bus expansion (steeranglefeedback)
    audio         16 kHz    cabin microphone (synthesised for this build)

Every model input is a fixed-length window resampled onto a common *event
grid* at 2 Hz.  2 Hz is not arbitrary: it is the nuScenes keyframe and
annotation rate, so grid points coincide with ground truth instead of
requiring interpolated labels.

Five rules are enforced here.  Each has a test in ``tests/test_align.py``.

1. Causal hold       a grid point takes the nearest *preceding* sample only.
                     No forward interpolation: a model that sees 40 ms into
                     the future scores well offline and is useless on a car.
2. Staleness budget  each channel declares how old a held sample may be.
                     Past that the grid point is masked.
3. Explicit masks    missing data is returned as a mask the model consumes,
                     never as a silent zero.
4. Split-safe norm   see :mod:`daystorm.data.norm` - statistics come from the
                     train split alone and travel with the checkpoint.
5. One clock         all timestamps are float seconds against a single
                     reference clock (nuScenes: the LIDAR_TOP timestamp),
                     asserted sorted and de-duplicated at construction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_SPECS",
    "AlignedWindow",
    "ChannelSpec",
    "Stream",
    "align_window",
    "make_grid",
    "resample_causal",
]


@dataclass(frozen=True)
class ChannelSpec:
    """Declares one modality's contract with the aligner.

    ``max_staleness_s`` is the budget that matters: it is roughly one and a
    half nominal sample periods, so a single dropped sample is tolerated and
    a real outage is not.
    """

    name: str
    dim: int
    max_staleness_s: float
    rate_hz: float | None = None  # nominal; documentation and sanity checks
    kind: str = "numeric"  # "numeric" (measured) or "reference" (index into an asset table)

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"{self.name}: dim must be positive, got {self.dim}")
        if self.max_staleness_s <= 0:
            raise ValueError(f"{self.name}: max_staleness_s must be positive")
        if self.kind not in ("numeric", "reference"):
            raise ValueError(f"{self.name}: kind must be numeric or reference")


#: The four Daystorm modalities.  Budgets are ~1.5 nominal periods.
DEFAULT_SPECS: tuple[ChannelSpec, ...] = (
    ChannelSpec("camera", dim=1, max_staleness_s=0.090, rate_hz=12.0, kind="reference"),
    ChannelSpec("can", dim=9, max_staleness_s=0.030, rate_hz=100.0),
    ChannelSpec("radar", dim=6, max_staleness_s=0.090, rate_hz=13.0),
    ChannelSpec("audio", dim=1, max_staleness_s=0.100, rate_hz=50.0, kind="reference"),
)


class Stream:
    """An irregularly sampled multivariate stream on the reference clock.

    Construction normalises the stream so that everything downstream can
    assume it: timestamps are sorted ascending and de-duplicated.  Duplicate
    timestamps keep the *last* value in original input order, which makes the
    result independent of how the caller happened to concatenate its shards.

    Parameters
    ----------
    t : array of float seconds, shape (N,)
    v : array, shape (N,) or (N, D)
    """

    __slots__ = ("name", "t", "v")

    def __init__(self, t: np.ndarray, v: np.ndarray, name: str = "") -> None:
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        v = np.asarray(v, dtype=np.float32)
        if v.ndim == 1:
            v = v[:, None]
        if v.ndim != 2:
            raise ValueError(f"{name}: values must be 1-D or 2-D, got shape {v.shape}")
        if len(t) != len(v):
            raise ValueError(f"{name}: {len(t)} timestamps but {len(v)} values")
        if not np.all(np.isfinite(t)):
            raise ValueError(f"{name}: timestamps must be finite")

        # Stable sort keeps original order within equal timestamps, so the
        # "keep last duplicate" rule below is deterministic.
        order = np.argsort(t, kind="stable")
        t, v = t[order], v[order]
        if len(t) > 1:
            keep = np.ones(len(t), dtype=bool)
            keep[:-1] = t[1:] != t[:-1]
            t, v = t[keep], v[keep]

        self.t = t
        self.v = np.ascontiguousarray(v)
        self.name = name

    @property
    def dim(self) -> int:
        return int(self.v.shape[1])

    def __len__(self) -> int:
        return int(self.t.shape[0])

    def __repr__(self) -> str:
        span = f"{self.t[0]:.3f}..{self.t[-1]:.3f}s" if len(self) else "empty"
        return f"Stream({self.name!r}, n={len(self)}, dim={self.dim}, {span})"

    @classmethod
    def empty(cls, dim: int, name: str = "") -> Stream:
        return cls(np.zeros(0), np.zeros((0, dim), dtype=np.float32), name)


@dataclass
class AlignedWindow:
    """One window of every modality, resampled onto a shared grid.

    Attributes
    ----------
    grid : (G,) float seconds on the reference clock.
    values : name -> (G, D) held values; masked entries are exactly 0.
    valid : name -> (G,) bool; True where a fresh-enough sample existed.
    age : name -> (G,) seconds since the held sample; inf where none exists.
    """

    grid: np.ndarray
    values: dict[str, np.ndarray]
    valid: dict[str, np.ndarray]
    age: dict[str, np.ndarray]
    t0: float

    @property
    def n_grid(self) -> int:
        return int(self.grid.shape[0])

    def features(self, order: Sequence[str] | None = None) -> np.ndarray:
        """Concatenate modalities along the feature axis -> (G, sum(D))."""
        names = list(order) if order is not None else list(self.values)
        return np.concatenate([self.values[n] for n in names], axis=1)

    def mask(self, order: Sequence[str] | None = None) -> np.ndarray:
        """Per-modality validity -> (G, n_modalities), float32 for the model."""
        names = list(order) if order is not None else list(self.values)
        return np.stack([self.valid[n] for n in names], axis=1).astype(np.float32)

    def coverage(self) -> dict[str, float]:
        """Fraction of grid points that are valid, per modality."""
        return {n: float(m.mean()) for n, m in self.valid.items()}

    def is_usable(self, min_coverage: float = 0.6) -> bool:
        """Reject windows too sparse to train on.

        A window where every camera point is masked teaches the model that
        vision is optional, so these are dropped at dataset build time rather
        than silently learned from.
        """
        cov = self.coverage()
        return bool(cov) and min(cov.values()) >= min_coverage


def make_grid(t0: float, duration_s: float = 2.0, rate_hz: float = 2.0) -> np.ndarray:
    """Event grid covering ``[t0, t0 + duration_s]`` inclusive.

    2 s at 2 Hz yields 5 points: t0, +0.5, +1.0, +1.5, +2.0.
    """
    if duration_s <= 0 or rate_hz <= 0:
        raise ValueError("duration_s and rate_hz must be positive")
    n = round(duration_s * rate_hz) + 1
    return float(t0) + np.arange(n, dtype=np.float64) / float(rate_hz)


def resample_causal(
    stream: Stream, grid: np.ndarray, max_staleness_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hold the nearest preceding sample onto ``grid``.

    A sample at exactly the grid time counts as available (age 0).  Nothing
    after the grid time can influence the result - that is rule 1, and
    ``test_future_samples_never_leak`` proves it by construction rather than
    by inspection.

    Returns
    -------
    values : (G, D) float32, zero where invalid
    valid  : (G,) bool
    age    : (G,) float64 seconds, inf where no preceding sample exists
    """
    grid = np.asarray(grid, dtype=np.float64)
    g = grid.shape[0]
    d = stream.dim

    values = np.zeros((g, d), dtype=np.float32)
    age = np.full(g, np.inf, dtype=np.float64)

    if len(stream) == 0:
        return values, np.zeros(g, dtype=bool), age

    # side="right" -> index of the last sample with t <= grid point.
    idx = np.searchsorted(stream.t, grid, side="right") - 1
    has_prev = idx >= 0
    safe = np.clip(idx, 0, None)

    values[has_prev] = stream.v[safe[has_prev]]
    age[has_prev] = grid[has_prev] - stream.t[safe[has_prev]]

    valid = has_prev & (age <= max_staleness_s)
    values[~valid] = 0.0  # rule 3: the mask carries the information, not a magic value
    return values, valid, age


def align_window(
    streams: Mapping[str, Stream],
    specs: Iterable[ChannelSpec] = DEFAULT_SPECS,
    t0: float = 0.0,
    duration_s: float = 2.0,
    rate_hz: float = 2.0,
) -> AlignedWindow:
    """Resample every declared modality onto one event grid.

    A modality absent from ``streams`` is treated as a total outage rather
    than an error: vehicles lose sensors, and the dataset should contain
    windows that say so.
    """
    specs = list(specs)
    grid = make_grid(t0, duration_s, rate_hz)

    values: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}
    age: dict[str, np.ndarray] = {}

    for spec in specs:
        stream = streams.get(spec.name)
        if stream is None:
            stream = Stream.empty(spec.dim, spec.name)
        elif stream.dim != spec.dim:
            raise ValueError(
                f"{spec.name}: spec declares dim={spec.dim} but stream has dim={stream.dim}"
            )
        v, m, a = resample_causal(stream, grid, spec.max_staleness_s)
        values[spec.name], valid[spec.name], age[spec.name] = v, m, a

    return AlignedWindow(grid=grid, values=values, valid=valid, age=age, t0=float(t0))
