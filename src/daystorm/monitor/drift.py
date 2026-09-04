"""Input drift monitoring, using the training statistics as the reference.

The normalisation statistics fitted on the train split are already a compact
description of what the model was trained on, so they double as the drift
reference for free - no second artifact to version, and no way for the monitor
to disagree with the model about what "normal" means.

Three signals, because they fail differently:

``z``          how far the channel distribution has moved. Catches a
               recalibrated sensor, a units change, a different vehicle.
``coverage``   how often channels are masked. Catches a failing bus or a
               camera dropping frames - the model keeps answering, quietly
               relying on fewer inputs.
``constancy``  a channel that has stopped varying at all. A stuck CAN signal
               reads as perfectly in-distribution to a z-score check while
               being completely dead, which is exactly the failure a fleet
               monitor exists to catch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from daystorm.data.align import AlignedWindow
from daystorm.data.norm import NormStats

__all__ = ["DriftMonitor", "DriftReport"]


@dataclass
class DriftReport:
    n_windows: int
    mean_abs_z: dict[str, float]
    coverage: dict[str, float]
    constant_channels: dict[str, list[int]]
    alerts: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.alerts

    def render(self) -> str:
        lines = [f"windows observed: {self.n_windows}"]
        for name in sorted(self.mean_abs_z):
            lines.append(
                f"  {name:<8} mean|z| {self.mean_abs_z[name]:5.2f}   "
                f"coverage {self.coverage.get(name, float('nan')):5.1%}"
            )
        for name in sorted(self.coverage):
            if name not in self.mean_abs_z:
                lines.append(f"  {name:<8} {'':16} coverage {self.coverage[name]:5.1%}")
        lines += [f"  ALERT: {a}" for a in self.alerts] or ["  no alerts"]
        return "\n".join(lines)


class DriftMonitor:
    """Accumulates windows and compares them against the training reference."""

    def __init__(
        self,
        stats: NormStats,
        z_threshold: float = 3.0,
        coverage_floor: float = 0.80,
        min_windows: int = 20,
    ):
        self.stats = stats
        self.z_threshold = z_threshold
        self.coverage_floor = coverage_floor
        self.min_windows = min_windows
        self._z_sum: dict[str, np.ndarray] = {}
        self._z_n: dict[str, int] = {}
        self._valid_sum: dict[str, float] = {}
        self._valid_n: dict[str, int] = {}
        self._min: dict[str, np.ndarray] = {}
        self._max: dict[str, np.ndarray] = {}
        self.n_windows = 0

    def observe(self, window: AlignedWindow) -> None:
        self.n_windows += 1
        for name, valid in window.valid.items():
            self._valid_sum[name] = self._valid_sum.get(name, 0.0) + float(valid.sum())
            self._valid_n[name] = self._valid_n.get(name, 0) + valid.size

        for name, mean in self.stats.mean.items():
            if name not in window.values:
                continue
            valid = window.valid[name]
            if not valid.any():
                continue
            raw = window.values[name][valid].astype(np.float64)
            z = np.abs((raw - mean) / self.stats.std[name])
            self._z_sum[name] = self._z_sum.get(name, np.zeros(z.shape[1])) + z.sum(axis=0)
            self._z_n[name] = self._z_n.get(name, 0) + z.shape[0]
            lo, hi = raw.min(axis=0), raw.max(axis=0)
            self._min[name] = np.minimum(self._min[name], lo) if name in self._min else lo
            self._max[name] = np.maximum(self._max[name], hi) if name in self._max else hi

    def report(self) -> DriftReport:
        mean_abs_z = {
            name: float((s / max(self._z_n[name], 1)).mean()) for name, s in self._z_sum.items()
        }
        coverage = {
            name: self._valid_sum[name] / max(self._valid_n[name], 1) for name in self._valid_sum
        }
        constant = {
            name: np.flatnonzero(self._max[name] - self._min[name] < 1e-9).tolist()
            for name in self._min
        }
        constant = {k: v for k, v in constant.items() if v}

        alerts: list[str] = []
        if self.n_windows < self.min_windows:
            # Refuse to alert on a handful of windows: a single unusual junction
            # would page someone at three in the morning.
            return DriftReport(self.n_windows, mean_abs_z, coverage, constant, alerts)

        for name, z in mean_abs_z.items():
            if z > self.z_threshold:
                alerts.append(f"{name}: mean|z|={z:.2f} exceeds {self.z_threshold} (distribution shift)")
        for name, cov in coverage.items():
            if cov < self.coverage_floor:
                alerts.append(f"{name}: coverage {cov:.1%} below {self.coverage_floor:.0%} (sensor dropping out)")
        for name, cols in constant.items():
            alerts.append(f"{name}: channels {cols} never varied across {self.n_windows} windows (stuck signal)")
        return DriftReport(self.n_windows, mean_abs_z, coverage, constant, alerts)
