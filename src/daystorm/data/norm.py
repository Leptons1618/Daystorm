"""Split-safe normalisation statistics (rule 4 of the alignment contract).

The classic leak in a multimodal pipeline is computing channel means over the
whole dataset and then splitting.  Here statistics are fitted on the train
split alone, written to a JSON artifact, and loaded alongside the checkpoint,
so an evaluation run physically cannot see validation data.

Only ``kind="numeric"`` channels are normalised.  Camera and audio carry
indices into an asset table, and normalising an index is meaningless.

Masked grid points are set to 0 *after* normalisation, which puts them at the
channel mean in normalised space.  The model is told which points those are
through the validity mask, so 0 is never load-bearing on its own.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .align import DEFAULT_SPECS, AlignedWindow, ChannelSpec

__all__ = ["NormStats"]

_EPS = 1e-6


@dataclass
class NormStats:
    """Per-channel mean and standard deviation for the numeric modalities."""

    mean: dict[str, np.ndarray] = field(default_factory=dict)
    std: dict[str, np.ndarray] = field(default_factory=dict)
    count: dict[str, int] = field(default_factory=dict)
    split: str = "train"
    created_utc: str = ""

    @classmethod
    def fit(
        cls,
        windows: Iterable[AlignedWindow],
        specs: Sequence[ChannelSpec] = DEFAULT_SPECS,
        split: str = "train",
    ) -> NormStats:
        """Accumulate statistics over *valid* grid points only.

        Masked points hold a stale or absent value; folding them in would drag
        every channel mean toward whatever the sensor last said before it
        died.
        """
        numeric = [s for s in specs if s.kind == "numeric"]
        total = {s.name: np.zeros(s.dim, dtype=np.float64) for s in numeric}
        total_sq = {s.name: np.zeros(s.dim, dtype=np.float64) for s in numeric}
        n = {s.name: 0 for s in numeric}

        for w in windows:
            for spec in numeric:
                if spec.name not in w.values:
                    continue
                m = w.valid[spec.name]
                if not m.any():
                    continue
                v = w.values[spec.name][m].astype(np.float64)
                total[spec.name] += v.sum(axis=0)
                total_sq[spec.name] += (v * v).sum(axis=0)
                n[spec.name] += int(m.sum())

        mean, std = {}, {}
        for spec in numeric:
            k = max(n[spec.name], 1)
            mu = total[spec.name] / k
            var = np.maximum(total_sq[spec.name] / k - mu * mu, 0.0)
            mean[spec.name] = mu.astype(np.float32)
            # A constant channel gets std 1, so it normalises to 0 instead of inf.
            std[spec.name] = np.maximum(np.sqrt(var), _EPS).astype(np.float32)

        if any(v == 0 for v in n.values()):
            empty = [k for k, v in n.items() if v == 0]
            raise ValueError(f"no valid samples for channels {empty}; cannot fit norm stats")

        return cls(
            mean=mean,
            std=std,
            count=n,
            split=split,
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def apply(self, window: AlignedWindow, in_place: bool = False) -> AlignedWindow:
        """Normalise a window's numeric channels; re-zero the masked points."""
        target = window if in_place else AlignedWindow(
            grid=window.grid,
            values={k: v.copy() for k, v in window.values.items()},
            valid=window.valid,
            age=window.age,
            t0=window.t0,
        )
        for name, mu in self.mean.items():
            if name not in target.values:
                continue
            v = target.values[name]
            if v.shape[1] != mu.shape[0]:
                raise ValueError(
                    f"{name}: stats have dim {mu.shape[0]} but window has {v.shape[1]}"
                )
            v -= mu
            v /= self.std[name]
            v[~target.valid[name]] = 0.0
        return target

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "split": self.split,
            "created_utc": self.created_utc,
            "count": self.count,
            "mean": {k: v.tolist() for k, v in self.mean.items()},
            "std": {k: v.tolist() for k, v in self.std.items()},
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> NormStats:
        d = json.loads(Path(path).read_text())
        return cls(
            mean={k: np.asarray(v, dtype=np.float32) for k, v in d["mean"].items()},
            std={k: np.asarray(v, dtype=np.float32) for k, v in d["std"].items()},
            count=d.get("count", {}),
            split=d.get("split", "train"),
            created_utc=d.get("created_utc", ""),
        )
