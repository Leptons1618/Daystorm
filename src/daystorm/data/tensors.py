"""Bridge from aligned windows to model tensors.

The camera and audio channels hold *indices* into an asset table (rule: a
reference channel is never normalised). Here those indices are exchanged for
embeddings from a cache that a frozen encoder filled in advance - see
``scripts/precompute_reference.py``. Training never runs SigLIP or Whisper,
which is what makes Stage A fit on a 16 GB T4.

Two cache backends:

``NpzCache``     the real one: memory-maps arrays written by the precompute
                 pass, keyed by scene.
``HashCache``    a deterministic stand-in used by the tests and the CPU
                 overfit gate. It is a fixed pseudo-random function of
                 (scene, modality, index), so it has the shape and the
                 determinism of a real cache and none of the meaning. Anything
                 that reports a metric must use ``NpzCache``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .align import DEFAULT_SPECS
from .norm import NormStats
from .windows import Sample

__all__ = ["FEATURE_DIMS", "REF_DIMS", "Cache", "HashCache", "NpzCache", "collate"]

#: Output widths of the frozen encoders Daystorm caches.
REF_DIMS = {"camera": 1152, "audio": 768}  # SigLIP-so400m, Whisper-small

FEATURE_DIMS = {
    "can": 9,
    "radar": 6,
    "camera": REF_DIMS["camera"],
    "audio": REF_DIMS["audio"],
}

MODALITIES = ("camera", "can", "radar", "audio")


class Cache(Protocol):
    def get(self, scene: int, modality: str, indices: np.ndarray) -> np.ndarray: ...


@dataclass
class HashCache:
    """Deterministic stand-in for a precomputed embedding cache."""

    dims: dict[str, int]
    scale: float = 0.5

    def get(self, scene: int, modality: str, indices: np.ndarray) -> np.ndarray:
        d = self.dims[modality]
        out = np.empty((len(indices), d), dtype=np.float32)
        for i, idx in enumerate(indices):
            key = (hash((scene, modality, int(idx))) & 0x7FFF_FFFF)
            out[i] = np.random.default_rng(key).standard_normal(d) * self.scale
        return out


@dataclass
class NpzCache:
    """Embeddings written by ``scripts/precompute_reference.py``.

    Layout: ``{root}/{modality}/{scene}.npy`` of shape (n_assets, dim).
    """

    root: Path
    dims: dict[str, int]

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self._open: dict[tuple[str, int], np.ndarray] = {}

    def get(self, scene: int, modality: str, indices: np.ndarray) -> np.ndarray:
        key = (modality, scene)
        if key not in self._open:
            path = self.root / modality / f"{scene}.npy"
            if not path.exists():
                raise FileNotFoundError(
                    f"missing cache {path}; run scripts/precompute_reference.py first"
                )
            self._open[key] = np.load(path, mmap_mode="r")
        table = self._open[key]
        clipped = np.clip(indices.astype(np.int64), 0, len(table) - 1)
        return np.asarray(table[clipped], dtype=np.float32)


def collate(
    samples: Sequence[Sample],
    cache: Cache,
    norm: NormStats | None = None,
) -> dict:
    """Stack samples into a batch of feature tensors, a mask, and the text.

    Returns numpy arrays; the trainer moves them to the device. Keeping torch
    out of this module means the data path is testable without torch
    installed.
    """
    import numpy as np  # local: keeps the module import-light

    order = [s.name for s in DEFAULT_SPECS]
    n_grid = samples[0].window.n_grid

    feats = {m: np.zeros((len(samples), n_grid, FEATURE_DIMS[m]), np.float32) for m in MODALITIES}
    mask = np.zeros((len(samples), n_grid, len(MODALITIES)), np.float32)

    for b, s in enumerate(samples):
        w = norm.apply(s.window) if norm is not None else s.window
        feats["can"][b] = w.values["can"]
        feats["radar"][b] = w.values["radar"]
        for m in ("camera", "audio"):
            idx = w.values[m][:, 0]
            feats[m][b] = cache.get(s.scene_seed, m, idx)
            feats[m][b][~w.valid[m]] = 0.0  # a masked reference has no embedding
        for j, m in enumerate(MODALITIES):
            mask[b, :, j] = w.valid[m].astype(np.float32)
        assert order  # modality order is fixed by DEFAULT_SPECS

    return {
        "features": feats,
        "mask": mask,
        "questions": [s.question for s in samples],
        "answers": [s.answer for s in samples],
        "events": [s.event for s in samples],
    }
