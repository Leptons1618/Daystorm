"""HTTP service around the aligner and the model.

The service takes *raw asynchronous streams* rather than a pre-aligned tensor.
That is the deliberate choice: alignment is where the correctness risk lives,
so it belongs on the server where it is versioned with the model, not in
whatever client happens to be calling. A client that aligns its own data can
silently use a different staleness budget and quietly move the model
off-distribution.

Every response carries the sensor coverage that produced it. An answer derived
from a window with the radar masked is a different kind of answer, and the
caller is entitled to know that without having to ask.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from daystorm.data.align import DEFAULT_SPECS, Stream, align_window
from daystorm.data.norm import NormStats
from daystorm.monitor.drift import DriftMonitor

MODALITIES = tuple(s.name for s in DEFAULT_SPECS)


class StreamIn(BaseModel):
    t: list[float] = Field(..., description="timestamps, seconds on the reference clock")
    v: list[list[float]] = Field(..., description="samples, shape (N, channels)")


class PredictRequest(BaseModel):
    t0: float = Field(0.0, description="window start on the reference clock")
    duration_s: float = 2.0
    streams: dict[str, StreamIn]


class PredictResponse(BaseModel):
    answer: str
    question: str
    coverage: dict[str, float]
    degraded: list[str]
    latency_ms: float
    model_version: str


class _State:
    """Loaded once at startup; absent in the smoke-test configuration."""

    def __init__(self) -> None:
        self.fusion: Any = None
        self.backbone: Any = None
        self.stats: NormStats | None = None
        self.monitor: DriftMonitor | None = None
        self.version: str = "unloaded"
        self.served: int = 0
        self.errors: int = 0
        self.latencies: list[float] = []


state = _State()


def _load(ckpt: str) -> None:
    import torch

    from daystorm.data.tensors import FEATURE_DIMS
    from daystorm.model.fusion import DaystormFusion
    from daystorm.train.backbone import build_backbone

    path = Path(ckpt)
    blob = torch.load(path / "fusion.pt", map_location="cpu", weights_only=False)
    state.stats = NormStats.load(path / "norm.json")
    name = blob.get("args", {}).get("backbone", "tiny")
    backbone = build_backbone(name)
    if blob.get("backbone") is not None:
        backbone.load_state_dict(blob["backbone"])
    backbone.eval()
    fusion = DaystormFusion(FEATURE_DIMS, d_model=backbone.d_model)
    fusion.load_state_dict(blob["fusion"])
    fusion.eval()
    state.backbone, state.fusion = backbone, fusion
    state.monitor = DriftMonitor(state.stats)
    state.version = f"{name}@{path.name}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt = os.environ.get("DAYSTORM_CKPT", "")
    if ckpt and Path(ckpt).exists():
        _load(ckpt)
    yield


app = FastAPI(title="Daystorm", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Liveness: the process is up. Never depends on the model."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Readiness: the model is loaded and can serve. Distinct from liveness so
    a rollout does not route traffic at a container still loading weights."""
    if state.fusion is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready", "model_version": state.version}


@app.get("/metrics")
def metrics() -> str:
    lat = sorted(state.latencies)
    p95 = lat[max(int(0.95 * len(lat)) - 1, 0)] if lat else 0.0
    lines = [
        f"daystorm_requests_total {state.served}",
        f"daystorm_errors_total {state.errors}",
        f"daystorm_latency_p95_ms {p95:.3f}",
        f"daystorm_model_loaded {int(state.fusion is not None)}",
    ]
    if state.monitor is not None:
        rep = state.monitor.report()
        lines.append(f"daystorm_drift_alerts {len(rep.alerts)}")
        for name, cov in rep.coverage.items():
            lines.append(f'daystorm_coverage{{modality="{name}"}} {cov:.4f}')
    return "\n".join(lines) + "\n"


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    # Validate the request before reporting service state: a client with a
    # malformed payload should learn that, not be told to retry later.
    unknown = set(req.streams) - set(MODALITIES)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"unknown modalities {sorted(unknown)}; expected a subset of {list(MODALITIES)}",
        )

    started = time.perf_counter()
    try:
        streams = {}
        for name, s in req.streams.items():
            if len(s.t) != len(s.v):
                raise HTTPException(
                    status_code=422, detail=f"{name}: {len(s.t)} timestamps but {len(s.v)} samples"
                )
            streams[name] = Stream(np.asarray(s.t), np.asarray(s.v, dtype=np.float32), name)
        window = align_window(streams, DEFAULT_SPECS, req.t0, req.duration_s)
    except HTTPException:
        state.errors += 1
        raise
    except ValueError as exc:
        state.errors += 1
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if state.fusion is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    answer, question = _run(window)
    if state.monitor is not None:
        state.monitor.observe(window)

    coverage = window.coverage()
    elapsed = (time.perf_counter() - started) * 1000.0
    state.served += 1
    state.latencies.append(elapsed)
    del state.latencies[:-1000]

    return PredictResponse(
        answer=answer,
        question=question,
        coverage={k: round(v, 3) for k, v in coverage.items()},
        degraded=sorted(k for k, v in coverage.items() if v < 1.0),
        latency_ms=round(elapsed, 2),
        model_version=state.version,
    )


def _run(window) -> tuple[str, str]:
    import torch

    from daystorm.data.tensors import FEATURE_DIMS, HashCache
    from daystorm.model.fusion import DaystormFusion

    cache = HashCache(FEATURE_DIMS)
    normed = state.stats.apply(window)
    order = DaystormFusion.MODALITIES
    feats, mask = {}, np.zeros((1, window.n_grid, len(order)), np.float32)
    for j, name in enumerate(order):
        if name in ("can", "radar"):
            feats[name] = torch.from_numpy(normed.values[name][None, ...])
        else:
            idx = normed.values[name][:, 0]
            emb = cache.get(0, name, idx)
            emb[~normed.valid[name]] = 0.0
            feats[name] = torch.from_numpy(emb[None, ...])
        mask[0, :, j] = normed.valid[name].astype(np.float32)

    question = "What is the vehicle doing?"
    with torch.no_grad():
        prefix = state.fusion(feats, torch.from_numpy(mask))
        answer = state.backbone.generate_one(prefix, question, max_new=200)
    return answer.strip(), question
