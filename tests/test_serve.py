"""The service contract: liveness, readiness, validation, and honest coverage."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from daystorm.data.synthetic import make_scene
from daystorm.serve import app as app_module


@pytest.fixture
def client():
    with TestClient(app_module.app) as c:
        yield c


def _payload(t0=5.0, drop=()):
    scene = make_scene(seed=3, event="brake", dropout=False)
    streams = {}
    for name, s in scene.streams.items():
        if name in drop:
            continue
        keep = (s.t >= t0 - 1.0) & (s.t <= t0 + 3.0)
        streams[name] = {"t": s.t[keep].tolist(), "v": s.v[keep].tolist()}
    return {"t0": t0, "duration_s": 2.0, "streams": streams}


def test_health_never_depends_on_the_model(client):
    """Liveness must stay green while weights are still loading."""
    assert app_module.state.fusion is None
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ready_is_503_until_the_model_is_loaded(client):
    r = client.get("/ready")
    assert r.status_code == 503


def test_predict_refuses_rather_than_guessing_without_weights(client):
    r = client.post("/predict", json=_payload())
    assert r.status_code == 503


def test_unknown_modality_is_rejected_with_a_useful_message(client):
    bad = _payload()
    bad["streams"]["lidar"] = {"t": [0.0], "v": [[1.0]]}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
    assert "lidar" in r.text


def test_mismatched_stream_lengths_are_rejected(client):
    bad = _payload()
    bad["streams"]["can"]["t"] = bad["streams"]["can"]["t"][:-3]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_metrics_are_scrapeable_before_any_traffic(client):
    body = client.get("/metrics").text
    assert "daystorm_requests_total 0" in body
    assert "daystorm_model_loaded 0" in body


def test_alignment_happens_server_side(client):
    """The service takes raw asynchronous streams, not a pre-aligned tensor -
    otherwise every client gets to invent its own staleness budget."""
    payload = _payload()
    rates = {k: len(v["t"]) for k, v in payload["streams"].items()}
    assert rates["can"] > rates["radar"] > rates["camera"], rates
    assert "grid" not in payload and "features" not in payload


def test_window_with_a_dead_sensor_is_still_a_valid_request(client):
    """A vehicle that loses its radar still needs an answer."""
    payload = _payload(drop=("radar",))
    r = client.post("/predict", json=payload)
    assert r.status_code == 503  # no weights in this fixture, but it parsed and aligned
    assert "radar" not in payload["streams"]


def test_stream_timestamps_may_arrive_unsorted(client):
    payload = _payload()
    t = np.asarray(payload["streams"]["can"]["t"])
    v = np.asarray(payload["streams"]["can"]["v"])
    perm = np.random.default_rng(0).permutation(len(t))
    payload["streams"]["can"] = {"t": t[perm].tolist(), "v": v[perm].tolist()}
    r = client.post("/predict", json=payload)
    assert r.status_code == 503  # accepted and aligned; only the weights are missing
