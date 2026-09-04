"""Phase 1 guarantees: causality survives the encoder, masks actually mask."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from daystorm.model.encoders import TelemetryTCN, TrackMLP
from daystorm.model.fusion import DaystormFusion, MaskedAttentionPool

DIMS = {"can": 9, "radar": 6, "camera": 1152, "audio": 768}
B, G, N_MOD = 3, 5, 4


def _batch(seed: int = 0):
    torch.manual_seed(seed)
    feats = {k: torch.randn(B, G, d) for k, d in DIMS.items()}
    return feats, torch.ones(B, G, N_MOD)


def test_fusion_emits_a_fixed_token_budget():
    m = DaystormFusion(DIMS, d_model=512)
    feats, mask = _batch()
    out = m(feats, mask)
    assert out.shape == (B, m.n_tokens, 512)
    assert m.n_tokens == 16
    assert torch.isfinite(out).all()


def test_token_budget_is_independent_of_masking():
    """Sequence length must not depend on data, or Phase 3 loses static shapes."""
    m = DaystormFusion(DIMS, d_model=256)
    feats, mask = _batch()
    full = m(feats, mask)
    mask[:, 2:, :] = 0.0
    sparse = m(feats, mask)
    assert full.shape == sparse.shape


def test_telemetry_encoder_is_causal():
    """Rule 1 does not stop holding at the tensor boundary."""
    enc = TelemetryTCN(in_dim=10, hidden=32).eval()
    x = torch.randn(1, G, 10)
    with torch.no_grad():
        before = enc(x)
        x[:, -1, :] += 10.0  # perturb only the last grid point
        after = enc(x)
    assert torch.allclose(before[:, :-1], after[:, :-1], atol=1e-6)
    assert not torch.allclose(before[:, -1], after[:, -1])


def test_track_encoder_is_pointwise_in_time():
    enc = TrackMLP(in_dim=7, hidden=32).eval()
    x = torch.randn(1, G, 7)
    with torch.no_grad():
        before = enc(x)
        x[:, 0, :] += 5.0
        after = enc(x)
    assert not torch.allclose(before[:, 0], after[:, 0])
    assert torch.allclose(before[:, 1:], after[:, 1:], atol=1e-6)


def test_pool_ignores_masked_positions():
    """Masked grid points must not reach the pooled summary at all."""
    pool = MaskedAttentionPool(hidden=32, n_queries=4).eval()
    h = torch.randn(2, G, 32)
    valid = torch.ones(2, G, dtype=torch.bool)
    valid[:, 3:] = False
    with torch.no_grad():
        before = pool(h, valid)
        h[:, 3:, :] += 100.0  # garbage behind the mask
        after = pool(h, valid)
    assert torch.allclose(before, after, atol=1e-6)


def test_fully_masked_modality_returns_the_absent_token():
    pool = MaskedAttentionPool(hidden=16, n_queries=3).eval()
    h = torch.randn(2, G, 16)
    valid = torch.zeros(2, G, dtype=torch.bool)
    with torch.no_grad():
        out = pool(h, valid)
    assert torch.isfinite(out).all(), "a dead sensor must not produce NaN"
    assert torch.allclose(out[0], pool.absent, atol=1e-6)
    assert torch.allclose(out[1], pool.absent, atol=1e-6)


def test_dead_sensor_output_does_not_depend_on_its_buffer_contents():
    m = DaystormFusion(DIMS, d_model=128).eval()
    feats, mask = _batch()
    mask[:, :, 3] = 0.0  # audio down for the whole window
    with torch.no_grad():
        before = m(feats, mask)
        feats["audio"] = torch.randn(B, G, DIMS["audio"]) * 50.0
        after = m(feats, mask)
    assert torch.allclose(before, after, atol=1e-5)


def test_no_nan_when_every_sensor_is_down():
    m = DaystormFusion(DIMS, d_model=64)
    feats, _ = _batch()
    out = m(feats, torch.zeros(B, G, N_MOD))
    assert torch.isfinite(out).all()


def test_gradients_reach_every_trainable_component():
    m = DaystormFusion(DIMS, d_model=64)
    feats, mask = _batch()
    m(feats, mask).square().mean().backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_modality_embeddings_separate_the_streams():
    """Camera tokens and CAN tokens must not be interchangeable."""
    m = DaystormFusion(DIMS, d_model=64).eval()
    assert not torch.allclose(m.modality_emb[0], m.modality_emb[1])


def test_trainable_budget_stays_small():
    """The point of the design is a small trainable surface on a 3B backbone."""
    m = DaystormFusion(DIMS, d_model=2048)
    millions = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
    assert millions < 12.0, f"fusion stack grew to {millions:.1f} M params"
