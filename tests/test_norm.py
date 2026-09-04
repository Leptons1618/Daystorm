"""Rule 4: normalisation statistics come from the train split and nowhere else."""

from __future__ import annotations

import numpy as np
import pytest

from daystorm.data.align import DEFAULT_SPECS, ChannelSpec, Stream, align_window
from daystorm.data.norm import NormStats
from daystorm.data.synthetic import make_scene


def _windows(seeds, dropout=True):
    out = []
    for s in seeds:
        scene = make_scene(seed=s, dropout=dropout)
        for t0 in np.arange(0.0, scene.duration_s - 2.0, 1.0):
            out.append(align_window(scene.streams, DEFAULT_SPECS, t0=float(t0)))
    return out


def test_fit_then_apply_gives_zero_mean_unit_std_on_train():
    ws = _windows(range(6))
    stats = NormStats.fit(ws, DEFAULT_SPECS)
    normed = [stats.apply(w) for w in ws]

    stacked = np.concatenate([w.values["can"][w.valid["can"]] for w in normed], axis=0)
    assert np.allclose(stacked.mean(axis=0), 0.0, atol=1e-3)
    assert np.allclose(stacked.std(axis=0), 1.0, atol=1e-2)


def test_reference_channels_are_not_normalised():
    ws = _windows(range(3))
    stats = NormStats.fit(ws, DEFAULT_SPECS)
    assert "camera" not in stats.mean and "audio" not in stats.mean
    before = ws[0].values["camera"].copy()
    after = stats.apply(ws[0]).values["camera"]
    assert np.array_equal(before, after)


def test_masked_points_excluded_from_the_fit():
    """A dead channel holding a huge stale value must not drag the mean."""
    grid_streams = {
        "can": Stream(np.array([0.0]), np.full((1, 9), 1000.0)),
        "radar": Stream(np.arange(0.0, 2.1, 0.05), np.ones((42, 6))),
    }
    specs = [s for s in DEFAULT_SPECS if s.name in ("can", "radar")]
    w = align_window(grid_streams, specs, t0=0.0)
    assert w.valid["can"].sum() == 1  # only t=0 is fresh enough

    stats = NormStats.fit([w], specs)
    assert stats.count["can"] == 1
    assert np.allclose(stats.mean["can"], 1000.0)


def test_masked_points_are_zero_after_apply():
    ws = _windows(range(4))
    stats = NormStats.fit(ws, DEFAULT_SPECS)
    for w in ws:
        n = stats.apply(w)
        for name in ("can", "radar"):
            assert np.array_equal(n.values[name][~n.valid[name]], 0.0 * n.values[name][~n.valid[name]])
            assert not n.values[name][~n.valid[name]].any()


def test_apply_does_not_mutate_by_default():
    ws = _windows(range(2))
    stats = NormStats.fit(ws, DEFAULT_SPECS)
    original = ws[0].values["can"].copy()
    stats.apply(ws[0])
    assert np.array_equal(ws[0].values["can"], original)
    stats.apply(ws[0], in_place=True)
    assert not np.array_equal(ws[0].values["can"], original)


def test_val_split_is_normalised_with_train_statistics(tmp_path):
    """The leak guard: fitting on val must be impossible by construction."""
    train, val = _windows(range(5)), _windows(range(100, 103))
    stats = NormStats.fit(train, DEFAULT_SPECS, split="train")
    assert stats.split == "train"

    path = stats.save(tmp_path / "norm.json")
    reloaded = NormStats.load(path)
    a = stats.apply(val[0]).values["can"]
    b = reloaded.apply(val[0]).values["can"]
    assert np.allclose(a, b)

    # Val, normalised with train stats, is near zero-mean but not exactly so.
    stacked = np.concatenate([reloaded.apply(w).values["can"][w.valid["can"]] for w in val])
    assert not np.allclose(stacked.mean(axis=0), 0.0, atol=1e-6)


def test_constant_channel_does_not_divide_by_zero():
    spec = ChannelSpec("can", dim=9, max_staleness_s=0.03)
    s = Stream(np.arange(0.0, 2.1, 0.01), np.full((210, 9), 3.0))
    w = align_window({"can": s}, [spec], t0=0.0)
    stats = NormStats.fit([w], [spec])
    out = stats.apply(w).values["can"]
    assert np.isfinite(out).all()
    assert np.allclose(out[w.valid["can"]], 0.0)


def test_fit_refuses_when_a_channel_has_no_valid_samples():
    specs = [ChannelSpec("can", dim=9, max_staleness_s=0.03)]
    w = align_window({}, specs, t0=0.0)
    with pytest.raises(ValueError, match="no valid samples"):
        NormStats.fit([w], specs)


def test_roundtrip_preserves_dtype_and_shape(tmp_path):
    stats = NormStats.fit(_windows(range(3)), DEFAULT_SPECS)
    reloaded = NormStats.load(stats.save(tmp_path / "n.json"))
    for k in stats.mean:
        assert reloaded.mean[k].dtype == np.float32
        assert reloaded.mean[k].shape == stats.mean[k].shape
        assert np.allclose(reloaded.mean[k], stats.mean[k], atol=1e-6)
