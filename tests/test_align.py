"""The five alignment rules, proven rather than asserted in a docstring.

These are the tests that matter most in this repo: temporal alignment is the
one line in the target role's requisition that a demo notebook cannot fake,
and a resampler that silently peeks into the future produces a model that
looks excellent offline and is unusable on a vehicle.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from daystorm.data.align import (
    DEFAULT_SPECS,
    AlignedWindow,
    ChannelSpec,
    Stream,
    align_window,
    make_grid,
    resample_causal,
)

GENEROUS = 10.0  # staleness budget large enough that masking never interferes

# Unique millisecond timestamps: keeps ordering properties unambiguous.
ms_times = st.lists(st.integers(0, 2000), min_size=1, max_size=80, unique=True)


def _stream(t_ms, dim=1, name="can"):
    t = np.asarray(sorted(t_ms), dtype=np.float64) / 1000.0
    v = np.arange(len(t) * dim, dtype=np.float32).reshape(len(t), dim)
    return Stream(t, v, name)


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------

def test_grid_is_inclusive_and_exact():
    g = make_grid(0.0, 2.0, 2.0)
    assert g.shape == (5,)
    assert np.allclose(g, [0.0, 0.5, 1.0, 1.5, 2.0])


def test_grid_offsets_from_t0():
    g = make_grid(1234.5, 2.0, 2.0)
    assert np.allclose(g - 1234.5, [0.0, 0.5, 1.0, 1.5, 2.0])


@pytest.mark.parametrize("bad", [(0.0, 2.0), (2.0, 0.0), (-1.0, 2.0)])
def test_grid_rejects_nonpositive(bad):
    with pytest.raises(ValueError):
        make_grid(0.0, *bad)


# --------------------------------------------------------------------------
# rule 1: causal hold
# --------------------------------------------------------------------------

def test_holds_nearest_preceding_sample():
    s = Stream(np.array([0.0, 0.4, 0.9, 1.9]), np.array([10.0, 20.0, 30.0, 40.0]))
    v, valid, _ = resample_causal(s, make_grid(0.0), GENEROUS)
    assert valid.all()
    assert np.allclose(v.ravel(), [10.0, 20.0, 30.0, 30.0, 40.0])


def test_sample_exactly_on_grid_point_counts_as_available():
    s = Stream(np.array([1.0]), np.array([7.0]))
    v, valid, age = resample_causal(s, make_grid(0.0), GENEROUS)
    assert valid[2] and age[2] == 0.0 and v[2, 0] == 7.0
    assert not valid[0] and not valid[1]  # nothing precedes t=0.0 or t=0.5


@given(times=ms_times, inject_ms=st.integers(0, 2000))
@settings(max_examples=250, deadline=None)
def test_future_samples_never_leak(times, inject_ms):
    """Rule 1, as a property: adding a sample at time T cannot change any grid
    point strictly before T. This is the guarantee that stops offline scores
    from being fiction."""
    grid = make_grid(0.0)
    before, _, _ = resample_causal(_stream(times), grid, GENEROUS)

    t = np.append(np.asarray(sorted(times), dtype=np.float64) / 1000.0, inject_ms / 1000.0)
    v = np.append(np.arange(len(times), dtype=np.float32), 999.0)[:, None]
    after, _, _ = resample_causal(Stream(t, v), grid, GENEROUS)

    untouched = grid < inject_ms / 1000.0
    assert np.allclose(before[untouched], after[untouched])


# --------------------------------------------------------------------------
# rule 2: staleness budget
# --------------------------------------------------------------------------

def test_staleness_budget_masks_stale_holds():
    s = Stream(np.array([0.0, 0.45]), np.array([1.0, 2.0]))
    _, valid, age = resample_causal(s, make_grid(0.0), max_staleness_s=0.10)
    assert valid.tolist() == [True, True, False, False, False]
    assert age[0] == 0.0 and np.isclose(age[1], 0.05)
    assert np.isclose(age[2], 0.55)


def test_budget_boundary_is_inclusive():
    s = Stream(np.array([0.4]), np.array([1.0]))
    _, valid, _ = resample_causal(s, make_grid(0.0), max_staleness_s=0.10)
    assert valid[1]  # age is exactly 0.10


def test_jitter_inside_budget_stays_valid():
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 2.05, 0.01)
    t = t + rng.uniform(-0.002, 0.002, t.shape)
    s = Stream(t, np.arange(len(t), dtype=np.float32))
    _, valid, _ = resample_causal(s, make_grid(0.0), max_staleness_s=0.030)
    assert valid[1:].all()


# --------------------------------------------------------------------------
# rule 3: explicit masks, never silent zeros
# --------------------------------------------------------------------------

def test_masked_points_are_exactly_zero():
    s = Stream(np.array([0.0]), np.array([[5.0, -5.0]]))
    v, valid, _ = resample_causal(s, make_grid(0.0), max_staleness_s=0.10)
    assert np.array_equal(v[~valid], np.zeros((4, 2), dtype=np.float32))


def test_dropout_span_masks_then_recovers():
    """A 0.8 s outage in the middle of the window."""
    t = np.concatenate([np.arange(0.0, 0.61, 0.05), np.arange(1.45, 2.05, 0.05)])
    s = Stream(t, np.arange(len(t), dtype=np.float32))
    _, valid, _ = resample_causal(s, make_grid(0.0), max_staleness_s=0.090)
    assert valid.tolist() == [True, True, False, True, True]


def test_empty_stream_is_all_invalid_not_a_crash():
    v, valid, age = resample_causal(Stream.empty(9, "can"), make_grid(0.0), GENEROUS)
    assert v.shape == (5, 9) and not valid.any() and np.isinf(age).all()


def test_missing_modality_is_treated_as_total_outage():
    w = align_window({}, DEFAULT_SPECS, t0=0.0)
    assert set(w.values) == {s.name for s in DEFAULT_SPECS}
    assert all(v == 0.0 for v in w.coverage().values())
    assert not w.is_usable()


# --------------------------------------------------------------------------
# rule 5: one clock, normalised input
# --------------------------------------------------------------------------

@given(times=ms_times, seed=st.integers(0, 10_000))
@settings(max_examples=200, deadline=None)
def test_input_order_does_not_change_output(times, seed):
    """Rule 5, as a property: how the caller concatenated its shards is not
    allowed to be load-bearing."""
    t = np.asarray(sorted(times), dtype=np.float64) / 1000.0
    v = np.arange(len(t), dtype=np.float32)[:, None]
    grid = make_grid(0.0)

    ordered, _, _ = resample_causal(Stream(t, v), grid, GENEROUS)
    perm = np.random.default_rng(seed).permutation(len(t))
    shuffled, _, _ = resample_causal(Stream(t[perm], v[perm]), grid, GENEROUS)
    assert np.array_equal(ordered, shuffled)


def test_duplicate_timestamps_keep_the_last_value():
    s = Stream(np.array([0.0, 1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0, 4.0]))
    assert len(s) == 2
    assert s.v[-1, 0] == 4.0


def test_non_finite_timestamps_rejected():
    with pytest.raises(ValueError):
        Stream(np.array([0.0, np.nan]), np.array([1.0, 2.0]))


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        Stream(np.array([0.0, 1.0]), np.array([1.0]))


def test_declared_dim_mismatch_rejected():
    spec = ChannelSpec("can", dim=9, max_staleness_s=0.03)
    with pytest.raises(ValueError, match="dim"):
        align_window({"can": Stream(np.array([0.0]), np.zeros((1, 4)))}, [spec])


# --------------------------------------------------------------------------
# window-level behaviour
# --------------------------------------------------------------------------

def test_features_and_mask_shapes_line_up():
    from daystorm.data.synthetic import make_scene

    scene = make_scene(seed=1, event="brake", dropout=False)
    w = align_window(scene.streams, DEFAULT_SPECS, t0=scene.event_t)
    total_dim = sum(s.dim for s in DEFAULT_SPECS)
    order = [s.name for s in DEFAULT_SPECS]
    assert w.features(order).shape == (5, total_dim)
    assert w.mask(order).shape == (5, len(DEFAULT_SPECS))
    assert w.mask(order).dtype == np.float32


def test_usable_threshold_rejects_sparse_windows():
    grid = make_grid(0.0)
    good = np.ones(5, dtype=bool)
    thin = np.array([True, False, False, False, False])
    w = AlignedWindow(
        grid=grid,
        values={"a": np.zeros((5, 1), np.float32), "b": np.zeros((5, 1), np.float32)},
        valid={"a": good, "b": thin},
        age={"a": np.zeros(5), "b": np.zeros(5)},
        t0=0.0,
    )
    assert w.coverage() == {"a": 1.0, "b": 0.2}
    assert not w.is_usable(min_coverage=0.6)
    assert w.is_usable(min_coverage=0.2)


def test_synthetic_brake_event_is_groundable():
    """The generator must actually produce the physics the labels claim."""
    from daystorm.data.synthetic import CAN_CHANNELS, make_scene

    scene = make_scene(seed=0, event="brake", dropout=False)
    w = align_window(scene.streams, DEFAULT_SPECS, t0=scene.event_t - 0.5)
    speed = w.values["can"][:, CAN_CHANNELS.index("speed_mps")]
    brake = w.values["can"][:, CAN_CHANNELS.index("brake")]
    lead = w.values["radar"][:, 0]
    assert speed[-1] < speed[0] - 1.5, "braking window must actually decelerate"
    assert brake.max() > 0.5, "brake pedal must be applied"
    assert lead[-1] < lead[0], "lead vehicle must be closing"
