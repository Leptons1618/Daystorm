"""The monitor has to fire on real degradation and stay quiet otherwise."""

from __future__ import annotations

import numpy as np

from daystorm.data.align import DEFAULT_SPECS, Stream, align_window
from daystorm.data.norm import NormStats
from daystorm.data.synthetic import make_scene
from daystorm.monitor.drift import DriftMonitor


def _windows(seeds, dropout=True, scale=1.0):
    out = []
    for s in seeds:
        scene = make_scene(seed=s, dropout=dropout)
        if scale != 1.0:
            st = scene.streams["can"]
            scene.streams["can"] = Stream(st.t, st.v * scale, "can")
        for t0 in np.arange(0.0, scene.duration_s - 2.0, 1.0):
            out.append(align_window(scene.streams, DEFAULT_SPECS, t0=float(t0)))
    return out


def _fitted(seeds=range(8)):
    return NormStats.fit(_windows(seeds))


def test_in_distribution_data_raises_no_alerts():
    stats = _fitted()
    mon = DriftMonitor(stats)
    for w in _windows(range(8, 14)):
        mon.observe(w)
    rep = mon.report()
    assert rep.ok, rep.alerts
    assert all(z < 3.0 for z in rep.mean_abs_z.values())


def test_scaled_channels_trip_the_z_alert():
    """A recalibrated sensor or a units change."""
    stats = _fitted()
    mon = DriftMonitor(stats)
    for w in _windows(range(8, 14), scale=25.0):
        mon.observe(w)
    rep = mon.report()
    assert not rep.ok
    assert any("distribution shift" in a for a in rep.alerts)


def test_dead_sensor_trips_the_coverage_alert():
    stats = _fitted()
    mon = DriftMonitor(stats)
    for scene_seed in range(8, 14):
        scene = make_scene(seed=scene_seed)
        scene.streams["radar"] = Stream.empty(6, "radar")
        for t0 in np.arange(0.0, 18.0, 1.0):
            mon.observe(align_window(scene.streams, DEFAULT_SPECS, t0=float(t0)))
    rep = mon.report()
    assert any("radar" in a and "coverage" in a for a in rep.alerts)


def test_stuck_signal_is_caught_even_though_z_looks_fine():
    """The failure a z-score check alone cannot see."""
    stats = _fitted()
    mon = DriftMonitor(stats)
    for scene_seed in range(8, 14):
        scene = make_scene(seed=scene_seed, dropout=False)
        st = scene.streams["can"]
        frozen = np.repeat(stats.mean["can"][None, :], len(st), axis=0)
        scene.streams["can"] = Stream(st.t, frozen, "can")
        for t0 in np.arange(0.0, 18.0, 1.0):
            mon.observe(align_window(scene.streams, DEFAULT_SPECS, t0=float(t0)))
    rep = mon.report()
    assert rep.mean_abs_z["can"] < 0.5, "a frozen-at-the-mean signal looks perfectly normal to z"
    assert any("stuck signal" in a for a in rep.alerts)


def test_monitor_stays_silent_below_the_minimum_sample_count():
    """One odd junction must not page anybody."""
    stats = _fitted()
    mon = DriftMonitor(stats, min_windows=50)
    for w in _windows(range(8, 9), scale=25.0):
        mon.observe(w)
    assert mon.report().ok


def test_report_renders_without_crashing_on_partial_data():
    stats = _fitted()
    mon = DriftMonitor(stats)
    mon.observe(_windows(range(8, 9))[0])
    text = mon.report().render()
    assert "windows observed" in text and "can" in text
