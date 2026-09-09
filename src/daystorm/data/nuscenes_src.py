"""Adapter: nuScenes + CAN bus expansion -> Daystorm ``Stream`` objects.

Everything downstream of this file is source-agnostic. Swapping
:mod:`daystorm.data.synthetic` for this module changes where the streams come
from and nothing else.

NOTE: this module needs the nuScenes data and devkit, so it is exercised by
``scripts/check_nuscenes.py`` against a real download rather than by the unit
tests. The tests run on the synthetic source so that ``make test`` works on a
clean checkout.

Reference clock: LIDAR_TOP sample timestamps, following the devkit convention.
nuScenes timestamps are integer microseconds; every Stream here is float
seconds relative to the scene's first LIDAR sample, which keeps float64
precision comfortable and makes windows portable across scenes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .align import Stream
from .synthetic import Scene

__all__ = ["list_scenes", "load_scene"]

_US = 1e-6

# nuScenes CAN bus expansion message -> the channels Daystorm consumes.
_CAN_LAYOUT = (
    ("steeranglefeedback", "value"),        # steering_angle_deg
    ("vehicle_monitor", "steering_speed"),  # steering_rate_dps
    ("vehicle_monitor", "throttle"),
    ("vehicle_monitor", "brake"),
    ("vehicle_monitor", "vehicle_speed"),
)


def list_scenes(dataroot: str | None = None, version: str = "v1.0-mini") -> list[str]:
    from nuscenes.nuscenes import NuScenes

    root = dataroot or os.environ.get("DAYSTORM_NUSCENES", "data/raw")
    nusc = NuScenes(version=version, dataroot=root, verbose=False)
    return [s["name"] for s in nusc.scene]


def load_scene(
    scene_name: str,
    dataroot: str | None = None,
    version: str = "v1.0-mini",
    camera: str = "CAM_FRONT",
) -> Scene:
    """Build one :class:`Scene` from a nuScenes scene name.

    Only ``CAM_FRONT`` is loaded by default. The six-camera surround costs 6x
    the vision tokens for a model that is being trained on a 16 GB T4, so the
    free-tier path uses one camera and the ablation in Phase 2 reports what
    that costs.
    """
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    from nuscenes.nuscenes import NuScenes

    root = Path(dataroot or os.environ.get("DAYSTORM_NUSCENES", "data/raw"))
    nusc = NuScenes(version=version, dataroot=str(root), verbose=False)

    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    first = nusc.get("sample", scene["first_sample_token"])
    t_ref = nusc.get("sample_data", first["data"]["LIDAR_TOP"])["timestamp"] * _US

    cam_t, cam_paths = _sample_data_series(nusc, scene, camera, t_ref)
    radar_t, radar_v = _radar_series(nusc, scene, t_ref)

    # The CAN bus is a separate download from the main nuScenes archive, and
    # most redistributions omit it - the Kaggle nuScenes-mini mirror has
    # samples, sweeps and maps but no can_bus/, which used to make every scene
    # here unloadable. A missing sensor is a first-class input in this pipeline:
    # an empty stream resamples to a zero-coverage window that the fusion mask
    # marks invalid, exactly as a dead bus would. So this degrades rather than
    # refuses - loudly, because the ablations say CAN is the modality whose
    # absence makes the model fabricate.
    try:
        can_t, can_v = _can_series(NuScenesCanBus(dataroot=str(root)), scene_name, t_ref)
    except Exception as exc:  # noqa: BLE001 - the devkit raises bare Exception here
        print(f"[nuscenes] CAN bus unavailable: {exc}")
        print("[nuscenes] Loading with the CAN stream empty, so every window is")
        print("[nuscenes] can-coverage 0. Camera and radar are unaffected. Do not")
        print("[nuscenes] quote a CAN ablation, or any grounding metric, from this.")
        # 9 columns, matching FEATURE_DIMS["can"] and _can_series' own no-data
        # return. Imported as a literal to keep this module's dependency on
        # `.align` and `.synthetic` alone.
        can_t, can_v = np.zeros(0), np.zeros((0, 9), dtype=np.float32)

    duration = float(scene["nbr_samples"]) * 0.5  # keyframes are 2 Hz
    streams = {
        "camera": Stream(cam_t, np.arange(len(cam_t), dtype=np.float32), "camera"),
        "can": Stream(can_t, can_v, "can"),
        "radar": Stream(radar_t, radar_v, "radar"),
        # No public driving corpus ships synchronised in-cabin audio; the audio
        # stream is attached separately by scripts/synthesise_audio.py and is
        # declared as synthetic in docs/model_card.md.
        "audio": Stream.empty(1, "audio"),
    }
    out = Scene(streams, event="unknown", event_t=0.0, duration_s=duration, seed=0)
    out.asset_paths = {"camera": cam_paths}  # type: ignore[attr-defined]
    return out


def _sample_data_series(nusc: Any, scene: dict, channel: str, t_ref: float):
    """Walk the sample_data linked list for one sensor channel."""
    sample = nusc.get("sample", scene["first_sample_token"])
    token = sample["data"][channel]
    times, paths = [], []
    while token:
        sd = nusc.get("sample_data", token)
        times.append(sd["timestamp"] * _US - t_ref)
        paths.append(sd["filename"])
        token = sd["next"]
    return np.asarray(times, dtype=np.float64), paths


def _radar_series(nusc: Any, scene: dict, t_ref: float):
    """Front radar reduced to the six kinematic channels Daystorm uses."""
    from nuscenes.utils.data_classes import RadarPointCloud

    sample = nusc.get("sample", scene["first_sample_token"])
    token = sample["data"]["RADAR_FRONT"]
    times, rows = [], []
    while token:
        sd = nusc.get("sample_data", token)
        pc = RadarPointCloud.from_file(str(Path(nusc.dataroot) / sd["filename"]))
        pts = pc.points  # (18, N): x, y, z, dyn_prop, id, rcs, vx, vy, ...
        if pts.shape[1] == 0:
            times.append(sd["timestamp"] * _US - t_ref)
            rows.append([80.0, 0.0, 80.0, 0.0, 0.0, 60.0])
        else:
            rng = np.hypot(pts[0], pts[1])
            k = int(np.argmin(rng))
            lead_range = float(rng[k])
            lead_rate = float(pts[6, k])  # vx, compensated radial velocity
            ttc = lead_range / -lead_rate if lead_rate < -0.1 else 60.0
            times.append(sd["timestamp"] * _US - t_ref)
            rows.append(
                [lead_range, lead_rate, float(rng.min()), lead_rate,
                 float(pts.shape[1]), float(np.clip(ttc, 0.0, 60.0))]
            )
        token = sd["next"]
    return np.asarray(times, dtype=np.float64), np.asarray(rows, dtype=np.float32)


def _can_series(can: Any, scene_name: str, t_ref: float):
    """Merge CAN messages onto one timebase, holding each channel causally.

    The expansion publishes each message type at its own rate, so this is the
    same causal-hold problem the aligner solves - reused here at 100 Hz rather
    than reimplemented.
    """
    from .align import make_grid, resample_causal

    per_channel: list[Stream] = []
    for message, field in _CAN_LAYOUT:
        try:
            msgs = can.get_messages(scene_name, message)
        except Exception:  # noqa: BLE001 - a scene without this message type
            per_channel.append(Stream.empty(1, message))
            continue
        t = np.asarray([m["utime"] * _US - t_ref for m in msgs], dtype=np.float64)
        v = np.asarray([float(m[field]) for m in msgs], dtype=np.float32)
        per_channel.append(Stream(t, v, message))

    spans = [s.t[-1] for s in per_channel if len(s)]
    if not spans:
        return np.zeros(0), np.zeros((0, 9), dtype=np.float32)

    grid = make_grid(0.0, duration_s=float(max(spans)), rate_hz=100.0)
    cols = [resample_causal(s, grid, max_staleness_s=0.5)[0] for s in per_channel]
    steer, steer_rate, throttle, brake, speed = (c[:, 0] for c in cols)

    dt = 1 / 100.0
    accel_x = np.gradient(speed, dt)
    yaw_rate = np.gradient(np.deg2rad(steer), dt) * 0.4  # crude bicycle-model proxy
    accel_y = np.deg2rad(yaw_rate) * speed
    stacked = np.stack(
        [steer, steer_rate, throttle, brake, speed, accel_x, accel_y, yaw_rate, speed],
        axis=1,
    ).astype(np.float32)
    return grid, stacked
