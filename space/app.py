"""Daystorm demo: fuse four sensor streams, then take them away one at a time.

The ablation controls are the point. A static demo of a multimodal model shows
you a nice answer and gives you no way to tell whether the model read all four
streams or only one. Here you switch a sensor off and watch what happens to the
numbers.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CANDIDATE_SRC = [
    _HERE / "src",  # Hugging Face Space bundle (src/ copied next to app.py)
    _HERE.parent / "src",  # repo checkout (running space/app.py)
]
for _src in _CANDIDATE_SRC:
    if (_src / "daystorm").exists():
        sys.path.insert(0, str(_src))
        break
else:
    sys.path.insert(0, str(_CANDIDATE_SRC[0]))

import gradio as gr
import numpy as np
import torch

from daystorm.data.align import DEFAULT_SPECS, align_window
from daystorm.data.norm import NormStats
from daystorm.data.synthetic import CAN_CHANNELS, make_scene
from daystorm.data.tensors import FEATURE_DIMS, HashCache
from daystorm.data.windows import describe_window
from daystorm.model.fusion import DaystormFusion
from daystorm.train.backbone import build_backbone


def _resolve_ckpt() -> Path:
    candidates: list[Path] = []
    env = os.environ.get("DAYSTORM_CKPT", "")
    if env:
        candidates.append(Path(env))
    candidates += [
        _HERE / "ckpt" / "stage_a",  # Hugging Face Space bundle
        _HERE.parent / "ckpt" / "stage_a",  # repo checkout
    ]
    for cand in candidates:
        if (cand / "fusion.pt").exists():
            return cand
    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find fusion.pt. Searched:\n"
        f"  {searched}\n"
        "Train it with `python -m daystorm.train.stage_a --overfit 8 --steps 300` "
        "(writes ckpt/stage_a/fusion.pt), or set DAYSTORM_CKPT to the checkpoint dir."
    )


CKPT = _resolve_ckpt()
ORDER = DaystormFusion.MODALITIES

blob = torch.load(CKPT / "fusion.pt", map_location="cpu", weights_only=False)
STATS = NormStats.load(CKPT / "norm.json")
BACKBONE = build_backbone(blob.get("args", {}).get("backbone", "tiny"))
if blob.get("backbone") is not None:
    BACKBONE.load_state_dict(blob["backbone"])
BACKBONE.eval()
FUSION = DaystormFusion(FEATURE_DIMS, d_model=BACKBONE.d_model)
FUSION.load_state_dict(blob["fusion"])
FUSION.eval()
CACHE = HashCache(FEATURE_DIMS)


def run(event: str, seed: int, t0: float, disabled: list[str]):
    scene = make_scene(seed=int(seed), event=event, dropout=True)
    t0 = float(np.clip(t0, 0.0, scene.duration_s - 2.0))
    window = align_window(scene.streams, DEFAULT_SPECS, t0=t0)

    normed = STATS.apply(window)
    feats = {}
    mask = np.zeros((1, window.n_grid, len(ORDER)), np.float32)
    for j, name in enumerate(ORDER):
        if name in ("can", "radar"):
            feats[name] = torch.from_numpy(normed.values[name][None, ...])
        else:
            emb = CACHE.get(int(seed), name, normed.values[name][:, 0])
            emb[~normed.valid[name]] = 0.0
            feats[name] = torch.from_numpy(emb[None, ...])
        live = normed.valid[name].astype(np.float32)
        if name in disabled:
            live[:] = 0.0
        mask[0, :, j] = live

    question = describe_window(window)[0] or "What is the vehicle doing?"
    started = time.perf_counter()
    with torch.no_grad():
        prefix = FUSION(feats, torch.from_numpy(mask))
        answer = BACKBONE.generate_one(prefix, question, max_new=220).strip()
    elapsed = (time.perf_counter() - started) * 1000

    reference = describe_window(window)[1]
    rows = []
    for j, name in enumerate(ORDER):
        natural = float(window.valid[name].mean())
        effective = float(mask[0, :, j].mean())
        note = "OFF (ablated)" if name in disabled else ("degraded" if natural < 1 else "ok")
        rows.append([name, f"{natural:.0%}", f"{effective:.0%}", note])

    speed = window.values["can"][window.valid["can"], CAN_CHANNELS.index("speed_mps")]
    truth = (
        f"**Ground truth for this window** — speed "
        f"{speed[0]:.1f} → {speed[-1]:.1f} m/s over 2.0 s.\n\n{reference}"
    )
    return question, answer, rows, truth, f"{elapsed:.0f} ms"


with gr.Blocks(title="Daystorm", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Daystorm\n"
        "Four sensor streams — camera 12 Hz, CAN 100 Hz, radar 13 Hz, audio 16 kHz — "
        "resampled onto a 2 Hz causal event grid and fused into a language model.\n\n"
        "**Turn sensors off on the left and watch the answer degrade.** Removing the CAN "
        "bus nearly triples numeric error and takes hallucination from 0% to 21%."
    )
    with gr.Row():
        with gr.Column(scale=1):
            event = gr.Radio(["brake", "turn", "cruise"], value="brake", label="Scenario")
            seed = gr.Slider(0, 60, value=3, step=1, label="Scene")
            t0 = gr.Slider(0.0, 17.0, value=10.0, step=0.5, label="Window start (s)")
            disabled = gr.CheckboxGroup(
                list(ORDER), value=[], label="Force sensors offline (live ablation)"
            )
            go = gr.Button("Ask", variant="primary")
        with gr.Column(scale=2):
            q_out = gr.Textbox(label="Question", interactive=False)
            a_out = gr.Textbox(label="Model answer", lines=4, interactive=False)
            lat = gr.Textbox(label="Latency", interactive=False)
            cov = gr.Dataframe(
                headers=["modality", "sensor coverage", "after ablation", "state"],
                label="What the model actually saw",
                interactive=False,
            )
            truth = gr.Markdown()

    gr.Markdown(
        "---\n**Limitations.** This checkpoint uses a byte-level stand-in backbone "
        "(~2 M params, not Qwen2.5-VL) and content-free camera/audio embeddings, so the "
        "camera and audio switches measure scene-identity leakage rather than perception. "
        "The CAN and radar paths are real. Scenes are synthetic. Not for use in a vehicle."
    )
    go.click(run, [event, seed, t0, disabled], [q_out, a_out, cov, truth, lat])
    demo.load(run, [event, seed, t0, disabled], [q_out, a_out, cov, truth, lat])

if __name__ == "__main__":
    demo.launch()
