# Daystorm

A four-modality driving-scene copilot. Camera, CAN bus, radar tracks and cabin
audio are resampled onto one event grid and fused into a language model, which
answers grounded questions about a two-second window of driving:

> **Q:** Why did the vehicle decelerate?
> **A:** Deceleration from 10.1 to 8.8 m/s over 2.0 s (-0.6 m/s²) with brake
> demand peaking at 0.80. The lead vehicle closed from 29 m to 16 m, minimum
> time-to-collision 2.4 s.

Built by grafting two new modality paths onto a released vision-language model
rather than training one from scratch: the backbone stays, the fusion layer is
new. Trainable surface is ~6 M parameters against a 3 B backbone.

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[dev,serve]"
make test            # 69 tests, no download required
make demo            # print one aligned window from a synthetic scene
make gate            # Phase 1 go/no-go: can the fusion stack carry information?
make eval            # held-out metrics + per-modality ablation + failure gallery
make bench           # stage-by-stage latency
```

## Requirement coverage

| Requirement | What proves it | Where |
|---|---|---|
| Fuse inputs from multiple modalities | Four streams into a shared token space; fixed 16-token budget per window | `model/fusion.py` |
| Pipelines for **temporal alignment** and normalization | 2 Hz causal event grid, per-channel staleness budgets, split-safe norm stats | `data/align.py`, `data/norm.py` |
| Adapt and fine-tune existing GenAI architectures | New encoders + projectors grafted onto a frozen backbone; two-stage recipe | `train/stage_a.py` |
| Hands-on foundation-model fine-tuning | QLoRA on the backbone in Stage B | `train/stage_b.py` *(phase 2)* |
| Vision-language and audio-text fusion | Both paths present, ablated per modality | `model/encoders.py` |
| Hands-on foundation-model fine-tuning | QLoRA adapters + two learning rates | `train/stage_b.py` |
| Model evaluation and testing | 69 tests; grounding metrics, ablation, failure gallery | `tests/`, `eval/` |
| Latency and production deployment | T4 dtype/quant sweep, compile, ONNX (verified), per-stage p50/p95/p99; FastAPI + Docker | `bench/`, `serve/` |
| Distributed training *(nice to have)* | FSDP/DDP wrapper, accelerate config, Kaggle 2×T4 kernel | `train/dist.py`, `kaggle/` |
| MLOps automation and monitoring *(nice to have)* | CI incl. a smoke train; drift monitor with a stuck-signal check | `.github/`, `monitor/` |

## The alignment contract

Four streams, four sample rates, none a multiple of another, all with dropouts.
Everything resamples onto a **2 Hz event grid** — the nuScenes keyframe and
annotation rate, so grid points coincide with ground truth instead of needing
interpolated labels.

| Rule | Enforced by | Test |
|---|---|---|
| Causal hold — nearest *preceding* sample only | `resample_causal` | `test_future_samples_never_leak` |
| Staleness budget — camera 90 ms, CAN 30 ms | `ChannelSpec.max_staleness_s` | `test_staleness_budget_masks_stale_holds` |
| Missing data is an input, never a silent zero | validity bit into every encoder | `test_dead_sensor_output_does_not_depend_on_its_buffer_contents` |
| Norm stats from the train split alone | `NormStats.fit` | `test_val_split_is_normalised_with_train_statistics` |
| One reference clock, order-independent | `Stream.__init__` | `test_input_order_does_not_change_output` |

Causality survives into the model: the telemetry encoder is a left-padded
dilated TCN, and its normalization is per-timestep `ChannelNorm` rather than
`GroupNorm` — GroupNorm normalizes over channels *and* time, which lets the
last grid point shift the statistics of every earlier one and silently breaks
causality. `test_telemetry_encoder_is_causal` catches exactly that.

## Running on free-tier Colab

Free Colab is a **T4 (16 GB), no A100, one GPU**, ~12.7 GB system RAM, an
ephemeral ~107 GB VM disk, and 15 GB of Google Drive. Three consequences drive
the design:

- **T4 is Turing.** No bfloat16 and no FlashAttention-2. Use fp16 compute dtype
  and `attn_implementation="sdpa"`; an A100 recipe copied verbatim fails here.
- **Frozen encoders run once, to disk.** `scripts/precompute_reference.py`
  caches SigLIP and Whisper outputs, so training never holds them in VRAM.
- **Sessions get preempted.** Checkpoint to Drive every 200 steps, resume by
  default, keep nothing that only exists on the VM.

| Phase | On free tier | Notes |
|---|---|---|
| 0 — data, alignment, tests | **Yes**, no GPU at all | Runs on any laptop; save the GPU quota |
| 1 — Stage A projector training | **Yes** | ~6–8 h *estimated* on T4, split across sessions; single camera, cached embeddings |
| 2 — Stage B QLoRA | **Marginal** | ~20–30 h *estimated* on T4 vs ~3–4 h on one A100. Possible, slow |
| 3 — quantization + latency | **Yes** | INT8/NF4 sweep is fine on T4 |
| 3 — distributed training | **No** | Free Colab gives one GPU. Use **Kaggle: 2× T4, 30 h/week, free** |
| Storage | **Partly** | mini + CAN expansion ≈ 4.5 GB fits in 15 GB Drive; the 30 GB trainval slice does not |

Timing figures marked *estimated* have not been measured yet — Phase 1 was
developed and gated on CPU. Phase 3 replaces every estimate with a measurement.

## Layout

```
src/daystorm/
  data/align.py       the event grid, causal hold, staleness masks
      norm.py         split-safe normalisation statistics
      windows.py      windows -> grounded question/answer supervision
      tensors.py      embedding cache + batching
      synthetic.py    a driving-scene generator, so the repo runs with no download
      nuscenes_src.py the real source: nuScenes + CAN bus expansion
  model/encoders.py   causal TCN (CAN), track MLP (radar), cached-embedding adapters
        fusion.py     masked attention pooling -> 16 tokens -> backbone width
  train/backbone.py   byte-level stand-in and the real HF backbone, one interface
        stage_a.py    projector alignment + the go/no-go gate
scripts/              colab sync/train wrappers, nuScenes download, precompute
```

## Results so far

Held-out split, 16 windows, scene-level partition, **byte-level backbone on CPU**.
Read `MODEL_CARD.md` before quoting any of this — the camera and audio caches
are content-free, so those two ablations measure scene-identity leakage rather
than perception.

| run | exact match | manoeuvre acc | numeric MAE | hallucination |
|---|---|---|---|---|
| full | 0.00 | 1.00 | 0.225 | 0.000 |
| no camera | 0.00 | 1.00 | 0.206 (−0.02) | 0.000 |
| **no CAN** | 0.00 | 1.00 | **0.663 (+0.44)** | **0.208 (+0.21)** |
| no radar | 0.00 | 1.00 | 0.323 (+0.10) | 0.000 |
| no audio | 0.00 | 1.00 | 0.419 (+0.19) | 0.000 |

Removing the CAN bus nearly triples numeric error and takes hallucination from
0% to 21%: with the telemetry gone the model starts inventing quantities. That
is the architecture behaving as designed.

### Measured on a Tesla T4

Full numbers in `reports/phase3_status.md`. The finding worth repeating:

**`torch.cuda.is_bf16_supported()` returns `True` on a T4, and bf16 is 20x
slower.** PyTorch counts emulation as support, so an A100 recipe copied onto
Turing does not error — it runs at a twentieth of the speed. Fusion stack, p50
at batch 32:

| config | eager | `torch.compile` | vs fp16 |
|---|---|---|---|
| fp32 | 4.868 ms | 3.370 ms | 1.35x slower |
| **fp16** | **3.597 ms** | **2.557 ms** | — |
| bf16 *(emulated)* | 74.149 ms | 72.829 ms | **20.6x slower** |
| nf4 | 6.271 ms | 7.126 ms | 1.74x slower |

nf4 losing to fp32 is expected and worth saying out loud: the fusion stack is
~6 M parameters, so dequantisation overhead swamps any memory saving. 4-bit is
for the 3 B backbone, not for this.

End-to-end, one window:

| stage | CPU p50 | T4 p50 | share (T4) |
|---|---|---|---|
| align | 0.04 ms | 0.15 ms | 0.1% |
| fuse | 0.82 ms | 3.41 ms | 1.5% |
| decode | 233.73 ms | 228.93 ms | 98.5% |

**The GPU barely helps** — 2% on decode — because the backbone is small enough
to be bound by per-token kernel launch latency, not compute. Batched decode or
a KV cache with CUDA graphs will move that number; a bigger GPU will not.

Training on T4: Stage A 700 steps in 38 s (CPU: 139 s, 3.7x); Stage B QLoRA
70 ms/step at 16 windows/step; the Phase 1 gate passes with a cleaner control
than on CPU (100% exact fused vs **0%** shuffled).

### How small can the frozen backbone get?

Full study in `reports/phase4_backbone_ladder.md`. Same gate, same 8 windows,
same 300 steps on every row, backbone frozen throughout — only the fusion stack
trains, so scale is the only variable that moves.

| backbone | trainable | peak VRAM | fused exact | control exact | gate |
|---|---|---|---|---|---|
| SmolLM2-135M-Instruct | 1.45% | 2.61 GB | **1.00** | 0.25 | **PASS** |
| SmolLM2-360M-Instruct | 0.73% | 4.32 GB | 0.00 | 0.00 | FAIL *(fp16 overflow)* |
| Qwen2.5-0.5B-Instruct | 0.51% | 6.13 GB | **1.00** | 0.00 | **PASS** |
| Qwen2.5-1.5B-Instruct | 0.27% | 10.95 GB | **1.00** | 0.00 | **PASS** |
| Qwen2.5-3B-Instruct (nf4) | 0.20% | 11.81 GB | **1.00** | 0.125 | **PASS** |

**The idea does not need scale.** A 135 M frozen instruct model reproduces all
eight answers verbatim from 16 fusion tokens, and fails to when those tokens
are shuffled across the batch. The one failure is arithmetic, not capacity:
SmolLM2-360M is the only backbone whose residual stream reaches the fp16
ceiling of 65504 (2 of 20 probed forward passes exceed it; the models either
side of it peak at 0.41x and 0.01x), so `GradScaler` skips every step and the
loss never leaves its starting value.

The binding constraint is VRAM, not compute — gradients reach the projectors
*through* the frozen decoder, so every layer's activations are kept for the
backward pass even though no weight in it updates.

## Status

- **Phase 0 — complete.** Alignment, normalization, windowing, the nuScenes
  adapter, and tests covering the five rules.
- **Phase 1 — complete.** Encoders, masked pooling, projectors, the Stage A
  trainer, and the overfit gate. Latest gate run (6 windows, byte-level
  backbone, CPU):

  | run | loss | exact match |
  |---|---|---|
  | fused (real sensor prefix) | 0.0002 | **100%** |
  | control (prefix shuffled across batch) | 0.0174 | 17% (chance) |

  The control is the point: the answers are ~97% shared boilerplate, so a
  loss margin proves nothing. Exact match requires the digits, and the
  digits only exist in the sensors.
- **Phase 2 — complete.** Stage B QLoRA (two learning rates, adapters only),
  the grounding metrics above, per-modality ablation, failure gallery.
- **Phase 3 — measured on a T4**, except the 2-GPU comparison. Quantization and
  dtype sweep, `torch.compile`, end-to-end latency, and ONNX export (verified to
  7.15e-07 across four cases including a fully-masked batch). The FSDP-vs-DDP
  run still has not produced a number: it now has two GPUs (Kaggle `GPU T4 x2`,
  account verified) and the distributed section emitted no output on that run,
  with a 0-byte kernel log. Local reproduction of the same command succeeds, so
  the cause is environment-specific — trail in `reports/phase3_status.md`.
- **Backbone ladder — complete.** Six frozen pretrained backbones from 135 M to
  3 B run through the Phase 1 gate on a T4; five pass, including the smallest
  real one. Latency, end-to-end timing and ONNX export reproduced on second
  hardware. Full study, method and negative results in
  `reports/phase4_backbone_ladder.md`.
- **Phase 4 — mostly complete.** FastAPI service (alignment server-side,
  liveness/readiness split, coverage in every response), Dockerfile, CI with a
  smoke train, drift monitor, and a Gradio demo with **live sensor ablation**
  (`space/app.py`, runs locally). Deploying it needs HF PRO — see
  `space/DEPLOY.md`. No demo recording yet.
- Full plan: `daystorm-plan.html`.

MIT licensed. Data: nuScenes (CC BY-NC-SA 4.0) is not redistributed here.
