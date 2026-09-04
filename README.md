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
uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
make test            # 43 tests, no download required
make demo            # print one aligned window from a synthetic scene
make gate            # Phase 1 go/no-go: can the fusion stack carry information?
```

## Requirement coverage

| Requirement | What proves it | Where |
|---|---|---|
| Fuse inputs from multiple modalities | Four streams into a shared token space; fixed 16-token budget per window | `model/fusion.py` |
| Pipelines for **temporal alignment** and normalization | 2 Hz causal event grid, per-channel staleness budgets, split-safe norm stats | `data/align.py`, `data/norm.py` |
| Adapt and fine-tune existing GenAI architectures | New encoders + projectors grafted onto a frozen backbone; two-stage recipe | `train/stage_a.py` |
| Hands-on foundation-model fine-tuning | QLoRA on the backbone in Stage B | `train/stage_b.py` *(phase 2)* |
| Vision-language and audio-text fusion | Both paths present, ablated per modality | `model/encoders.py` |
| Model evaluation and testing | 43 tests incl. property tests for causality and masking | `tests/` |
| Latency and production deployment | Quantization sweep, p50/p95, FastAPI service | *(phase 3-4)* |
| Distributed training *(nice to have)* | FSDP config + a real 2-GPU run | *(phase 3)* |
| MLOps automation and monitoring *(nice to have)* | MLflow, CI, embedding-drift monitor | *(phase 4)* |

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

## Status

- **Phase 0 — complete.** Alignment, normalization, windowing, the nuScenes
  adapter, and 32 tests covering the five rules.
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
- Phases 2–4 are specified in `daystorm-plan.html`.

MIT licensed. Data: nuScenes (CC BY-NC-SA 4.0) is not redistributed here.
