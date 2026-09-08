# Model card — Daystorm

## What it is

A four-modality driving-scene copilot. Camera, CAN bus, radar tracks and cabin
audio are resampled onto a 2 Hz causal event grid, pooled to 16 tokens, and
projected into a language backbone that answers grounded questions about a
two-second window.

## Intended use

A portfolio and research artifact demonstrating multimodal sensor fusion,
temporal alignment, and the training/evaluation/serving path around them.

## Out of scope

**Not for use in or near a vehicle**, in the loop or as a driver aid. It has
never been evaluated on real driving data, it has no safety case, and the
current checkpoint cannot perceive anything — see the limitations below.

## Limitations — read these before quoting any number

The metrics in `reports/` are real measurements of a real pipeline. They are
**not** evidence that the model perceives driving scenes. Four reasons, in
descending order of importance:

1. **Camera and audio embeddings carry no content.** The current checkpoint
   uses `HashCache`, a fixed pseudo-random function of `(scene, modality,
   index)`. It has the shape and determinism of a real embedding cache and
   none of the meaning. Worse, it *identifies the scene*, which a model can
   memorise — so the camera and audio ablations measure scene-identity
   leakage, not perception. Real numbers require `scripts/precompute_reference.py`
   against actual nuScenes frames and a re-run.

2. **Every metric below uses a byte-level stand-in, not Qwen2.5-VL.**
   `TinyBackbone` is ~2 M randomly-initialised parameters with no language
   prior, and it produced every ablation and latency number in this card.
   `HFBackbone` — the real path — has now been run on a Tesla T4 across a
   ladder of six frozen pretrained backbones from 135 M to 3 B, and the Phase 1
   gate passes on five of them (`reports/phase4_backbone_ladder.md`). That
   establishes the fusion channel carries sensor information through a real
   frozen language model. It does **not** re-measure the held-out metrics
   below, which remain stand-in numbers.

3. **The data is synthetic.** `daystorm.data.synthetic` generates plausible
   kinematics — urban speeds, realistic decelerations, following distances —
   but it is not driving. `daystorm.data.nuscenes_src` implements the real
   source and is untested against a real download.

4. **The audio modality has no real source anywhere.** No public driving
   corpus ships synchronised in-cabin audio. The audio path is a validated
   *fusion mechanism*, trained on synthesised signals. It is not evidence of
   audio understanding, and the ablation below should be read with that in
   mind.

## Measured results

Held-out split of 16 windows, scene-level partition, byte-level backbone.
Two independent runs (CPU-trained and T4-trained checkpoints) are reported
because they disagree in the details and agree on the conclusion.

CPU-trained checkpoint:

| run | exact match | manoeuvre acc | numeric MAE | hallucination |
|---|---|---|---|---|
| full | 0.00 | 1.00 | 0.225 | 0.000 |
| no camera | 0.00 | 1.00 | 0.206 (−0.02) | 0.000 |
| **no CAN** | 0.00 | 1.00 | **0.663 (+0.44)** | **0.208 (+0.21)** |
| no radar | 0.00 | 1.00 | 0.323 (+0.10) | 0.000 |
| no audio | 0.00 | 1.00 | 0.419 (+0.19) | 0.000 |

T4-trained checkpoint (same seeds, same split):

| run | exact match | manoeuvre acc | numeric MAE | hallucination |
|---|---|---|---|---|
| full | 0.00 | 1.00 | 0.260 | 0.021 |
| no camera | 0.00 | 1.00 | 0.373 (+0.11) | 0.000 |
| **no CAN** | 0.00 | 1.00 | **0.454 (+0.19)** | **0.208 (+0.19)** |
| no radar | 0.00 | 1.00 | 0.323 (+0.06) | 0.021 |
| no audio | 0.00 | 1.00 | 0.398 (+0.14) | 0.000 |

Reading these honestly:

- **CAN is doing the work, in both runs.** It is the only ablation that makes
  the model *fabricate*: hallucination 0%→21% (CPU) and 2%→21% (T4), with the
  largest numeric-error increase in both. That is the result the architecture
  predicts, and it reproduces across two independently trained checkpoints.
- **Camera's contribution is not stable** across runs (−0.02 on CPU, +0.11 on
  T4). Expected: its embeddings are meaningless by construction, so whatever
  the model extracts is scene identity, and how much it leans on that varies
  with initialisation. Neither number is evidence about vision.
- **Audio "helping" is a leak, not a capability.** With content-free
  embeddings, the only thing the audio channel can supply is scene identity.
  Do not read +0.19 as audio understanding.
- **Exact match is 0.00.** The model gets the phrasing and the manoeuvre right
  every time and the digits slightly wrong (10.1 vs 10.2 m/s). Honest for a
  2 M-parameter random-init backbone on held-out scenes.
- **Coverage honesty is 0.00.** The model never reports degraded sensors, even
  though references do. A real gap, not a rounding artifact.

## Latency

Single window, byte-level backbone, Tesla T4 (sm_75) and CPU:

| stage | CPU p50 | T4 p50 | share (T4) |
|---|---|---|---|
| align | 0.04 ms | 0.15 ms | 0.1% |
| fuse | 0.82 ms | 3.41 ms | 1.5% |
| decode | 233.73 ms | 228.93 ms | 98.5% |

Decode is the only sequential stage and dominates completely. The GPU improves
it by 2%, because at this model size decode is bound by per-token kernel launch
latency rather than compute — so the next optimisation is batched decode or a
KV cache with CUDA graphs, not more GPU.

Fusion-stack dtype sweep on the T4 (p50, batch 32): fp16 3.597 ms, fp32
4.868 ms, nf4 6.271 ms, bf16 **74.149 ms**. `torch.cuda.is_bf16_supported()`
returns `True` on Turing because it counts emulation; there are no bf16 tensor
cores, so bf16 runs 20.6x slower than fp16 instead of failing loudly. Full
table in `reports/phase3_status.md`. Reproduced independently on a second T4
and a different torch build at 19.4x — `reports/phase4_backbone_ladder.md`.

## Frozen-backbone gate

Phase 1's go/no-go gate, run on a T4 with the backbone frozen and only the
fusion stack trainable. PASS = all eight windows reproduced verbatim from the
fusion prefix, and not reproduced when that prefix is shuffled across the
batch.

| backbone | trainable | peak VRAM | fused exact | control exact | gate |
|---|---|---|---|---|---|
| SmolLM2-135M-Instruct | 1.45% | 2.61 GB | 1.00 | 0.25 | **PASS** |
| SmolLM2-360M-Instruct | 0.73% | 4.32 GB | 0.00 | 0.00 | FAIL *(fp16 overflow)* |
| Qwen2.5-0.5B-Instruct | 0.51% | 6.13 GB | 1.00 | 0.00 | **PASS** |
| Qwen2.5-1.5B-Instruct | 0.27% | 10.95 GB | 1.00 | 0.00 | **PASS** |
| Qwen2.5-3B-Instruct (nf4) | 0.20% | 11.81 GB | 1.00 | 0.125 | **PASS** |

This is a deliberate 8-window overfit and says nothing about generalisation.
It is the necessary condition — if the fusion tokens cannot carry information
at all, nothing downstream matters — and it is now met through a real frozen
LM at 135 M parameters. The 360M failure is a numerics artifact, not a
capacity limit; see the report.

## Training data

Synthetic scenes from `daystorm.data.synthetic` (seeded, reproducible).
The real path targets nuScenes v1.0-mini plus the CAN bus expansion
(CC BY-NC-SA 4.0, non-commercial), which is not redistributed in this repo.

## Ethical and safety notes

Driving models fail in ways that hurt people. The two design choices here that
exist for that reason: missing sensor data is an explicit input rather than a
silent zero, so a degraded window is distinguishable from a normal one; and
the hallucination metric is reported alongside accuracy, because a confident
wrong time-to-collision is more dangerous than an abstention.
