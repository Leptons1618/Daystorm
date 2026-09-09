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
   against actual nuScenes frames and a re-run — which in turn requires the CAN
   bus expansion, because the labels are derived from CAN (see limitation 3).

2. **Most metrics below use a byte-level stand-in, not Qwen2.5-VL.**
   `TinyBackbone` is ~2 M randomly-initialised parameters with no language
   prior, and it produced every latency number in this card and two of the
   three ablation tables. `HFBackbone` — the real path — has been run on a
   Tesla T4 across a ladder of five frozen pretrained backbones from 135 M to
   3 B (`docs/findings.md`); the Phase 1 gate passes on all of
   them, and the third ablation table below is a frozen Qwen2.5-0.5B. What is
   still missing at the real scale is Qwen2.5-VL itself and anything larger
   than 0.5 B on the held-out split.

   Related: **every held-out number here is Stage A only.** The eval harness
   scores a Stage A checkpoint, so LoRA is not in any of these tables — Phase
   2's central claim is trained and checkpointed but not yet measured on
   held-out data.

3. **The data is synthetic.** `daystorm.data.synthetic` generates plausible
   kinematics — urban speeds, realistic decelerations, following distances —
   but it is not driving. `daystorm.data.nuscenes_src` has now been run against
   a real nuScenes v1.0-mini mount: the devkit installs, the metadata parses,
   `list_scenes()` returns all ten scenes and camera and radar load. What that
   mirror does **not** carry is the CAN bus expansion, a separate download most
   redistributions omit. A scene loads today with the CAN stream empty and
   masked invalid, and that is more than a missing ablation: `describe_window`
   derives the question, answer and manoeuvre tag from CAN and radar, so without
   CAN every real window is discarded before training sees it. The CAN expansion
   is the prerequisite for real data of any kind here. No metric in this card
   comes from real data yet.

4. **The audio modality has no real source anywhere.** No public driving
   corpus ships synchronised in-cabin audio. The audio path is a validated
   *fusion mechanism*, trained on synthesised signals. It is not evidence of
   audio understanding, and the ablation below should be read with that in
   mind.

## Measured results

Held-out split of 16 windows, scene-level partition. Three independent runs are
reported — two byte-level checkpoints (CPU- and T4-trained) and one frozen
Qwen2.5-0.5B — because they disagree in the details and agree on the
conclusion.

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

Qwen2.5-0.5B-Instruct, frozen, Stage A only (same harness, same split):

| run | exact match | manoeuvre acc | numeric MAE | hallucination |
|---|---|---|---|---|
| full | 0.00 | 1.00 | 0.290 | 0.042 |
| no camera | 0.06 | 1.00 | 0.471 (+0.18) | 0.021 |
| **no CAN** | 0.00 | 1.00 | **1.079 (+0.79)** | **0.250 (+0.21)** |
| no radar | 0.00 | 1.00 | 3.460 (+3.17) | 0.091 |
| no audio | 0.00 | 1.00 | 3.826 (+3.54) | 0.140 |

Reading these honestly:

- **CAN is doing the work, in all three runs.** It is the ablation that makes
  the model *fabricate*: hallucination 0%→21% (CPU), 2%→21% (T4) and 4%→25% on
  a frozen 0.5 B pretrained backbone. Three checkpoints, two of which share no
  weights, no vocabulary and no architecture, agree.
- **A real backbone is not more accurate here** (MAE 0.290 against 0.260) and
  degrades much harder when a sensor drops (+3.2 radar, +3.5 audio). With
  content-free camera and audio embeddings, the bigger model appears to lean on
  scene identity harder — which is an argument for real embeddings, not a
  finding about perception. n=16 and numeric MAE is unbounded.
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
latency rather than compute. Adding a KV cache to the byte-level decode — an
O(n³) → O(n²) change — bought only 1.18x (219.2 → 185.7 ms on the same 102
tokens), which is the same finding from the other direction: the next
optimisation is CUDA graphs or batched decode — attacking launches, not FLOPs —
and not more GPU.

Fusion-stack dtype sweep on the T4 (p50, batch 32): fp16 3.597 ms, fp32
4.868 ms, nf4 6.271 ms, bf16 **74.149 ms**. `torch.cuda.is_bf16_supported()`
returns `True` on Turing because it counts emulation; there are no bf16 tensor
cores, so bf16 runs 20.6x slower than fp16 instead of failing loudly. Full
table in `docs/benchmarks.md`. Reproduced independently on a second T4
and a different torch build at 19.2x — `docs/findings.md`.

## Frozen-backbone gate

Phase 1's go/no-go gate, run on a T4 with the backbone frozen and only the
fusion stack trainable. PASS = all eight windows reproduced verbatim from the
fusion prefix, and not reproduced when that prefix is shuffled across the
batch.

| backbone | trainable | peak VRAM | fused exact | control exact | gate |
|---|---|---|---|---|---|
| SmolLM2-135M-Instruct | 1.45% | 2.61 GB | 1.00 | 0.125 | **PASS** |
| SmolLM2-360M-Instruct *(fp16)* | 0.73% | 4.32 GB | 0.00 | 0.25 | FAIL *(fp16 overflow)* |
| SmolLM2-360M-Instruct *(fp32)* | 0.73% | 6.07 GB | 1.00 | 0.125 | **PASS** |
| Qwen2.5-0.5B-Instruct | 0.51% | 6.13 GB | 1.00 | 0.00 | **PASS** |
| Qwen2.5-1.5B-Instruct | 0.27% | 10.95 GB | 1.00 | 0.00 | **PASS** |
| Qwen2.5-3B-Instruct (nf4) | 0.20% | 11.81 GB | 1.00 | 0.125 | **PASS** |
| facebook/opt-125m | 1.55% | 1.80 GB | 1.00 | 0.125 | **PASS** |
| Qwen2.5-0.5B-Instruct *(nf4)* | 0.51% | 5.58 GB | 1.00 | 0.125 | **PASS** |

This is a deliberate 8-window overfit and says nothing about generalisation.
It is the necessary condition — if the fusion tokens cannot carry information
at all, nothing downstream matters — and it is now met through a real frozen
LM at 125 M parameters, across three model families, including one that was
never instruction-tuned. The 360M failure is a numerics artifact, not a
capacity limit: the same model in fp32 passes at 1.00 exact. See the report.

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
