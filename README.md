# Daystorm

Four asynchronous sensor streams — camera, CAN bus, radar tracks, cabin audio —
resampled onto a 2 Hz causal event grid, pooled to **16 tokens**, and projected
into a **frozen** pretrained language model that answers grounded questions
about a two-second driving window.

> **Q:** Why did the vehicle decelerate?
> **A:** Deceleration from 10.1 to 8.8 m/s over 2.0 s (-0.6 m/s²) with brake
> demand peaking at 0.80. The lead vehicle closed from 29 m to 16 m, minimum
> time-to-collision 2.4 s.

Nothing is trained from scratch. The backbone is a released causal LM whose
weights never move in Stage A; the new surface is ~6 M parameters of encoders,
masked pooling and projectors. Stage B adds QLoRA adapters to the backbone.

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[dev,serve]"
make test     # 71 tests, no download, no GPU
make gate     # the go/no-go experiment below
make eval     # held-out metrics + per-modality ablation + failure gallery
```

Research report: [`docs/findings.md`](docs/findings.md) · limits and intended
use: [`docs/model_card.md`](docs/model_card.md) · GPU measurements:
[`docs/benchmarks.md`](docs/benchmarks.md) · raw evidence: [`reports/`](reports/).

---

## The problem

A driving stack produces streams that share nothing but a clock: CAN at 100 Hz,
radar at 13 Hz, camera at 12 Hz, audio at 50 Hz, each with dropouts and its own
latency. A language model wants one ordered token sequence. Between those two
facts sit three problems that are easy to get quietly wrong:

1. **Temporal alignment.** Resampling onto a common grid is trivial to write and
   trivial to write *acausally*. Any interpolation that touches a future sample
   leaks the answer into the input, and the resulting metrics look excellent.
   The leak is invisible in aggregate loss.
2. **Representation.** Nine channels of telemetry at 100 Hz and a 1152-dim SigLIP
   embedding have to arrive in the same token space, on a fixed budget, with
   *missing* data represented as something other than zeros — because a dead
   sensor and a sensor reading zero are different situations and only one of them
   should change the answer.
3. **Grounding.** Answers about driving are ~97% boilerplate. A model that
   memorises phrasing scores a low loss while reading none of the sensors. Any
   claim that "the fusion works" needs an experiment that can distinguish those
   two states, not a loss curve.

And all of it has to run on hardware that costs nothing.

## Goal

Establish, with a falsifiable experiment at each step, that:

- sensor information survives compression to 16 tokens and is *readable by a
  frozen decoder that was never trained on it*;
- each modality's contribution is measurable, not asserted;
- the smallest backbone this works on is small enough to be useful;
- the whole path — align, train, evaluate, quantise, serve — reproduces on free
  hardware from a clean checkout.

Not goals: state-of-the-art driving VQA, a driver aid, or anything with a safety
case. See [`docs/model_card.md`](docs/model_card.md).

## Constraints

| Constraint | Consequence |
|---|---|
| Free-tier GPU is a **Tesla T4** (Turing, 16 GB) | No bfloat16 tensor cores, no FlashAttention-2. An A100 recipe runs here without erroring, 20x slower |
| Free Colab gives **one** GPU | Distributed work has to move to Kaggle (2x T4, 30 h/week) |
| Sessions get preempted | Every long run checkpoints and resumes, or it is not a run |
| Local development machine is CPU-only and weak | The gate must pass on a laptop; all training and inference goes to Kaggle |
| nuScenes is **CC BY-NC-SA 4.0** | No data redistributed here. The repo must be useful with no download at all |
| The nuScenes **CAN bus expansion** is a separate download that no public mirror carries | The real-data path is blocked on it — see Finding 8 |
| No public driving corpus ships synchronised in-cabin audio | The audio path is a validated mechanism trained on synthesised signals, not evidence of audio understanding |

## How the constraints were met

| Constraint | What was done | What it cost |
|---|---|---|
| No GPU for development | `TinyBackbone`: a 2 M byte-level decoder that trains in seconds on a CPU, behind the same interface as the real backbone, so `stage_a.py` never branches | Every early number describes plumbing, not capability — which is why the ladder below exists |
| No data | `daystorm.data.synthetic` generates seeded kinematics; `HashCache` stands in for frozen vision/audio encoders | The stand-in *identifies the scene*, so camera and audio ablations measure identity leakage, not perception |
| 16 GB of VRAM | Frozen encoders run once to disk (`scripts/precompute_reference.py`); NF4 for the 3 B row; fp16 compute and `attn_implementation="sdpa"` throughout | Gradients still reach the projectors *through* the frozen decoder, so activations dominate VRAM regardless |
| Turing has no bf16 | Measured the dtype sweep instead of trusting `torch.cuda.is_bf16_supported()`, which returns `True` on a T4 | One benchmark run; it saved every subsequent one |
| One GPU on Colab | A self-contained Kaggle kernel (`kaggle/`) with a section menu — `--only smoke,dist` runs 5 minutes instead of 70 — and `machine_shape` pinned in `kernel-metadata.json` so the accelerator is not a manual browser step | A push script and a temp-copy trick so `--only` cannot clobber the working tree |
| Preemption | `--ckpt-every`, resume-by-default, nothing kept that only exists on the VM | FSDP made this harder before it made it work: `state_dict()` is a collective, and calling it inside a rank-0 guard deadlocks |
| Free-tier honesty | Every claim in this README traces to a JSON or a log under `reports/` | Several claims got weaker when measured. They are still here, with the measurement |

## Strengths

- **The gate is a real experiment, not a metric.** It trains twice: once with
  the fusion prefix, once with the same prefixes shuffled across the batch, so
  window *i* gets window *j*'s sensors. PASS requires fused exact ≥ 0.90 **and**
  control exact ≤ 0.50. A model reciting boilerplate fails it by construction.
- **Causality is enforced and tested, not assumed.** Causal hold with per-channel
  staleness budgets, and `ChannelNorm` rather than `GroupNorm` in the telemetry
  encoder — GroupNorm normalises over channels *and* time, letting the last grid
  point move the statistics of every earlier one. `test_telemetry_encoder_is_causal`
  fails if that regresses.
- **Missing data is a first-class input.** A validity bit reaches every encoder;
  `test_dead_sensor_output_does_not_depend_on_its_buffer_contents` pins it.
  Every HTTP response reports the coverage that produced it.
- **Checkpoints are self-describing.** The backbone id, quantisation and dtype
  travel with the weights, so evaluation and serving rebuild the model exactly
  as training built it. Loading an fp16-trained projector against a 4-bit
  backbone would otherwise produce numbers instead of an error.
- **Alignment lives server-side.** The service takes raw asynchronous streams,
  not a pre-aligned tensor — a client cannot silently use a different staleness
  budget and move the model off-distribution.
- **Everything reproduces twice.** The full backbone ladder was run again on
  fresh hardware: all seven verdicts hold, `fused_exact` identical, peak VRAM
  identical to 0.01 GB.

## Shortcomings

Read these before quoting any number below.

- **No real data yet.** Every metric is synthetic. `nuscenes_src.py` loads real
  scenes off a real v1.0-mini mount — camera, radar and metadata all parse — but
  the CAN expansion is missing from every public mirror, and `describe_window`
  derives the question, answer and manoeuvre tag *from* CAN. Without it every
  real window is discarded before a camera embedding is used.
- **Camera and audio embeddings carry no content.** `HashCache` is a fixed
  pseudo-random function of `(scene, modality, index)`. It has the shape and
  determinism of a real cache and none of the meaning.
- **The gate is a deliberate 8-window overfit.** It is a necessary condition —
  if 16 tokens cannot carry the answer at all, nothing downstream matters — and
  it says nothing about generalisation.
- **Every held-out number is Stage A only.** The eval harness scores a Stage A
  checkpoint; LoRA adapters are trained and checkpointed but not yet measured on
  held-out data.
- **Exact match on held-out windows is 0.00.** The model gets phrasing and
  manoeuvre right and the digits slightly wrong (10.1 vs 10.2 m/s).
- **Coverage honesty is 0.00.** The model never volunteers that a sensor was
  degraded, even when the reference answer does. A real gap.
- **n = 16 on the held-out split, n = 8 on the gate.** Control exact is visibly
  noisy at that size; one ladder row moved 0.000 → 0.375 between identical runs.
- **The 2-GPU study runs on a 2 M-parameter model**, which is the regime where
  sharding is least interesting. It says nothing about FSDP where it matters.
- **No published weights.** See [Using a trained model](#using-a-trained-model);
  the recipe is the artifact, and a checkpoint trained against a stand-in
  embedding cache would not be worth downloading.

## Findings

Every table here comes from a file under `reports/`. Method and full trail in
[`docs/findings.md`](docs/findings.md).

### 1. A frozen 125 M decoder reads the fusion prefix

Same gate, same 8 windows, same 300 steps, backbone frozen — only the fusion
stack trains, so scale is the only variable that moves.

| backbone | trainable | peak VRAM | fused exact | control exact | gate |
|---|---|---|---|---|---|
| `facebook/opt-125m` | 1.55% | 1.80 GB | **1.00** | 0.125 | **PASS** |
| SmolLM2-135M-Instruct | 1.45% | 2.61 GB | **1.00** | 0.125 | **PASS** |
| SmolLM2-360M-Instruct *(fp16)* | 0.73% | 4.32 GB | 0.00 | 0.25 | FAIL *(fp16 overflow)* |
| SmolLM2-360M-Instruct *(fp32)* | 0.73% | 6.07 GB | **1.00** | 0.125 | **PASS** |
| Qwen2.5-0.5B-Instruct | 0.51% | 6.13 GB | **1.00** | 0.00 | **PASS** |
| Qwen2.5-0.5B-Instruct *(nf4)* | 0.51% | 5.58 GB | **1.00** | 0.125 | **PASS** |
| Qwen2.5-1.5B-Instruct | 0.27% | 10.95 GB | **1.00** | 0.00 | **PASS** |
| Qwen2.5-3B-Instruct *(nf4)* | 0.20% | 11.81 GB | **1.00** | 0.125 | **PASS** |

Three model families, one of which (`opt-125m`) was never instruction-tuned.

### 2. The single failure is arithmetic, not capacity

SmolLM2-360M is the only backbone whose residual stream reaches the fp16 ceiling
of 65504 — 2 of 20 probed forward passes exceed it, while the models either side
peak at 0.41x and 0.01x of it. `GradScaler` then skips every step and the loss
goes NaN. The same model, same data, same 300 steps in fp32 passes at 1.00
exact, for 1.4x the VRAM and 1.7x the wall clock. Stage A now prints a named
diagnosis after 25 consecutive skipped updates instead of letting a run die flat.

### 3. `is_bf16_supported()` returns True on a T4, and bf16 is 20x slower

Fusion stack, p50 at batch 32:

| config | eager | `torch.compile` | vs fp16 |
|---|---|---|---|
| fp32 | 4.868 ms | 3.370 ms | 1.35x slower |
| **fp16** | **3.597 ms** | **2.557 ms** | — |
| bf16 *(emulated)* | 74.149 ms | 72.829 ms | **20.6x slower** |
| nf4 | 6.271 ms | 7.126 ms | 1.74x slower |

PyTorch counts emulation as support. Reproduced on second hardware and a
different torch build at 19.2x.

### 4. Decode is launch-bound, and a KV cache proves it

| stage | CPU p50 | T4 p50 | share (T4) |
|---|---|---|---|
| align | 0.04 ms | 0.15 ms | 0.1% |
| fuse | 0.82 ms | 3.41 ms | 1.5% |
| decode | 233.73 ms | 228.93 ms | 98.5% |

The GPU improves decode by 2%. Adding a KV cache — an asymptotic O(n³) → O(n²)
change — moved 102 tokens from 219.2 ms to 185.7 ms, **1.18x**, against a ~7%
run-to-run noise floor. Removing arithmetic from a launch-bound loop returns
almost nothing.

### 5. Sharding costs 1.5%; the speedup was fp16

Stage B, 300 steps, batch 16 per rank, Kaggle 2x T4:

| config | ms/step | windows/s | vs 1 GPU |
|---|---|---|---|
| 1 GPU | 63 | 254 | 1.00x |
| DDP (fp32) | 67 | 478 | **1.88x** |
| FSDP (fp32) | 68 | 471 | **1.85x** |
| FSDP (fp16) | 48 | 667 | **2.63x** |

The first version of this table showed FSDP at 2.42x, and superlinear scaling on
two GPUs is a bug report, not a result: `dist.py` was applying mixed precision to
the FSDP arm only. All three runs finish within 0.0006 loss of each other.

### 6. CAN is the modality whose absence makes the model fabricate

Held-out split, 16 windows, scene-level partition, Qwen2.5-0.5B frozen, Stage A
only:

| run | exact | manoeuvre acc | numeric MAE | hallucination |
|---|---|---|---|---|
| full | 0.00 | 1.00 | 0.290 | 0.042 |
| no camera | 0.06 | 1.00 | 0.471 (+0.18) | 0.021 |
| **no CAN** | 0.00 | 1.00 | **1.079 (+0.79)** | **0.250 (+0.21)** |
| no radar | 0.00 | 1.00 | 3.460 (+3.17) | 0.091 |
| no audio | 0.00 | 1.00 | 3.826 (+3.54) | 0.140 |

Hallucination on CAN removal: 0% → 21% (byte-level, CPU), 2% → 21% (byte-level,
T4), 4% → 25% (Qwen2.5-0.5B). Three checkpoints, two of which share no weights,
no vocabulary and no architecture.

### 7. NF4 is not worth it below 3 B

Same model, same gate, only the quantisation changes:

| Qwen2.5-0.5B | fused loss | exact | peak VRAM | wall |
|---|---|---|---|---|
| fp16 | 0.00092 | 1.00 | 6.13 GB | 218 s |
| nf4 | 0.00083 | 1.00 | **5.58 GB (−9%)** | **471 s (2.2x)** |

### 8. What actually blocks real data

The public nuScenes-mini mirror carries `v1.0-mini`, `samples`, `sweeps` and
`maps` — and no `can_bus/`. No public mirror found does. Since `describe_window`
derives the supervision itself from CAN and radar and returns nothing when
either is invalid, `build_samples` drops every real window before a camera
embedding is ever read.

## What the findings imply

- **The architecture's claim does not need scale, and does not need instruction
  tuning.** What the gate exercises is a pretrained causal decoder reading a
  learned prefix. 125 M is enough; a model that never saw instruction data is
  enough. That moves the interesting question away from "which backbone" and
  onto the sensor path, where the remaining errors are.
- **The binding resource is activations, not weights.** Gradients reach the
  projectors *through* the frozen decoder, so every layer's activations are
  retained even though no weight in it updates. That single fact explains the
  VRAM ladder, why NF4 saves only 9%, and why quantising below 3 B is a pure
  wall-clock loss.
- **The next latency win is not hardware and not FLOPs.** Two independent
  measurements — a GPU that buys 2% and a KV cache that buys 1.18x — say decode
  is bound by per-token kernel launches. CUDA graphs or batched decode; a bigger
  card would change nothing.
- **Sharding is a memory tool, and this model has no memory problem.** FSDP costs
  1.5% against DDP at 2 M parameters and buys headroom nothing here needs. It
  earns its place only once the backbone itself is trained.
- **A frozen pretrained backbone is not automatically *better*.** Qwen2.5-0.5B
  is slightly worse on MAE than the 2 M stand-in (0.290 vs 0.260) and degrades
  far harder when a sensor drops. With content-free camera and audio caches, the
  larger model leans on scene identity harder — an argument for real embeddings,
  not a finding about perception.
- **The grounding result is the durable one.** Removing CAN triples numeric error
  and multiplies fabrication by 5-10x across three checkpoints that share nothing
  architecturally. That is the architecture behaving as designed, and it is the
  result most likely to survive contact with real data.
- **CAN is not merely the strongest ablation, it is the label source.** Which
  reorders the roadmap: real vision embeddings are worth nothing until the CAN
  expansion exists, so that download is step one and everything else is step two.

## Using a trained model

**There are no published weights, deliberately.** A checkpoint trained against
`HashCache` embeddings has learned a scene-identity hash, not perception —
downloading it would be worse than training one, which takes seconds. The
artifact here is the recipe; these commands produce a checkpoint you can then
serve, quantise, export or ablate.

**Train one** (CPU is fine for the byte-level backbone; the real ones want a T4):

```bash
# byte-level stand-in, a couple of minutes on a laptop CPU
python -m daystorm.train.stage_a --scenes 40 --steps 600 --out ckpt/stage_a

# a real frozen backbone, a few minutes on a T4
python -m daystorm.train.stage_a --backbone Qwen/Qwen2.5-0.5B-Instruct \
    --scenes 40 --steps 600 --out ckpt/qwen05b

# QLoRA on top of either: adapters and fusion at two learning rates
python -m daystorm.train.stage_b --stage-a ckpt/qwen05b --steps 600 --out ckpt/qwen05b_b
```

The checkpoint records the backbone id, quantisation and dtype, so every command
below rebuilds the model the way training built it — `--backbone` only overrides.

**Score it** on held-out scenes, with a per-modality ablation and a failure
gallery:

```bash
python -m daystorm.eval.harness --ckpt ckpt/qwen05b --max-eval 16 --out reports/local
```

**Serve it.** The service takes raw asynchronous streams and aligns them itself:

```bash
DAYSTORM_CKPT=ckpt/stage_a uvicorn daystorm.serve.app:app --port 8000

curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "t0": 5.0, "duration_s": 2.0,
  "streams": {"can": {"t": [4.99, 5.01],
                      "v": [[0,0,0.1,0.0,10.1, 0.0,0,0,10.1],
                            [0,0,0.0,0.8, 8.8,-0.6,0,0, 8.8]]}}
}'
```

`t` is seconds on the reference clock and `v` is `(N, channels)` — 9 for CAN,
6 for radar, 1 for the cached camera and audio references. Modalities you omit
come back as coverage 0 in the response rather than as an error, which is the
point of the design. `/health` is liveness, `/ready` is readiness (503 until
weights load), `/metrics` is Prometheus text including per-modality coverage and
drift alerts.

**Container** (the image bakes in `ckpt/`, so train first):

```bash
docker build -t daystorm . && docker run -p 8000:8000 daystorm
```

**Export the fusion stack to ONNX** — verified to 7.15e-07 across four cases
including a fully-masked batch:

```bash
python -m daystorm.bench.export_onnx --out reports/local/fusion.onnx
```

**Load it in Python.** `src/daystorm/serve/app.py` (`_load` and `_run`) is the
shortest complete example — checkpoint to answer in about twenty lines:

```python
import torch
from daystorm.data.norm import NormStats
from daystorm.data.tensors import FEATURE_DIMS
from daystorm.model.fusion import DaystormFusion
from daystorm.train.backbone import backbone_kwargs, build_backbone

blob = torch.load("ckpt/stage_a/fusion.pt", map_location="cpu", weights_only=False)
name = blob["args"]["backbone"]
backbone = build_backbone(name, **backbone_kwargs(blob["args"], name))
if blob.get("backbone") is not None:   # the tiny backbone trains; the real ones stay frozen
    backbone.load_state_dict(blob["backbone"])
fusion = DaystormFusion(FEATURE_DIMS, d_model=backbone.d_model)
fusion.load_state_dict(blob["fusion"])
stats = NormStats.load("ckpt/stage_a/norm.json")
```

GPU runs — Colab, and the 2x T4 Kaggle kernel with its section menu — are in
[`docs/running.md`](docs/running.md).

## Layout

```
src/daystorm/
  data/align.py       the event grid, causal hold, staleness masks
      norm.py         split-safe normalisation statistics
      windows.py      windows -> grounded question/answer supervision
      tensors.py      embedding cache + batching
      synthetic.py    seeded scene generator, so the repo runs with no download
      nuscenes_src.py the real source: nuScenes + CAN bus expansion
  model/encoders.py   causal TCN (CAN), track MLP (radar), cached-embedding adapters
        fusion.py     masked attention pooling -> 16 tokens -> backbone width
  train/backbone.py   byte-level stand-in and the real HF backbone, one interface
        stage_a.py    projector alignment + the go/no-go gate
        stage_b.py    QLoRA, two learning rates, FSDP/DDP
        dist.py       the distributed wrapper
  eval/harness.py     held-out metrics, ablation, failure gallery
  bench/              latency, end-to-end, ONNX export
  serve/app.py        FastAPI: aligns server-side, reports coverage
  monitor/drift.py    coverage and stuck-signal checks
docs/                 findings, benchmarks, model card, runbook
reports/              raw evidence for every number quoted (see reports/README.md)
kaggle/               the 2x T4 kernel and its metadata
scripts/              Colab/Kaggle wrappers, nuScenes download, encoder precompute
space/                Gradio demo with live sensor ablation
```

## Contributing

Issues and PRs welcome. What is genuinely useful here, roughly in order:

1. **A legal route to the nuScenes CAN bus expansion**, or a label derivation
   that uses ego pose instead — the main archive does ship pose, and either
   unblocks the entire real-data path.
2. **Real frozen-encoder embeddings** via `scripts/precompute_reference.py`, to
   replace `HashCache` and make the camera and audio ablations mean something.
3. **Backbones below 125 M**, or families not yet covered. One row is about a
   minute on a T4:
   `python -m daystorm.train.stage_a --overfit 8 --steps 300 --backbone <id>`.
4. **Anything that fails a test that should have caught it.** A PR that adds the
   failing test first is the most welcome kind.

Ground rules, none of them unusual:

- `make test` and `make lint` pass. CI additionally runs a Stage A → Stage B
  smoke train, because a repo whose training script broke three commits ago is
  the standard failure mode of an ML project and unit tests never notice.
- **A number in a document must be reproducible from a file in `reports/`.**
  New claims arrive with the run that produced them.
- Negative results stay. `docs/findings.md` keeps a section of them on purpose.
- Measurements name their hardware. "Fast" is not a unit.

## Future scope

Ordered by what unblocks the most, not by what is most interesting:

1. **The CAN bus expansion** — downloaded from nuscenes.org and republished
   privately, or replaced by supervision derived from ego pose. Everything real
   is downstream of this.
2. **Real vision embeddings**, then a `--source nuscenes` path through Stage A,
   then the first held-out metrics that mean anything.
3. **Stage B adapters in the eval harness**, so Phase 2's central claim gets a
   held-out number instead of a training curve.
4. **A held-out gate.** The current one is an 8-window overfit by design; a
   scene-level version would test generalisation of the fusion channel itself.
5. **CUDA graphs or batched decode**, the only latency work the measurements
   actually support.
6. **Coverage honesty as a training objective.** The model never reports a
   degraded sensor; the metric exists and sits at 0.00.
7. **A vision-language backbone** (Qwen2.5-VL), once camera embeddings are real —
   grafting onto a model that already has a vision tower is the natural next
   rung of the ladder.

## License

MIT. nuScenes (CC BY-NC-SA 4.0, non-commercial) is **not** redistributed here;
`scripts/download_nuscenes.sh` fetches it under its own terms.
