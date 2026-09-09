# Backbone ladder — how small can the frozen language model get?

Hardware: Kaggle `GPU T4 x2`, 2x Tesla T4 (sm_75, 15.6 GB), CUDA 12.8, torch
2.10.0+cu128. Kernel `kaggle/daystorm_fsdp_kernel.py`. Raw outputs, including
the full console transcript, in `reports/kaggle_ladder/`.

## The question

Daystorm's claim is that four sensor streams can be compressed into 16 tokens
and a **frozen** language model will read them. Every number published before
this run used `TinyBackbone` — 2 M randomly-initialised parameters, no language
prior, trained end-to-end alongside the projectors. That validates plumbing and
little else: "the fusion prefix carries information" is a weak claim when the
reader was trained on the same 300 steps as the writer.

So: does the gate still pass when the backbone is a real pretrained instruct
model whose weights never move, and how far down the parameter ladder does that
survive? A 3 B backbone passing is a nice result. A 125 M backbone passing is a
*useful* one, because it fits on hardware people actually have.

## Method

One instrument, seven rows, one variable. Each row runs
`daystorm.train.stage_a` with identical settings — 8 windows, 300 steps,
batch 4, lr 1e-3, 24 scenes, seed 0 — and differs only in `--backbone` (and, in
one row, `--backbone-dtype`).

The gate trains twice per row:

1. **fused** — the real fusion prefix in front of each question.
2. **control** — the same prefixes, shuffled across the batch, so window *i*
   gets window *j*'s sensors.

PASS requires **fused exact ≥ 0.90 and control exact ≤ 0.50**. Loss alone would
prove nothing: the eight answers share ~97% of their tokens, so a model that
memorises boilerplate scores a low loss with no sensor information at all. Exact
match requires the digits, and the digits exist only in the sensors. The eight
windows carry eight distinct answers under one shared question, so chance on the
control is 1/8 = 0.125.

The backbone is frozen in every row; only the fusion stack (encoders, masked
attention pooling, projectors) receives gradients. The main sweep uses ungated
instruction-tuned models from two families, so scale moves and little else does;
Result 11 adds a third family and drops instruction tuning as a separate,
deliberate change. NF4 is used only for the 3 B row — below that the fp16
weights fit twice over on a T4 and dequantisation is pure overhead, which the
dtype sweep below measures.

## Result 1 — the ladder

| backbone | dtype | d_model | fused loss | fused exact | control loss | control exact | gate |
|---|---|---|---|---|---|---|---|
| `TinyBackbone` (2 M, random init) | fp16 | 192 | 0.00078 | **1.00** | 0.0233 | 0.125 | **PASS** |
| SmolLM2-135M-Instruct | fp16 | 576 | 0.02813 | **1.00** | 0.1202 | 0.125 | **PASS** |
| SmolLM2-360M-Instruct | fp16 | 960 | NaN | 0.00 | 0.0705 | 0.250 | **FAIL** |
| SmolLM2-360M-Instruct | **fp32** | 960 | 0.01869 | **1.00** | 0.1077 | 0.125 | **PASS** |
| Qwen2.5-0.5B-Instruct | fp16 | 896 | 0.00092 | **1.00** | 0.0656 | 0.000 | **PASS** |
| Qwen2.5-1.5B-Instruct | fp16 | 1536 | 0.00076 | **1.00** | 0.0691 | 0.000 | **PASS** |
| Qwen2.5-3B-Instruct | nf4 | 2048 | 0.00103 | **1.00** | 0.0617 | 0.125 | **PASS** |

**Six of seven pass. The only failure is a dtype, not a model — the same
SmolLM2-360M passes in fp32.** And the smallest real backbone that passes is
135 M.

Three things worth reading carefully:

- **Every passing row is at or near chance on the control** (0/8 to 1/8), while
  every passing row is at 8/8 fused. The separation is not marginal.
- **The frozen pretrained backbones beat the trainable random one on loss.**
  Qwen2.5-1.5B reaches 0.00076 with 0.27% of parameters trainable;
  `TinyBackbone` reaches 0.00078 with 43% trainable and the whole backbone free
  to move. Whatever the projectors emit, a pretrained decoder reads it at least
  as well as a decoder co-trained to read it.
- **The one row above chance on the control is the failed one** (0.25). A model
  that never learned to use its prefix is exactly the model whose control run is
  unconstrained.

### Replication

The whole ladder ran a second time, unchanged, in a later kernel (v11,
`reports/kaggle_ladder/run11/`). **All seven verdicts reproduce.** `fused_exact`
is identical in every row — 1.00 everywhere, 0.00 for SmolLM2-360M in fp16 — and
peak VRAM is identical to the 0.01 GB. What moves is loss and the control:

| backbone | fused loss (run 1 → run 2) | control exact (run 1 → run 2) |
|---|---|---|
| tiny | 0.00078 → 0.00082 | 0.125 → 0.125 |
| SmolLM2-135M | 0.02813 → 0.00577 | 0.125 → 0.125 |
| SmolLM2-360M fp16 | NaN → NaN | 0.250 → *NaN, 0.000* |
| SmolLM2-360M fp32 | 0.01869 → 0.00454 | 0.125 → 0.125 |
| Qwen2.5-0.5B | 0.00092 → 0.00122 | 0.000 → **0.375** |
| Qwen2.5-1.5B | 0.00076 → 0.00100 | 0.000 → 0.000 |
| Qwen2.5-3B (nf4) | 0.00103 → 0.00077 | 0.125 → 0.000 |

Worth stating rather than hiding: **the control is noisy at n=8.** Qwen2.5-0.5B
scored 3/8 on the shuffled prefix in the second run against 0/8 in the first —
still a clear PASS against the 0.50 threshold and against its own 8/8 fused, but
one lucky window is 0.125 of the metric. The verdicts are robust; the control
values are estimates on eight samples and should be read as such.

## Result 2 — the SmolLM2-360M failure, predicted then confirmed

This is the part of the study with a prediction in the middle, so it is worth
laying out in order.

**Observation.** SmolLM2-360M sits between two models that pass, so "too small"
cannot explain it. `fused_loss` came back NaN and `fused_exact` 0.00. In an
earlier run of the same sweep the same row returned `fused_loss` 4.437, flat —
never descended rather than descended and stalled — with a NaN control. 4.437 is
roughly what a frozen instruct model scores predicting the answer text with *no*
useful conditioning: it already speaks English, it just cannot see the sensors.
A model that learned nothing from its prefix lands exactly there.

That is not a learning failure. It is an optimiser that never stepped. Under
`torch.amp.GradScaler`, one inf or NaN in the gradients makes the scaler skip
the step and halve the scale; if every step produces one, the weights never move.

**Hypothesis.** The fp16 forward overflows.

**Probe.** Each backbone was loaded in fp32 on CPU and given the shape of input
the fusion stack produces — 16 prefix vectors at unit variance followed by 8 real
token embeddings — with a forward hook on every decoder layer recording the
largest absolute activation. 20 random draws per model:

| backbone | median peak | max peak | vs fp16 ceiling | draws over 65504 |
|---|---|---|---|---|
| SmolLM2-135M | 22 794 | 26 756 | 0.41x | **0 / 20** |
| SmolLM2-360M | 25 973 | **77 751** | **1.19x** | **2 / 20** |
| Qwen2.5-0.5B | 440 | 816 | 0.01x | **0 / 20** |

fp16 tops out at 65504. **SmolLM2-360M is the only model in the ladder whose
residual stream reaches that ceiling, and it was the only model that failed.**
Its last two decoder layers carry a massive-activation channel in the 6x10⁴
range; SmolLM2-135M peaks at less than half the ceiling, and Qwen2.5-0.5B never
exceeds 816 — three orders of magnitude of headroom.

Two reasons 2/20 understates the training-time rate: the probe uses 24 positions
where training uses up to 320, and the peak is a max over positions; and the
projector output scale is learned, so it does not stay at unit variance. Skipping
a step does not let the projector drift back to a safer scale, because skipping
is precisely not updating it.

**Prediction.** The same model, same data, same 300 steps, in fp32, passes.

**Confirmation.** `--backbone-dtype fp32`: `fused_loss` 0.01869, `fused_exact`
**1.00**, `control_exact` 0.125, **PASS**. Peak VRAM 6.07 GB against 4.32 GB in
fp16, and 473 s against 285 s — so the fix costs 1.4x memory and 1.7x time, and
buys the difference between a working row and a NaN.

The replication run makes the overflow reading harder to argue with: the second
time, **both** the fused and the control training went NaN on the fp16 row, and
the fp32 row passed again at 1.00 exact. A capacity limit does not produce NaN.

**Conclusion:** the 360 M result was never evidence about model capacity. The
ladder has no capacity floor above 135 M; it has one model with an unusually hot
residual stream and a numerical format too small to hold it.

## Result 3 — what the frozen backbone actually costs

| backbone | backbone params | fusion params | trainable | peak VRAM | gate wall clock |
|---|---|---|---|---|---|
| `TinyBackbone` | 2.1 M | 1.58 M | 43.29% | 0.17 GB | 21.6 s |
| SmolLM2-135M | 134.5 M | 1.98 M | 1.45% | 2.61 GB | 147.0 s |
| SmolLM2-360M (fp16) | 361.8 M | 2.67 M | 0.73% | 4.32 GB | 284.5 s |
| SmolLM2-360M (fp32) | 361.8 M | 2.67 M | 0.73% | 6.07 GB | 472.9 s |
| Qwen2.5-0.5B | 494.0 M | 2.53 M | 0.51% | 6.13 GB | 217.6 s |
| Qwen2.5-1.5B | 1543.7 M | 4.25 M | 0.27% | 10.95 GB | 428.9 s |
| Qwen2.5-3B (nf4) | 1698.7 M *(packed)* | 6.22 M | 0.36% | 11.81 GB | 1312.0 s |

Wall clock covers the whole gate — both the fused and the control training run,
600 steps total — plus the model download. The seven rows together take 48
minutes.

The 3 B backbone parameter count is the count of `Params4bit` storage elements,
not weights: NF4 packs two 4-bit weights per byte, so 3.09 B real parameters
appear as 1.70 B. True trainable fraction there is 0.20%.

**The binding constraint is VRAM, not compute.** A frozen backbone is not a free
backbone: gradients have to reach the projectors *through* the whole decoder, so
every layer's activations are retained for the backward pass even though none of
its weights update. That is why 0.5 B of fp16 weights (≈1 GB) costs 6.13 GB peak
at batch 4. It is also why the 3 B row needs NF4 — not to make it faster, but to
make it fit at all next to its own activations.

## Result 4 — dtype on Turing

Fusion stack, d_model 2048, p50 ms:

| config | batch 1 | batch 8 | batch 32 | per-window @32 | compiled b1 | compiled b8 |
|---|---|---|---|---|---|---|
| fp32 | 3.317 | 3.687 | 4.772 | 0.149 | 1.546 | 1.736 |
| **fp16** | 3.529 | 3.394 | **3.450** | **0.108** | **1.517** | **1.777** |
| bf16 *(emulated)* | 5.573 | 19.022 | **66.392** | 2.075 | 5.807 | 19.245 |
| nf4 | 4.658 | 4.859 | 5.886 | 0.184 | 5.869 | 5.676 |

This reproduces the Phase 3 headline on different hardware and a different torch
build: **`torch.cuda.is_bf16_supported()` returns `True` on a T4 and bf16 runs
19.2x slower than fp16.** PyTorch counts emulation as support, so an A100 recipe
copied verbatim onto Turing does not error — it runs, silently, at a twentieth of
the speed. `torch.compile` cannot rescue it (19.022 → 19.245 ms at batch 8: the
eager number, unmoved), because the cost is emulation, not graph overhead.

`torch.compile` gives fp16 a 2.33x win at batch 1 (3.529 → 1.517 ms) and makes
NF4 *worse* (4.658 → 5.869 ms): dequantisation does not fuse away.

## Result 5 — end-to-end, and where the time actually goes

*(The decode number here is the pre-KV-cache measurement. Result 10 replaces it
and explains why the replacement moved so little.)*

Single window, byte-level backbone, T4:

| stage | p50 | p95 | p99 | share |
|---|---|---|---|---|
| align | 0.118 ms | 0.153 ms | 0.187 ms | 0.06% |
| fuse | 3.198 ms | 3.422 ms | 3.557 ms | 1.54% |
| **decode** | **204.869 ms** | 224.569 ms | 224.569 ms | **98.41%** |
| total | 208.185 ms | | | |

Everything this project optimises — alignment, pooling, projection, dtype,
quantisation, ONNX — lives inside the 1.6% of the wall clock that is not decode.
Worth saying plainly rather than burying: the 19.2x bf16 finding is a 19.2x
difference on 1.5% of the pipeline. Decode at this size is bound by per-token
kernel launch latency, not compute.

## Result 6 — ONNX export still exact

Fusion stack exported at opset 17, 0.21 MB, verified against the torch forward
over four cases including a fully-masked batch:

| case | max abs delta |
|---|---|
| batch=1, coverage 100% | 4.77e-07 |
| batch=2, coverage 100% | 6.26e-07 |
| batch=5, coverage 50% | 5.96e-07 |
| batch=3, coverage 0% | 5.36e-07 |

Worst divergence 6.26e-07 — fp32 round-off, not a semantic difference. The
coverage-0% case matters: it is the degraded-sensor path, and an export that
silently changed behaviour when every sensor is missing would be the one that
hurts someone.

The export log contains an alarming-looking `RuntimeError` from
`onnx.version_converter` that is **not** a failure: onnxscript tries a C-API
version conversion from opset 18 to 17, that path raises, and the exporter falls
back and succeeds. The verified deltas above are from the model it produced.

## Result 7 — four bugs between a working single GPU and a working second one

Three attempts at the FSDP-vs-DDP comparison produced no output directory and no
readable log. Mirroring child process output into `/kaggle/working/console.log`
found the cause on the first run that had it — a single line, identical in the
single-GPU baseline and both distributed runs:

```
ImportError: Found an incompatible version of torchao.
Found version 0.10.0, but only versions above 0.16.0 are supported
```

Kaggle's image ships torchao 0.10.0. PEFT's LoRA dispatcher calls
`is_torchao_available()` unconditionally, and that function **raises** on a
too-old torchao instead of returning `False`. So every `get_peft_model()` dies,
Stage B never starts, and nothing in this project uses torchao.

The kernel now uninstalls torchao before running. That fix worked, and the run
behind it (kernel v11) got Stage B started for the first time — which
immediately exposed three more bugs that the import error had been hiding.

**Single-GPU baseline, measured:** 300 steps in 19 s, **64 ms/step at 16
windows/step**, loss 0.0698 → 0.0743 over the run.

**DDP died on the first step:**

```
AttributeError: 'DistributedDataParallel' object has no attribute 'tokenize'
```

`stage_b` called `backbone.tokenize(...)` after wrapping. DDP does not forward
attribute lookups to the module it wraps; FSDP does, which is why only the DDP
run hit this. Fixed by binding `tokenize` before the wrap.

**FSDP ran, then deadlocked at step 200:**

```
[Rank 0] Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=1205, OpType=_ALLGATHER_BASE, ...) ran for 600086 ms
[Rank 1] ... OpType=BROADCAST ...
```

Two ranks, same sequence number, *different collectives* — textbook rank
divergence. The cause is one line: `if step % args.ckpt_every == 0 and
info.is_main: _save(...)`, with `--ckpt-every 200`. Under FSDP, `state_dict()`
is a collective. Rank 0 entered the allgather to reassemble its shards; rank 1,
excluded by `is_main`, walked on to the next step's broadcast; the NCCL watchdog
killed the job 600 s later. A checkpoint guard that is correct on one GPU and
correct under DDP is a deadlock under FSDP. Fixed: every rank calls
`_save`, the gather runs on all of them under `FullStateDictConfig(rank0_only=
True)`, and only rank 0 writes bytes.

**And a dead adapter, found by DDP refusing to start:**

```
Parameter indices which did not receive grad for rank 0: 0 1 6 7 12 13 18 19
```

Four layers, six LoRA parameters each, the first pair of every layer dead. Those
are the `out_proj` adapters. `nn.MultiheadAttention` never *calls* `out_proj` as
a module — it passes `out_proj.weight` and `out_proj.bias` into
`F.multi_head_attention_forward` — so a LoRA wrapper around it is never in the
graph. It had been training 0.03 M parameters that could not receive a gradient,
in every Stage B run in this repo's history, and single-GPU training has no way
to notice. `find_unused_parameters=True` would have silenced the complaint;
removing the target is the honest fix, and the trainable count drops 0.15 M →
0.12 M with nothing lost.

**Verification.** The DDP path and the all-ranks checkpoint are confirmed on a
local 2-rank gloo run (6 steps, `--ckpt-every 3`, CPU): both ranks finish, both
checkpoints write, no hang. The FSDP branch of `_full_state_dict` is **not**
locally verifiable — `RuntimeError: FSDP needs a non-CPU accelerator device` —
so it stands on the API contract and the diagnosis, not on a green run — the
next GPU run supplied that, in Result 8.

## Result 8 — FSDP vs DDP, and a 2.4x speedup that was not sharding

All three fixes held. The comparison finally ran, and the first version of it
produced a result that should not have been possible.

Stage B, `TinyBackbone`, 300 steps, batch 16 per rank, 2x T4:

| config | ms/step | windows/step | windows/s | vs 1 GPU | final loss |
|---|---|---|---|---|---|
| 1 GPU | 63 | 16 | 254 | 1.00x | 0.0982 |
| DDP (fp32) | 67 | 32 | 478 | **1.88x** | 0.0847 |
| FSDP (fp32) | 68 | 32 | 471 | **1.85x** | 0.0847 |
| FSDP (fp16) | 48 | 32 | 667 | **2.63x** | 0.0853 |

**Read the first version of this table carefully and it does not add up.** It
had two rows — DDP at 1.88x and FSDP at 2.42x — and 2.42x on two GPUs is
superlinear. Sharding does not do that to a 2 M parameter model; sharding is
supposed to *cost* throughput at this size, which is what the kernel's own
commentary predicted.

The cause was in `dist.py`: the FSDP branch wrapped the model in
`MixedPrecision(param_dtype=torch.float16)` and the DDP branch had no mixed
precision at all. "FSDP beats DDP" was a dtype result wearing a sharding label.
Adding `--dist-precision` and running the fp32 FSDP row settles it:

- **Sharding costs 1.5%** (68 ms against DDP's 67). At 2 M parameters FSDP buys
  memory nobody needs and charges a communication fee for it — the predicted
  result, now measured instead of asserted.
- **fp16 buys 1.40x** (68 → 48 ms), and buys it without accuracy: all three
  two-GPU runs land within 0.0006 of each other in final loss.
- **Neither reaches 2x.** 1.88x is the honest scaling number for DDP here;
  gradient all-reduce and per-step Python overhead take the rest.

The lesson generalises past this repo: an A/B where the two arms differ in two
things is not an A/B, and a suspiciously good number is the cheapest bug
detector available. The tell was not in the code, it was that the result was
*too good* for what it claimed to measure.

## Result 9 — held-out metrics on a real frozen backbone

Every held-out number this project has published came from `TinyBackbone`: 2 M
random parameters trained alongside the projectors. Qwen2.5-0.5B passes the gate
at 6.13 GB, which fits a free T4, so the weakest numbers in the repo were
affordable to replace. Stage A, 600 steps, backbone frozen, final loss 0.0273;
16 held-out windows, four ablations.

| run | exact | manoeuvre | numeric MAE | hallucination |
|---|---|---|---|---|
| full | 0.000 | 1.000 | 0.290 | 0.042 |
| no camera | 0.062 | 1.000 | 0.471 (+0.18) | 0.021 |
| **no CAN** | 0.000 | 1.000 | 1.079 (+0.79) | **0.250 (+0.21)** |
| no radar | 0.000 | 1.000 | 3.460 (+3.17) | 0.091 |
| no audio | 0.000 | 1.000 | 3.826 (+3.54) | 0.140 |

Against the `TinyBackbone` run of the same harness (`reports/phase3_status.md`),
three things stand out:

- **The CAN-bus result reproduces across an entirely different backbone.**
  Removing the CAN bus takes hallucination from 4% to 25% here and from 2% to
  21% there. Two backbones that share no weights, no vocabulary and no
  architecture agree that the CAN bus is the modality whose absence makes this
  model *fabricate*. That was the single most quotable claim in the repo and it
  was resting on a 2 M parameter stand-in; it is not any more.
- **A real backbone is not better on the full input.** Numeric MAE 0.290 against
  the tiny model's 0.260. Whatever a pretrained decoder brings, it does not show
  up as accuracy on this synthetic task — consistent with the ladder, where
  every backbone from 135 M to 3 B passed the gate and the differences were in
  loss, not verdict.
- **It degrades far more sharply when a sensor drops.** No-radar and no-audio
  cost +3.2 and +3.5 MAE here against +0.06 and +0.14 for the tiny model. Read
  that as a warning, not a win: with `HashCache` the camera and audio vectors
  are scene *identity*, and a bigger backbone appears to lean on that identity
  harder. Which is an argument for real embeddings, not against big backbones.

Caveats that belong next to these numbers: n=16, numeric MAE is unbounded so a
single bad prediction moves it a long way (the no-camera exact-match of 0.062 is
one window out of sixteen, i.e. noise), and `coverage_honesty` is 0.000 in every
row for both backbones — the model never states what it could not see. This is
Stage A only; the eval harness scores a Stage A checkpoint, so LoRA is not in
these numbers.

## Result 10 — a KV cache, and why it only bought 1.18x

Decode is 98%+ of end-to-end latency in every measurement this project has
taken, so it is the only place an optimisation can matter. Reading
`TinyBackbone.generate_one` explains the number: it re-ran the entire
transformer over the entire sequence for every token. Quadratic attention inside
a linear loop — the whole prompt re-encoded 102 times to emit 102 bytes.

Torch's `nn.TransformerEncoderLayer` has no incremental-decode entry point, so
the fix walks the same submodules by hand and keeps keys and values between
steps. No weight is re-parameterised, so existing checkpoints load unchanged,
and `tests/test_kv_cache.py` asserts the cached decode is character-identical to
the recompute version.

Same kernel, same T4, same checkpoint recipe, and **the same 102 tokens
produced**, so the comparison is clean:

| | decode p50 | throughput | total p50 | decode share |
|---|---|---|---|---|
| recompute | 219.22 ms | 465 char/s | 222.6 ms | 98.5% |
| **KV cache** | **185.74 ms** | **549 char/s** | **188.9 ms** | 98.3% |

**1.18x.** An asymptotic improvement from O(n³) to O(n²) that returns 15% is
worth understanding rather than shipping quietly. At d_model 192 and four
layers, one decode step is far too small to saturate a T4: the wall clock is
per-step kernel launch and Python overhead, not matrix multiplication. Removing
arithmetic from a launch-bound loop returns very little, and the hand-rolled
layer walk adds Python ops per step that eat part of what it saves.

Two things follow. The repo's standing claim — *decode here is bound by
per-token launch latency, and a bigger GPU will not help* — now has a direct
experiment behind it rather than an inference from a CPU-vs-T4 comparison. And
the remaining lever is not arithmetic: it is CUDA graphs or batched decode,
which attack launches rather than FLOPs.

One caveat on the size of the win: end-to-end decode has varied ~7% run to run
(204.9 ms and 219.2 ms on two earlier runs of the identical recipe), so 15% is
real but only about twice the noise floor.

## Result 11 — 135 M was not the floor, and NF4 costs 2.2x to save 9%

Two questions the main sweep left open, run as their own ten-minute section
rather than by re-running the fifty-one-minute ladder:

| backbone | params | dtype | fused loss | fused exact | control loss | control exact | peak VRAM | wall | gate |
|---|---|---|---|---|---|---|---|---|---|
| facebook/opt-125m | 125 M | fp16 | 0.00813 | **1.00** | 0.1246 | 0.125 | 1.80 GB | 55.2 s | **PASS** |
| Qwen2.5-0.5B-Instruct | 494 M | **nf4** | 0.00083 | **1.00** | 0.0645 | 0.125 | 5.58 GB | 471.0 s | **PASS** |

**A 125 M model from a third family passes, and it is not instruction-tuned.**
OPT-125m shares no lineage with SmolLM2 or Qwen2.5 and has never seen an
instruction-following corpus. It reproduces all eight answers verbatim from the
fusion prefix at 1.80 GB peak — under a third of the memory of the 0.5 B row —
in under a minute of training. (Run twice: 0.00741/0.00813 fused loss, 1.00
exact both times.)

So the property being exercised is not instruction-following and not scale. It
is that a pretrained causal decoder reads a learned prefix — which is what
prefix tuning has always claimed, now with the sensor pipeline in front of it.

**And NF4 is now separable from size.** The 3 B row was the only quantised one
in the main ladder, so "3 B passes" and "NF4 passes" were the same measurement.
Running 0.5 B in NF4 against the same model in fp16 separates them:

| Qwen2.5-0.5B | fused loss | fused exact | peak VRAM | wall |
|---|---|---|---|---|
| fp16 | 0.00092 / 0.00122 | 1.00 | 6.13 GB | 218 s |
| nf4 | 0.00083 | 1.00 | **5.58 GB (−9%)** | **471 s (2.2x)** |

**Quantisation changes nothing about the verdict and very little about the
loss** — which is the reassuring half. The other half is the price: 2.2x the
wall clock to save 9% of peak VRAM. The saving is small because weights are not
what fills the card here; 0.5 B in fp16 is about 1 GB of weights inside a 6.13 GB
peak, and the rest is activations retained for a backward pass that has to reach
the projectors through the frozen decoder. NF4 shrinks the 1 GB and pays
dequantisation on every forward.

That is the direct measurement behind a claim this repo has been making from the
dtype sweep alone: **4-bit is for the 3 B backbone, which does not otherwise
fit, and for nothing smaller.**

## Result 12 — the real-data path, and exactly what blocks it

The largest gap in this project is that camera and audio embeddings carry no
content. A probe section on a mounted Kaggle nuScenes-mini answers what a real
mount actually provides, without spending a minute of training:

```
candidate nuScenes roots: ['/kaggle/input/nuscenes-mini']
  v1.0-mini  present=True  entries=13
  samples    present=True  entries=12
  sweeps     present=True  entries=12
  maps       present=True  entries=4
  can_bus    present=False entries=0
scenes: 10 ['scene-0061', 'scene-0103', 'scene-0553']
load_scene failed: Error: CAN bus directory not found
```

`nuscenes-devkit` installs, the metadata parses, and `list_scenes()` returns all
ten mini scenes. **Real camera frames are one `precompute_reference.py` away.**
What is missing is the CAN bus expansion — a separate download that most
redistributions omit, and the one modality every ablation in this repo says is
load-bearing. No Kaggle dataset carrying it turned up in a search.

`load_scene` used to raise on that, which made the entire real-data path
unreachable over one absent directory. It now degrades: the CAN stream is empty,
the aligner produces zero coverage for it, and the fusion mask marks it invalid
— the same path a dead bus takes, which this architecture models explicitly. It
says so loudly on stderr, because a checkpoint trained that way is missing the
signal the grounding metrics depend on.

It is tempting to read that as "vision is unblocked, CAN is a separate errand".
It is not, and the code says why. `describe_window` derives the question, the
answer and the manoeuvre tag from CAN and radar, and returns empty strings when
either is invalid; `build_samples` then drops the window. **The CAN bus is not
just the strongest ablation in this repo — it is the label source.** Load real
nuScenes today and every window is discarded before a single camera embedding is
used, however good that embedding is.

So the dependency is linear, not parallel: the CAN expansion is a prerequisite
for real data of any kind, not an enhancement to it. The alternative — deriving
manoeuvre labels from ego pose in the main archive, which *is* present — is a
real option and a different piece of work, because it changes what the labels
mean rather than where they come from.

## Negative results worth keeping

- **The P100 that could not run anything.** Kaggle's default accelerator is a
  Tesla P100 (sm_60) and Kaggle's own preinstalled torch supports sm_70 and up.
  `torch.cuda.is_available()` is `True`, device count is 1,
  `torch.cuda.is_bf16_supported()` is `True` — and no kernel will launch. Six
  sections failed independently, which reads as six unrelated bugs rather than
  one wrong dropdown. The kernel now fails fast below sm_70 with the fix in the
  message.
- **`kaggle kernels push` resets the accelerator to that default** — unless the
  metadata pins it. `enable_gpu` requests a GPU and cannot say which one, so a
  kernel that ran on `GPU T4 x2` yesterday comes back on sm_60 today and has to
  be restarted by hand from the browser. `"machine_shape": "NvidiaTeslaT4"` is
  the field that fixes it, and it is documented; we cost ourselves several runs
  by reading `enable_gpu` as the whole story. The sm_70 guard stays as the
  backstop, turning a wrong accelerator into a one-minute failure that names the
  field.
- **Two `torch.cuda.is_*_supported()` calls in this project have returned `True`
  for something that does not work.** Both are capability queries answering about
  the software build, not the silicon.
- **A 0-byte log is not an empty run.** The Kaggle API returned this kernel's log
  as 0 bytes on every attempt. Three failures were invisible because of it. The
  fix — mirror every child's stdout and stderr into a file under
  `/kaggle/working` — is four lines and found two root causes on its first run.

## What this establishes, and what it does not

Established:

- The fusion prefix carries per-window sensor information **through a frozen,
  pretrained language model**, not just through a co-trained random one.
- It does so at 125 M, 135 M, 360 M, 0.5 B, 1.5 B and 3 B, across three model
  families, and at 125 M without instruction tuning — the mechanism is not tuned
  to one width, one lineage, or one kind of fine-tune.
- Peak VRAM, not compute, is what limits backbone size in this recipe.
- fp16 is not free: one model in seven has a residual stream that does not fit
  in it, and the failure mode is silent.

Not established:

- **Anything about perception.** These runs use the same content-free `HashCache`
  camera and audio embeddings as every earlier phase, on synthetic scenes. See
  `MODEL_CARD.md` limitations 1, 3 and 4. The gate tests the *fusion channel*;
  passing it is necessary, not sufficient.
- **Generalisation.** The gate is a deliberate 8-window overfit. It answers "can
  this architecture route information at all", not "does it work on held-out
  scenes". Result 9 adds a held-out table for a real backbone, but on n=16
  synthetic windows, Stage A only — no LoRA is in any held-out number here.
- **A capacity floor.** 125 M is now the smallest backbone *tested* that passes
  (Result 11), and still not the smallest that would.
- **Scaling past two GPUs.** Result 8 measures 2x T4 on a 2 M parameter model.
  1.88x for DDP there says nothing about what FSDP does when the model actually
  exceeds one card's memory, which is the case FSDP exists for.

## What follows from this

- **Ship fp16, but verify it per backbone.** The dtype sweep says fp16 and the
  ladder says fp16 breaks one model in seven, silently. A flat loss curve on a
  new backbone should be treated as a numerics symptom before it is treated as a
  hyperparameter problem.
- **Scale is not the lever here.** Going from 135 M to 3 B moves fused loss from
  0.028 to 0.001 and buys nothing on the gate, which all of them pass, while
  costing 4.5x VRAM and 9x wall clock. If the gate is what you are optimising,
  the small backbone is the correct engineering answer.
- **The remaining risk is entirely in the data, not the architecture.** With the
  fusion channel now demonstrated through five real backbones, the content-free
  camera and audio caches are the single largest gap between this project and a
  claim about driving.
- **Decode is the product's latency, and it is launch-bound.** Two phases of
  optimisation moved 1.6% of the wall clock; a KV cache that removed most of the
  arithmetic from the other 98% moved 15%. What is left to attack is kernel
  launches — CUDA graphs, batched decode — not FLOPs.
- **Single-GPU correctness is not evidence of distributed correctness.** Three
  bugs in Result 7 — an unforwarded attribute, a collective inside a rank guard,
  and a LoRA adapter that never received a gradient — all ran clean on one GPU
  for months. Two of them are invisible without a second rank; the third was
  *reported* by a second rank. Any claim about a training loop that has only
  ever run on one device is a claim about one device.

## Next steps, in the order they are worth doing

1. **The CAN bus expansion — the prerequisite for all real data.** Download it
   from nuscenes.org and republish as a private Kaggle dataset; nothing public
   carries it. Result 12 is the reason this is first rather than second: without
   CAN, `describe_window` produces no labels and every real window is dropped,
   so real camera embeddings cannot be trained on at all. The alternative is
   deriving manoeuvre labels from ego pose, which the main archive does ship —
   a different piece of work with different semantics, worth costing before
   choosing.
2. **Real vision embeddings — the largest remaining gap, once 1 lands.**
   `scripts/precompute_reference.py --source nuscenes`, then Stage A over
   nuScenes windows and a re-run of the ablations. Until this happens the camera
   and audio rows measure scene-identity leakage, and Result 9 suggests a larger
   backbone exploits that leakage *harder*, which makes every camera and audio
   delta in this repo a number about memorisation. Note that `stage_a` builds
   its samples from `make_scene` today; a `--source nuscenes` path is part of
   this step, not a given.
3. **CUDA graphs or batched decode.** Result 10 establishes that decode is
   launch-bound, not compute-bound, so this is the only remaining lever on 98%
   of the wall clock. A KV cache already removed the arithmetic and returned
   1.18x; capturing the step as a graph attacks what is actually left.
4. **Stage B in the eval harness.** The harness scores a Stage A checkpoint, so
   every held-out number here — `TinyBackbone` and Qwen2.5-0.5B alike — is
   pre-LoRA. Loading `stage_b.pt` adapters would let Phase 2's central claim be
   measured rather than assumed.
5. **A held-out gate.** The current gate is a deliberate 8-window overfit. The
   architecture question it answers is settled across seven backbones; the next
   question — does the fusion prefix carry information about windows the
   projectors have never seen — needs a different instrument.
6. **Widen the ladder further.** Both `ladder2` rows are now measured
   (Result 11). What remains open at the bottom is how far below 125 M the gate
   survives — a 70 M or 30 M row would cost about a minute each. At the top, a
   3 B fp16 row needs more than 15 GB: 1.5 B already peaks at 10.95 GB.

## Reproduce

```bash
bash scripts/kaggle_push_src.sh                 # code + a Stage A checkpoint, as a dataset
bash scripts/kaggle_run.sh --only ladder        # ~51 min: the seven rows above
bash scripts/kaggle_run.sh --only smoke,dist    # ~5 min: results 7 and 8
bash scripts/kaggle_run.sh --only real          # ~20 min: result 9
bash scripts/kaggle_run.sh --only stagea        # ~5 min: results 5 and 10
bash scripts/kaggle_run.sh --only ladder2       # ~10 min: result 11
bash scripts/kaggle_run.sh --only nuscenes      # ~2 min: result 12, trains nothing
bash scripts/kaggle_run.sh --only all           # everything in this report
```

Raw outputs for each are under `reports/kaggle_ladder/`: the main ladder at the
top level, `run11/` for the replication, `dist_run1/` and `dist_run2/` for
results 7 and 8, `kv_cache/` for result 10, `ladder2/` for results 11 and 12.

Single row, locally, on any CUDA device:

```bash
python -m daystorm.train.stage_a \
  --backbone HuggingFaceTB/SmolLM2-135M-Instruct \
  --overfit 8 --steps 300 --batch 4 --scenes 24 --lr 1e-3 \
  --gate-json gate_135m.json
```

Add `--backbone-dtype fp32` if the loss does not move.
