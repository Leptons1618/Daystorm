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
survive? A 3 B backbone passing is a nice result. A 135 M backbone passing is a
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
attention pooling, projectors) receives gradients. Backbones are ungated and
instruction-tuned, two families, so scale moves and little else does. NF4 is
used only for the 3 B row — below that the fp16 weights fit twice over on a T4
and dequantisation is pure overhead, which the dtype sweep below measures.

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

## Result 7 — still no distributed number, and now we know why

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

Nothing was wrong with `stage_b`, `train/dist.py`, torchrun, or NCCL — the two
distributed runs got as far as spawning both ranks and initialising the process
group before hitting the same import. The kernel now uninstalls torchao before
running. The comparison is one run away.

## Negative results worth keeping

- **The P100 that could not run anything.** Kaggle's default accelerator is a
  Tesla P100 (sm_60) and Kaggle's own preinstalled torch supports sm_70 and up.
  `torch.cuda.is_available()` is `True`, device count is 1,
  `torch.cuda.is_bf16_supported()` is `True` — and no kernel will launch. Six
  sections failed independently, which reads as six unrelated bugs rather than
  one wrong dropdown. The kernel now fails fast below sm_70 with the fix in the
  message.
- **`kaggle kernels push` resets the accelerator to that default.** A kernel that
  ran on `GPU T4 x2` yesterday comes back on sm_60 today, and
  `kernel-metadata.json` can request a GPU but cannot say which one. Every push
  needs the accelerator set again in the browser. The guard above turns this from
  an hour of confusing output into a one-minute failure that says what to do.
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
- It does so at 135 M parameters, and at 360 M, 0.5 B, 1.5 B and 3 B — the
  mechanism is not tuned to one width.
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
  scenes". No held-out metric in this repo has been measured with a real
  backbone.
- **A capacity floor.** 135 M is the smallest backbone *tested* that passes, not
  the smallest that would.
- **Any distributed throughput claim.** See Result 7.

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
- **Decode is the product's latency.** Two phases of optimisation have moved
  1.6% of the wall clock.

## Next steps, in the order they are worth doing

1. **Re-run the distributed comparison** with torchao removed. Everything else
   is in place; this is the one measurement the README promises and does not
   have.
2. **Real embeddings.** `scripts/precompute_reference.py` against actual nuScenes
   frames, then re-run the ablations. Until this happens the camera and audio
   rows measure scene-identity leakage and the model card has to keep saying so.
3. **Held-out metrics with a real backbone.** Every published metric uses
   `TinyBackbone`. The 0.5 B row passes the gate at 6.13 GB, which fits a free
   T4 — so a full Stage A + Stage B + eval on Qwen2.5-0.5B is affordable and
   would replace the weakest numbers in the repo.
4. **Batched decode or a KV cache with CUDA graphs.** The only change that can
   move the 98.4%.
5. **An fp16 overflow guard in `stage_a`.** The diagnosis took a CPU probe and a
   confirming GPU run; a check on the first forward's activation magnitude, or on
   consecutive skipped `GradScaler` steps, would have printed it in one line.
6. **Widen the ladder** — a 3 B row in fp16 rather than NF4 to separate the
   quantisation variable from the size variable, and a second sub-200 M family to
   test whether 135 M is a floor or just the smallest thing tried.

## Reproduce

```bash
bash scripts/kaggle_push_src.sh        # publish src/daystorm as the kernel's dataset
bash scripts/kaggle_run.sh             # push and run; then set Accelerator to GPU T4 x2
                                       # in the browser and Save & Run All
```

Single row, locally, on any CUDA device:

```bash
python -m daystorm.train.stage_a \
  --backbone HuggingFaceTB/SmolLM2-135M-Instruct \
  --overfit 8 --steps 300 --batch 4 --scenes 24 --lr 1e-3 \
  --gate-json gate_135m.json
```

Add `--backbone-dtype fp32` if the loss does not move.
