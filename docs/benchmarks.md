# GPU benchmarks — measured on a Tesla T4

Hardware: Tesla T4, sm_75, 15.6 GB, CUDA 12.8, torch 2.11.0+cu128, via
`colab new -s daystorm --gpu T4`. T4 is deliberate: the README makes claims
about free-tier Colab, and free-tier Colab is a T4.

## Headline: `is_bf16_supported()` returns True on a T4, and bf16 is 20x slower

This is the finding worth carrying into an interview. PyTorch reports bf16 as
supported on sm_75 because it counts *emulation*. Turing has no bf16 tensor
cores, so an A100 recipe copied verbatim does not error — it runs, silently, at
a twentieth of the speed.

Fusion stack, d_model 2048, p50 latency (ms), eager / `torch.compile`:

| config | batch 1 | batch 8 | batch 32 | per-window @32 | compiled @32 |
|---|---|---|---|---|---|
| fp32 | 3.585 | 3.976 | 4.868 | 0.152 | 3.370 |
| **fp16** | 3.534 | 3.586 | **3.597** | **0.112** | **2.557** |
| bf16 *(emulated)* | 5.863 | 21.250 | 74.149 | 2.317 | 72.829 |
| nf4 | 5.014 | 5.166 | 6.271 | 0.196 | 7.126 |

Three readings:

- **fp16 is the configuration to ship on this hardware** — 1.35x over fp32
  eager, 1.9x with compile.
- **bf16 costs 20.6x against fp16** at batch 32, and `torch.compile` cannot
  rescue it (74.1 → 72.8 ms): the cost is emulation, not graph overhead.
- **nf4 is slower than fp32 here, and compile makes it worse** (6.27 → 7.13 ms).
  Expected and worth stating plainly: the fusion stack is ~6 M parameters, so
  dequantisation overhead dominates any memory saving. 4-bit is for the 3 B
  backbone, not for this.

`torch.compile` gains where it helps: fp16 batch 1 3.534 → 1.607 ms (2.2x),
fp32 batch 8 3.976 → 2.057 ms (1.9x).

## End-to-end, T4 vs CPU

| stage | CPU p50 | T4 p50 | share (T4) |
|---|---|---|---|
| align | 0.04 ms | 0.15 ms | 0.1% |
| fuse | 0.82 ms | 3.41 ms | 1.5% |
| decode | 233.73 ms | 228.93 ms | 98.5% |
| **total** | 234.59 ms | 232.49 ms | |

**The GPU barely helps.** Decode moved 233.73 → 228.93 ms, a 2% gain, because
the byte-level backbone is small enough that decode is bound by per-token
kernel launch latency rather than compute. This is a real result and it sets
the priority for the next optimisation pass: batching decode, or a KV cache
with CUDA graphs, will move this number and a bigger GPU will not.

## Training throughput

| run | hardware | result |
|---|---|---|
| Phase 1 gate | T4 | **PASS** — fused 100% exact, control **0%** (CPU control was 17%) |
| Stage A, 700 steps | T4 | 38 s (CPU: 139 s) — **3.7x** |
| Stage B QLoRA, 300 steps | T4 | 21 s, 70 ms/step at 16 windows/step |
| ONNX export | T4 | verified, worst divergence 7.15e-07 over 4 cases |

## Held-out metrics, T4-trained checkpoint

| run | exact | manoeuvre | numeric MAE | hallucination |
|---|---|---|---|---|
| full | 0.00 | 1.00 | 0.260 | 0.021 |
| no camera | 0.00 | 1.00 | 0.373 (+0.11) | 0.000 |
| **no CAN** | 0.00 | 1.00 | **0.454 (+0.19)** | **0.208 (+0.19)** |
| no radar | 0.00 | 1.00 | 0.323 (+0.06) | 0.021 |
| no audio | 0.00 | 1.00 | 0.398 (+0.14) | 0.000 |

Removing the CAN bus is still the only ablation that makes the model
*fabricate*: hallucination 2% → 21%. Camera and audio rows remain
scene-identity leakage, not perception — see `docs/model_card.md`.

## The same harness on a real frozen backbone

Qwen2.5-0.5B-Instruct, frozen, Stage A 600 steps (final loss 0.0273), 16
held-out windows:

| run | exact | manoeuvre | numeric MAE | hallucination |
|---|---|---|---|---|
| full | 0.000 | 1.000 | 0.290 | 0.042 |
| no camera | 0.062 | 1.000 | 0.471 (+0.18) | 0.021 |
| **no CAN** | 0.000 | 1.000 | 1.079 (+0.79) | **0.250 (+0.21)** |
| no radar | 0.000 | 1.000 | 3.460 (+3.17) | 0.091 |
| no audio | 0.000 | 1.000 | 3.826 (+3.54) | 0.140 |

**The CAN-bus finding reproduces across an entirely different backbone**:
hallucination 4% → 25% here, 2% → 21% above. Two models sharing no weights, no
vocabulary and no architecture agree on which missing sensor makes this model
fabricate.

Two honest caveats. The real backbone is *not* more accurate on the full input
(MAE 0.290 against 0.260), and it degrades far more sharply when a sensor drops
(+3.2 and +3.5 MAE for radar and audio). With `HashCache` the camera and audio
vectors are scene identity, so the larger model appears to lean on that identity
harder — an argument for real embeddings, not against real backbones. n=16 and
numeric MAE is unbounded, so single predictions move it a long way.

## The 2-GPU comparison — measured, after eight attempts

Stage B, `TinyBackbone`, 300 steps, batch 16 per rank, Kaggle 2x T4:

| config | ms/step | windows/step | windows/s | vs 1 GPU | final loss |
|---|---|---|---|---|---|
| 1 GPU | 63 | 16 | 254 | 1.00x | 0.0982 |
| DDP (fp32) | 67 | 32 | 478 | **1.88x** | 0.0847 |
| FSDP (fp32) | 68 | 32 | 471 | **1.85x** | 0.0847 |
| FSDP (fp16) | 48 | 32 | 667 | **2.63x** | 0.0853 |

**Sharding costs 1.5% and buys memory nothing here needs** — the predicted
result, now measured. **fp16 buys 1.40x** and costs no accuracy: the three
two-GPU runs finish within 0.0006 of each other.

The first version of this table had two rows, DDP at 1.88x and FSDP at 2.42x,
and 2.42x on two GPUs is superlinear. `dist.py` was wrapping FSDP in
`MixedPrecision(param_dtype=fp16)` while DDP got no mixed precision at all, so
"FSDP beats DDP" was a dtype result wearing a sharding label. `--dist-precision`
separates them. A result too good for what it claims to measure is a bug report.

Getting here took eight attempts. The blockers, in the order they were hit,
because most of them are environment traps rather than code:

1. **HF Jobs** — `a10g-largex2` is exactly right at $3.00/h, and the CLI works,
   but the account has no pre-paid credits: `402 Pre-paid credit balance is
   insufficient`.
2. **Kaggle (2x T4, free, 30 h/week)** — the worker came up `torch
   2.10.0+cpu, devices 0` with no internet. `enable_gpu` and `enable_internet`
   do not take effect until the account is **phone-verified**. The kernel
   refused to continue rather than emit a CPU number labelled as distributed.
   *Resolved:* account phone-verified.
3. **Kaggle, second attempt** — ran on the default accelerator, a **Tesla
   P100**, which is sm_60. Kaggle's own preinstalled torch supports sm_70 and
   up, so `torch.cuda.is_available()` was `True` and no kernel could launch.
   Every section failed independently. *Resolved:* accelerator set to
   `GPU T4 x2`; the kernel now fails fast below sm_70.
4. **Kaggle, third attempt (`GPU T4 x2`)** — the backbone ladder, latency
   sweep, end-to-end timing and ONNX export all completed
   (`docs/findings.md`). The distributed section still emitted
   no `sb_1gpu` / `sb_ddp` / `sb_fsdp` output directory, and the Kaggle API
   returns the kernel log as 0 bytes, so the cause was invisible. A local
   CPU reproduction of the same `stage_b` invocation against the Kaggle-produced
   Stage A checkpoint **succeeds** (5 steps, 497 ms/step, LoRA 0.15 M + fusion
   1.58 M), which correctly ruled out a bug in `stage_b` but pointed at torchrun
   and NCCL, and both were innocent — see 6. The kernel now mirrors every child
   process's stdout and stderr into `/kaggle/working/console.log`, which comes
   back with the outputs, so the next run does not depend on the log endpoint
   working.
5. **Kaggle, fourth attempt** — the first run of the mirrored log, and it
   immediately earned its keep: `kaggle kernels push` **resets the accelerator
   to the default P100**, so a kernel that ran on `GPU T4 x2` yesterday comes
   back on sm_60 today with nothing in the CLI to say so. `kernel-metadata.json`
   can request a GPU but not which one. The sm_70 guard caught it in under a
   minute and `console.log` recorded the reason. Every push now needs the
   accelerator set again in the browser before the run means anything.
6. **Kaggle, fifth attempt (`GPU T4 x2`, complete run)** — everything except the
   distributed section finished, and the mirrored log finally named the cause.
   One line, identical in the single-GPU baseline and in both distributed runs:

   ```
   ImportError: Found an incompatible version of torchao.
   Found version 0.10.0, but only versions above 0.16.0 are supported
   ```

   Kaggle's image ships torchao 0.10.0. PEFT's LoRA dispatcher calls
   `is_torchao_available()` unconditionally, and that function **raises** on a
   too-old torchao rather than returning `False`, so `get_peft_model()` dies in
   `stage_b.attach_lora` before a single step runs. Nothing in this project uses
   torchao. Both distributed runs had already spawned two ranks and initialised
   the process group, so torchrun and NCCL were never the problem. The kernel now
   runs `pip uninstall -y torchao` before anything else — upgrading it instead
   would drag torch along behind it.
7. **Kaggle, sixth attempt (`GPU T4 x2`, kernel v11)** — the torchao fix worked
   and Stage B started for the first time, which exposed three bugs the import
   error had been hiding. The **single-GPU baseline is now measured on a T4: 300
   steps in 19 s, 64 ms/step at 16 windows/step.** Then:

   - **DDP**, step 1: `AttributeError: 'DistributedDataParallel' object has no
     attribute 'tokenize'`. `stage_b` called `backbone.tokenize(...)` after
     wrapping; DDP does not forward attribute lookups to the wrapped module,
     FSDP does. *Fixed* — bind `tokenize` before the wrap.
   - **FSDP**, step 200: NCCL watchdog timeout after 600 s, rank 0 in
     `_ALLGATHER_BASE` and rank 1 in `BROADCAST` at the same sequence number.
     `if step % args.ckpt_every == 0 and info.is_main: _save(...)` put a
     *collective* — FSDP `state_dict()` — inside a rank-0 guard, so rank 0
     waited for a gather rank 1 was never going to join. *Fixed* — every rank
     calls `_save` under `FullStateDictConfig(rank0_only=True)`, only rank 0
     writes.
   - **Dead LoRA adapters**, reported by DDP: `Parameter indices which did not
     receive grad for rank 0: 0 1 6 7 12 13 18 19` — the `out_proj` adapter in
     each of the four layers. `nn.MultiheadAttention` passes `out_proj.weight`
     into `F.multi_head_attention_forward` rather than calling the module, so
     LoRA there is never in the graph. Every Stage B run in this repo's history
     carried 0.03 M parameters that could not receive a gradient. *Fixed* —
     `out_proj` removed from `LORA_TARGETS["tiny"]`, trainable 0.15 M → 0.12 M.

   The DDP path and the all-rank checkpoint are verified on a local 2-rank gloo
   run (6 steps, `--ckpt-every 3`, CPU): both ranks finish, both checkpoints
   write, no hang. The FSDP branch cannot be verified locally — `RuntimeError:
   FSDP needs a non-CPU accelerator device` — so it rests on the API contract
   until the next GPU run.
8. **Kaggle, seventh attempt — the comparison ran.** `smoke` passed first
   (20 steps on all three paths, ~1 min), `smoke_fsdp/stage_b.pt` proving the
   FSDP gather works on a GPU, and `dist` produced the table at the top of this
   section. `"machine_shape": "NvidiaTeslaT4"` held the accelerator across the
   push, so no browser step was needed. Elapsed: about five minutes, against
   ~70 for every previous attempt.
9. **Colab** — one GPU per session; cannot produce the comparison at all.

## How the kernel is run now

Four of the six attempts above spent ~70 minutes re-measuring work that was
already published in order to reach the three minutes that were actually
broken — and then failed there, so the next attempt paid the 70 minutes again.
The kernel is now a menu of independent sections rather than a pipeline:

```bash
scripts/kaggle_push_src.sh            # code + a Stage A checkpoint
scripts/kaggle_run.sh --only dist     # ~5 min, not ~70
scripts/kaggle_run.sh --only all      # the whole thing, when that is the point
```

Four changes make that safe:

- **`"machine_shape": "NvidiaTeslaT4"`** in `kernel-metadata.json` pins the
  accelerator across pushes. `enable_gpu` alone silently resets to the sm_60
  P100 on every push, which is the trap in item 5.
- **The Stage A checkpoint ships in the source dataset.** `dist` needed
  `stagea` to have run first, which is why a throughput measurement was paying
  for a training run. 15 MB in the dataset against ~15 min of GPU time per run.
- **A `smoke` section runs first**: 20 steps of every distributed path with
  `--ckpt-every 10`, so a mid-run checkpoint actually happens. All three bugs in
  item 7 surface there in about a minute, and the queue stops if it fails
  rather than measuring a broken build.
- **`DAYSTORM_DIST_TIMEOUT_S`, default 120.** NCCL's default collective timeout
  is 10 minutes, so the FSDP deadlock cost 10 minutes of a session before
  printing anything. The run is dead either way; 2 minutes is still orders of
  magnitude above any collective in this model.

## Also blocked

**Gradio Space deploy.** Hosting a Gradio Space on free `cpu-basic` requires HF
**PRO**; static Spaces are free but cannot run PyTorch. The app is complete and
runs locally (`python space/app.py`) — see `space/DEPLOY.md`.
