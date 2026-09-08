# Phase 3 — measured on a Tesla T4

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
scene-identity leakage, not perception — see `MODEL_CARD.md`.

## Still blocked

**The 2-GPU FSDP vs DDP comparison.** It needs two GPUs and free Colab gives
one. The code is written and the single-GPU baseline is measured (70 ms/step,
16 windows/step), so the comparison is one working 2-GPU session away.

Routes and their blockers, in the order they were tried:

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
   (`reports/phase4_backbone_ladder.md`). The distributed section still emitted
   no `sb_1gpu` / `sb_ddp` / `sb_fsdp` output directory, and the Kaggle API
   returns the kernel log as 0 bytes, so the cause is not yet known. A local
   CPU reproduction of the same `stage_b` invocation against the Kaggle-produced
   Stage A checkpoint **succeeds** (5 steps, 497 ms/step, LoRA 0.15 M + fusion
   1.58 M), so this is environment-specific — torchrun or NCCL on the Kaggle
   worker — and not a bug in `stage_b`. The kernel now mirrors every child
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
6. **Colab** — one GPU per session; cannot produce the comparison at all.

## Also blocked

**Gradio Space deploy.** Hosting a Gradio Space on free `cpu-basic` requires HF
**PRO**; static Spaces are free but cannot run PyTorch. The app is complete and
runs locally (`python space/app.py`) — see `space/DEPLOY.md`.
