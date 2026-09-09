# Running Daystorm on free hardware

Everything in `README.md` was produced on free tiers: a laptop CPU, one Colab
T4, and Kaggle's 2x T4 worker. This is the operational detail — what fits where,
and how the GPU runs are actually launched.

## What fits on which tier

| Phase | Free tier | Notes |
|---|---|---|
| Alignment, normalisation, windowing, tests | **Yes**, no GPU | Runs on any laptop; save the GPU quota for training |
| The go/no-go gate, byte-level backbone | **Yes**, no GPU | ~2 minutes of CPU. This is why `TinyBackbone` exists |
| The gate on a real frozen backbone | **Yes**, one T4 | ~1 minute per row up to 1.5 B; the 3 B row needs NF4 |
| Stage A, real backbone, full run | **Yes**, one T4 | 600 steps on Qwen2.5-0.5B is ~4 minutes |
| Stage B QLoRA | **Marginal** on Colab | Runs, slowly. Kaggle's second GPU is the better use of the quota |
| Quantisation and latency sweep | **Yes**, one T4 | INT8/NF4/compile all fine |
| Distributed training | **No** on Colab | One GPU per session. Use **Kaggle: 2x T4, 30 h/week** |
| Storage | **Partly** | nuScenes mini + CAN expansion ≈ 4.5 GB fits in 15 GB of Drive; the 30 GB trainval slice does not |

Three T4 facts that decide the recipe, all measured in `benchmarks.md`:

- **Turing has no bfloat16 tensor cores**, and `torch.cuda.is_bf16_supported()`
  returns `True` anyway because it counts emulation. Use fp16.
- **No FlashAttention-2.** `attn_implementation="sdpa"`.
- **Frozen encoders belong on disk, not in VRAM.** `scripts/precompute_reference.py`
  caches SigLIP and Whisper outputs once.

## Colab

```bash
bash scripts/colab_bootstrap.sh    # venv, deps, CPU-or-CUDA torch
bash scripts/colab_sync.sh         # repo <-> Drive, so preemption costs nothing
bash scripts/colab_train.sh        # Stage A, checkpointing to Drive
```

Sessions get preempted, so every long run takes `--ckpt-every` and resumes by
default. Nothing that only exists on the VM is worth keeping.

## Kaggle: the 2x T4 kernel

`kaggle/daystorm_fsdp_kernel.py` is self-contained and organised as a **menu of
independent sections**, not a pipeline. That is a deliberate correction: four
early attempts each spent ~70 minutes re-measuring published work in order to
reach the three minutes that were broken, then failed there and paid the 70
minutes again on the next attempt.

```bash
bash scripts/kaggle_push_src.sh          # ship src/ + a Stage A checkpoint as a dataset
bash scripts/kaggle_run.sh --only dist   # ~5 min
bash scripts/kaggle_run.sh --only all    # everything, ~90 min
```

`kaggle_run.sh` pushes, polls until the kernel reports a terminal status, and
pulls the outputs into `reports/kaggle_run/`.

| `--only` | Time | Produces |
|---|---|---|
| `smoke` | ~1 min | 20 steps of every distributed path, `--ckpt-every 10`; the queue stops here if it fails |
| `stagea` | ~5 min | Stage A, end-to-end latency, the KV-cache comparison |
| `latency`, `onnx` | ~3 min | dtype sweep, `torch.compile`, ONNX export and verification |
| `dist` | ~4 min | 1 GPU vs DDP vs FSDP, fp32 and fp16 |
| `real` | ~20 min | Stage A on Qwen2.5-0.5B plus the held-out eval and ablation |
| `ladder` | ~51 min | the seven-row backbone ladder |
| `ladder2` | ~10 min | `opt-125m`, and NF4 at 0.5 B against fp16 |
| `nuscenes` | ~2 min | the real-data probe; trains nothing |

Four things make the menu safe, each of them a bug that had to happen first:

- **`"machine_shape": "NvidiaTeslaT4"`** in `kernel-metadata.json` pins the
  accelerator across pushes. `enable_gpu` alone silently resets to Kaggle's
  default P100, which is sm_60 — below the sm_70 floor of Kaggle's own torch
  build, so the run dies at the first CUDA call.
- **The Stage A checkpoint ships inside the source dataset.** `dist` needed
  `stagea` to have run first, which is how a throughput measurement ended up
  paying for a training run. 15 MB in a dataset against ~15 minutes of GPU time
  per attempt.
- **`smoke` runs first** and stops the queue on failure, so nothing measures a
  broken build.
- **`DAYSTORM_DIST_TIMEOUT_S`, default 120.** NCCL's default collective timeout
  is 10 minutes; an FSDP deadlock used to cost that much of a session before
  printing anything.

Push the source before pushing the kernel. Otherwise the kernel runs the
*previous* `src/daystorm`, and a newly added flag comes back as `unrecognized
arguments` from inside a subprocess whose output nobody is reading.

## One row locally, on any CUDA device

```bash
python -m daystorm.train.stage_a --overfit 8 --steps 300 --batch 4 --lr 1e-3 \
    --backbone HuggingFaceTB/SmolLM2-135M-Instruct \
    --gate-json reports/local/gate_smol135m.json
```

## Real data

`scripts/download_nuscenes.sh` fetches nuScenes under its own licence (CC
BY-NC-SA 4.0, non-commercial); nothing is redistributed in this repo. Point the
loader at it with `DAYSTORM_NUSCENES=/path/to/data`.

The v1.0-mini archive alone is not enough: the CAN bus expansion is a separate
download, and `describe_window` derives its supervision from CAN. Without it,
`load_scene` degrades loudly to an empty CAN stream and every window is dropped
before training sees it. See Finding 8 in `README.md`.

## The Gradio demo

`space/app.py` runs locally with live per-sensor ablation:

```bash
python space/app.py
```

Hosting it on Hugging Face needs a PRO account — free `cpu-basic` Gradio Spaces
are not available and static Spaces cannot run PyTorch. `space/DEPLOY.md` has
the detail.
