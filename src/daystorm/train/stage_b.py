"""Stage B: unfreeze the backbone through LoRA and fit on grounded QA.

Stage A taught the projectors to emit tokens the backbone can read. Stage B
lets the backbone move too - but only through low-rank adapters, so the base
weights are untouched, the checkpoint is tens of megabytes instead of six
gigabytes, and the whole thing fits in 4-bit on a 16 GB card.

Two learning rates, deliberately. The fusion stack is already trained and only
needs refinement, so it runs an order of magnitude below the freshly
initialised LoRA adapters. Training both at the LoRA rate is the standard way
to destroy a good Stage A checkpoint in the first hundred steps.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from daystorm.data.norm import NormStats
from daystorm.data.synthetic import make_scene
from daystorm.data.tensors import FEATURE_DIMS, HashCache, NpzCache, collate
from daystorm.data.windows import build_samples, split_by_scene
from daystorm.model.fusion import DaystormFusion
from daystorm.train.backbone import backbone_kwargs, build_backbone
from daystorm.train.dist import all_reduce_mean, init_distributed, shutdown, wrap

# Attention and MLP projections. The names differ per architecture; the byte
# level stand-in exposes torch's own TransformerEncoderLayer submodules.
#
# `out_proj` is deliberately absent from the tiny row. It exists on
# nn.MultiheadAttention, but MHA never *calls* it as a module - it hands
# out_proj.weight and out_proj.bias to F.multi_head_attention_forward - so a
# LoRA adapter wrapped around it is never in the graph and never receives a
# gradient. Single-GPU training does not notice; DDP does, and refuses to start
# with "Parameter indices which did not receive grad". They were dead weights,
# not a lost capability.
LORA_TARGETS = {
    "tiny": ["linear1", "linear2"],
    "default": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def attach_lora(backbone, name: str, rank: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model

    targets = LORA_TARGETS["tiny" if name in ("tiny", "none") else "default"]
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=targets,
        task_type=None,  # a bare nn.Module, not a HF task head
    )
    return get_peft_model(backbone, cfg)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default="",
                   help="defaults to the backbone the Stage A checkpoint was trained with")
    p.add_argument("--stage-a", default="ckpt/stage_a", help="checkpoint to start from")
    p.add_argument("--scenes", type=int, default=40)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--lr-fusion", type=float, default=2e-5, help="an order below the adapters")
    p.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--cache", default="")
    p.add_argument("--out", default="ckpt/stage_b")
    p.add_argument("--ckpt-every", type=int, default=200, help="survive Colab preemption")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--shard", choices=("none", "ddp", "fsdp"), default="none",
                   help="multi-GPU strategy; launch with torchrun --nproc_per_node=N")
    p.add_argument("--dist-precision", choices=("fp16", "fp32"), default="fp16",
                   help="FSDP parameter dtype. fp32 makes FSDP comparable to DDP, "
                        "which has no mixed precision here")
    args = p.parse_args()

    info = init_distributed()
    if info.enabled:
        args.device = info.device
    # Every rank must draw a different batch, or N GPUs do one GPU of work.
    torch.manual_seed(args.seed + info.rank)
    samples = build_samples([make_scene(seed=i) for i in range(args.scenes)])
    train, val = split_by_scene(samples, val_fraction=0.2, seed=args.seed)

    stage_a = Path(args.stage_a)
    blob = torch.load(stage_a / "fusion.pt", map_location=args.device, weights_only=False)
    stats = NormStats.load(stage_a / "norm.json")  # train-split stats travel with the model
    cache = NpzCache(args.cache, FEATURE_DIMS) if args.cache else HashCache(FEATURE_DIMS)

    # The backbone Stage A used, unless overridden: mixing a Stage A projector
    # with a differently quantised backbone is silent, not an error.
    args.backbone = args.backbone or blob.get("args", {}).get("backbone", "tiny")
    backbone = build_backbone(
        args.backbone, device=args.device, **backbone_kwargs(blob.get("args", {}), args.backbone)
    )
    if blob.get("backbone") is not None:
        backbone.load_state_dict(blob["backbone"])
    d_model = backbone.d_model

    fusion = DaystormFusion(FEATURE_DIMS, d_model=d_model).to(args.device)
    fusion.load_state_dict(blob["fusion"])

    backbone = attach_lora(backbone, args.backbone, args.lora_r, args.lora_alpha, args.lora_dropout)
    n_lora = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    n_fusion = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)
    print(
        f"[rank {info.rank}/{info.world_size}] "
        f"train {len(train)} / val {len(val)} | backbone {args.backbone} d_model={d_model}\n"
        f"trainable: LoRA {n_lora / 1e6:.2f} M + fusion {n_fusion / 1e6:.2f} M "
        f"| frozen backbone {n_frozen / 1e6:.1f} M "
        f"({100 * (n_lora + n_fusion) / max(n_lora + n_fusion + n_frozen, 1):.1f}% trainable)"
    )

    opt = torch.optim.AdamW(
        [
            {"params": [q for q in backbone.parameters() if q.requires_grad], "lr": args.lr_lora},
            {"params": list(fusion.parameters()), "lr": args.lr_fusion},
        ],
        weight_decay=0.01,
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr_lora, args.lr_fusion], total_steps=args.steps, pct_start=0.1
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Bind before wrapping: DDP does not forward attribute lookups to the module
    # it wraps, so a post-wrap backbone.tokenize() raises AttributeError on both
    # ranks. FSDP happens to forward them, which is why only the DDP run died.
    tokenize = backbone.tokenize
    if args.shard != "none":
        backbone = wrap(backbone, info, args.shard, args.dist_precision)
        fusion = wrap(fusion, info, "ddp")  # 6 M params: replicate, do not shard

    rng = np.random.default_rng(args.seed + info.rank)
    started, history = time.time(), []

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.accum):
            idx = rng.choice(len(train), size=min(args.batch, len(train)), replace=False)
            batch = collate([train[i] for i in idx], cache, stats)
            feats = {k: torch.from_numpy(v).to(args.device) for k, v in batch["features"].items()}
            mask = torch.from_numpy(batch["mask"]).to(args.device)
            ids, labels = tokenize(batch["questions"], batch["answers"], args.device)
            loss = backbone(fusion(feats, mask), ids, labels) / args.accum
            loss.backward()
            total += float(loss.item()) * args.accum
        torch.nn.utils.clip_grad_norm_(
            [q for q in backbone.parameters() if q.requires_grad] + list(fusion.parameters()), 1.0
        )
        opt.step()
        sched.step()
        total = all_reduce_mean(total, info)  # log the real loss, not rank 0's
        history.append(total)

        if info.is_main and (step % max(args.steps // 10, 1) == 0 or step == 1):
            per_step = (time.time() - started) / step
            print(f"  step {step:>5}/{args.steps}  loss {total:.4f}  "
                  f"{per_step * 1000:.0f} ms/step  {time.time() - started:.0f}s")
        if step % args.ckpt_every == 0:
            _save(out, fusion, backbone, stats, args, history, info)

    _save(out, fusion, backbone, stats, args, history, info)
    if info.is_main:
        elapsed = time.time() - started
        print(f"\nsaved LoRA adapters + fusion stack to {out}")
        print(f"world_size={info.world_size}  {args.steps} steps in {elapsed:.0f}s  "
              f"({1000 * elapsed / args.steps:.0f} ms/step, "
              f"{args.batch * args.accum * info.world_size} windows/step)")
    shutdown(info)
    return 0


def _full_state_dict(module):
    """Gather one unsharded state dict, with keys matching the single-GPU run.

    Under FSDP this is a **collective** - every rank has to call it. Guarding the
    whole save with `if info.is_main` left rank 0 alone in an allgather while
    rank 1 walked on to the next step, and the NCCL watchdog took the job down
    600 s later. Unwrapping DDP matters for a different reason: its `module.`
    key prefix would make the checkpoint unloadable by the single-GPU path.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if isinstance(module, FSDP):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, cfg):
            return module.state_dict()
    return getattr(module, "module", module).state_dict()


def _save(out: Path, fusion, backbone, stats: NormStats, args, history: list[float], info) -> None:
    """Checkpoint everything trainable. Base weights are never written.

    Called by every rank - see `_full_state_dict` - but only rank 0 writes.
    """
    fusion_sd = _full_state_dict(fusion)
    backbone_sd = _full_state_dict(backbone)
    if not info.is_main:
        return
    torch.save(
        {
            "fusion": fusion_sd,
            "lora": {k: v for k, v in backbone_sd.items() if "lora_" in k},
            "backbone": backbone_sd if args.backbone in ("tiny", "none") else None,
            "args": vars(args),
        },
        out / "stage_b.pt",
    )
    stats.save(out / "norm.json")
    (out / "history.json").write_text(json.dumps({"loss": history}, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
