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
from daystorm.train.backbone import build_backbone
from daystorm.train.dist import all_reduce_mean, init_distributed, shutdown, wrap

# Attention and MLP projections. The names differ per architecture; the byte
# level stand-in exposes torch's own TransformerEncoderLayer submodules.
LORA_TARGETS = {
    "tiny": ["out_proj", "linear1", "linear2"],
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
    p.add_argument("--backbone", default="tiny")
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

    backbone = build_backbone(args.backbone).to(args.device)
    if blob.get("backbone") is not None:
        backbone.load_state_dict(blob["backbone"])
    d_model = backbone.d_model

    fusion = DaystormFusion(FEATURE_DIMS, d_model=d_model).to(args.device)
    fusion.load_state_dict(blob["fusion"])

    backbone = attach_lora(backbone, args.backbone, args.lora_r, args.lora_alpha, args.lora_dropout)
    n_lora = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    n_fusion = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)
    if not info.is_main:
        pass
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
    if args.shard != "none":
        backbone = wrap(backbone, info, args.shard)
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
            ids, labels = backbone.tokenize(batch["questions"], batch["answers"], args.device)
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
        if step % args.ckpt_every == 0 and info.is_main:
            _save(out, fusion, backbone, stats, args, history)

    if info.is_main:
        _save(out, fusion, backbone, stats, args, history)
        elapsed = time.time() - started
        print(f"\nsaved LoRA adapters + fusion stack to {out}")
        print(f"world_size={info.world_size}  {args.steps} steps in {elapsed:.0f}s  "
              f"({1000 * elapsed / args.steps:.0f} ms/step, "
              f"{args.batch * args.accum * info.world_size} windows/step)")
    shutdown(info)
    return 0


def _save(out: Path, fusion, backbone, stats: NormStats, args, history: list[float]) -> None:
    """Checkpoint everything trainable. Base weights are never written."""
    torch.save(
        {
            "fusion": fusion.state_dict(),
            "lora": {k: v for k, v in backbone.state_dict().items() if "lora_" in k},
            "backbone": backbone.state_dict() if args.backbone in ("tiny", "none") else None,
            "args": vars(args),
        },
        out / "stage_b.pt",
    )
    stats.save(out / "norm.json")
    (out / "history.json").write_text(json.dumps({"loss": history}, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
