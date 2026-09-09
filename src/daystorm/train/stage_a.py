"""Stage A: teach the projectors to speak the backbone's token language.

Everything is frozen except the fusion stack - two new encoders, four pools,
the projector. The backbone never updates in this stage; if the loss falls,
it fell because the fusion tokens became informative, which is the only thing
Stage A is trying to establish.

The ``--overfit`` gate is the Phase 1 go/no-go. Before spending six hours of
T4 time, prove the model can drive the loss to ~0 on a handful of windows. A
stack that cannot memorise eight examples has a plumbing bug - a detached
tensor, a mask inverted, a projector that never receives gradient - and no
amount of training data will fix it.

A loss near zero is not sufficient on its own, though, and neither is a loss
margin against a control. These answers are ~97% shared boilerplate; only a
few digits carry sensor information, so cross-entropy averaged over the whole
string dilutes the signal into noise - a run where the sensors are ignored
still lands within 0.02 of one where they are not.

The gate therefore decodes. It greedily generates each answer twice: once
with the real fusion prefix, and once with the prefix shuffled across the
batch so every window is paired with another window's tokens. Exact match
is not dilutable - getting "10.1 to 8.8 m/s" right requires the digits, and
the digits only exist in the sensors. PASS needs near-perfect reconstruction
with the real prefix and clear failure without it.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from daystorm.data.norm import NormStats
from daystorm.data.synthetic import make_scene
from daystorm.data.tensors import FEATURE_DIMS, HashCache, NpzCache, collate
from daystorm.data.windows import Sample, build_samples, split_by_scene
from daystorm.model.fusion import DaystormFusion
from daystorm.train.backbone import build_backbone


def gate_samples(train: list[Sample], n: int) -> list[Sample]:
    """Pick N windows that only the sensors can tell apart.

    The naive choice - the first N training samples - is worthless here.
    Windows slide with 1 s stride over a 2 s span, so consecutive samples
    from one scene are near-duplicates that share a question and very nearly
    share an answer; a model that always emits the majority answer scores
    ~0 with the fusion path severed, and the gate passes something broken.

    So: take the largest question group, keep one sample per distinct answer,
    and spread across scenes. Every sample then carries the same prompt and a
    different target, which leaves the sensor tokens as the only thing in the
    input capable of telling them apart.
    """
    from collections import defaultdict

    by_question: dict[str, list[Sample]] = defaultdict(list)
    for s in train:
        by_question[s.question].append(s)
    group = max(by_question.values(), key=len)

    seen_answers: set[str] = set()
    seen_scenes: set[int] = set()
    picked: list[Sample] = []
    for pass_no in (0, 1):  # first pass: one per scene; second: fill the rest
        for s in group:
            if len(picked) >= n:
                break
            if s.answer in seen_answers:
                continue
            if pass_no == 0 and s.scene_seed in seen_scenes:
                continue
            picked.append(s)
            seen_answers.add(s.answer)
            seen_scenes.add(s.scene_seed)
    return picked


def exact_match(fusion, backbone, samples, cache, stats, args, shuffle: bool) -> float:
    """Fraction of windows whose answer is reproduced verbatim by greedy decode."""
    fusion.eval()
    batch = collate(samples, cache, stats)
    feats, mask = to_torch(batch, args.device)
    with torch.no_grad():
        prefix = fusion(feats, mask)
        if shuffle:
            prefix = prefix[torch.roll(torch.arange(prefix.shape[0]), 1)]
        hits = sum(
            backbone.generate_one(prefix[i : i + 1], batch["questions"][i]).strip()
            == batch["answers"][i].strip()
            for i in range(len(samples))
        )
    fusion.train()
    return hits / len(samples)


def to_torch(batch: dict, device: str):
    feats = {k: torch.from_numpy(v).to(device) for k, v in batch["features"].items()}
    return feats, torch.from_numpy(batch["mask"]).to(device)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default="tiny", help="'tiny' or a HF model id")
    p.add_argument("--load-4bit", action="store_true",
                   help="NF4-quantise a pretrained backbone; fp16 otherwise")
    p.add_argument("--backbone-dtype", default="auto", choices=("auto", "fp16", "fp32", "bf16"),
                   help="'auto' is fp16 on CUDA, fp32 on CPU. Use fp32 if the loss "
                        "sits flat: some models overflow fp16 in the residual stream")
    p.add_argument("--scenes", type=int, default=24)
    p.add_argument("--overfit", type=int, default=0, help="run the go/no-go gate on N samples")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--cache", default="", help="NpzCache root; empty uses HashCache")
    p.add_argument("--out", default="ckpt/stage_a")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gate-threshold", type=float, default=0.05)
    p.add_argument("--gate-exact", type=float, default=0.90,
                   help="required exact-match rate with the real fusion prefix")
    p.add_argument("--gate-control-max", type=float, default=0.50,
                   help="maximum exact-match rate tolerated with a shuffled prefix")
    p.add_argument("--gate-json", default="",
                   help="write the gate verdict and cost of this backbone here")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    t_start = time.time()
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(args.seed)
    scenes = [make_scene(seed=i) for i in range(args.scenes)]
    samples = build_samples(scenes)
    train, val = split_by_scene(samples, val_fraction=0.2, seed=args.seed)
    stats = NormStats.fit([s.window for s in train], split="train")
    cache = NpzCache(args.cache, FEATURE_DIMS) if args.cache else HashCache(FEATURE_DIMS)

    if args.overfit:
        train = gate_samples(train, args.overfit)
        val = train
        n_ans = len({s.answer for s in train})
        n_q = len({s.question for s in train})
        print(
            f"[gate] {len(train)} samples | {n_q} distinct question(s) | "
            f"{n_ans} distinct answers | scenes {sorted({s.scene_seed for s in train})}"
        )
        if n_ans < len(train):
            print("[gate] WARNING: duplicate answers weaken the control; add scenes")

    # A pretrained backbone is frozen in Stage A; the byte-level stand-in has no
    # pretrained language prior at all, so freezing it would leave nothing able
    # to decode and the gate would fail for a reason that says nothing about
    # the fusion stack.
    freeze_backbone = args.backbone not in ("tiny", "none")

    def run(shuffle_prefix: bool) -> float:
        torch.manual_seed(args.seed)
        kw = (
            {}
            if args.backbone in ("tiny", "none")
            else {"load_in_4bit": args.load_4bit, "dtype": args.backbone_dtype}
        )
        backbone = build_backbone(args.backbone, **kw)
        if not freeze_backbone:
            backbone = backbone.to(args.device)  # HFBackbone is placed by device_map
        backbone.requires_grad_(not freeze_backbone)
        fusion = DaystormFusion(FEATURE_DIMS, d_model=backbone.d_model).to(args.device)

        params = list(fusion.parameters())
        if not freeze_backbone:
            params += list(backbone.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        rng = np.random.default_rng(args.seed)
        # A frozen pretrained backbone runs in fp16 (a T4 has no bf16). Gradients
        # travel through it to reach the fp32 fusion stack, and fp16 gradients
        # underflow to zero unscaled - the loss then sits flat and reads as a
        # fusion bug rather than the numerics problem it is.
        scaler = torch.amp.GradScaler(
            "cuda", enabled=freeze_backbone and str(args.device).startswith("cuda")
        )
        started, last = time.time(), float("nan")
        tag = "control" if shuffle_prefix else "fused  "
        skipped = nonfinite = 0

        for step in range(1, args.steps + 1):
            idx = rng.choice(len(train), size=min(args.batch, len(train)), replace=False)
            batch = collate([train[i] for i in idx], cache, stats)
            feats, mask = to_torch(batch, args.device)
            ids, labels = backbone.tokenize(
                batch["questions"], batch["answers"], args.device
            )

            prefix = fusion(feats, mask)
            if shuffle_prefix:
                # Pair every window with a different window's tokens.
                roll = torch.roll(torch.arange(prefix.shape[0], device=prefix.device), 1)
                prefix = prefix[roll]
            loss = backbone(prefix, ids, labels)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            prev_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            sched.step()
            last = float(loss.item())

            # GradScaler halves the scale and *skips* the optimizer step on any
            # non-finite gradient. A few of those at the start is the scale
            # calibrating; a long unbroken run of them means every forward is
            # overflowing, the weights are not moving, and the loss sits flat or
            # goes NaN with nothing in the output to say why. Diagnosing that
            # from the outside cost a CPU activation probe and a confirming GPU
            # run - see docs/findings.md, result 2.
            skipped = skipped + 1 if scaler.get_scale() < prev_scale else 0
            nonfinite = nonfinite + 1 if not math.isfinite(last) else 0
            if 25 in (skipped, nonfinite):
                print(f"  [{tag}] WARNING: 25 consecutive steps with "
                      f"{'skipped updates' if skipped else 'a non-finite loss'}. "
                      f"This backbone is overflowing fp16 in its residual stream; "
                      f"re-run with --backbone-dtype fp32.", flush=True)

            if step % max(args.steps // 8, 1) == 0 or step == 1:
                print(f"  [{tag}] step {step:>5}/{args.steps}  loss {last:.4f}  "
                      f"{time.time() - started:.0f}s")

        exact = float("nan")
        if args.overfit:
            exact = exact_match(fusion, backbone, train, cache, stats, args, shuffle_prefix)
            print(f"  [{tag}] exact match {exact:.0%} on {len(train)} windows")
        if not shuffle_prefix:
            run.fusion, run.backbone = fusion, backbone  # type: ignore[attr-defined]
        return last, exact

    probe = DaystormFusion(FEATURE_DIMS, d_model=1)
    trainable = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    print(
        f"train {len(train)} / val {len(val)} samples | backbone {args.backbone} "
        f"| fusion ~{trainable / 1e6:.2f} M trainable | frozen={freeze_backbone} "
        f"| {args.device}"
    )

    last, exact = run(shuffle_prefix=False)
    fusion = run.fusion  # type: ignore[attr-defined]
    backbone = run.backbone  # type: ignore[attr-defined]

    if args.overfit:
        ctrl_loss, ctrl_exact = run(shuffle_prefix=True)
        ok = (
            last < args.gate_threshold
            and exact >= args.gate_exact
            and ctrl_exact <= args.gate_control_max
        )
        print(f"\n[gate] fused    loss {last:.4f}   exact {exact:.0%}   "
              f"(need loss < {args.gate_threshold}, exact >= {args.gate_exact:.0%})")
        print(f"[gate] control  loss {ctrl_loss:.4f}   exact {ctrl_exact:.0%}   "
              f"(need exact <= {args.gate_control_max:.0%})")
        print(f"[gate] {'PASS' if ok else 'FAIL'}")
        if args.gate_json:
            cuda = str(args.device).startswith("cuda")
            Path(args.gate_json).write_text(
                json.dumps(
                    {
                        "backbone": args.backbone,
                        "d_model": int(backbone.d_model),
                        "load_in_4bit": bool(args.load_4bit),
                        "backbone_dtype": args.backbone_dtype,
                        "backbone_params": sum(q.numel() for q in backbone.parameters()),
                        "fusion_params": sum(q.numel() for q in fusion.parameters()),
                        "samples": len(train),
                        "steps": args.steps,
                        "batch": args.batch,
                        "fused_loss": last,
                        "fused_exact": exact,
                        "control_loss": ctrl_loss,
                        "control_exact": ctrl_exact,
                        "pass": bool(ok),
                        "wall_s": round(time.time() - t_start, 1),
                        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
                        if cuda
                        else None,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"[gate] verdict -> {args.gate_json}")
        if last >= args.gate_threshold or exact < args.gate_exact:
            print("[gate] cannot reconstruct a handful of windows: look for a detached")
            print("[gate] tensor, an inverted mask, or a projector with no gradient.")
        elif ctrl_exact > args.gate_control_max:
            print("[gate] the answers are recoverable without the sensors - the task is")
            print("[gate] guessable from the prompt and fusion is decorative. Fix the")
            print("[gate] gate sample selection before trusting any Stage A result.")
        return 0 if ok else 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # The byte-level backbone trains jointly, so it is part of the checkpoint.
    # A pretrained backbone is frozen and reloaded by id instead.
    torch.save(
        {
            "fusion": fusion.state_dict(),
            "backbone": None if freeze_backbone else backbone.state_dict(),
            "args": vars(args),
        },
        out / "fusion.pt",
    )
    stats.save(out / "norm.json")
    (out / "meta.json").write_text(
        json.dumps(
            {"final_loss": last, "trainable_params": trainable, "backbone": args.backbone},
            indent=2,
        )
        + "\n"
    )
    print(f"\nsaved fusion stack + train-split norm stats to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
