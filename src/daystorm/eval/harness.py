"""Run the model over a held-out split, ablate each modality, show failures.

The ablation is the load-bearing part. A multimodal model that scores well is
not evidence that it is multimodal - it may be reading one stream and ignoring
three. Zeroing each modality's validity mask in turn and re-measuring is what
turns "we fused four inputs" into a number per input.

Nothing here shares state with training: the split comes from the same seeded
scene-level partition, the normalisation statistics are loaded from the
checkpoint, and the model is never shown a validation scene during Stage A.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from daystorm.data.norm import NormStats
from daystorm.data.synthetic import make_scene
from daystorm.data.tensors import FEATURE_DIMS, HashCache, NpzCache, collate
from daystorm.data.windows import Sample, build_samples, split_by_scene
from daystorm.eval.metrics import Prediction, evaluate, numbers_in
from daystorm.model.fusion import DaystormFusion
from daystorm.train.backbone import backbone_kwargs, build_backbone

__all__ = ["ablate", "failure_gallery", "predict"]

MODALITIES = DaystormFusion.MODALITIES


@torch.no_grad()
def predict(
    fusion: DaystormFusion,
    backbone,
    samples: list[Sample],
    cache,
    stats: NormStats,
    device: str = "cpu",
    drop: tuple[str, ...] = (),
    max_new: int = 220,
) -> list[Prediction]:
    """Decode every sample, optionally with some modalities forced offline."""
    fusion.eval()
    batch = collate(samples, cache, stats)
    feats = {k: torch.from_numpy(v).to(device) for k, v in batch["features"].items()}
    mask = torch.from_numpy(batch["mask"]).to(device)

    for name in drop:
        mask[:, :, MODALITIES.index(name)] = 0.0

    prefix = fusion(feats, mask)
    out = []
    for i, s in enumerate(samples):
        text = backbone.generate_one(prefix[i : i + 1], batch["questions"][i], max_new=max_new)
        out.append(
            Prediction(
                window=s.window,
                question=s.question,
                reference=s.answer,
                prediction=text.strip(),
                event=s.event,
                ablated=drop,
            )
        )
    return out


def ablate(fusion, backbone, samples, cache, stats, device="cpu", max_new=220) -> dict:
    """Full model, then one run per modality with that modality forced offline."""
    results: dict[str, dict[str, float]] = {}
    base = predict(fusion, backbone, samples, cache, stats, device, (), max_new)
    results["full"] = evaluate(base)
    for name in MODALITIES:
        preds = predict(fusion, backbone, samples, cache, stats, device, (name,), max_new)
        results[f"no_{name}"] = evaluate(preds)
    return results, base


def failure_gallery(preds: list[Prediction], n: int = 10) -> list[dict]:
    """The worst cases, ranked by how far the numbers are off.

    A portfolio without a failure section reads as either untested or
    dishonest, and this is the section an interviewer opens first.
    """
    scored = []
    for p in preds:
        pn, rn = numbers_in(p.prediction), numbers_in(p.reference)
        err = max((abs(a - b) for a, b in zip(pn, rn)), default=0.0)
        if len(pn) != len(rn):
            err = max(err, 999.0)  # wrong shape of answer is worse than a wrong digit
        scored.append((err, p))
    scored.sort(key=lambda x: -x[0])
    return [
        {
            "worst_abs_error": round(err, 3),
            "event": p.event,
            "question": p.question,
            "reference": p.reference,
            "prediction": p.prediction,
        }
        for err, p in scored[:n]
        if err > 0
    ]


def _fmt(results: dict) -> str:
    keys = ["exact_match", "manoeuvre_accuracy", "numeric_mae", "hallucination_rate"]
    head = f"{'run':<12}" + "".join(f"{k.replace('_', ' '):>22}" for k in keys)
    lines = [head, "-" * len(head)]
    base = results.get("full", {})
    for name, m in results.items():
        row = f"{name:<12}"
        for k in keys:
            v = m.get(k, float("nan"))
            delta = ""
            if name != "full" and k in base:
                d = v - base[k]
                if abs(d) > 1e-9:
                    delta = f" ({d:+.2f})"
            row += f"{v:>16.3f}{delta:>6}"
        lines.append(row)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="ckpt/stage_a")
    ap.add_argument("--backbone", default="",
                    help="defaults to the backbone the checkpoint was trained with")
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--max-eval", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=220)
    ap.add_argument("--cache", default="")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt)
    blob = torch.load(ckpt_dir / "fusion.pt", map_location=args.device, weights_only=False)
    stats = NormStats.load(ckpt_dir / "norm.json")

    samples = build_samples([make_scene(seed=i) for i in range(args.scenes)])
    _, val = split_by_scene(samples, val_fraction=0.2, seed=args.seed)
    val = val[: args.max_eval]
    cache = NpzCache(args.cache, FEATURE_DIMS) if args.cache else HashCache(FEATURE_DIMS)
    if not args.cache:
        print(
            "WARNING: HashCache in use. Camera and audio embeddings are a fixed\n"
            "  pseudo-random function of (scene, index), so they carry no visual or\n"
            "  acoustic content - but they DO identify the scene, which a model can\n"
            "  memorise. Camera and audio ablations below are therefore measuring\n"
            "  scene-identity leakage, not perception. Re-run with --cache pointing\n"
            "  at scripts/precompute_reference.py output before quoting any number.\n"
        )

    # Same backbone, same quantisation, same dtype as Stage A - read from the
    # checkpoint rather than retyped, because a mismatch here changes every
    # number below without raising anything.
    args.backbone = args.backbone or blob.get("args", {}).get("backbone", "tiny")
    backbone = build_backbone(
        args.backbone, device=args.device, **backbone_kwargs(blob.get("args", {}), args.backbone)
    )
    if blob.get("backbone") is not None:
        backbone.load_state_dict(blob["backbone"])
    backbone.eval()
    fusion = DaystormFusion(FEATURE_DIMS, d_model=backbone.d_model).to(args.device)
    fusion.load_state_dict(blob["fusion"])

    print(f"evaluating {len(val)} held-out windows from {args.scenes} scenes "
          f"| backbone {args.backbone} | {args.device}\n")
    results, base = ablate(fusion, backbone, val, cache, stats, args.device, args.max_new)
    print(_fmt(results))

    gallery = failure_gallery(base, n=8)
    print(f"\n{len(gallery)} imperfect predictions of {len(base)}")
    for g in gallery[:3]:
        print(f"\n  [{g['event']}] worst numeric error {g['worst_abs_error']}")
        print(f"    ref:  {g['reference'][:150]}")
        print(f"    pred: {g['prediction'][:150]}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "failures.json").write_text(json.dumps(gallery, indent=2) + "\n")
    (out / "ablation.md").write_text(
        "# Modality ablation\n\n"
        "Each row forces one modality offline for the whole window by zeroing its\n"
        "validity mask, exactly as a dead sensor would. Deltas are against `full`.\n\n"
        "```\n" + _fmt(results) + "\n```\n"
    )
    print(f"\nwrote {out}/eval.json, {out}/ablation.md, {out}/failures.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
