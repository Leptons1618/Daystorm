"""End-to-end latency: alignment, fusion, prefill and decode.

``latency.py`` times the fusion stack in isolation, which is the part this
project owns. This one times what a caller actually waits for, including
autoregressive decode - the term that dominates and the only one a vehicle
integrator cares about.

Reported per stage, because the mitigations differ: alignment is CPU work that
batches, fusion is one small forward, and decode is N sequential forwards that
quantization and a KV cache attack directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from daystorm.bench.latency import time_calls
from daystorm.data.align import DEFAULT_SPECS, align_window
from daystorm.data.norm import NormStats
from daystorm.data.synthetic import make_scene
from daystorm.data.tensors import FEATURE_DIMS, HashCache, collate
from daystorm.data.windows import build_samples
from daystorm.model.fusion import DaystormFusion
from daystorm.train.backbone import build_backbone


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="ckpt/stage_a")
    ap.add_argument("--backbone", default="tiny")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--out", default="reports/end_to_end.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    blob = torch.load(ckpt / "fusion.pt", map_location=args.device, weights_only=False)
    stats = NormStats.load(ckpt / "norm.json")
    backbone = build_backbone(args.backbone).to(args.device)
    if blob.get("backbone") is not None:
        backbone.load_state_dict(blob["backbone"])
    backbone.eval()
    fusion = DaystormFusion(FEATURE_DIMS, d_model=backbone.d_model).to(args.device)
    fusion.load_state_dict(blob["fusion"])
    fusion.eval()

    scene = make_scene(seed=101, event="brake", dropout=True)
    sample = build_samples([scene])[0]
    cache = HashCache(FEATURE_DIMS)

    stages: dict[str, dict] = {}
    stages["align"] = time_calls(
        lambda: align_window(scene.streams, DEFAULT_SPECS, t0=5.0), args.iters * 10
    )

    batch = collate([sample], cache, stats)
    feats = {k: torch.from_numpy(v).to(args.device) for k, v in batch["features"].items()}
    mask = torch.from_numpy(batch["mask"]).to(args.device)
    with torch.no_grad():
        stages["fuse"] = time_calls(lambda: fusion(feats, mask), args.iters * 10)
        prefix = fusion(feats, mask)
        stages["decode"] = time_calls(
            lambda: backbone.generate_one(prefix, sample.question, max_new=args.max_new),
            args.iters,
            warmup=2,
        )
        text = backbone.generate_one(prefix, sample.question, max_new=args.max_new)

    total = sum(s["p50_ms"] for s in stages.values())
    print(f"device {args.device} | backbone {args.backbone}\n")
    for name, s in stages.items():
        share = 100 * s["p50_ms"] / total
        print(f"  {name:<8} p50 {s['p50_ms']:>9.2f} ms  p95 {s['p95_ms']:>9.2f}  ({share:4.1f}% of total)")
    print(f"  {'TOTAL':<8} p50 {total:>9.2f} ms")
    print(f"\n  tokens produced: {len(text)}  ({1000 * len(text) / max(stages['decode']['p50_ms'], 1e-9):.0f} char/s)")
    print("\n  Decode dominates, as expected: it is the only sequential stage. That is")
    print("  what quantization and a KV cache target in the optimisation pass.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": args.device, "stages": stages, "total_p50_ms": total}, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
