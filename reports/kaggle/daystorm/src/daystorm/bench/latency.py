"""Latency, measured the way a vehicle integrator would ask for it.

Mean latency is the wrong headline: an ADAS-adjacent component is specified on
its tail, so this reports p50, p95 and p99 and treats the p99 as the number
that matters. Every configuration is measured on the same windows, after
warm-up, with the device synchronised - a CUDA timing that forgets to
synchronise measures how fast Python can enqueue work, not how fast the model
runs.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from daystorm.data.norm import NormStats
from daystorm.data.synthetic import make_scene
from daystorm.data.tensors import FEATURE_DIMS, HashCache, collate
from daystorm.data.windows import build_samples
from daystorm.model.fusion import DaystormFusion


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_calls(fn, n: int, warmup: int = 5) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync()
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return {
        "p50_ms": round(statistics.median(samples), 3),
        "p95_ms": round(samples[max(int(0.95 * len(samples)) - 1, 0)], 3),
        "p99_ms": round(samples[max(int(0.99 * len(samples)) - 1, 0)], 3),
        "mean_ms": round(statistics.fmean(samples), 3),
        "n": n,
    }


def to_nf4(model: torch.nn.Module) -> torch.nn.Module:
    """Replace every nn.Linear with a bitsandbytes 4-bit layer, in place.

    Without this the nf4 row silently measured fp16 - the config carried a
    quant flag that nothing read, so the benchmark reported a quantization
    speed-up that had never happened. A benchmark that lies in your favour is
    worse than no benchmark.
    """
    import bitsandbytes as bnb

    for name, child in list(model.named_children()):
        if isinstance(child, torch.nn.Linear):
            q = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=torch.float16,
                quant_type="nf4",
            )
            q.load_state_dict(child.state_dict())
            setattr(model, name, q)
        else:
            to_nf4(child)
    return model


def available_configs(device: str) -> list[tuple[str, dict]]:
    """Only offer configurations this machine can actually run.

    A benchmark table with rows that silently fell back to fp32 is worse than
    a shorter table, so unsupported configurations are reported as skipped
    with the reason attached.
    """
    configs: list[tuple[str, dict]] = [("fp32", {"dtype": torch.float32})]
    if device.startswith("cuda"):
        configs.append(("fp16", {"dtype": torch.float16}))
        if torch.cuda.is_bf16_supported():
            configs.append(("bf16", {"dtype": torch.bfloat16}))
        else:
            configs.append(("bf16", {"skip": "not supported (Turing: T4 has no bf16)"}))
        try:
            import bitsandbytes  # noqa: F401

            configs.append(("nf4", {"quant": "nf4"}))
        except ImportError:
            configs.append(("nf4", {"skip": "bitsandbytes not installed"}))
    else:
        for name in ("fp16", "bf16", "nf4"):
            configs.append((name, {"skip": f"requires CUDA, running on {device}"}))
    return configs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d-model", type=int, default=2048, help="backbone hidden size")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--compile", action="store_true", help="also measure torch.compile")
    ap.add_argument("--out", default="reports/latency.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    samples = build_samples([make_scene(seed=i) for i in range(8)])
    stats = NormStats.fit([s.window for s in samples])
    cache = HashCache(FEATURE_DIMS)

    results: dict = {
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else platform.processor() or "cpu",
        "torch": torch.__version__,
        "d_model": args.d_model,
        "runs": {},
    }
    print(f"device: {results['device']}  |  torch {torch.__version__}\n")

    for name, cfg in available_configs(args.device):
        if "skip" in cfg:
            print(f"{name:<10} skipped: {cfg['skip']}")
            results["runs"][name] = {"skipped": cfg["skip"]}
            continue

        quant = cfg.get("quant")
        dtype = cfg.get("dtype", torch.float16)
        model = DaystormFusion(FEATURE_DIMS, d_model=args.d_model).eval()
        if quant == "nf4":
            model = to_nf4(model).to(args.device)
            dtype = torch.float16  # activations stay fp16; weights are the 4-bit part
        else:
            model = model.to(args.device, dtype)
        if args.compile:
            model = torch.compile(model)

        results["runs"][name] = {}
        for bs in args.batch:
            batch = collate(samples[:bs], cache, stats)
            feats = {
                k: torch.from_numpy(v).to(args.device, dtype) for k, v in batch["features"].items()
            }
            mask = torch.from_numpy(batch["mask"]).to(args.device, dtype)

            with torch.no_grad():
                # bind the loop variables: a bare closure here would time
                # whatever the last iteration left behind
                timing = time_calls(
                    lambda m=model, f=feats, k=mask: m(f, k), args.iters
                )
            timing["batch"] = bs
            timing["per_window_ms"] = round(timing["p50_ms"] / bs, 3)
            results["runs"][name][f"batch{bs}"] = timing
            print(
                f"{name:<10} batch {bs:<3} "
                f"p50 {timing['p50_ms']:>8.3f} ms  p95 {timing['p95_ms']:>8.3f}  "
                f"p99 {timing['p99_ms']:>8.3f}  per-window {timing['per_window_ms']:>7.3f}"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out}")
    print("note: this times the fusion stack only. End-to-end latency adds backbone")
    print("      prefill and decode, which is measured by bench/end_to_end.py on a GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
