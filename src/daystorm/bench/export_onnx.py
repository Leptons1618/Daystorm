"""Export the fusion stack to ONNX and prove the export is numerically faithful.

Exporting is the easy half. The half that matters is checking that the exported
graph produces the same numbers as the PyTorch module it came from: a silent
divergence here is the classic way a "deployed" model ends up behaving
differently from the one that was evaluated.

Only the fusion stack is exported. The backbone is a language model with a KV
cache and autoregressive decode - ONNX is the wrong tool for it, and the
runtimes that serve it well (vLLM, TensorRT-LLM) take the HF checkpoint
directly. Being explicit about that boundary is better than exporting
everything and pretending the hard part is solved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from daystorm.data.tensors import FEATURE_DIMS
from daystorm.model.fusion import DaystormFusion

ORDER = DaystormFusion.MODALITIES


class FusionONNX(nn.Module):
    """Positional-argument wrapper: ONNX has no notion of a dict input."""

    def __init__(self, fusion: DaystormFusion):
        super().__init__()
        self.fusion = fusion

    def forward(self, camera, can, radar, audio, mask):
        return self.fusion(
            {"camera": camera, "can": can, "radar": radar, "audio": audio}, mask
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="", help="optional stage_a checkpoint to export")
    ap.add_argument("--d-model", type=int, default=2048)
    ap.add_argument("--out", default="reports/daystorm_fusion.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    args = ap.parse_args()

    fusion = DaystormFusion(FEATURE_DIMS, d_model=args.d_model).eval()
    if args.ckpt:
        blob = torch.load(Path(args.ckpt) / "fusion.pt", map_location="cpu", weights_only=False)
        fusion = DaystormFusion(FEATURE_DIMS, d_model=blob["args"].get("d_model", args.d_model))
        fusion.load_state_dict(blob["fusion"])
        fusion.eval()

    wrapper = FusionONNX(fusion).eval()
    g = 5
    example = tuple(torch.randn(2, g, FEATURE_DIMS[m]) for m in ORDER) + (
        torch.ones(2, g, len(ORDER)),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        example,
        str(out),
        input_names=[*ORDER, "mask"],
        output_names=["tokens"],
        # Batch is dynamic; the grid is deliberately NOT. A fixed 5-point grid
        # and a fixed 16-token budget are what let the serving runtime
        # pre-allocate and what make torch.compile worth anything.
        dynamic_axes={name: {0: "batch"} for name in [*ORDER, "mask", "tokens"]},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    size_mb = out.stat().st_size / 1e6
    print(f"exported {out}  ({size_mb:.1f} MB, opset {args.opset})")

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    report = {"file": str(out), "size_mb": round(size_mb, 2), "opset": args.opset, "checks": []}

    worst = 0.0
    for batch, coverage in ((1, 1.0), (2, 1.0), (5, 0.5), (3, 0.0)):
        feats = [torch.randn(batch, g, FEATURE_DIMS[m]) for m in ORDER]
        mask = (torch.rand(batch, g, len(ORDER)) < coverage).float()
        with torch.no_grad():
            expected = wrapper(*feats, mask).numpy()
        got = sess.run(
            None,
            {**{m: f.numpy() for m, f in zip(ORDER, feats)}, "mask": mask.numpy()},
        )[0]

        delta = float(np.abs(expected - got).max())
        worst = max(worst, delta)
        ok = delta <= args.tolerance
        label = f"batch={batch} coverage={coverage:.0%}"
        print(f"  {label:<26} max|delta| {delta:.2e}  {'ok' if ok else 'FAIL'}")
        report["checks"].append({"case": label, "max_abs_delta": delta, "ok": ok})

    report["worst_abs_delta"] = worst
    report["passed"] = worst <= args.tolerance
    # next to the model, not a hardcoded reports/ that may not exist here
    summary = out.with_name(out.stem + "_export.json")
    summary.write_text(json.dumps(report, indent=2) + "\n")
    print(f"summary -> {summary}")

    if not report["passed"]:
        print(f"\nFAIL: worst divergence {worst:.2e} exceeds {args.tolerance}")
        return 1
    print(f"\nexport verified: worst divergence {worst:.2e} across 4 cases,")
    print("including a fully-masked batch, which is where the absent-token path lives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
