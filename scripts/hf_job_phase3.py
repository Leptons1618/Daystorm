# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch", "numpy", "huggingface_hub",
#   "onnx", "onnxruntime", "onnxscript", "bitsandbytes",
# ]
# ///
"""Phase 3 on a real GPU: dtype sweep, quantization, compile, ONNX, end-to-end.

Run with:
    hf jobs uv run --flavor t4-small scripts/hf_job_phase3.py

t4-small is chosen deliberately over something faster: the README makes claims
about what free-tier Colab can do, and free Colab is a T4. Measuring on an A100
would produce prettier numbers about hardware nobody reading this repo has.
"""

import os
import subprocess
import sys

REPO = os.environ.get("DAYSTORM_SRC", "Lept0n5/daystorm-src")
SRC = "/tmp/daystorm"


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def main() -> int:
    import torch
    from huggingface_hub import snapshot_download

    banner("hardware")
    n = torch.cuda.device_count()
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  devices {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  {p.total_memory / 1e9:.1f} GB  sm_{p.major}{p.minor}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported() if n else 'n/a'}")
    if n == 0:
        print("ERROR: no GPU visible", file=sys.stderr)
        return 1

    snapshot_download(repo_id=REPO, repo_type="dataset", local_dir=SRC)
    sys.path.insert(0, f"{SRC}/src")

    def run(mod: str, *argv: str) -> None:
        subprocess.run(
            [sys.executable, "-m", mod, *argv],
            cwd=SRC,
            check=False,
            env={**os.environ, "PYTHONPATH": f"{SRC}/src"},
        )

    banner("fusion latency: dtype and quantization sweep")
    run("daystorm.bench.latency", "--d-model", "2048", "--batch", "1", "8", "32",
        "--iters", "80", "--out", "/tmp/latency.json")

    banner("same sweep with torch.compile")
    run("daystorm.bench.latency", "--d-model", "2048", "--batch", "1", "8",
        "--iters", "60", "--compile", "--out", "/tmp/latency_compiled.json")

    banner("ONNX export + numerical verification")
    run("daystorm.bench.export_onnx", "--d-model", "2048", "--out", "/tmp/fusion.onnx")

    banner("stage A on GPU (short) then end-to-end latency")
    run("daystorm.train.stage_a", "--scenes", "40", "--steps", "400",
        "--batch", "16", "--lr", "1e-3", "--out", "/tmp/ckpt")
    run("daystorm.bench.end_to_end", "--ckpt", "/tmp/ckpt", "--iters", "20",
        "--max-new", "200", "--out", "/tmp/end_to_end.json")

    banner("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
