"""Daystorm Phase 3 on Kaggle GPUs.

Free Colab and HF Jobs are both unavailable for this work - Colab gives one
GPU and HF Jobs needs pre-paid credits - so the GPU measurements for this
project come from Kaggle, which grants 2x T4 for 30 h/week at no cost. That is
also the right hardware to measure on: the README makes claims about what a
free-tier T4 can do, and an A100 would produce prettier numbers about hardware
nobody reading this repo has.

The kernel does as much as the session allows rather than refusing outright:
the single-GPU sweep always runs, and the distributed comparison runs only if
two GPUs are actually visible. A one-GPU number labelled "distributed" would
make the whole artifact worthless, so that case is reported, not faked.

Set the accelerator to "GPU T4 x2" in the notebook settings for the full run.
"""

import os
import subprocess
import sys

RAW = "/kaggle/input/daystorm-src"
SRC = "/kaggle/working/daystorm"
STEPS = int(os.environ.get("DAYSTORM_STEPS", "300"))
BATCH = int(os.environ.get("DAYSTORM_BATCH", "16"))
ENV = {**os.environ, "PYTHONPATH": f"{SRC}/src", "TOKENIZERS_PARALLELISM": "false"}


def banner(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}", flush=True)


def run(*argv, workdir="/kaggle/working"):
    print(f"$ {' '.join(str(a) for a in argv)}", flush=True)
    r = subprocess.run([str(a) for a in argv], cwd=workdir, env=ENV, check=False)
    if r.returncode != 0:
        print(f"[exit {r.returncode}]", flush=True)
    return r.returncode


def stage_source():
    """Locate the source wherever Kaggle mounted it, rather than guessing.

    The mount layout is not stable across dataset types and push methods - a
    hardcoded /kaggle/input/<slug> failed twice - so search for the package
    itself and derive the path from that.
    """
    import glob
    import shutil
    import zipfile

    for archive in glob.glob("/kaggle/input/**/src.zip", recursive=True):
        with zipfile.ZipFile(archive) as z:
            z.extractall("/kaggle/working/unzipped")

    hits = glob.glob("/kaggle/input/**/daystorm/__init__.py", recursive=True)
    hits += glob.glob("/kaggle/working/unzipped/**/daystorm/__init__.py", recursive=True)
    if not hits:
        listing = subprocess.run(
            ["find", "/kaggle/input", "-maxdepth", "4"],
            capture_output=True, text=True, check=False,
        ).stdout
        raise SystemExit(f"daystorm package not found. /kaggle/input tree:\n{listing[:4000]}")

    src_root = os.path.dirname(os.path.dirname(hits[0]))  # .../src
    os.makedirs(SRC, exist_ok=True)
    shutil.copytree(src_root, f"{SRC}/src", dirs_exist_ok=True)
    print(f"found package at {src_root}", flush=True)
    print(f"staged -> {SRC}/src : {sorted(os.listdir(f'{SRC}/src'))}", flush=True)


def main():
    stage_source()
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "bitsandbytes", "peft",
                    "onnx", "onnxruntime", "onnxscript"], check=False)

    import torch

    banner("hardware")
    n = torch.cuda.device_count()
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  devices {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  {p.total_memory / 1e9:.1f} GB  sm_{p.major}{p.minor}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported() if n else 'n/a'}")
    if n == 0:
        print("ERROR: no GPU. Enable the accelerator in notebook settings.", file=sys.stderr)
        return 1

    py = sys.executable

    banner("1/5  fusion latency: dtype + nf4 quantization sweep")
    run(py, "-m", "daystorm.bench.latency", "--d-model", "2048",
        "--batch", "1", "8", "32", "--iters", "80", "--out", "/kaggle/working/latency.json")

    banner("2/5  same sweep under torch.compile")
    run(py, "-m", "daystorm.bench.latency", "--d-model", "2048",
        "--batch", "1", "8", "--iters", "60", "--compile",
        "--out", "/kaggle/working/latency_compiled.json")

    banner("3/5  ONNX export + numerical verification")
    run(py, "-m", "daystorm.bench.export_onnx", "--d-model", "2048",
        "--out", "/kaggle/working/fusion.onnx")

    banner("4/5  Stage A on GPU, then end-to-end latency")
    run(py, "-m", "daystorm.train.stage_a", "--scenes", "40", "--steps", "500",
        "--batch", "16", "--lr", "1e-3", "--out", "/kaggle/working/ckpt")
    run(py, "-m", "daystorm.bench.end_to_end", "--ckpt", "/kaggle/working/ckpt",
        "--iters", "20", "--max-new", "200", "--out", "/kaggle/working/end_to_end.json")

    banner("5/5  distributed Stage B")
    common = ["-m", "daystorm.train.stage_b", "--backbone", "tiny", "--steps", str(STEPS),
              "--batch", str(BATCH), "--scenes", "60", "--stage-a", "/kaggle/working/ckpt"]
    print(">>> single-GPU baseline", flush=True)
    ENV["CUDA_VISIBLE_DEVICES"] = "0"
    run(py, *common, "--shard", "none", "--out", "/kaggle/working/sb_1gpu")
    ENV.pop("CUDA_VISIBLE_DEVICES")

    if n < 2:
        print(
            f"\nSKIPPED the two-GPU comparison: only {n} GPU visible.\n"
            "Set the accelerator to 'GPU T4 x2' and re-run. Not emitting a\n"
            "single-GPU number labelled as distributed.",
            flush=True,
        )
        return 0

    print("\n>>> 2 GPU, DDP", flush=True)
    run(py, "-m", "torch.distributed.run", "--nproc_per_node=2", "--standalone",
        *common, "--shard", "ddp", "--out", "/kaggle/working/sb_ddp")

    print("\n>>> 2 GPU, FSDP", flush=True)
    run(py, "-m", "torch.distributed.run", "--nproc_per_node=2", "--standalone",
        *common, "--shard", "fsdp", "--out", "/kaggle/working/sb_fsdp")

    print(
        "\nRead ms/step against windows/step: both distributed runs process 2x the\n"
        "windows per step, so equal ms/step is a 2x throughput win. FSDP is expected\n"
        "to LOSE to DDP on a model this small - it trades throughput for memory it\n"
        "does not need here - and reporting that is the point of running both.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
