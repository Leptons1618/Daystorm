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

# Backbone ladder for the Stage A gate. The question is where the fusion idea
# stops working as the language model shrinks: the projectors have to make a
# frozen LM emit exact digits, and a smaller LM is a weaker decoder of them.
# All ungated, all instruction-tuned, one family per size where possible, so
# scale is the only variable that moves. 4-bit only for the 3B row - below that
# the fp16 weights fit twice over on a T4 and dequantisation is pure overhead.
SWEEP = [
    # (model id, 4-bit, dtype, output tag)
    ("tiny", False, "auto", "tiny"),
    ("HuggingFaceTB/SmolLM2-135M-Instruct", False, "auto", "SmolLM2-135M-Instruct"),
    ("HuggingFaceTB/SmolLM2-360M-Instruct", False, "auto", "SmolLM2-360M-Instruct"),
    # The fp16 row above failed with a flat loss and a NaN control. Its residual
    # stream peaks near the fp16 ceiling of 65504, so one inf makes GradScaler
    # skip every step. fp32 is the control that separates "overflowed" from
    # "too small to learn this" - see reports/phase4_backbone_ladder.md.
    ("HuggingFaceTB/SmolLM2-360M-Instruct", False, "fp32", "SmolLM2-360M-Instruct-fp32"),
    ("Qwen/Qwen2.5-0.5B-Instruct", False, "auto", "Qwen2.5-0.5B-Instruct"),
    ("Qwen/Qwen2.5-1.5B-Instruct", False, "auto", "Qwen2.5-1.5B-Instruct"),
    ("Qwen/Qwen2.5-3B-Instruct", True, "auto", "Qwen2.5-3B-Instruct"),
]


# The Kaggle API has returned this kernel's own log as 0 bytes on every run so
# far, which makes a section that failed indistinguishable from one that never
# started. Everything printed here is mirrored into an output file, which comes
# back with `kaggle kernels output` whatever the log endpoint decides to do.
CONSOLE = "/kaggle/working/console.log"


def echo(text):
    print(text, flush=True)
    with open(CONSOLE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def banner(title):
    echo(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def run(*argv, workdir="/kaggle/working"):
    echo(f"$ {' '.join(str(a) for a in argv)}")
    with open(CONSOLE, "a", encoding="utf-8") as f:
        r = subprocess.run(
            [str(a) for a in argv], cwd=workdir, env=ENV, check=False,
            stdout=f, stderr=subprocess.STDOUT,
        )
    if r.returncode != 0:
        echo(f"[exit {r.returncode}]")
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

    # Whatever the upload named the archive: `kaggle datasets version -r zip`
    # names it after the folder, and Kaggle only sometimes expands it for you.
    for archive in glob.glob("/kaggle/input/**/*.zip", recursive=True):
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
    # Kaggle's image ships torchao 0.10.0, and peft's LoRA dispatcher calls
    # is_torchao_available() unconditionally - which *raises* on a too-old
    # torchao rather than returning False, so every get_peft_model() dies with
    # an ImportError about a library this project never uses. Removing it is the
    # fix; upgrading it would drag torch along behind it.
    subprocess.run([sys.executable, "-m", "pip", "-q", "uninstall", "-y", "torchao"],
                   check=False)

    import torch

    banner("hardware")
    n = torch.cuda.device_count()
    echo(f"torch {torch.__version__}  cuda {torch.version.cuda}  devices {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        echo(f"  [{i}] {p.name}  {p.total_memory / 1e9:.1f} GB  sm_{p.major}{p.minor}")
    echo(f"bf16 supported: {torch.cuda.is_bf16_supported() if n else 'n/a'}")
    if n == 0:
        print("ERROR: no GPU. Enable the accelerator in notebook settings.", file=sys.stderr)
        return 1

    # Kaggle's default accelerator is a P100 (sm_60) and Kaggle's own preinstalled
    # torch is built for sm_70 and up. Every kernel launch then fails one section
    # at a time, which reads as six unrelated bugs rather than one wrong dropdown.
    # Fail here instead, with the fix in the message.
    #
    # Worth knowing: `kaggle kernels push` resets the accelerator to that
    # default. A configuration that worked yesterday comes back on a P100 today
    # for no reason visible from the CLI, because kernel-metadata.json can
    # request a GPU but cannot say which one.
    caps = [torch.cuda.get_device_capability(i) for i in range(n)]
    if min(caps) < (7, 0):
        worst = min(caps)
        echo(
            f"\nERROR: sm_{worst[0]}{worst[1]} is below the sm_70 floor of this torch"
            f" build.\nNothing will run on it - the GPU is present and unusable."
            f"\nSet the accelerator to 'GPU T4 x2' in the notebook settings"
            f" (Session options -> Accelerator) and re-run."
            f"\nEvery `kaggle kernels push` resets this to the default P100, so"
            f" it has to be set again after each push."
        )
        return 1

    py = sys.executable

    banner("1/6  fusion latency: dtype + nf4 quantization sweep")
    run(py, "-m", "daystorm.bench.latency", "--d-model", "2048",
        "--batch", "1", "8", "32", "--iters", "80", "--out", "/kaggle/working/latency.json")

    banner("2/6  same sweep under torch.compile")
    run(py, "-m", "daystorm.bench.latency", "--d-model", "2048",
        "--batch", "1", "8", "--iters", "60", "--compile",
        "--out", "/kaggle/working/latency_compiled.json")

    banner("3/6  ONNX export + numerical verification")
    run(py, "-m", "daystorm.bench.export_onnx", "--d-model", "2048",
        "--out", "/kaggle/working/fusion.onnx")

    banner("4/6  Stage A on GPU, then end-to-end latency")
    run(py, "-m", "daystorm.train.stage_a", "--scenes", "40", "--steps", "500",
        "--batch", "16", "--lr", "1e-3", "--out", "/kaggle/working/ckpt")
    run(py, "-m", "daystorm.bench.end_to_end", "--ckpt", "/kaggle/working/ckpt",
        "--iters", "20", "--max-new", "200", "--out", "/kaggle/working/end_to_end.json")

    banner("5/6  distributed Stage B")
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
    else:
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

    banner("6/6  backbone ladder: how small can the language model get?")
    print(
        "Same gate, same 8 windows, same 300 steps, same batch on every row, so\n"
        "backbone size is the only thing that moves. PASS means the frozen LM\n"
        "reproduced all eight answers verbatim from the fusion prefix, and failed\n"
        "to when that prefix was shuffled across the batch.\n",
        flush=True,
    )
    for model_id, four_bit, dtype, tag in SWEEP:
        print(f"\n>>> {model_id}{' (nf4)' if four_bit else ''} [{dtype}]", flush=True)
        argv = [py, "-m", "daystorm.train.stage_a", "--backbone", model_id,
                "--overfit", "8", "--steps", "300", "--batch", "4", "--scenes", "24",
                "--lr", "1e-3", "--backbone-dtype", dtype,
                "--gate-json", f"/kaggle/working/gate_{tag}.json"]
        if four_bit:
            argv.append("--load-4bit")
        # A row that OOMs or fails to download must not take the sweep down with
        # it; a missing gate_*.json is the record that the row produced nothing.
        run(*argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
