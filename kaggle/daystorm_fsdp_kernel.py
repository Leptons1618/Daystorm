"""Daystorm GPU measurements on Kaggle.

Free Colab and HF Jobs are both unavailable for this work - Colab gives one
GPU and HF Jobs needs pre-paid credits - so the GPU measurements for this
project come from Kaggle, which grants 2x T4 for 30 h/week at no cost. That is
also the right hardware to measure on: the README makes claims about what a
free-tier T4 can do, and an A100 would produce prettier numbers about hardware
nobody reading this repo has.

The kernel is a menu, not a pipeline. Every section is independently runnable
and `SECTIONS` picks which ones execute, because the first four attempts at the
distributed comparison each paid ~90 minutes of already-measured work before
reaching the three minutes that were actually broken. `scripts/kaggle_run.sh
--only dist` rewrites that line before pushing.

Two rules that fall out of the same lesson:

1. **Cheap and risky first.** `smoke` is a 20-step distributed run with a forced
   mid-run checkpoint. It exercises every code path the measured `dist` section
   uses, in about a minute, and the rest of the queue is skipped if it fails.
2. **Nothing is recomputed to feed something else.** The Stage A checkpoint
   arrives in the source dataset, so `dist` does not need `stagea` to have run.
"""

import glob
import os
import subprocess
import sys

SRC = "/kaggle/working/daystorm"
UNZIP = "/kaggle/working/unzipped"
STEPS = int(os.environ.get("DAYSTORM_STEPS", "300"))
BATCH = int(os.environ.get("DAYSTORM_BATCH", "16"))
ENV = {**os.environ, "PYTHONPATH": f"{SRC}/src", "TOKENIZERS_PARALLELISM": "false"}

# Sections to run, in the order listed below. "all" runs everything.
# scripts/kaggle_run.sh --only <names> rewrites this line, so keep it one line.
SECTIONS = "smoke,dist"  # RUN_SECTIONS

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
    # The fp16 row above fails with a NaN loss, twice, reproducibly. Its residual
    # stream peaks near the fp16 ceiling of 65504, so one inf makes GradScaler
    # skip every step. fp32 is the control that separates "overflowed" from
    # "too small to learn this" - see docs/findings.md.
    ("HuggingFaceTB/SmolLM2-360M-Instruct", False, "fp32", "SmolLM2-360M-Instruct-fp32"),
    ("Qwen/Qwen2.5-0.5B-Instruct", False, "auto", "Qwen2.5-0.5B-Instruct"),
    ("Qwen/Qwen2.5-1.5B-Instruct", False, "auto", "Qwen2.5-1.5B-Instruct"),
    ("Qwen/Qwen2.5-3B-Instruct", True, "auto", "Qwen2.5-3B-Instruct"),
]

# Two rows that answer what the first sweep left open, as their own section so
# asking costs ten minutes instead of re-running the fifty-one the seven rows
# above have already cost twice.
SWEEP_EXTRA = [
    # 0.5B in NF4 against the same model in fp16 isolates what quantisation
    # costs the gate. The 3B row is the only quantised one above, so "3B passes"
    # and "NF4 passes" are currently the same measurement. Asking at 0.5B costs
    # four minutes; a 3B fp16 row would cost twenty-five and would likely OOM on
    # a T4 anyway, since 1.5B already peaks at 10.95 GB.
    ("Qwen/Qwen2.5-0.5B-Instruct", True, "auto", "Qwen2.5-0.5B-Instruct-nf4"),
    # A second family under 200M, to test whether 135M is a floor or just the
    # smallest thing tried. Not instruction-tuned, deliberately: if it passes,
    # the gate does not need instruction tuning either.
    ("facebook/opt-125m", False, "auto", "opt-125m"),
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
    import shutil
    import zipfile

    # Whatever the upload named the archives: `kaggle datasets version -r zip`
    # zips each top-level folder with paths relative to it, so two archives can
    # both hold a file called norm.json. Each unpacks into its own directory.
    for archive in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        stem = os.path.splitext(os.path.basename(archive))[0]
        with zipfile.ZipFile(archive) as z:
            z.extractall(f"{UNZIP}/{stem}")

    hits = glob.glob("/kaggle/input/**/daystorm/__init__.py", recursive=True)
    hits += glob.glob(f"{UNZIP}/**/daystorm/__init__.py", recursive=True)
    if not hits:
        listing = subprocess.run(
            ["find", "/kaggle/input", "-maxdepth", "4"],
            capture_output=True, text=True, check=False,
        ).stdout
        raise SystemExit(f"daystorm package not found. /kaggle/input tree:\n{listing[:4000]}")

    src_root = os.path.dirname(os.path.dirname(hits[0]))  # .../src
    os.makedirs(SRC, exist_ok=True)
    shutil.copytree(src_root, f"{SRC}/src", dirs_exist_ok=True)
    echo(f"found package at {src_root}")


def stage_a_dir():
    """Where Stage A's checkpoint lives, preferring the one shipped in the data.

    `dist` needs a Stage A checkpoint but does not care which one - it measures
    throughput, not quality. Shipping one in the source dataset means the
    distributed sections do not have to re-train it, which is the difference
    between a 4-minute run and a 20-minute one.
    """
    # Both places: Kaggle sometimes serves an uploaded archive expanded under
    # /kaggle/input and sometimes as the zip itself, and which one you get is
    # not stable across pushes.
    hits = glob.glob("/kaggle/input/**/fusion.pt", recursive=True)
    hits += glob.glob(f"{UNZIP}/**/fusion.pt", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    return "/kaggle/working/ckpt"


def sec_latency(py, n):
    banner("latency: dtype + nf4 quantization sweep, eager then compiled")
    run(py, "-m", "daystorm.bench.latency", "--d-model", "2048",
        "--batch", "1", "8", "32", "--iters", "80", "--out", "/kaggle/working/latency.json")
    run(py, "-m", "daystorm.bench.latency", "--d-model", "2048",
        "--batch", "1", "8", "--iters", "60", "--compile",
        "--out", "/kaggle/working/latency_compiled.json")


def sec_onnx(py, n):
    banner("ONNX export + numerical verification")
    run(py, "-m", "daystorm.bench.export_onnx", "--d-model", "2048",
        "--out", "/kaggle/working/fusion.onnx")


def sec_stagea(py, n):
    banner("Stage A on GPU, then end-to-end latency")
    run(py, "-m", "daystorm.train.stage_a", "--scenes", "40", "--steps", "500",
        "--batch", "16", "--lr", "1e-3", "--out", "/kaggle/working/ckpt")
    run(py, "-m", "daystorm.bench.end_to_end", "--ckpt", "/kaggle/working/ckpt",
        "--iters", "20", "--max-new", "200", "--out", "/kaggle/working/end_to_end.json")


def _stage_b(py, steps, batch, scenes, ckpt_every, shard, out, ranks=1, precision="fp16"):
    argv = [py]
    if ranks > 1:
        argv += ["-m", "torch.distributed.run", f"--nproc_per_node={ranks}", "--standalone"]
    argv += ["-m", "daystorm.train.stage_b", "--backbone", "tiny", "--steps", str(steps),
             "--batch", str(batch), "--scenes", str(scenes), "--ckpt-every", str(ckpt_every),
             "--stage-a", stage_a_dir(), "--shard", shard, "--out", out,
             "--dist-precision", precision]
    return run(*argv)


def sec_smoke(py, n):
    """20 steps of every distributed path, with a checkpoint forced mid-run.

    This exists because three separate bugs - an attribute DDP does not forward,
    an FSDP collective inside a rank-0 checkpoint guard, and a LoRA adapter that
    never received a gradient - each cost a full pipeline run to discover. All
    three surface here inside a minute. `--ckpt-every 10` is the point: the FSDP
    deadlock only appeared when a checkpoint landed mid-run.
    """
    banner("smoke: 20-step distributed dry run (fails fast, before anything costly)")
    if n < 2:
        echo(f"only {n} GPU visible - smoke-testing the single-GPU path alone")
    ok = _stage_b(py, 20, 4, 12, 10, "none", "/kaggle/working/smoke_1gpu") == 0
    if n >= 2:
        for shard in ("ddp", "fsdp"):
            ok &= _stage_b(py, 20, 4, 12, 10, shard, f"/kaggle/working/smoke_{shard}",
                           ranks=2) == 0
    echo(f"smoke: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def sec_dist(py, n):
    banner("distributed Stage B: 1 GPU vs DDP vs FSDP")
    # ckpt-every above steps on purpose: this section measures throughput, and a
    # mid-run checkpoint under FSDP is a collective. `smoke` already proved the
    # checkpoint path works, and the final save exercises it once more here.
    every = STEPS + 1
    echo(">>> single-GPU baseline")
    ENV["CUDA_VISIBLE_DEVICES"] = "0"
    _stage_b(py, STEPS, BATCH, 60, every, "none", "/kaggle/working/sb_1gpu")
    ENV.pop("CUDA_VISIBLE_DEVICES")

    if n < 2:
        echo(f"\nSKIPPED the two-GPU comparison: only {n} GPU visible.\n"
             "Set machine_shape to NvidiaTeslaT4 and re-run. Not emitting a\n"
             "single-GPU number labelled as distributed.")
        return 1

    # Three two-GPU rows, not two. DDP here runs in fp32, so an FSDP row in fp16
    # differs from it by dtype *and* sharding - and the first measurement duly
    # came back with FSDP faster, which is not a thing sharding does to a 2 M
    # parameter model. The fp32 FSDP row is the one that isolates sharding.
    for tag, shard, precision in (("ddp", "ddp", "fp32"),
                                  ("fsdp_fp32", "fsdp", "fp32"),
                                  ("fsdp", "fsdp", "fp16")):
        echo(f"\n>>> 2 GPU, {shard.upper()} [{precision}]")
        _stage_b(py, STEPS, BATCH, 60, every, shard, f"/kaggle/working/sb_{tag}",
                 ranks=2, precision=precision)

    echo("\nRead ms/step against windows/step: both distributed runs process 2x the\n"
         "windows per step, so equal ms/step is a 2x throughput win. FSDP is expected\n"
         "to LOSE to DDP on a model this small - it trades throughput for memory it\n"
         "does not need here - and reporting that is the point of running both.")
    return 0


def sec_real(py, n):
    """Stage A and held-out evaluation on a real frozen backbone.

    Every published held-out metric in this repo comes from `TinyBackbone`: 2 M
    randomly initialised parameters trained alongside the projectors. Those are
    the weakest numbers here. Qwen2.5-0.5B passes the gate at 6.13 GB, which
    fits a free T4, so replacing them is affordable - the eval harness now reads
    the backbone and its quantisation out of the checkpoint, so the projector is
    never scored against a differently loaded backbone than it trained on.

    Stage B is deliberately not in this section: the harness scores a Stage A
    checkpoint, so running LoRA here would burn GPU time on a checkpoint nothing
    reads.
    """
    banner("real backbone: Stage A + held-out eval, Qwen2.5-0.5B-Instruct (frozen)")
    ckpt = "/kaggle/working/ckpt_qwen05b"
    if run(py, "-m", "daystorm.train.stage_a", "--backbone", "Qwen/Qwen2.5-0.5B-Instruct",
           "--scenes", "40", "--steps", "600", "--batch", "8", "--lr", "1e-3",
           "--out", ckpt):
        echo("Stage A failed; skipping the eval that would have scored it.")
        return 1
    return run(py, "-m", "daystorm.eval.harness", "--ckpt", ckpt, "--scenes", "40",
               "--max-eval", "16", "--max-new", "220",
               "--out", "/kaggle/working/eval_qwen05b")


NUSCENES_PROBE = r"""
import glob, os, sys
roots = sorted({os.path.dirname(p) for p in
                glob.glob("/kaggle/input/**/v1.0-mini", recursive=True)})
print("candidate nuScenes roots:", roots)
if not roots:
    print("no v1.0-mini directory under /kaggle/input")
    sys.exit(1)
root = roots[0]
for sub in ("v1.0-mini", "samples", "sweeps", "maps", "can_bus"):
    path = os.path.join(root, sub)
    present = os.path.isdir(path)
    n = len(os.listdir(path)) if present else 0
    print(f"  {sub:<10} present={present} entries={n}")
os.environ["DAYSTORM_NUSCENES"] = root
try:
    from daystorm.data.nuscenes_src import list_scenes, load_scene
except Exception as e:
    print("import failed:", type(e).__name__, e)
    sys.exit(2)
try:
    names = list_scenes(root)
    print("scenes:", len(names), names[:3])
except Exception as e:
    print("list_scenes failed:", type(e).__name__, e)
    sys.exit(3)
try:
    scene = load_scene(names[0], dataroot=root)
    for k, v in scene.streams.items():
        print(f"  stream {k:<8} n={len(v)}")
    print("  asset_paths:", {k: len(v) for k, v in getattr(scene, "asset_paths", {}).items()})
except Exception as e:
    print("load_scene failed:", type(e).__name__, e)
    sys.exit(4)
print("nuScenes path is usable end to end")
"""


def sec_nuscenes(py, n):
    """Find out what a real nuScenes mount actually provides. Trains nothing.

    Real camera and audio embeddings are the largest remaining gap in this
    project - every ablation row for those two modalities currently measures
    scene-identity leakage, not perception, and the frozen 0.5B run suggests a
    bigger backbone leans on that leakage harder. Before spending GPU time on
    it, this answers the cheap questions: does the devkit install here, is the
    CAN bus expansion present (the modality the ablations say matters most), and
    does `daystorm.data.nuscenes_src` load a scene off this layout at all.
    """
    banner("nuScenes probe: is a real-data path available on this worker?")
    subprocess.run([py, "-m", "pip", "-q", "install", "nuscenes-devkit"], check=False)
    return run(py, "-c", NUSCENES_PROBE)


def sec_ladder2(py, n):
    return _ladder(py, SWEEP_EXTRA, "backbone ladder, follow-up rows")


def sec_ladder(py, n):
    return _ladder(py, SWEEP, "backbone ladder: how small can the language model get?")


def _ladder(py, sweep, title):
    banner(title)
    echo("Same gate, same 8 windows, same 300 steps, same batch on every row, so\n"
         "backbone size is the only thing that moves. PASS means the frozen LM\n"
         "reproduced all eight answers verbatim from the fusion prefix, and failed\n"
         "to when that prefix was shuffled across the batch.\n")
    for model_id, four_bit, dtype, tag in sweep:
        echo(f"\n>>> {model_id}{' (nf4)' if four_bit else ''} [{dtype}]")
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


ALL_SECTIONS = [
    ("smoke", sec_smoke),
    ("latency", sec_latency),
    ("onnx", sec_onnx),
    ("stagea", sec_stagea),
    ("dist", sec_dist),
    ("real", sec_real),
    ("nuscenes", sec_nuscenes),
    ("ladder", sec_ladder),
    ("ladder2", sec_ladder2),
]


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

    wanted = [s.strip() for s in SECTIONS.split(",") if s.strip()]
    queue = [(k, f) for k, f in ALL_SECTIONS if "all" in wanted or k in wanted]
    unknown = set(wanted) - {k for k, _ in ALL_SECTIONS} - {"all"}

    banner("hardware")
    n = torch.cuda.device_count()
    echo(f"torch {torch.__version__}  cuda {torch.version.cuda}  devices {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        echo(f"  [{i}] {p.name}  {p.total_memory / 1e9:.1f} GB  sm_{p.major}{p.minor}")
    echo(f"bf16 supported: {torch.cuda.is_bf16_supported() if n else 'n/a'}")
    echo(f"stage A checkpoint: {stage_a_dir()}")
    echo(f"sections: {' '.join(k for k, _ in queue) or '(none)'}")
    if unknown:
        echo(f"WARNING: unknown section(s) ignored: {sorted(unknown)}")
    if n == 0:
        echo("ERROR: no GPU. Set enable_gpu in kernel-metadata.json.")
        return 1

    # Kaggle's default accelerator is a P100 (sm_60) and Kaggle's own preinstalled
    # torch is built for sm_70 and up. Every kernel launch then fails one section
    # at a time, which reads as six unrelated bugs rather than one wrong dropdown.
    # Fail here instead, with the fix in the message.
    #
    # `"machine_shape": "NvidiaTeslaT4"` in kernel-metadata.json is what pins the
    # accelerator across pushes. Without it a push resets to that P100 default -
    # enable_gpu requests a GPU but cannot say which one - and the run has to be
    # started again by hand from the browser.
    caps = [torch.cuda.get_device_capability(i) for i in range(n)]
    if min(caps) < (7, 0):
        worst = min(caps)
        echo(f"\nERROR: sm_{worst[0]}{worst[1]} is below the sm_70 floor of this torch"
             f" build.\nNothing will run on it - the GPU is present and unusable."
             f'\nSet "machine_shape": "NvidiaTeslaT4" in kernel-metadata.json and'
             f" re-push,\nor set Accelerator to 'GPU T4 x2' in the notebook settings.")
        return 1

    py = sys.executable
    for i, (name, fn) in enumerate(queue, 1):
        echo(f"\n### section {i}/{len(queue)}: {name}")
        if fn(py, n) and name == "smoke":
            echo("\nSTOPPING: the smoke run failed, so every section after it would\n"
                 "be measuring a broken build. Fix, re-push, run smoke again.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
