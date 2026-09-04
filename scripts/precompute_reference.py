"""Run the frozen encoders once and cache their outputs.

SigLIP and Whisper are frozen for the whole of Stage A, so their outputs are a
pure function of the data. Computing them inside the training loop would mean
paying for them on every epoch and holding two extra encoders in 16 GB of T4
memory that is already tight. Running them once, to disk, is the difference
between Stage A fitting on a free tier and not.

    python scripts/precompute_reference.py --scenes 40 --out cache/

Writes ``cache/{modality}/{scene}.npy`` of shape (n_assets, dim), which is the
layout ``daystorm.data.tensors.NpzCache`` expects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", type=int, default=24)
    ap.add_argument("--out", default="cache")
    ap.add_argument("--source", choices=("synthetic", "nuscenes"), default="synthetic")
    ap.add_argument("--vision-model", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--audio-model", default="openai/whisper-small")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)

    if args.source == "synthetic":
        # No images or audio exist for synthetic scenes, so this writes cache
        # files with the right shapes and a fixed seed. It exercises the exact
        # NpzCache path the real run uses; it does not produce meaningful
        # embeddings and anything reporting a metric must use --source nuscenes.
        from daystorm.data.synthetic import make_scene
        from daystorm.data.tensors import REF_DIMS

        print("[warn] synthetic source: shapes are real, embeddings are not.")
        for seed in range(args.scenes):
            scene = make_scene(seed=seed)
            for modality, dim in REF_DIMS.items():
                n = max(len(scene.streams[modality]), 1)
                rng = np.random.default_rng(abs(hash((seed, modality))) % 2**31)
                arr = (rng.standard_normal((n, dim)) * 0.5).astype(np.float32)
                _write(out / modality / f"{seed}.npy", arr)
        print(f"--> wrote {args.scenes} scenes x {len(REF_DIMS)} modalities to {out}")
        return 0

    from transformers import AutoModel, AutoProcessor

    from daystorm.data.nuscenes_src import list_scenes, load_scene

    vision = AutoModel.from_pretrained(args.vision_model, torch_dtype=torch.float16).to(device)
    vision.requires_grad_(False).eval()
    vproc = AutoProcessor.from_pretrained(args.vision_model)

    names = list_scenes()[: args.scenes]
    for i, name in enumerate(names):
        scene = load_scene(name)
        paths = getattr(scene, "asset_paths", {}).get("camera", [])
        embs = []
        for start in range(0, len(paths), args.batch):
            chunk = _load_images(paths[start : start + args.batch])
            inputs = vproc(images=chunk, return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                feats = vision.get_image_features(**inputs)
            embs.append(feats.float().cpu().numpy())
        if embs:
            _write(out / "camera" / f"{i}.npy", np.concatenate(embs).astype(np.float32))
        print(f"  [{i + 1}/{len(names)}] {name}: {len(paths)} frames")

    print(f"--> vision cache written to {out / 'camera'}")
    print("[note] audio: no public driving corpus ships in-cabin audio; run")
    print("       scripts/synthesise_audio.py first, then re-run with --audio-only.")
    return 0


def _load_images(paths):
    from PIL import Image

    return [Image.open(p).convert("RGB") for p in paths]


def _write(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


if __name__ == "__main__":
    raise SystemExit(main())
