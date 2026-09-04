# Deploying the Space

The demo is complete and runs locally:

```bash
uv pip install gradio
python space/app.py
```

Deploying it to Hugging Face is **blocked on account tier, not on code**:
hosting a Gradio Space on free `cpu-basic` requires a PRO subscription
(static Spaces are free; a PyTorch demo cannot be static). The attempt and its
exact error are recorded here rather than quietly dropped from the plan.

Once PRO is active:

```bash
hf repos create <user>/daystorm --type space --sdk gradio --exist-ok
mkdir -p /tmp/space && cp -r space/* src ckpt /tmp/space/
hf upload <user>/daystorm /tmp/space . --type space
```

The Space bundles `src/` and `ckpt/stage_a` (15 MB) directly, so it has no
runtime dependency on this repo.
