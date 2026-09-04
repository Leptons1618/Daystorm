---
title: Daystorm
emoji: 🛞
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: Multimodal driving-scene copilot with live sensor ablation
---

# Daystorm

A four-modality driving-scene copilot: camera, CAN bus, radar tracks and cabin
audio are resampled onto a 2 Hz causal event grid and fused into a language
model that answers questions about a two-second driving window.

**Switch sensors off in the sidebar and watch the answer degrade.** That is the
whole point of the architecture, and it is the thing a static demo cannot show.

Read the limitations before believing any number: this checkpoint uses a
byte-level stand-in backbone and content-free camera/audio embeddings. The CAN
and radar paths are real. See `MODEL_CARD.md` in the repo.
