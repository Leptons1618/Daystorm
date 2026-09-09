"""Every LoRA adapter has to be in the backward graph.

A LoRA target whose module is never *called* - `nn.MultiheadAttention.out_proj`
is the one that bit us - trains nothing and shows no symptom on a single GPU.
DDP is what eventually reported it, at the cost of a Kaggle run. This is the
one-process version of that check.
"""

import torch

from daystorm.train.backbone import build_backbone
from daystorm.train.stage_b import attach_lora


def test_every_lora_parameter_receives_gradient():
    backbone = attach_lora(build_backbone("tiny"), "tiny", rank=8, alpha=16, dropout=0.0)
    ids, labels = backbone.tokenize(["how fast?"], ["12 m/s"], "cpu")
    prefix = torch.randn(1, 16, backbone.d_model)

    backbone(prefix, ids, labels).backward()

    dead = [n for n, p in backbone.named_parameters()
            if p.requires_grad and "lora_" in n and p.grad is None]
    assert not dead, f"LoRA parameters outside the graph: {dead}"
