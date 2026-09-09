"""The KV-cached decode must produce exactly what recomputation produced.

`TinyBackbone.generate_one` walks the transformer layers by hand so it can keep
keys and values between steps. That is the kind of optimisation that is easy to
get subtly wrong - an off-by-one in the positional slice, a missing causal mask
during prefill - and wrong here means every generated string changes while
nothing raises.
"""

import torch

from daystorm.train.backbone import BOS, EOS, TinyBackbone


def _recompute(bb, prefix, question, max_new):
    """The obvious implementation: re-run everything, every token."""
    ids = [BOS] + list(f"Q: {question}\nA: ".encode())
    produced = []
    for _ in range(max_new):
        x = torch.cat([prefix, bb.embed(torch.tensor([ids]))], dim=1)
        x = x + bb.pos[:, : x.shape[1]]
        n = x.shape[1]
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
        nxt = int(bb.head(bb.body(x, mask=causal))[0, -1].argmax())
        if nxt == EOS:
            break
        ids.append(nxt)
        produced.append(nxt)
    return bytes(b for b in produced if b < 256).decode("utf-8", "replace")


def test_cached_decode_matches_recomputation():
    torch.manual_seed(0)
    bb = TinyBackbone(d_model=32, layers=2, heads=4, max_len=64).eval()
    prefix = torch.randn(1, 16, 32)

    for question in ("what is the ego speed?", "any obstacle ahead?"):
        assert bb.generate_one(prefix, question, max_new=24) == _recompute(
            bb, prefix, question, max_new=24
        )
