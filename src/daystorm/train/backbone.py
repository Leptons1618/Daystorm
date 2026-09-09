"""Language backbones behind one interface.

``TinyBackbone`` is a byte-level decoder that trains on a CPU in seconds. It
exists so the Phase 1 overfit gate can run on a clean checkout with no
download and no GPU: the gate is a test of the *plumbing* - do fusion tokens
actually reach the loss and move it - and that question does not need 3
billion parameters to answer.

``HFBackbone`` is the real one. Same interface, so ``stage_a.py`` does not
branch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["HFBackbone", "TinyBackbone", "build_backbone"]

PAD, BOS, EOS, VOCAB = 256, 257, 258, 259


def encode(text: str, max_len: int) -> list[int]:
    ids = [BOS] + list(text.encode("utf-8"))[: max_len - 2] + [EOS]
    return ids + [PAD] * (max_len - len(ids))


class TinyBackbone(nn.Module):
    """Byte-level causal transformer. Trainable, tiny, and honest about it."""

    def __init__(self, d_model: int = 192, layers: int = 4, heads: int = 4, max_len: int = 320):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embed = nn.Embedding(VOCAB, d_model)
        self.pos = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, d_model * 4, batch_first=True, norm_first=True, dropout=0.0
        )
        self.body = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d_model, VOCAB)

    def tokenize(self, questions, answers, device):
        """Label only the answer: the model is graded on what it produces."""
        ids, labels = [], []
        for q, a in zip(questions, answers):
            q_ids = list(f"Q: {q}\nA: ".encode())
            a_ids = list(f"{a}".encode()) + [EOS]
            seq = ([BOS] + q_ids + a_ids)[: self.max_len]
            lab = ([-100] * (1 + len(q_ids)) + a_ids)[: self.max_len]
            pad = self.max_len - len(seq)
            ids.append(seq + [PAD] * pad)
            labels.append(lab + [-100] * pad)
        return (
            torch.tensor(ids, device=device),
            torch.tensor(labels, device=device),
        )

    def _step(self, x: torch.Tensor, cache: list) -> tuple[torch.Tensor, list]:
        """Push new positions through every layer, extending the KV cache.

        Torch's `nn.TransformerEncoderLayer` has no incremental-decode entry
        point, so this walks the same submodules by hand. The weights are the
        ones the layer already owns - nothing is re-parameterised, and existing
        checkpoints load unchanged.
        """
        updated = []
        for layer, (past_k, past_v) in zip(self.body.layers, cache):
            assert layer.norm_first, "pre-norm block assumed below"
            attn = layer.self_attn
            h = layer.norm1(x)
            q, k, v = F.linear(h, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
            if past_k is not None:
                k, v = torch.cat([past_k, k], 1), torch.cat([past_v, v], 1)
            updated.append((k, v))

            b, t, d = q.shape
            heads = attn.num_heads

            def split(z):
                return z.view(b, -1, heads, d // heads).transpose(1, 2)

            # is_causal only when several new positions arrive at once (the
            # prefill). A single new token attends over the whole cache, which
            # is exactly the past, so it needs no mask.
            out = F.scaled_dot_product_attention(
                split(q), split(k), split(v), is_causal=t > 1
            )
            x = x + attn.out_proj(out.transpose(1, 2).reshape(b, t, d))
            x = x + layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))
        return x, updated

    @torch.no_grad()
    def generate_one(self, prefix: torch.Tensor, question: str, max_new: int = 300) -> str:
        """Greedy decode a single window, with a KV cache. Used by the gate.

        The obvious version re-runs the whole stack over the whole sequence for
        every token - quadratic attention inside a linear loop, and it is where
        98.4% of the end-to-end latency went in `reports/phase3_status.md`.
        Caching keys and values means each step attends over the past instead of
        recomputing it.
        """
        ids = [BOS] + list(f"Q: {question}\nA: ".encode())
        tok = torch.tensor([ids], device=prefix.device)
        x = torch.cat([prefix, self.embed(tok)], dim=1)
        pos = x.shape[1]
        x = x + self.pos[:, :pos]

        cache: list = [(None, None)] * len(self.body.layers)
        produced: list[int] = []
        for _ in range(max_new):
            h, cache = self._step(x, cache)
            nxt = int(self.head(h)[0, -1].argmax())
            if nxt == EOS or pos >= self.pos.shape[1]:
                break
            produced.append(nxt)
            x = self.embed(torch.tensor([[nxt]], device=prefix.device))
            x = x + self.pos[:, pos : pos + 1]
            pos += 1
        return bytes(b for b in produced if b < 256).decode("utf-8", "replace")

    def forward(self, prefix: torch.Tensor, input_ids: torch.Tensor, labels: torch.Tensor):
        t = prefix.shape[1]
        text = self.embed(input_ids)
        x = torch.cat([prefix, text], dim=1)
        x = x + self.pos[:, : x.shape[1]]
        n = x.shape[1]
        causal = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), 1)
        h = self.body(x, mask=causal)
        # hidden at index t+j-1 predicts text token j
        logits = self.head(h[:, t - 1 : n - 1])
        return F.cross_entropy(
            logits.reshape(-1, VOCAB), labels.reshape(-1), ignore_index=-100
        )


class HFBackbone(nn.Module):
    """A frozen Hugging Face causal LM fed through ``inputs_embeds``.

    Stage A freezes this entirely; Stage B attaches LoRA adapters to it. The
    4-bit path uses fp16 compute because a free-tier T4 is Turing and has no
    bfloat16 and no FlashAttention-2 - the two settings that silently waste a
    day if you copy an A100 recipe.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        load_in_4bit: bool = True,
        dtype: str = "auto",
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # One model, one device. `device_map="auto"` spreads a model across every
        # visible GPU, and this forward assumes a single one: on a 2x T4 worker
        # it put the 0.5B NF4 embedding table on cuda:1 while `input_ids` were on
        # cuda:0, and died with "Expected all tensors to be on the same device".
        # Whether it splits depends on free memory at load time, so the same row
        # passes alone and fails after a section that left memory held - the
        # worst kind of intermittent. Pinning is also what the distributed path
        # wants: one rank owns its local device and nothing else.
        placement = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
        kwargs: dict = {"device_map": placement, "attn_implementation": "sdpa"}
        resolved = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "auto": torch.float16 if torch.cuda.is_available() else torch.float32,
        }[dtype]
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=resolved,
            )
        else:
            # Without a quantization config transformers loads in fp32, which on a
            # T4 doubles the VRAM and roughly doubles the step time for nothing.
            # Only on CUDA, though: CPU fp16 matmul has no fast kernel and falls
            # back to something an order of magnitude slower than fp32.
            #
            # fp32 is not merely the slow option. Some models carry residual-stream
            # activations within a factor of two of the fp16 ceiling of 65504 - a
            # forward pass then produces inf, every gradient is non-finite, and the
            # loss sits flat while GradScaler skips every step. SmolLM2-360M peaks
            # at 60694 on this path; see reports/phase4_backbone_ladder.md.
            kwargs["dtype"] = resolved
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            self.lm = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except TypeError:  # transformers < 4.56 spells the dtype argument differently
            if "dtype" not in kwargs:
                raise
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            self.lm = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.lm.requires_grad_(False)
        self.d_model = int(self.lm.config.hidden_size)
        self.max_len = 320

    def tokenize(self, questions, answers, device):
        ids, labels = [], []
        for q, a in zip(questions, answers):
            q_ids = self.tok(f"Q: {q}\nA: ", add_special_tokens=False).input_ids
            a_ids = self.tok(a, add_special_tokens=False).input_ids + [self.tok.eos_token_id]
            seq = (q_ids + a_ids)[: self.max_len]
            lab = ([-100] * len(q_ids) + a_ids)[: self.max_len]
            pad = self.max_len - len(seq)
            ids.append(seq + [self.tok.pad_token_id] * pad)
            labels.append(lab + [-100] * pad)
        return torch.tensor(ids, device=device), torch.tensor(labels, device=device)

    @torch.no_grad()
    def generate_one(self, prefix: torch.Tensor, question: str, max_new: int = 300) -> str:
        q = self.tok(f"Q: {question}\nA: ", return_tensors="pt").input_ids.to(prefix.device)
        embeds = torch.cat([prefix.to(self.lm.dtype), self.lm.get_input_embeddings()(q)], dim=1)
        out = self.lm.generate(
            inputs_embeds=embeds, max_new_tokens=max_new, do_sample=False,
            pad_token_id=self.tok.pad_token_id,
        )
        return self.tok.decode(out[0], skip_special_tokens=True)

    def forward(self, prefix: torch.Tensor, input_ids: torch.Tensor, labels: torch.Tensor):
        text = self.lm.get_input_embeddings()(input_ids)
        embeds = torch.cat([prefix.to(text.dtype), text], dim=1)
        prefix_labels = torch.full(
            (labels.shape[0], prefix.shape[1]), -100, device=labels.device, dtype=labels.dtype
        )
        out = self.lm(
            inputs_embeds=embeds,
            labels=torch.cat([prefix_labels, labels], dim=1),
            attention_mask=torch.ones(embeds.shape[:2], device=embeds.device, dtype=torch.long),
        )
        return out.loss


def build_backbone(name: str, device: str | None = None, **kw):
    """Build a backbone and place it on `device`.

    Placement is the caller-facing reason this helper takes a device at all.
    HFBackbone is placed by `device_map="auto"` inside `from_pretrained`, and a
    later `.to(device)` either fights that placement or, for a 4-bit model,
    raises outright - so the device is honoured for the tiny stand-in and
    ignored for the real ones. Every call site had been getting that rule
    slightly wrong on its own.
    """
    if name in ("tiny", "none"):
        kw.pop("load_in_4bit", None)
        kw.pop("dtype", None)
        model = TinyBackbone(**kw)
        return model.to(device) if device else model
    return HFBackbone(name, **kw)


def backbone_kwargs(train_args: dict, name: str) -> dict:
    """Rebuild a backbone the way Stage A built it.

    Stage A stores `vars(args)` in its checkpoint, so quantisation and dtype
    travel with the weights instead of being retyped on every later command
    line. Loading an fp16-trained projector against a 4-bit backbone produces
    numbers rather than an error, which is the failure worth designing out.
    """
    if name in ("tiny", "none"):
        return {}
    return {
        "load_in_4bit": bool(train_args.get("load_4bit", False)),
        "dtype": train_args.get("backbone_dtype", "auto"),
    }
