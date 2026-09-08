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

    @torch.no_grad()
    def generate_one(self, prefix: torch.Tensor, question: str, max_new: int = 300) -> str:
        """Greedy decode a single window. Used by the Phase 1 gate."""
        ids = [BOS] + list(f"Q: {question}\nA: ".encode())
        produced: list[int] = []
        for _ in range(max_new):
            tok = torch.tensor([ids], device=prefix.device)
            x = torch.cat([prefix, self.embed(tok)], dim=1)
            x = x + self.pos[:, : x.shape[1]]
            n = x.shape[1]
            causal = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), 1)
            nxt = int(self.head(self.body(x, mask=causal))[0, -1].argmax())
            if nxt == EOS:
                break
            ids.append(nxt)
            produced.append(nxt)
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

        kwargs: dict = {"device_map": "auto", "attn_implementation": "sdpa"}
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


def build_backbone(name: str, **kw):
    if name in ("tiny", "none"):
        return TinyBackbone(**kw)
    return HFBackbone(name, **kw)
