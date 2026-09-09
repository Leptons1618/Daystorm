"""Multi-GPU helpers for the Stage B run.

Scope note, stated plainly because overclaiming here is easy: two T4s is not
large-scale training. What this file buys is a *real* distributed run -
process group, sharded parameters, gradient synchronisation, rank-aware
checkpointing - measured rather than described, on hardware that is free
(Kaggle gives 2x T4 for 30 h/week). The scaling write-up reports what actually
changed and says where the analogy to a 64-GPU job stops.

Why fp16 FSDP on a small backbone rather than FSDP over the 4-bit QLoRA setup:
bitsandbytes 4-bit parameters do not shard cleanly under FSDP, and the
combination is a known source of silent correctness problems. Choosing a
smaller backbone in fp16 gives a distributed run whose numbers mean something,
instead of a 4-bit one whose numbers cannot be trusted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch


@dataclass
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    enabled: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> str:
        if not torch.cuda.is_available():
            return "cpu"
        return f"cuda:{self.local_rank}"


def init_distributed() -> DistInfo:
    """Join the process group when launched under torchrun; otherwise no-op."""
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) < 2:
        return DistInfo()

    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world = int(os.environ["WORLD_SIZE"])
    # NCCL's default collective timeout is 10 minutes. A rank divergence - one
    # rank inside a collective the other never joins - costs exactly that before
    # anything is printed, and the run is dead either way. Two minutes is still
    # orders of magnitude above any collective in this model and turns a
    # deadlock into a fast, legible failure.
    timeout = timedelta(seconds=int(os.environ.get("DAYSTORM_DIST_TIMEOUT_S", "120")))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world, timeout=timeout)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return DistInfo(rank=rank, local_rank=local_rank, world_size=world, enabled=True)


def shutdown(info: DistInfo) -> None:
    if info.enabled:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()


def wrap(module: torch.nn.Module, info: DistInfo, strategy: str = "fsdp",
         precision: str = "fp16") -> torch.nn.Module:
    """Wrap for data-parallel or fully-sharded training.

    ``ddp`` replicates parameters and all-reduces gradients: right for the
    small fusion stack, where sharding 6 M parameters costs more in
    communication than it saves in memory.

    ``fsdp`` shards parameters, gradients and optimizer state across ranks:
    right for the backbone, which is the only thing here big enough to care.
    """
    if not info.enabled:
        return module

    if strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        device_ids = [info.local_rank] if torch.cuda.is_available() else None
        return DDP(module, device_ids=device_ids, find_unused_parameters=False)

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

    # T4 is Turing: fp16, never bf16.
    #
    # `precision` exists so FSDP-vs-DDP can be read as a statement about
    # sharding. DDP here has no mixed precision, so an FSDP wrapped in fp16 is
    # two changes at once, and the first measured comparison duly showed FSDP
    # *faster* than DDP on a model far too small to benefit from sharding - a
    # dtype result wearing a sharding label.
    mixed = None
    if precision == "fp16":
        mixed = MixedPrecision(
            param_dtype=torch.float16, reduce_dtype=torch.float32, buffer_dtype=torch.float16
        )
    return FSDP(
        module,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed,
        device_id=info.local_rank if torch.cuda.is_available() else None,
        use_orig_params=True,  # required for per-parameter-group learning rates
    )


def all_reduce_mean(value: float, info: DistInfo) -> float:
    """Average a scalar across ranks so the logged loss is the real one."""
    if not info.enabled:
        return value
    import torch.distributed as dist

    t = torch.tensor([value], device=info.device if torch.cuda.is_available() else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item()) / info.world_size
