# reports/ — raw evidence

Machine-generated output only. Every number quoted in `README.md`,
`docs/findings.md`, `docs/benchmarks.md` and `docs/model_card.md` comes from a
file in here, and nothing in here is hand-edited (the one exception is
`kaggle/kv_cache/`, which says so in its own README).

| Path | Hardware | What it holds |
|---|---|---|
| `cpu/` | laptop CPU | first held-out eval, ablation, failure gallery, latency, ONNX check |
| `t4/` | Tesla T4 (Colab/HF) | the same harness on a GPU: dtype sweep, compile, end-to-end |
| `kaggle/run9/` | Kaggle 2x T4 | the seven-row backbone ladder |
| `kaggle/run11/` | Kaggle 2x T4 | the ladder run a second time, unchanged — the replication |
| `kaggle/dist_run1/` | Kaggle 2x T4 | first two-GPU attempt: the four bugs |
| `kaggle/dist_run2/` | Kaggle 2x T4 | FSDP vs DDP with the dtype confound removed, plus the Qwen2.5-0.5B eval |
| `kaggle/kv_cache/` | Kaggle T4 | decode with and without a KV cache |
| `kaggle/ladder2/` | Kaggle 2x T4 | `opt-125m`, and NF4 at 0.5 B against the same model in fp16 |

`ckpt/` inside a run directory is the Stage A checkpoint metadata that run
scored — kept because a metric is not reproducible without knowing which
normalisation statistics produced it.

The exported ONNX graph is not tracked: it is 7.5 MB of regenerable weights.
`python -m daystorm.bench.export_onnx` rebuilds it, and `cpu/onnx_export.json`
holds the verification numbers it must reproduce.
