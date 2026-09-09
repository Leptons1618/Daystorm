# KV-cached decode, end-to-end on a T4

`end_to_end.json` is transcribed verbatim from the kernel output of the
`stagea` section run after `TinyBackbone.generate_one` gained a KV cache. The
kernel has since been re-pushed for other sections, so the original output
directory is no longer retrievable from the Kaggle API; the numbers here are
the ones read out of it, not a re-run.

The pre-cache comparison is in `../run11/console.log` (decode p50 219.22 ms,
465 char/s) against this run's 185.74 ms and 549 char/s — the same 102 output
tokens in both. Analysis in `docs/findings.md`, result 10.
