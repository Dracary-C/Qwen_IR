# Vendored TPGDiff Runtime

This directory contains the runtime source needed by Qwen_IR's Conditional
UNet, optional IR-SDE objective, structure prior, and legacy TPGDiff adapter.

- Upstream project: TPGDiff (`leoyjTu/TPGDiff`)
- Runtime ancestry: Image Restoration SDE / DA-CLIP code by Ziwei Luo et al.
- License: see `LICENSE` in this directory
- Vendored on: 2026-07-02

The upstream `LICENSE` currently contains unresolved Git conflict markers in
its copyright line. It is preserved verbatim so the vendored snapshot does not
silently rewrite upstream attribution. Resolve the copyright holder wording
with the upstream authors before a formal release if needed.

Excluded from this snapshot:

- model checkpoints and datasets;
- Python bytecode and cache directories;
- `data/ucdpsf.pkl` (a large degradation-kernel data file not needed by the
  Qwen_IR training, evaluation, or inference paths).

