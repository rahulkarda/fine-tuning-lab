# fine-tuning-lab

Experiments fine-tuning small open models (Phi, Qwen, Llama-3 family) on focused datasets. Notes-as-you-go style — config files, training scripts, eval comparisons.

## Goals

- Reproducible LoRA + full-parameter fine-tuning recipes for small open models
- Eval harness that runs before/after on the same prompts so deltas are visible
- Track which hyperparameter changes actually moved the needle
- Stay on a single consumer GPU footprint (≤24GB)

## Status

Scaffolding. See [ROADMAP.md](ROADMAP.md).

## Setup

```bash
pip install -r requirements.txt
```
