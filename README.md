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

## CLI Usage

Run experiments and dataset stats from the command line:

```bash
python -m src.cli train --config configs/my_experiment.yaml
```
- Loads experiment config YAML (see example in configs/)
- Prints config and (currently) stub for training
- Add `--dry` to print config and exit without running training

Print quick dataset stats:

```bash
python -m src.cli stats --config configs/my_experiment.yaml
```
- Loads dataset from path specified in config
- Prints token length distribution, label balance, and other stats

## Example configs

See `configs/` for sample experiment YAMLs. Edit for your own datasets/models.
