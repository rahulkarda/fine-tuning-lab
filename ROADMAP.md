# Roadmap

## Phase 1: dataset tooling
- [x] JSONL loader with schema validation
- [x] Prompt formatter (chat templates per model family)
- [x] Dataset stats helper (token length distribution, label balance)
- [x] Train/val split with seed control

## Phase 2: training loop
- [x] Minimal HF Trainer wrapper with LoRA
- [x] Config-driven runs (one YAML per experiment)
- [x] Resume-from-checkpoint
- [x] Memory-efficient gradient checkpointing toggle

## Phase 3: eval
- [x] Loss-on-held-out-set
- [x] Generation quality probe (fixed prompt set)
- [x] Diff viewer: base model vs. tuned model on same prompts
- [x] Aggregate metrics dashboard

## Phase 4: experiments
- [ ] Phi-3-mini on instruction data
- [ ] Qwen-2.5 on a coding-style dataset
- [ ] LoRA rank sweep on one fixed dataset
- [ ] Learning rate finder

## Phase 5: polish
- [ ] CLI entrypoint
- [x] Unit tests for data utilities
- [ ] Notebook walkthrough
