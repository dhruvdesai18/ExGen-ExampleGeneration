## Requirements
- CUDA GPU >= 4GB VRAM
- Python 3.14
- [`uv`](https://docs.astral.sh/uv/)

## Setup Guide
- `uv sync`
- Install the CUDA training/inference stack only on supported platforms with `uv sync --extra gpu`

## Common Commands
- Run ad-hoc scripts with `uv run python <path-to-script>.py`
- Open a project shell with `uv run -- python`

## Quantisation
- Run `uv sync --extra gpu` first on a supported CUDA platform
- Run `uv run python quantisation/quantise_ministral.py` to quantise and save Ministral-3-3B-Instruct-2512-BF16 locally in NF4 precision
- Host inference using transformers in `uv run python inference/quantised_inference.py`

## Fine-tuning
- Run `uv sync --extra gpu` first on a supported CUDA platform
- Run `uv run python finetuning/finetune.py` to train the LoRA adapter for the quantised model
- Run tensorboard --logdir ./exgen-qlora-checkpoint/runs
- Run sample inference with `uv run python inference/exgen_qlora_inference.py`
- Full test split generation with `uv run python inference/full_qft_generation.py`
