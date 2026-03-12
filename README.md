## Requirements
- CUDA GPU >= 4GB VRAM
- Python 3.13

## Setup Guide
- python -m venv .venv
- .venv/Scripts/activate
- pip install -r requirements.txt

## Quantisation
- Run quantise_ministral.py to quantise and save Ministral-3-3B-Instruct-2512-BF16 locally in NF4 precision
- Host inference using transformers in quantised_inference.py

## Fine-tuning
- Run finetune.py to train the LoRA adapter for the quantised model
- Host inference using transformers in qlora_inference.py