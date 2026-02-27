## Requirements
- CUDA GPU >=8GB VRAM
- Python 3.13
## Setup Guide
- python -m venv .venv
- .venv/Scripts/activate
- pip install -r requirements.txt

## Quantisation
- Run quantise_ministral.py to quantise and save Ministral-3-3B-Instruct-2512-BF16 locally in NF4 precision