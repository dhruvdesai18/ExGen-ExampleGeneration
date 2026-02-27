# Import required libraries
import torch
from transformers import (
    BitsAndBytesConfig,
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
)
from peft import prepare_model_for_kbit_training


# Quantisation config for 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # load in 4-bit
    bnb_4bit_quant_type="nf4",  # normally distributed weights
    bnb_4bit_use_double_quant=True,  # quantise scale factors
    bnb_4bit_compute_dtype=torch.float16,  # 16-bit during computation
)

# Load the base model with quant config
base_model_id = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
base_model = Mistral3ForConditionalGeneration.from_pretrained(
    base_model_id, device_map="auto", quantization_config=bnb_config
)

# Load the tokenizer
tokeniser = MistralCommonBackend.from_pretrained(base_model_id)
# Enable gradient checkpointing (for limited memory)
base_model.gradient_checkpointing_enable()
# Quantisation-aware training (for improved effectiveness)
base_model = prepare_model_for_kbit_training(base_model)
# Save the quantised model and tokenizer
base_model.save_pretrained("Ministral-3-3B-Instruct-2512-bnb-nf4")
tokeniser.save_pretrained("Ministral-3-3B-Instruct-2512-bnb-nf4")
