# Import required libraries
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training


# Quantisation config for 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # load in 4-bit
    bnb_4bit_quant_type="nf4",  # normally distributed weights
    bnb_4bit_use_double_quant=True,  # quantise scale factors
    bnb_4bit_compute_dtype=torch.float16,  # 16-bit during computation
)

# Load the base model with quant config
base_model_id = "google/gemma-3-270m"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, device_map="auto", quantization_config=bnb_config
)

# Load the tokenizer
tokeniser = AutoTokenizer.from_pretrained(base_model_id)
# Enable gradient checkpointing (for limited memory)
base_model.gradient_checkpointing_enable()
# Quantisation-aware training (for improved effectiveness)
base_model = prepare_model_for_kbit_training(base_model)
# Save the quantised model and tokenizer
base_model.save_pretrained("gemma-3-270m-bnb-nf4")
tokeniser.save_pretrained("gemma-3-270m-bnb-nf4")
