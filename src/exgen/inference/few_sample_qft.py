import re

import pandas as pd
from peft import PeftModel
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

from exgen.config import load_yaml_config


N_SHOTS = 3
MODEL_CONFIG_PATH = "configs/models/ministral_qft/v1.yaml"
model_config = load_yaml_config(MODEL_CONFIG_PATH)
max_new_tokens = int(model_config["max_tokens"])
temperature = float(model_config["temperature"])
top_p = float(model_config["top_p"])

# Load base quantised model
base_model = Mistral3ForConditionalGeneration.from_pretrained(
    "Ministral-3-3B-Instruct-2512-bnb-nf4", device_map="auto"
)

# Load trained LoRA adapter and tokeniser
model = PeftModel.from_pretrained(base_model, "exgen-qlora-adapter")
tokeniser = MistralCommonBackend.from_pretrained("exgen-qlora-adapter")
print(f"Model loaded on device: {model.device}")

# Sample a random entry from the test set
df = pd.read_parquet("data/test/test_items.parquet")
entry = df.sample(n=1).iloc[0]

# Load train set and sample N same-language shots
train_df = pd.read_parquet("data/train/train_items.parquet")
shot_pool = train_df[train_df["language"] == entry["language"]]
shots = shot_pool.sample(n=N_SHOTS).to_dict("records")

# Build the prompt — prepend few-shot examples before the target
messages = []
for shot in shots:
    messages.append(
        {
            "role": "user",
            "content": (
                "Generate an example sentence for this dictionary entry:\n"
                f"<LANGUAGE>{shot["language"]}</LANGUAGE>\n"
                f"<WORD>{shot["word"]}</WORD>\n"
                f"<POS>{shot["pos"]}</POS>\n"
                f"<DEFINITION>{shot["definition"]}</DEFINITION>"
            ),
        }
    )
    messages.append(
        {
            "role": "assistant",
            "content": f"<EXAMPLE>{shot["sentence"]}</EXAMPLE>",
        }
    )

messages.append(
    {
        "role": "user",
        "content": (
            "Generate an example sentence for this dictionary entry:\n"
            f"<LANGUAGE>{entry["language"]}</LANGUAGE>\n"
            f"<WORD>{entry["word"]}</WORD>\n"
            f"<POS>{entry["pos"]}</POS>\n"
            f"<DEFINITION>{entry["definition"]}</DEFINITION>"
        ),
    }
)
messages.append(
    {
        "role": "assistant",
        "content": "<EXAMPLE>",
    }
)

inputs = tokeniser.apply_chat_template(
    messages,
    return_tensors="pt",
    return_dict=True,
    continue_final_message=True,
).to(model.device)

# Remove trailing EOS token
eos_id = tokeniser.eos_token_id
for key in inputs:
    if inputs[key].shape[-1] > 0 and inputs[key][0, -1].item() == eos_id:
        inputs[key] = inputs[key][:, :-1]

input_ids = inputs["input_ids"][0]
print(f"\nUser Prompt:\n{tokeniser.decode(input_ids, skip_special_tokens=False)}")

# Generate
output = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=True,
    temperature=temperature,
    top_p=top_p,
)[0]
response = tokeniser.decode(output[len(input_ids) :], skip_special_tokens=True)

# Extract example from tags
match = re.search(r"(.*?)\s*</EXAMPLE>", response, flags=re.DOTALL)
example = match.group(1).strip() if match else response.strip()
print(f"\nGenerated Output:\n{example}")
