import json
import random
import re

from peft import PeftModel
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

# Load base quantised model
base_model = Mistral3ForConditionalGeneration.from_pretrained(
    "Ministral-3-3B-Instruct-2512-bnb-nf4", device_map="auto"
)

# Load trained LoRA adapter and tokeniser
model = PeftModel.from_pretrained(base_model, "exgen-qlora-adapter")
tokeniser = MistralCommonBackend.from_pretrained("exgen-qlora-adapter")
print(f"Model loaded on device: {model.device}")

# Sample a random entry from examples.jsonl
with open("data/examples.jsonl", "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]
entry = random.choice(entries)

# Build prompt
language = entry["language"]
word = entry["word"]
pos = entry["pos"]
definition = entry["definition"]
messages = [
    {
        "role": "user",
        "content": (
            "Generate an example sentence for this dictionary entry:\n"
            f"<LANGUAGE>{language}</LANGUAGE>\n"
            f"<WORD>{word}</WORD>\n"
            f"<POS>{pos}</POS>\n"
            f"<DEFINITION>{definition}</DEFINITION>"
        ),
    },
]
inputs = tokeniser.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
).to(model.device)
input_ids = inputs["input_ids"][0]
print(f"\nUser Prompt:\n{tokeniser.decode(input_ids, skip_special_tokens=False)}")

# Generate
output = model.generate(
    **inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9
)[0]
response = tokeniser.decode(output[len(input_ids) :], skip_special_tokens=True)

# Extract example from tags
match = re.search(r"<EXAMPLE>\s*(.*?)\s*</EXAMPLE>", response, flags=re.DOTALL)
example = match.group(1).strip() if match else response.strip()
print(f"\nGenerated Output:\n{example}")
