# Import the necessary libraries

from transformers import (
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
)
from peft import PeftModel


# Load the base quantised model
base_model_id = "Ministral-3-3B-Instruct-2512-bnb-nf4"
base_model = Mistral3ForConditionalGeneration.from_pretrained(
    base_model_id, device_map="auto"
)

# Load the trained LoRA adapter and tokeniser
exgen_qlora_id = "exgen-qlora"
model = PeftModel.from_pretrained(base_model, exgen_qlora_id)
tokeniser = MistralCommonBackend.from_pretrained(exgen_qlora_id)

# Print device information
print(f"Model loaded on device: {model.device}")

# Generate an example for "fire"
messages = [
    {
        "role": "user",
        "content": (
            "Generate an example sentence for this dictionary entry:\n"
            "<LANGUAGE>en</LANGUAGE>\n"
            "<WORD>fire</WORD>\n"
            "<POS>noun</POS>\n"
            "<DEFINITION>(countable) An instance of this chemical reaction, especially when intentionally created and maintained in a specific location to a useful end (such as a campfire or a hearth fire).</DEFINITION>"
        ),
    },
]
user_prompt = tokeniser.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
).to(model.device)
output = model.generate(
    **user_prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9
)[0]
response = tokeniser.decode(
    output[len(user_prompt["input_ids"][0]) :], skip_special_tokens=True
)
# Extract the example from <EXAMPLE>...</EXAMPLE> tags if present
if "<EXAMPLE>" in response and "</EXAMPLE>" in response:
    example = response.split("<EXAMPLE>")[1].split("</EXAMPLE>")[0]
else:
    example = response.strip()
print(f"\nExample: {example}")
