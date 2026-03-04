# Import the necessary libraries

from transformers import (
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
)


# Load the base quantised model and tokeniser
model_id = "Ministral-3-3B-Instruct-2512-bnb-nf4"
model = Mistral3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")
tokeniser = MistralCommonBackend.from_pretrained(model_id)

# Print device information
print(f"Model loaded on device: {model.device}")

# Take user input and generate a response
while True:
    messages = [{"role": "user", "content": input("Enter your prompt: ")}]
    user_prompt = tokeniser.apply_chat_template(
        messages, return_tensors="pt", return_dict=True
    ).to(model.device)
    output = model.generate(
        **user_prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9
    )[0]
    print(
        tokeniser.decode(
            output[len(user_prompt["input_ids"][0]) :], skip_special_tokens=True
        )
    )
