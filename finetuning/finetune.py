# Import the necessary libraries
from transformers import (
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig


# Load the quantised model and tokeniser
base_model_id = "Ministral-3-3B-Instruct-2512-bnb-nf4"
base_model = Mistral3ForConditionalGeneration.from_pretrained(
    base_model_id,
    device_map="auto",
)
tokeniser = MistralCommonBackend.from_pretrained(base_model_id)

# Load the dataset
dataset = load_dataset("json", data_files="data/examples.jsonl", split="train")


# Format the dataset
def format_sample(sample):
    messages = [
        {
            "role": "user",
            "content": (
                "Generate an example sentence for this dictionary entry:\n"
                f"<LANGUAGE>{sample["language"]}</LANGUAGE>\n"
                f"<WORD>{sample["word"]}</WORD>\n"
                f"<POS>{sample["pos"]}</POS>\n"
                f"<DEFINITION>{sample["definition"]}</DEFINITION>"
            ),
        },
        {
            "role": "assistant",
            "content": f"<EXAMPLE>{sample["example"]}</EXAMPLE>",
        },
    ]
    return {
        "text": tokeniser.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    }


formatted_dataset = dataset.map(format_sample)

# Lora Configuration
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()

# Training Configuration
training_args = SFTConfig(
    output_dir="./exgen-qlora-checkpoint",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    report_to="none",
    dataset_text_field="text",
    max_length=512,
)

# Model Training
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=formatted_dataset,
    processing_class=tokeniser,
)
trainer.train()

# Save the trained LoRA adapter and tokeniser
peft_model.save_pretrained("exgen-qlora")
tokeniser.save_pretrained("exgen-qlora")
