import json
import random

def generate_prompt_from_fetched_data(target_word, jsonl_file="examples.jsonl", level="intermediate"):
    examples = []

    try:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # Ensure case-insensitive matching (e.g., "Bank" == "bank")
                if data["word"].strip().lower() == target_word.strip().lower():
                    examples.append(data["example"])
    except FileNotFoundError:
        # IMPROVEMENT 1: Raise an actual error so the AI doesn't read the error text
        raise FileNotFoundError(f"{jsonl_file} not found. Please run fetch.py first.")

    if not examples:
        raise ValueError(f"No examples found for '{target_word}' in the fetched data.")

    # IMPROVEMENT 3: Pick up to 3 random examples for better variety
    shots = random.sample(examples, min(3, len(examples)))

    formatted_examples = "\n".join(
        [f"{i+1}. {example}" for i, example in enumerate(shots)]
    )

    # IMPROVEMENT 2: Split into "system" and "user" to perfectly match Cathal's script
    system_instruction = "You are an expert English teacher."
    
    user_prompt = f"""Generate exactly 5 sentences using the word '{target_word}' suitable for an {level} learner.

Dictionary examples:
{formatted_examples}

Rules:
- Make the context very clear
- Someone should guess the word even if hidden
- Number the sentences 1 to 5
"""

    # This format plugs directly into Mistral's apply_chat_template!
    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt}
    ]


if __name__ == "__main__":
    word = input("Enter target word: ")
    
    try:
        messages = generate_prompt_from_fetched_data(word)
        print("\nGenerated Few-Shot Messages for Mistral:\n")
        # Pretty print the dictionary so it's easy to read in the terminal
        print(json.dumps(messages, indent=2))
    except Exception as e:
        print(f"\n❌ Script Stopped: {e}")