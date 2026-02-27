# Fetch examples from https://freedictionaryapi.com/

import requests


# Fetch word data from the API
def fetch_word(word, language):
    url = f"https://freedictionaryapi.com/api/v1/entries/{language}/{word}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


# Extract examples from the API response and display them
def extract_examples(data):
    examples = []
    for entry in data.get("entries", []):
        for sense in entry.get("senses", []):
            for example in sense.get("examples", []):
                examples.append(example)
    return examples


# Main function to fetch and display examples for a given word
if __name__ == "__main__":
    word = input("Enter a word: ")
    try:
        data = fetch_word(word, "en")
        examples = extract_examples(data)
        for example in examples:
            print(example)
    except requests.exceptions.HTTPError as e:
        print(f"Word not found or API error: {e}")
    except Exception as e:
        print(f"Error: {e}")
