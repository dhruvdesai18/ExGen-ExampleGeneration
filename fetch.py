# Fetch examples from https://freedictionaryapi.com/

import json
import requests


# Fetch word data from the API
def fetch_word(language, word):
    url = f"https://freedictionaryapi.com/api/v1/entries/{language}/{word}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


# Extract examples from the API response and display them
def extract_examples(data):
    results = []
    for entry in data.get("entries", []):
        pos = entry.get("partOfSpeech")
        for sense in entry.get("senses", []):
            definition = sense.get("definition")
            examples = sense.get("examples", [])
            if examples:
                results.append((pos, definition, examples))
    return results


# Save results to a JSONL file
def save_to_jsonl(results, filename, language, word):
    with open(filename, "w") as f:
        for pos, definition, examples in results:
            for example in examples:
                record = {
                    "language": language,  # language of the word
                    "word": word,  # the word itself
                    "pos": pos,  # part of speech
                    "definition": definition,  # definition of the word
                    "example": example,  # example sentence using the word
                }
                f.write(json.dumps(record) + "\n")


# Main function to fetch and display examples for a given word with part of speech and definition
if __name__ == "__main__":
    language = "en"
    word = input("Enter a word: ")
    try:
        data = fetch_word(language, word)
        results = extract_examples(data)
        for pos, definition, examples in results:
            print(f"Part of Speech: {pos}")
            print(f"Definition: {definition}")
            for example in examples:
                print(f"Example: {example}")
            print()
        save_to_jsonl(
            results, "examples.jsonl", language, word
        )  # TODO: iterate over all words and languages and save to a single file
    except requests.exceptions.HTTPError as e:
        print(f"Word not found or API error: {e}")
    except Exception as e:
        print(f"Error: {e}")
