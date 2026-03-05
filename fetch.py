# Fetch examples from https://freedictionaryapi.com/

import json
import pandas as pd
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
                results.append((pos, definition, examples[0]))
    return results


# Append results to a JSONL file
def save_to_jsonl(results, filename, language, word, mode="a"):
    with open(filename, mode, encoding="utf-8") as f:
        for pos, definition, example in results:
            record = {
                "language": language,  # language of the word
                "word": word,  # the word itself
                "pos": pos,  # part of speech
                "definition": definition,  # definition of the word
                "example": example,  # example sentence using the word
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# Load dictionary CSV into a DataFrame
def load_dictionary(filepath):
    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    return df


# Main function to iterate over dictionary.csv and fetch examples
if __name__ == "__main__":
    import time

    dictionary_file = "dictionary.csv"
    output_file = "examples.jsonl"

    df = load_dictionary(dictionary_file)
    languages = list(df.columns)

    print(f"Languages: {languages}")
    print(f"Words per language: {len(df)}")

    # Clear the output file before starting
    open(output_file, "w").close()

    for _, row in df.iterrows():
        for lang in languages:
            word = row[lang]
            if pd.isna(word) or not word:
                continue
            print(f"  [{lang}] Fetching: {word} ... ", end="")
            try:
                data = fetch_word(lang, word)
                results = extract_examples(data)
                if results:
                    save_to_jsonl(results, output_file, lang, word)
                    print(f"{len(results)} example(s) saved.")
                else:
                    print("no examples found.")
            except requests.exceptions.HTTPError as e:
                print(f"HTTP error: {e}")
            except Exception as e:
                print(f"error: {e}")
            time.sleep(0.1)  # delay between API calls

    print(f"\nCompleted. All examples saved to {output_file}")
