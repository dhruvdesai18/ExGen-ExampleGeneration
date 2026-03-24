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


# Extract part-of-speech and definitions from the API response
def extract_entries(data):
    results = []
    for entry in data.get("entries", []):
        pos = entry.get("partOfSpeech")
        for sense in entry.get("senses", []):
            definition = sense.get("definition")
            examples = sense.get("examples", [])
            # Keep only the first example when multiple are available
            example = examples[0] if examples else None

            # Skip senses that would produce incomplete records
            if pos and definition and example:
                results.append((pos, definition, example))
    return results


# Append results to a JSONL file
def save_to_jsonl(results, filename, language, word, start_id=1, mode="a"):
    next_id = start_id

    def _is_missing(value):
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        return False

    with open(filename, mode, encoding="utf-8") as f:
        for pos, definition, example in results:
            record = {
                "id": next_id,
                "language": language,  # language of the word
                "word": word,  # the word itself
                "pos": pos,  # part of speech
                "definition": definition,  # definition of the word
                "source": "freedictionary.com",
                "example": example,
            }

            # Do not write entries with any missing required field
            required_fields = [
                record["id"],
                record["language"],
                record["word"],
                record["pos"],
                record["definition"],
                record["source"],
                record["example"],
            ]
            if any(_is_missing(field) for field in required_fields):
                continue

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            next_id += 1

    return next_id


# Load dictionary CSV into a DataFrame
def load_dictionary(filepath):
    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    return df


# Main function to iterate over dictionary.csv and fetch examples
if __name__ == "__main__":

    dictionary_file = "data/dictionary.csv"
    output_file = "data/examples.jsonl"

    df = load_dictionary(dictionary_file)
    languages = list(df.columns)

    print(f"Languages: {languages}")
    print(f"Words per language: {len(df)}")

    # Clear the output file before starting
    open(output_file, "w").close()

    next_id = 1

    for _, row in df.iterrows():
        for lang in languages:
            word = row[lang]
            if pd.isna(word) or not word:
                continue
            print(f"  [{lang}] Fetching: {word} ... ", end="")
            try:
                data = fetch_word(lang, word)
                results = extract_entries(data)
                if results:
                    next_id = save_to_jsonl(
                        results,
                        output_file,
                        lang,
                        word,
                        start_id=next_id,
                    )
                    print(f"{len(results)} record(s) saved.")
                else:
                    print("no records found.")
            except requests.exceptions.HTTPError as e:
                print(f"HTTP error: {e}")
            except Exception as e:
                print(f"error: {e}")
            # time.sleep(0.1)  # delay between API calls

    print(f"\nCompleted. All examples saved to {output_file}")
