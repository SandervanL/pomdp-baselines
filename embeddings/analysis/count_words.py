import json
from collections import Counter
import re
from typing import List, Dict


def count_word_occurrences(sentences: List[str]) -> Counter:
    # Combine all sentences into a single string
    all_text = " ".join(sentences)

    # Remove non-alphabetic characters and split the text into words
    words = re.findall(r"\b\w+\b", all_text.lower())

    # Count occurrences of each word
    word_counts = Counter(words)

    return word_counts


def main(file_path: str):
    # Load JSON data from file
    with open(file_path, "r") as json_file:
        data: List[Dict[str, str]] = json.load(json_file)

    # Extract sentences from the dictionaries
    sentences = [item["sentence"] for item in data]

    # Count word occurrences
    word_occurrences = count_word_occurrences(sentences)

    # Sort and print word occurrences in descending order
    for word, count in sorted(
        word_occurrences.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"{word}\t{count}")


def filter_non_left(file_path: str):
    with open(file_path, "r") as json_file:
        data: List[Dict[str, str]] = json.load(json_file)

    sentences = [
        item["sentence"] for item in data if "left" not in item["sentence"].lower()
    ]
    print(sentences)
    print(len(data))


if __name__ == "__main__":
    main(
        "C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\repos\\pomdp-baselines\\embeddings\sentences\\all_directions.json"
    )
