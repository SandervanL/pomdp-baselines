import json
import re
from collections import defaultdict

with open("words.json", "r") as file:
    words: list[str] = [
        word[0] for word_group in json.load(file) for word in word_group
    ]

with open("directions.json", "r") as file:
    directions: list[str] = [
        direction
        for direction_group in json.load(file)
        for direction in direction_group
    ]


def create_hash(task: dict) -> int:
    word_index = words.index(task["word"][0])  # 0 - 120
    direction_index = directions.index(task["direction"])  #  0 - 16
    blocked = int(task["blocked"])
    negation = int(task["negation"])

    return (word_index << 7) | (direction_index << 2) | (blocked << 1) | negation


def odd_one_out(file_path: str):
    with open(file_path, "r") as file:
        tasks: list[dict] = json.load(file)

    hash_to_obj = {}
    hash_to_sentence = defaultdict(lambda: [])

    for task in tasks:
        hash = create_hash(task)
        hash_to_obj[hash] = task
        hash_to_sentence[hash].append(task["sentence"])

    for hash_to_sentence, sentences in hash_to_sentence.items():
        if len(sentences) > 5:
            print(f"Sentences: {sentences}")
            print(f"Object: {hash_to_obj[hash]}")


def filter_tasks(in_path: str, out_path: str):
    with open(in_path, "r") as file:
        tasks: list[dict] = json.load(file)

    for task in tasks:
        task["word"] = task["word"][0]
        sentence = re.sub(r"^[0-9. \")(]", "", task["sentence"]).strip()
        sentence = re.sub(r"[.\"]$", "", sentence).strip()
        task["sentence"] = sentence

    with open(out_path, "w") as file:
        json.dump(tasks, file)


if __name__ == "__main__":
    # odd_one_out("sentences_4_directions.json")
    filter_tasks("sentences_4_directions.json", "sentences_4_directions2.json")
