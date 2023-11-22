import json

from embeddings.real import MazeTask


def main(file: str):
    with open(file, "r") as file:
        tasks: list[dict] = json.load(file)

    words = set(task["word"] for task in tasks)
    leaked_words = []
    for task in tasks:
        for upper_word in words:
            word = upper_word.lower()
            task_word = task["word"].lower()
            if task_word == word:
                continue

            # Add sentence to search for full word
            search_word = f" {word} "
            sentence = f" {task['sentence'].lower()} "
            if search_word in sentence:
                leaked_words.append([task["word"], upper_word, task["sentence"]])

    # Sort leaked_words alphabetically by word
    leaked_words.sort(key=lambda x: x[1])
    leaked_words.sort(key=lambda x: x[0])
    # Rewrite for python
    print(" \\\\\n".join(" & ".join(leakage) for leakage in leaked_words))


if __name__ == "__main__":
    main("embeddings/one_direction/sentences.json")
