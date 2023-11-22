import json
import os


def find_task_files(root_dir: str):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        tasks_file = None

        for filename in filenames:
            if filename.endswith("tasks.json"):
                tasks_file = os.path.join(dirpath, filename)

        if tasks_file:
            yield os.path.join(dirpath, tasks_file)


def get_word_index(words: list[list[str]], word: str) -> int:
    for index, word_list in enumerate(words):
        if word in word_list:
            return index
    raise ValueError(f"Could not find word {word}")


def main(root_dir: str):
    with open("words.json", "r") as file:
        words: list[list[list[str]]] = json.load(file)

    for task_file in find_task_files(root_dir):
        with open(task_file, "r") as file:
            tasks = json.load(file)

        for task in tasks:
            word_index = get_word_index(words, task["word"])
            task["blocked"] = (word_index == 0) ^ task["negation"]

        with open(task_file, "w") as file:
            json.dump(tasks, file, indent=4)


if __name__ == "__main__":
    main("C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\embeddings2")
