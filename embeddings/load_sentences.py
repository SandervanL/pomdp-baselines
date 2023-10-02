import json
import re

import openai
from dotenv import load_dotenv

import os


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_prompt(word: str, num_sentences: int):
    return f"Create {num_sentences} sentences that tell someone else there is a {word} to his/her right"


def get_sentence(word: str) -> list[str]:
    print(word)
    prompt = generate_prompt(word, 30)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return [
        sentence
        for sentence in response.choices[0].message.content.split("\n")
        if len(sentence) > 0
    ]


def anti_direction(direction: int) -> int:
    return ((direction + 1) % 2) + 2 * (direction // 2)


def get_task_configs(config_type: str):
    if config_type == "one":
        yield {
            "short_direction": 2,
            "short_hook_direction": 2,
            "long_direction": 3,
            "long_hook_direction": 3,
        }
    elif config_type == "two":
        yield {
            "short_direction": 2,
            "short_hook_direction": 2,
            "long_direction": 3,
            "long_hook_direction": 3,
        }
        yield {
            "short_direction": 3,
            "short_hook_direction": 3,
            "long_direction": 2,
            "long_hook_direction": 2,
        }
    elif config_type == "all":
        for short_direction in range(4):
            for short_hook_direction in range(4):
                for long_direction in range(4):
                    for long_hook_direction in range(4):
                        if (
                            short_direction == long_direction
                            or short_hook_direction == long_direction
                            or anti_direction(short_direction) == short_hook_direction
                            or anti_direction(long_direction) == long_hook_direction
                        ):
                            continue
                        yield {
                            "short_direction": short_direction,
                            "short_hook_direction": short_hook_direction,
                            "long_direction": long_direction,
                            "long_hook_direction": long_hook_direction,
                        }


def main(file_path: str, out_file: str, config_type: str = "all"):
    with open(file_path, "r") as file:
        words = json.load(file)

    with open("progress.csv", "w") as file:
        file.write("word,sentence,metadata\n")

    words = [[word[0] for word in word_list] for word_list in words]

    sentences = []
    for task_dict in get_task_configs(config_type):
        for word_index, word_list in enumerate(words):
            for word in word_list:
                for sentence in get_sentence(word):
                    filtered_sentence = filter_sentence(sentence)
                    with open("progress.csv", "a") as file:
                        file.write(
                            f'"{word}","{filtered_sentence}","{json.dumps(task_dict)}\n'
                        )

                    task = dict.copy(task_dict)
                    task["blocked"] = word_index == 0
                    task["sentence"] = filtered_sentence
                    task["word"] = word
                    sentences.append(task)

    with open(out_file, "w") as file:
        json.dump(sentences, file)


def filter_sentence(sentence: str) -> str:
    return (
        re.sub(r"[0-9.]", "", sentence)
        .replace("\u2019", "'")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace('"', "")
        .strip()
    )


if __name__ == "__main__":
    main(
        "embeddings/words.json",
        "embeddings/all_directions.json",
    )
