import json
import re

import openai
from dotenv import load_dotenv

import os


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_prompt(word: str, num_sentences: int):
    return f"Create {num_sentences} sentences that tell someone else there is a {word} to his/her left"


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


def main(file_path: str, out_file: str):
    with open(file_path, "r") as file:
        words = json.load(file)

    words = [[word[0] for word in word_list] for word_list in words]

    sentences = [
        [
            [filter_sentence(sentence), word]
            for word in word_list
            for sentence in get_sentence(word)
        ]
        for word_list in words
    ]

    with open(out_file, "w") as file:
        json.dump(sentences, file)


if __name__ == "__main__":
    main("light_heavy_words.json", "light_heavy_sentences.json")


def filter_sentence(sentence: str) -> str:
    return (
        re.sub(r"[0-9.]", "", sentence)
        .replace("\u2019", "'")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace('"', "")
        .strip()
    )
