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


def main(file_path: str, out_file: str):
    with open(file_path, "r") as file:
        words = json.load(file)

    with open("progress.csv", "w") as file:
        file.write("word,sentence\n")

    words = [[word[0] for word in word_list] for word_list in words]

    sentences = []
    for word_index, word_list in enumerate(words):
        sentences.append([])
        for word in word_list:
            for sentence in get_sentence(word):
                with open("progress.csv", "a") as file:
                    file.write(f'"{word}","{sentence}"\n')

                sentences[word_index].append([filter_sentence(sentence), word])

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
        "embeddings/light_vs_heavy/words.json",
        "embeddings/light_vs_heavy/right_sentences.json",
    )
