import json

import openai
from dotenv import load_dotenv

import os


load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]


def generate_prompt(word: str, num_sentences: int):
    return f"Create {num_sentences} sentences that tell someone else there is a {word} to his/her left"


def get_sentence(word: str):
    prompt = generate_prompt(word, 5)
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        temperature=0.7,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        stop=["\n"],
    )
    return response.choices[0].text


def main(file_path: str, out_file: str):
    with open(file_path, "r") as file:
        words = json.load(file)

    words = [[word[0] for word in word_list] for word_list in words]

    sentences = [
        [[sentence, word] for word in word_list for sentence in get_sentence(word)]
        for word_list in words
    ]

    with open(out_file, "w") as file:
        json.dump(sentences, file)


if __name__ == "__main__":
    main("light_heavy_words.json", "light_heavy_sentences.json")
