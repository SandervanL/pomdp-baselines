import json
from collections import defaultdict

from embeddings.real import SentenceList


def main(sentence_files: list[str]):
    sentence_list: list[SentenceList] = []
    for sentence_file in sentence_files:
        with open(sentence_file, "r") as file:
            sentence_list.append(json.load(file))

    count = defaultdict(lambda: 0)
    for sentences in sentence_list:
        for sentence_type in sentences:
            for sentence, word in sentence_type:
                new_sentence = (
                    sentence.replace(word, "").replace("left", "").replace("right", "")
                )
                count[new_sentence] += 1

    total_dup = 0
    for sentence, c in count.items():
        if c > 1:
            print(sentence, c)
            total_dup += c - 1

    print("\nTotal duplicates: " + str(total_dup))


if __name__ == "__main__":
    main(
        [
            "embeddings/light_vs_heavy/right_sentences2.json",
        ]
    )
