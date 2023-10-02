import json

from torch import Tensor

from embeddings.real import MazeTask


def main():
    with open("embeddings/one_direction/sentences.json", "r") as file:
        sentences = json.load(file)

    tasks = []
    for index, sentence_type in enumerate(sentences):
        for sentence, word in sentence_type:
            task = dict(
                blocked=index == 0,
                task_type=index,
                word=word,
                sentence=sentence,
                short_direction=2,
                short_hook_direction=2,
                long_direction=3,
                long_hook_direction=3,
            )
            tasks.append(task)

    with open("embeddings/two_directions/right_sentences.json", "r") as file:
        sentences = json.load(file)

    for index, sentence_type in enumerate(sentences):
        for sentence, word in sentence_type:
            task = dict(
                blocked=index == 0,
                task_type=index,
                word=word,
                sentence=sentence,
                short_direction=3,
                short_hook_direction=3,
                long_direction=2,
                long_hook_direction=2,
            )
            tasks.append(task)

    with open("embeddings/two_directions/sentences.json", "w") as file:
        json.dump(tasks, file)


if __name__ == "__main__":
    main()
