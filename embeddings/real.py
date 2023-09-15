import json
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor
import dill


@dataclass
class MazeTask:
    """Maze task class."""

    embedding: Tensor
    right_direction: bool
    blocked: bool
    task_type: int  # unique number for each type (high vs low, heavy vs light, etc)
    word: str


def create_embeddings(
    inputs: tuple[list[str], list[str]], embedder: Callable[[str], Tensor]
) -> list[MazeTask]:
    tasks = []
    for sentence in inputs[0]:
        tasks.append(
            MazeTask(
                embedder(sentence.strip().lower()),
                False,
                False,
                0,
            )
        )

    for sentence in inputs[1]:
        tasks.append(MazeTask(embedder(sentence.strip().lower()), False, True, 1))

    return [task for task in tasks if task.embedding is not None]


simcse_model = None


def get_simcse_embedding(sentence: str) -> Tensor:
    global simcse_model
    from simcse import SimCSE

    if simcse_model is None:
        simcse_model = SimCSE("princeton-nlp/sup-simcse-roberta-large")

    return simcse_model.encode(sentence)


vectors = None


def get_word2vec_embedding(sentence: str) -> Optional[Tensor]:
    global vectors
    import gensim.downloader

    if vectors is None:
        vectors = gensim.downloader.load("word2vec-google-news-300")

    embed_sum: Optional[Tensor] = None
    words = [word for word in sentence.split(" ")]
    length = 0
    for word in words:
        if word in vectors.key_to_index:
            length += 1
            if embed_sum is None:
                embed_sum = vectors.vectors[vectors.key_to_index[word]]
            else:
                embed_sum += vectors.vectors[vectors.key_to_index[word]]
        else:
            print(f"Word {word} not in word2vec")

    if length == 0:
        print(f"Sentence {sentence} has no words in word2vec")
        return None
    return torch.from_numpy(embed_sum / length)


def main(input_file: str, output_file: str, use_word2vec: bool = False):
    with open(input_file, "r") as file:
        inputs: input_file = json.load(file)

    embedder = get_word2vec_embedding if use_word2vec else get_simcse_embedding
    tasks = create_embeddings(inputs, embedder)
    with open(output_file, "wb") as file:
        dill.dump(tasks, file)


if __name__ == "__main__":
    main("light_heavy.json", "light_heavy_word2vec.dill", True)
    main("light_heavy.json", "light_heavy_simcse.dill", False)
