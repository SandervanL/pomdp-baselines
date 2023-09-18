import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
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
    inputs: tuple[list[tuple[str, str]], list[tuple[str, str]]],
    embedder: Callable[[str], Tensor],
) -> list[MazeTask]:
    tasks = [
        MazeTask(
            embedder(sentence.strip().lower()),
            False,
            index == 0,
            index,
            word.strip().lower(),
        )
        for index, sentence_list in enumerate(inputs)
        for sentence, word in sentence_list
        if len(sentence.strip()) > 0
    ]

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

    print(f"Sentence: '{sentence}'")

    if vectors is None:
        vectors = gensim.downloader.load("word2vec-google-news-300")

    embed_sum: np.ndarray = np.zeros_like(vectors[0])
    filtered_sentence = re.sub(r"[.,!?\'\"()-]", " ", sentence)
    filtered_sentence = re.sub(r"[^a-zA-Z ]", "", filtered_sentence)
    words = [
        word
        for word in filtered_sentence.split(" ")
        if len(word) > 1 and word not in ["to", "and", "of"]
    ]
    length = 0
    for word in words:
        if word in vectors and not np.any(np.isinf(vectors[word])):
            length += 1
            embed_sum += vectors[word]
        else:
            print(f"Word '{word}' not in word2vec or infinite")

    if length == 0:
        print(f"Sentence '{sentence}' has no words in word2vec")
        return None

    result = torch.from_numpy(embed_sum / length)
    if torch.where(result == torch.inf)[0].shape[0] > 0:
        print(f"Inf in {result}")
        return None
    return result


def main(input_file: str, output_file: str, use_word2vec: bool = False):
    with open(input_file, "r") as file:
        inputs: tuple[list[tuple[str, str]], list[tuple[str, str]]] = json.load(file)

    embedder = get_word2vec_embedding if use_word2vec else get_simcse_embedding
    tasks = create_embeddings(inputs, embedder)
    with open(output_file, "wb") as file:
        dill.dump(tasks, file)


if __name__ == "__main__":
    # main("light_heavy_sentences.json", "light_heavy_word2vec_sentences.dill", True)
    main("light_heavy_sentences.json", "light_heavy_simcse_sentences.dill", False)
