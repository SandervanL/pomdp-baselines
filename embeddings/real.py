import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor
import dill

SentenceList = tuple[list[tuple[str, str]], list[tuple[str, str]]]


@dataclass
class MazeTask:
    """Maze task class."""

    embedding: Tensor
    right_direction: bool
    blocked: bool
    task_type: int  # unique number for each type (high vs low, heavy vs light, etc)
    word: str
    sentence: str


def create_sentence_embedding(
    inputs: SentenceList,
    embedder: Callable[[str], Tensor],
) -> list[MazeTask]:
    tasks = [
        MazeTask(
            embedder(sentence.strip().lower()),
            False,
            index == 0,
            index,
            word.strip().lower(),
            sentence.strip().lower(),
        )
        for index, sentence_list in enumerate(inputs)
        for sentence, word in sentence_list
        if len(sentence.strip()) > 0
    ]

    return [task for task in tasks if task.embedding is not None]


def create_word_embedding(
    inputs: SentenceList, embedder: Callable[[str], Tensor]
) -> list[MazeTask]:
    unique_words = [{task[1] for task in tasks} for tasks in inputs]
    tasks = [
        MazeTask(
            embedder(word.strip().lower()),
            False,
            index == 0,
            index,
            word.strip().lower(),
            word.strip().lower(),
        )
        for index, sentence_list in enumerate(unique_words)
        for word in sentence_list
        if len(word.strip()) > 0
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


def main(
    folder: str, input_file: str, use_word2vec: bool = False, sentences: bool = True
):
    with open(os.path.join(folder, input_file), "r") as file:
        inputs: SentenceList = json.load(file)

    embedder = get_word2vec_embedding if use_word2vec else get_simcse_embedding
    creator = create_sentence_embedding if sentences else create_word_embedding

    tasks = creator(inputs, embedder)

    output_file = f"{'sentences' if sentences else 'words'}_{'word2vec' if use_word2vec else 'simcse'}2.dill"
    print(f"Saving to {output_file}")
    with open(os.path.join(folder, output_file), "wb") as file:
        dill.dump(tasks, file)


if __name__ == "__main__":
    folder = "embeddings/light_vs_heavy"
    input_file = "sentences.json"
    for use_word2vec in [True, False]:
        for sentences in [True, False]:
            main(folder, input_file, use_word2vec, sentences)
