import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional, Literal

import numpy as np
import torch
from torch import Tensor
import dill

Direction = Literal["north", "south", "west", "east"]


@dataclass
class MazeTask:
    """Maze task class."""

    embedding: Tensor
    blocked: bool
    task_type: int  # unique number for each type (high vs low, heavy vs light, etc)
    word: str
    sentence: str
    object_type: str
    negation: bool
    direction: str

    # Directions of the map hallways
    short_direction: int
    short_hook_direction: int
    long_direction: int
    long_hook_direction: int


def create_sentence_embedding(
    sentences: list[dict],
    embedder: Callable[[str], Tensor],
) -> list[MazeTask]:
    tasks = [
        MazeTask(embedding=embedder(sentence["sentence"].strip().lower()), **sentence)
        for sentence in sentences
    ]

    return [task for task in tasks if task.embedding is not None]


word_buffer = []


def task_hash(task: dict) -> int:
    global word_buffer
    word = task["word"]
    if word in word_buffer:
        word_index = word_buffer.index(word)
    else:
        word_buffer.append(word)
        word_index = len(word_buffer) - 1
    return (
        (int(task["blocked"]))
        | (int(task["task_type"]) << 8)
        | (int(task["short_direction"]) << 16)
        | (int(task["short_hook_direction"]) << 24)
        | (int(task["long_direction"]) << 32)
        | (int(task["long_hook_direction"]) << 40)
        | (word_index << 48)
    )


def create_word_embedding(attribute: str):
    def embedding_creator(
        tasks: list[dict], embedder: Callable[[str], Tensor]
    ) -> list[MazeTask]:
        unique_words = {task_hash(task): task for task in tasks}
        tasks = []
        for task in unique_words.values():
            new_task = task.copy()
            new_task["sentence"] = task[attribute]
            maze_task = MazeTask(
                embedding=embedder(task[attribute].strip().lower()), **new_task
            )
            if maze_task.embedding is not None:
                tasks.append(maze_task)

        return tasks

    return embedding_creator


simcse_model = None


def multiple_simcse_embeddings(sentences: list[str], model: str) -> Tensor:
    global simcse_model
    from simcse import SimCSE

    if simcse_model is None:
        simcse_model = SimCSE(model)

    return simcse_model.encode(sentences)


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
    folder: str,
    input_file: str,
    use_word2vec: bool = False,
    type_name: Literal["sentences", "words", "object_type"] = "sentences",
):
    with open(os.path.join(folder, input_file), "r") as file:
        inputs: list[dict] = json.load(file)

    embedder = get_word2vec_embedding if use_word2vec else get_simcse_embedding
    creators = {
        "sentences": create_sentence_embedding,
        "words": create_word_embedding("word"),
        "object_type": create_word_embedding("object_type"),
    }
    creator = creators[type_name]
    tasks = creator(inputs, embedder)

    output_file = f"{type_name}_{'word2vec' if use_word2vec else 'simcse'}.dill"
    print(f"Saving to {output_file}")
    with open(os.path.join(folder, output_file), "wb") as file:
        dill.dump(tasks, file)


def directions_index(directions: list[list[str]], direction: str) -> int:
    for index, direction_list in enumerate(directions):
        if direction in direction_list:
            return index
    raise f"Could not find direction {direction}"


def main_multiple_directions(in_file: str):
    with open(in_file, "r") as file:
        tasks: list[dict] = json.load(file)

    with open("directions.json", "r") as file:
        directions: list[list[str]] = json.load(file)

    sentences = [task["sentence"] for task in tasks]
    embeddings: Tensor = multiple_simcse_embeddings(
        sentences, "princeton-nlp/sup-simcse-roberta-large"
    )

    all_directions = []
    for task, embedding in zip(tasks, embeddings):
        short_direction = directions_index(directions, task["direction"])
        for long_direction in range(0, 4):
            if long_direction == short_direction:
                continue
            maze_task = MazeTask(
                embedding=embedding,
                short_direction=short_direction,
                short_hook_direction=short_direction,
                long_direction=long_direction,
                long_hook_direction=long_direction,
                **task,
            )
            all_directions.append(maze_task)

    with open("all_directions_negation.dill", "wb") as file:
        dill.dump(all_directions, file)

    just_directions = [task for task in all_directions if not task.negation]
    with open("all_directions.dill", "wb") as file:
        dill.dump(just_directions, file)

    leftright_directions = [
        task for task in just_directions if task.short_direction >= 2
    ]
    with open("leftright_directions.dill", "wb") as file:
        dill.dump(leftright_directions, file)

    left_directions = [
        task for task in leftright_directions if task.long_direction == 2
    ]
    with open("left_directions.dill", "wb") as file:
        dill.dump(left_directions, file)


if __name__ == "__main__":
    folders = ["embeddings/two_directions/"]
    input_file = "sentences.json"
    for folder in folders:
        for use_word2vec in [True, False]:
            for sentences in ["sentences", "words", "object_type"]:
                main(folder, input_file, use_word2vec, sentences)
