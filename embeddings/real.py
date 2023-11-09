import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional, Literal

import numpy as np
import torch
from torch import Tensor
import dill

from envs.meta.maze.MazeTask import MazeTask
from torchkit import pytorch_utils as ptu

Direction = Literal["north", "south", "west", "east"]


def create_sentence_embedding(
    tasks: list[dict],
    embedder: Callable[[list[str]], Tensor],
) -> list[MazeTask]:
    embeddings = embedder([sentence["sentence"].strip().lower() for sentence in tasks])

    result_tasks: list[Optional[MazeTask]] = [None] * len(tasks)
    for index, (task, embedding) in enumerate(zip(tasks, embeddings)):
        result_tasks[index] = MazeTask(embedding=embedding, **task)

    return result_tasks


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
        tasks: list[dict], embedder: Callable[[list[str]], Tensor]
    ) -> list[MazeTask]:
        # Find unique values for the attribute
        unique_words = {task_hash(task): task for task in tasks}
        embeddings = embedder(
            [task[attribute].strip().lower() for task in unique_words.values()]
        )

        # Create tasks with the embeddings
        tasks: list[Optional[MazeTask]] = [None] * len(unique_words)
        for index, (task, embedding) in enumerate(
            zip(unique_words.values(), embeddings)
        ):
            maze_task = MazeTask(embedding=embedding, **task)
            maze_task.sentence = task[attribute]
            tasks[index] = maze_task

        return tasks

    return embedding_creator


simcse_model = None


def get_simcse_embedding(sentences: list[str]) -> Tensor:
    global simcse_model
    from simcse import SimCSE

    if simcse_model is None:
        simcse_model = SimCSE("princeton-nlp/sup-simcse-roberta-large")

    return simcse_model.encode(sentences)


word2vec_model = None


def get_word2vec_embedding(
    positional: bool = False,
) -> Callable[[list[str]], Tensor]:
    def word2vec_embedding(sentences: list[str]) -> Tensor:
        global word2vec_model
        import gensim.downloader

        # Create positional encodings
        max_words = count_max_words(sentences)
        positional_encodings = (
            positional_encoding_matrix(max_words, 300)
            if positional
            else ptu.ones((max_words, 300))
        )

        if word2vec_model is None:
            word2vec_model = gensim.downloader.load("word2vec-google-news-300")

        result_tensor = ptu.zeros((len(sentences), 300))
        unknown_words = set()
        for sentence_index, sentence in enumerate(sentences):
            embeddings = ptu.zeros((max_words, 300))
            filtered_sentence = filter_sentence(sentence)

            # Extract the word embeddings
            length = 0
            for index, word in enumerate(filtered_sentence.split(" ")):
                if len(word) == 0 or word in ["to", "and", "of"]:
                    continue
                if word in word2vec_model and not np.any(
                    np.isinf(word2vec_model[word])
                ):
                    length += 1
                    embeddings[index, :] = ptu.from_numpy(np.copy(word2vec_model[word]))
                else:
                    unknown_words.add(word)

            if length == 0:
                raise ValueError(f"Sentence '{sentence}' has no words in word2vec")

            # Multiply with the positional encodings
            emb_and_pos = torch.mul(embeddings, positional_encodings)
            result_tensor[sentence_index, :] = torch.sum(emb_and_pos, dim=0) / length

            # Check for inf
            if torch.where(emb_and_pos == torch.inf)[0].shape[0] > 0:
                raise ValueError(f"Sentence '{sentence}' has inf in result")

        if len(unknown_words) > 0:
            print(f"Unknown words: {unknown_words}")
        return result_tensor

    return word2vec_embedding


def create_embeddings(
    folder: str,
    input_file: str,
    use_word2vec: bool = False,
    type_name: Literal["sentences", "words", "object_type"] = "sentences",
    positional: bool = False,
):
    model_name = "word2vec" if use_word2vec else "simcse"
    pos_name = "_pos" if positional else ""
    output_file = f"{type_name}_{model_name}{pos_name}"
    print(f"Creating embeddings for {output_file}")

    with open(os.path.join(folder, input_file), "r") as file:
        inputs: list[dict] = json.load(file)

    embedder = (
        get_word2vec_embedding(positional) if use_word2vec else get_simcse_embedding
    )
    creators = {
        "sentences": create_sentence_embedding,
        "words": create_word_embedding("word"),
        "object_type": create_word_embedding("object_type"),
    }
    creator = creators[type_name]
    tasks = creator(inputs, embedder)
    save_tasks(tasks, os.path.join(folder, output_file))


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
    embeddings = get_simcse_embedding(sentences)

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

    print("breakpoint")
    save_tasks(all_directions, "all_directions_negation")

    just_directions = [task for task in all_directions if not task.negation]
    save_tasks(just_directions, "all_directions")

    leftright_directions = [
        task for task in just_directions if task.short_direction >= 2
    ]
    save_tasks(leftright_directions, "leftright_directions")

    left_directions = [
        task for task in leftright_directions if task.long_direction == 2
    ]
    save_tasks(left_directions, "left_directions")


def save_tasks(tasks: list[MazeTask], file_path: str):
    os.makedirs(file_path, exist_ok=True)
    embeddings = torch.stack([task.embedding for task in tasks])
    torch.save(embeddings, os.path.join(file_path, "embeddings.pt"))

    without_embeddings = [None] * len(tasks)
    for index, task in enumerate(tasks):
        without_embeddings[index] = dict.copy(task.__dict__)
        del without_embeddings[index]["embedding"]

    with open(os.path.join(file_path, "tasks.json"), "w") as file:
        json.dump(without_embeddings, file)


def positional_encoding_matrix(n_words: int, n_dimensions: int) -> Tensor:
    """
    Calculates the positional encoding matrix n_words, n_dimensions.
    Follows the formula from Attention Is All You Need: https://arxiv.org/pdf/1706.03762.pdf
    Args:
        n_words: The maximum number of words in a sentence.
        n_dimensions: The number of dimensions of the positional encoding.

    Returns:
        Matrix with the positional encodings
    """
    positional_encoding = ptu.zeros((n_words, n_dimensions))
    dimension = ptu.arange(0, n_dimensions, 2)
    for pos in range(n_words):
        # Offset pos with 1 to make sure sin(x) != 0
        sin_contents = (pos + 1) / (10000 ** (2 * dimension / n_dimensions))
        positional_encoding[pos, dimension] = torch.sin(sin_contents)
        positional_encoding[pos, dimension + 1] = torch.cos(sin_contents)

    return positional_encoding


def count_max_words(sentences: list[str]) -> int:
    max_words = 0
    for sentence in sentences:
        max_words = max(len(filter_sentence(sentence).split(" ")), max_words)
    return max_words


def filter_sentence(sentence: str) -> str:
    filtered_sentence = re.sub(r"[.,!?\'\"()-]", " ", sentence)
    filtered_sentence = re.sub(r"[^a-zA-Z ]", "", filtered_sentence)
    return filtered_sentence


def main():
    ptu.set_gpu_mode(True)
    base_folder = (
        "C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\embeddings2"
    )
    folders = [
        os.path.join(base_folder, "one_direction"),
        os.path.join(base_folder, "two_directions"),
    ]
    input_file = "sentences.json"
    for folder in folders:
        for use_word2vec in [True, False]:
            for type_name in ["sentences", "words", "object_type"]:
                positionals = [True, False] if use_word2vec else [False]
                for positional in positionals:
                    create_embeddings(
                        folder, input_file, use_word2vec, type_name, positional
                    )


if __name__ == "__main__":
    main()
    # main_multiple_directions("sentences_4_directions.json")
