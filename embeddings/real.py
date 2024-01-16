import json
import os
import re
from typing import Callable, Optional, Literal, Set

import numpy as np
import torch
from torch import Tensor

from embeddings.save_tasks import save_tasks
from envs.meta.maze.MazeTask import MazeTask
from torchkit import pytorch_utils as ptu

Direction = Literal["north", "south", "west", "east"]


def create_sentence_embedding(
    tasks: list[dict],
    embedder: Callable[[list[str]], Tensor],
) -> list[MazeTask]:
    """
    Create embeddings for the given tasks.
    Args:
        tasks: The tasks to create embeddings for.
        embedder: The function to create embeddings for the sentences.

    Returns:
        List of tasks with the embeddings.
    """
    embeddings = embedder([sentence["sentence"].strip().lower() for sentence in tasks])

    result_tasks: list[Optional[MazeTask]] = [None] * len(tasks)
    for index, (task, embedding) in enumerate(zip(tasks, embeddings)):
        result_tasks[index] = MazeTask(embedding=embedding, **task)

    return result_tasks


word_buffer = []


def task_hash(task: dict) -> int:
    """
    Create a hash for the given task.
    It does so based on the word, blocked, task_type, short_direction, short_hook_direction,
    long_direction, and long_hook_direction.
    Args:
        task: The task to create a hash for.

    Returns:
        The hash for the task.
    """
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
    """
    Create a function that creates embeddings for the given attribute.
    Args:
        attribute: The MazeTask attribute to create embeddings for (word, task_type).

    Returns:
        Function that creates embeddings for the given attribute.
    """

    def embedding_creator(
        tasks: list[dict], embedder: Callable[[list[str]], Tensor]
    ) -> list[MazeTask]:
        """
        Create embeddings for the given attribute.
        Args:
            tasks: The tasks to create embeddings for.
            embedder: The function to create embeddings for the attribute.

        Returns:
            List of tasks with the embeddings.
        """
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


sbert_model = None


def get_sbert_embedding(sentences: list[str]) -> Tensor:
    """
    Create embeddings for the given sentences using SBERT.
    Args:
        sentences: The sentences to create embeddings for.

    Returns:
        Tensor with sentence embeddings (len(sentences), 768)
    """
    global sbert_model

    if sbert_model is None:
        set_huggingface_cache_dir()
        from sentence_transformers import SentenceTransformer

        sbert_model = SentenceTransformer(
            "all-mpnet-base-v2",
            device=ptu.device,
            cache_folder="D:\\Afstuderen\\.cache",
        )

    return ptu.from_numpy(sbert_model.encode(sentences))


simcse_model = None


def get_simcse_embedding(sentences: list[str]) -> Tensor:
    """
    Create embeddings for the given sentences using SimCSE.
    Args:
        sentences: The sentences to create embeddings for.

    Returns:
        Tensor with sentence embeddings (len(sentences), 1024)
    """
    global simcse_model

    if simcse_model is None:
        set_huggingface_cache_dir()
        from simcse import SimCSE

        simcse_model = SimCSE("princeton-nlp/sup-simcse-roberta-large")

    return simcse_model.encode(sentences)


def set_huggingface_cache_dir():
    os.environ["HF_HOME"] = "D:\\Afstuderen\\.cache"


infersent_model = None


def get_infersent_embedding(sentences: list[str]) -> Tensor:
    """
    Create embeddings for the given sentences using InferSent.
    Args:
        sentences: The sentences to create embeddings for.

    Returns:
        Tensor with sentence embeddings (len(sentences), 4096)
    """
    global infersent_model

    if infersent_model is None:
        from infersent import InferSent
        import nltk

        nltk.download("punkt")
        infersent_model = InferSent(
            {
                "bsize": 64,
                "word_emb_dim": 300,
                "enc_lstm_dim": 2048,
                "pool_type": "max",
                "dpout_model": 0.0,
                "version": 2,
            }
        )
        base_folder = "D:\\Afstuderen\\infersent"
        infersent_model.load_state_dict(
            torch.load(os.path.join(base_folder, "infersent2.pkl"))
        )
        infersent_model.set_w2v_path(os.path.join(base_folder, "crawl-300d-2M.vec"))
        if ptu.gpu_enabled():
            infersent_model.cuda()

    infersent_model.build_vocab(sentences, tokenize=True)
    embeddings = infersent_model.encode(sentences, tokenize=True, verbose=True)
    return ptu.from_numpy(np.copy(embeddings))


word2vec_model = None


def get_word2vec_embedding(
    positional: bool = False,
) -> Callable[[list[str]], Tensor]:
    """
    Create a function that creates embeddings for the given sentences using Word2Vec.
    Args:
        positional: Whether to add positional encodings to the Word2Vec embeddings.

    Returns:
        Function that creates embeddings for the given sentences.
    """

    def word2vec_embedding(sentences: list[str]) -> Tensor:
        """
        Create embeddings for the given sentences using Word2Vec.
        Args:
            sentences: The sentences to create embeddings for.

        Returns:
            Tensor with sentence embeddings (len(sentences), 300)
        """
        # Create positional encodings
        max_words = count_max_words(sentences)
        positional_encodings = (
            positional_encoding_matrix(max_words, 300)
            if positional
            else ptu.ones((max_words, 300))
        )

        # Create the embeddings for each sentence
        result_tensor = ptu.zeros((len(sentences), 300))
        unknown_words = set()
        for sentence_index, sentence in enumerate(sentences):
            result_tensor[sentence_index, :], words = _get_word2vec_embedding(
                sentence, positional_encodings
            )
            unknown_words.update(words)

        if len(unknown_words) > 0:
            print(f"Unknown words: {unknown_words}")

        return result_tensor

    return word2vec_embedding


def _get_word2vec_embedding(
    sentence: str, positional_encodings: Tensor
) -> (Tensor, Set):
    """
    Create an embedding for the given sentence using Word2Vec.
    Args:
        sentence: The sentence to create an embedding for.
        positional_encodings: The positional encodings to multiply with the word embeddings.

    Returns:
        Tensor with the sentence embedding (300,)
    """
    global word2vec_model

    if word2vec_model is None:
        os.environ["GENSIM_DATA_DIR"] = "D:\\Afstuderen\\gensim-data"
        import gensim.downloader

        word2vec_model = gensim.downloader.load("word2vec-google-news-300")

    embeddings = ptu.zeros(positional_encodings.shape)
    filtered_sentence = filter_sentence(sentence)

    # Extract the word embeddings
    unknown_words = set()
    length = 0
    for index, word in enumerate(filtered_sentence.split(" ")):
        if len(word) == 0 or word in ["to", "and", "of"]:
            continue
        if word in word2vec_model and not np.any(np.isinf(word2vec_model[word])):
            length += 1
            embeddings[index, :] = ptu.from_numpy(np.copy(word2vec_model[word]))
        else:
            unknown_words.add(word)

    if length == 0:
        raise ValueError(f"Sentence '{sentence}' has no words in word2vec")

    # Multiply with the positional encodings, and sum
    positioned_embeddings = torch.mul(embeddings, positional_encodings)
    embedding = torch.sum(positioned_embeddings, dim=0) / length

    # Check for inf
    if torch.where(embedding == torch.inf)[0].shape[0] > 0:
        raise ValueError(f"Sentence '{sentence}' has inf in result")

    return embedding, unknown_words


def create_embeddings(
    output_folder: str,
    input_file: str,
    model_name: Literal["simcse", "sbert", "infersent", "word2vec"] = "simcse",
    type_name: Literal["sentences", "words", "object_type"] = "sentences",
):
    """
    Create embeddings for the given type.
    Args:
        output_folder: The folder to save the embeddings to.
        input_file: The file with the tasks.
        model_name: The embedding model to use.
        type_name: The type of embedding to create.
    """
    embedders = {
        "simcse": get_simcse_embedding,
        "sbert": get_sbert_embedding,
        "infersent": get_infersent_embedding,
        "word2vec": get_word2vec_embedding(False),
        "word2vec_pos": get_word2vec_embedding(True),
    }
    assert model_name in embedders
    output_dirname = f"{type_name}_{model_name}"
    print(f"Creating embeddings to {output_dirname} for {input_file}")

    with open(input_file, "r") as file:
        inputs: list[dict] = json.load(file)

    creators = {
        "sentences": create_sentence_embedding,
        "words": create_word_embedding("word"),
        "object_type": create_word_embedding("object_type"),
    }
    creator = creators[type_name]
    tasks = creator(inputs, embedders[model_name])
    save_tasks(tasks, os.path.join(output_folder, output_dirname))


def directions_index(directions: list[list[str]], direction: str) -> int:
    """Get the index of the list that contains the direction."""
    for index, direction_list in enumerate(directions):
        if direction in direction_list:
            return index
    raise f"Could not find direction {direction}"


def main_multiple_directions(in_file: str):
    """
    Create embeddings for all directions and all combinations of directions.
    Args:
        in_file: The file with the tasks.
    """
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
    """Count the maximum number of words in a sentence."""
    max_words = 0
    for sentence in sentences:
        max_words = max(len(filter_sentence(sentence).split(" ")), max_words)
    return max_words


def filter_sentence(sentence: str) -> str:
    """Remove punctuation and non-alphabetical characters from sentence."""
    filtered_sentence = re.sub(r"[.,!?\'\"()-]", " ", sentence)
    filtered_sentence = re.sub(r"[^a-zA-Z ]", "", filtered_sentence)
    return filtered_sentence


def main():
    """
    Create embeddings for words, sentences, and object types using SimCSE and Word2Vec.
    Positional encodings can be added to the Word2Vec embeddings.
    """
    ptu.set_gpu_mode(True)
    base_folder = (
        "C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\embeddings2"
    )
    inputs = [
        # "one_direction",
        # "two_directions",
        # "all_directions",
        "all_directions_decoupled",
    ]
    for input_name in inputs:
        input_file = f"sentences/{input_name}.json"
        output_folder = os.path.join(base_folder, input_name)
        for type_name in ["sentences"]:
            embedding_models = (
                ["infersent", "sbert"]
                if type_name in ["sentences", "words"]
                else ["simcse"]
            )
            for embedding_model in embedding_models:
                create_embeddings(output_folder, input_file, embedding_model, type_name)


if __name__ == "__main__":
    # add_directionality("sentences/all_directions.json")
    main()
    # main_multiple_directions("sentences_4_directions.json")
