import json
import os

import torch

from envs.meta.maze.MazeTask import MazeTask


def save_tasks(tasks: list[MazeTask], file_path: str):
    """
    Save the tasks to a file. Stores the embeddings in a separate file.
    Args:
        tasks: The tasks to save.
        file_path: The directory to save the tasks to.
    """
    os.makedirs(file_path, exist_ok=True)
    embeddings = torch.stack([task.embedding for task in tasks])
    torch.save(embeddings.cpu(), os.path.join(file_path, "embeddings.pt"))

    without_embeddings = [None] * len(tasks)
    for index, task in enumerate(tasks):
        without_embeddings[index] = dict.copy(task.__dict__)
        del without_embeddings[index]["embedding"]

    with open(os.path.join(file_path, "tasks.json"), "w") as file:
        json.dump(without_embeddings, file, indent=4)
