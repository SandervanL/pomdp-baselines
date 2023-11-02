from collections import defaultdict
from functools import reduce
from typing import Literal

import dill
import numpy as np
import torch
import wandb
from torch import nn, optim, Tensor
from tqdm import tqdm

from embeddings.real import MazeTask
from torchkit import pytorch_utils as ptu


def build_classifier(input_dim: int, output_dim: int, hidden_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def main(input_file: str, output_csv: str):
    # start a new wandb run to track this script

    with open(input_file, "rb") as file:
        tasks: list[MazeTask] = dill.load(file)

    ptu.set_gpu_mode(True)
    x = torch.cat([task.embedding.unsqueeze(0) for task in tasks], dim=0).to(ptu.device)
    y = torch.tensor([task.task_type for task in tasks]).to(ptu.device)

    for train_test_split in [0.1, 0.3, 0.5, 0.8, 1]:
        for task_selection in ["random", "random-word", "random-within-word"]:
            train_indices, eval_indices = get_train_test(
                tasks, task_selection, train_test_split
            )
            print(f"Split: {train_test_split}, selection: {task_selection}")
            epochs = 40
            num_seeds = 48
            result_data = ptu.zeros((num_seeds, epochs, 4))
            for seed in tqdm(range(num_seeds)):
                torch.manual_seed(seed)
                train_x, train_y = x[train_indices], y[train_indices]
                eval_x, eval_y = x[eval_indices], y[eval_indices]

                if eval_x.shape[0] == 0:
                    eval_x, eval_y = train_x, train_y

                n_classes = len({task.task_type for task in tasks})
                classifier = build_classifier(
                    tasks[0].embedding.shape[0], n_classes, 128
                ).to(ptu.device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(classifier.parameters(), lr=0.001)

                for epoch in range(epochs):
                    optimizer.zero_grad()
                    output = classifier(train_x)
                    loss = criterion(output, train_y)
                    loss.backward()
                    optimizer.step()
                    train_loss = (loss / train_x.shape[0]).unsqueeze(dim=0).detach()

                    train_accuracy = (
                        ((output.argmax(dim=-1) == train_y).sum() / train_x.shape[0])
                        .unsqueeze(dim=0)
                        .detach()
                    )

                    with torch.no_grad():
                        output = classifier(eval_x)
                        eval_accuracy = (
                            ((output.argmax(dim=-1) == eval_y).sum() / eval_x.shape[0])
                            .unsqueeze(dim=0)
                            .detach()
                        )
                        eval_loss = (
                            (criterion(output, eval_y) / eval_x.shape[0])
                            .unsqueeze(dim=0)
                            .detach()
                        )

                    result_data[seed, epoch, :] = torch.concatenate(
                        [train_loss, train_accuracy, eval_loss, eval_accuracy]
                    )

            result_data = result_data.to("cpu")
            with open(output_csv, "at") as file:
                for seed in range(num_seeds):
                    for epoch in range(epochs):
                        (
                            train_loss,
                            train_accuracy,
                            eval_loss,
                            eval_accuracy,
                        ) = result_data[seed, epoch, :]
                        file.write(
                            f"{input_file},{train_test_split},{task_selection},{epoch},{train_loss},{train_accuracy},{eval_loss},{eval_accuracy}\n"
                        )


def get_train_test(
    tasks: list[MazeTask], task_selection: str, split: float
) -> tuple[Tensor, Tensor]:
    # NOTE: This is off-policy varibad's setting, i.e. limited training tasks
    # split to train/eval tasks. If train_test_split is None, then num_train_tasks
    # and num_eval_tasks must be specified.

    selection_to_task_key = {
        "random": "all",
        "random-word": "word",
        "random-within-word": "word",
    }
    if task_selection == "even":
        shuffled_tasks = ptu.randperm(len(tasks))
        return shuffled_tasks, shuffled_tasks
    if task_selection not in selection_to_task_key:
        raise ValueError(f"Unknown task selection '{task_selection}'")

    task_key = selection_to_task_key[task_selection]
    tasks_by_class = get_tasks_by_type(tasks, task_key)
    if task_selection == "random-word":
        tasks_by_class = tasks_by_class.transpose(0, 1)
        tasks_per_group, num_words = tasks_by_class.shape
        shuffled_indices = ptu.randperm(num_words)
        shuffled_tasks = tasks_by_class[:, shuffled_indices]
        tasks_per_group = num_words
    else:
        num_groups, tasks_per_group = tasks_by_class.shape
        shuffled_indices = torch.stack(
            [ptu.randperm(tasks_per_group) for _ in range(num_groups)]
        )
        shuffled_tasks = torch.gather(tasks_by_class, dim=1, index=shuffled_indices)

    # Select which words to include in test and train
    num_train_sentences = int(split * tasks_per_group)

    # Select how many sentences to include in test and train
    train_tasks = shuffled_tasks[:, :num_train_sentences].reshape(-1)
    eval_tasks = shuffled_tasks[:, num_train_sentences:].reshape(-1)

    return train_tasks, eval_tasks


def get_tasks_by_type(
    tasks: list[MazeTask], key: str = Literal["task_type", "word", "all"]
) -> Tensor:  # (num_keys, num_tasks_per_key)
    if key == "all":
        return ptu.arange(len(tasks)).unsqueeze(dim=0)

    tasks_by_class_dict: dict[int, list[int]] = defaultdict(lambda: [])
    for index, task in enumerate(tasks):
        tasks_by_class_dict[getattr(task, key)].append(index)

    return ptu.tensor(list(tasks_by_class_dict.values()))


if __name__ == "__main__":
    files = [
        "C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\embeddings/one_direction/sentences_simcse.dill",
        # "embeddings/one_direction/sentences_word2vec.dill",
        # "embeddings/one_direction/words_simcse.dill",
        # "embeddings/one_direction/words_word2vec.dill",
        # "embeddings/one_direction/perfect.dill",
    ]
    output_csv = "C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\Logs\\classifier\\generalization\\progress-filled.csv"
    with open(output_csv, "wt") as csv_file:
        csv_file.write(
            "file,split,task_selection,z/env_steps,metrics/train_loss,metrics/train_accuracy,metrics/eval_loss,metrics/eval_accuracy\n"
        )
    for file in files:
        main(file, output_csv)
        # wandb.finish()
