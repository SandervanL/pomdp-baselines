from collections import defaultdict
from functools import reduce

import dill
import numpy as np
import torch
import wandb
from torch import nn, optim

from embeddings.real import MazeTask


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
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="Classify Vectors",
    #     # track hyperparameters and run metadata
    #     config={"file": input_file[input_file.rindex("/") + 1 :]},
    # )

    with open(input_file, "rb") as file:
        tasks: list[MazeTask] = dill.load(file)

    task_types: dict[int, list[MazeTask]] = defaultdict(lambda: [])
    for task in tasks:
        task_types[task.task_type].append(task)

    for key, value in task_types.items():
        orig_len = len(value)
        while len(task_types[key]) < 1800:
            task_types[key] += value[:orig_len]

    tasks = []
    for task_list in task_types.values():
        tasks += task_list

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.cat([task.embedding.unsqueeze(0) for task in tasks], dim=0).to(device)
    y = torch.tensor([task.task_type for task in tasks]).to(device)

    for train_test_split_int in range(1, 11):
        train_test_split = train_test_split_int / 10
        for _ in range(48):
            train_set_size = int(train_test_split * len(tasks))
            shuffled_tasks = np.random.permutation(list(range(len(tasks))))
            train_tasks = shuffled_tasks[:train_set_size]
            eval_tasks = shuffled_tasks[train_set_size:]
            train_x, train_y = x[train_tasks], y[train_tasks]
            eval_x, eval_y = x[eval_tasks], y[eval_tasks]

            n_classes = len({task.task_type for task in tasks})
            classifier = build_classifier(
                tasks[0].embedding.shape[0], n_classes, 128
            ).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(classifier.parameters(), lr=0.001)

            for epoch in range(20):
                optimizer.zero_grad()
                output = classifier(train_x)
                loss = criterion(output, train_y)
                loss.backward()
                optimizer.step()
                train_loss = loss.item() / len(train_tasks)
                train_accuracy = (output.argmax(dim=-1) == train_y).sum().item() / len(
                    train_tasks
                )

                if len(eval_tasks) > 0:
                    with torch.no_grad():
                        output = classifier(eval_x)
                        eval_accuracy = (
                            output.argmax(dim=-1) == eval_y
                        ).sum().item() / len(eval_tasks)
                        eval_loss = criterion(output, eval_y) / len(eval_tasks)
                else:
                    eval_accuracy = train_accuracy
                    eval_loss = train_loss

                with open(output_csv, "at") as file:
                    file.write(
                        f"{input_file},{train_test_split},{epoch},{train_loss},{train_accuracy},{eval_loss},{eval_accuracy}\n"
                    )
                # wandb.log(
                #     {
                #         "train": {"loss": train_loss, "accuracy": train_accuracy},
                #         "eval": {"loss": eval_loss, "accuracy": eval_accuracy},
                #     }
                # )
                print(
                    f"Epoch {epoch} train loss: {train_loss}, accuracy: {train_accuracy},"
                    f" eval loss: {eval_loss}, accuracy: {eval_accuracy}"
                )


if __name__ == "__main__":
    files = [
        "embeddings/one_direction/sentences_simcse.dill",
        # "embeddings/one_direction/sentences_word2vec.dill",
        # "embeddings/one_direction/words_simcse.dill",
        # "embeddings/one_direction/words_word2vec.dill",
        # "embeddings/one_direction/perfect.dill",
    ]
    output_csv = "C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\Logs\\classifier\\generalization\\progress-filled.csv"
    with open(output_csv, "wt") as csv_file:
        csv_file.write(
            "file,split,z/env_steps,metrics/train_loss,metrics/train_accuracy,metrics/eval_loss,metrics/eval_accuracy\n"
        )
    for file in files:
        main(file, output_csv)
        # wandb.finish()
