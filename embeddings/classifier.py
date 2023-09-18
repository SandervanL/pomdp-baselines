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


def main(input_file: str):
    # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="Classify Vectors",
    #     # track hyperparameters and run metadata
    #     config={"file": input_file},
    # )
    with open(input_file, "rb") as file:
        tasks: list[MazeTask] = dill.load(file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.cat([task.embedding.unsqueeze(0) for task in tasks], dim=0).to(device)
    y = torch.tensor([task.task_type for task in tasks]).to(device)

    train_set_size = int(0.8 * len(tasks))
    shuffled_tasks = np.random.permutation(list(range(len(tasks))))
    train_tasks = shuffled_tasks[:train_set_size]
    eval_tasks = shuffled_tasks[train_set_size:]
    train_x, train_y = x[train_tasks], y[train_tasks]
    eval_x, eval_y = x[eval_tasks], y[eval_tasks]

    n_classes = len({task.task_type for task in tasks})
    classifier = build_classifier(tasks[0].embedding.shape[0], n_classes, 128).to(
        device
    )
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

        output = classifier(eval_x)
        eval_accuracy = (output.argmax(dim=-1) == eval_y).sum().item() / len(eval_tasks)
        eval_loss = criterion(output, eval_y) / len(eval_tasks)

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
        "data/light_vs_heavy/sentences_simcse.dill",
        # "data/light_vs_heavy/sentences_word2vec.dill",
        # "data/light_vs_heavy/words_simcse.dill",
        # "data/light_vs_heavy/words_word2vec.dill",
    ]
    for file in files:
        for _ in range(10):
            main(file)
            wandb.finish()
