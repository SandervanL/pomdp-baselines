import json
from typing import Optional

import torch

from embeddings.real import get_simcse_embedding
from embeddings.save_tasks import save_tasks
from envs.meta.maze.MazeTask import MazeTask
from envs.meta.maze.MultitaskMaze import load_tasks_dir

BASE_OUT_DIR = (
    "C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\embeddings\\paper"
)


def convert_and_save_tasks(tasks: list[dict], out_dir: str):
    sentences = [task["sentence"] for task in tasks]
    embeddings = get_simcse_embedding(sentences)

    result: list[Optional[MazeTask]] = [None] * (
        3 * len(tasks)
    )  # 3 for long directions
    for outer_index, task in enumerate(tasks):
        inner_index = -1
        for long_direction in range(4):
            if long_direction == task["short_direction"]:
                continue
            inner_index += 1
            index = outer_index * 3 + inner_index
            result[index] = MazeTask(
                **dict.copy(task), embedding=embeddings[outer_index, :]
            )
            result[index].long_direction = long_direction
            result[index].long_hook_direction = long_direction
            result[index].task_type = result[index].short_direction * 4 + int(
                result[index].blocked
            )

    save_tasks(result, out_dir)


def left_orig():
    input_file = "sentences/left_direction.json"
    output_dir = f"{BASE_OUT_DIR}\\left_orig_sentences"

    with open(input_file, "r") as f:
        tasks = json.load(f)

    convert_and_save_tasks(tasks, output_dir)


def left_lr_orig():
    input_file = "sentences/left_right_directions.json"
    output_dir = f"{BASE_OUT_DIR}\\left_lr_orig_sentences"

    with open(input_file, "r") as f:
        tasks = [task for task in json.load(f) if task["direction"] == "left"]

    convert_and_save_tasks(tasks, output_dir)


def right_orig():
    input_file = "sentences/left_right_directions.json"
    output_dir = f"{BASE_OUT_DIR}\\right_orig_sentences"

    with open(input_file, "r") as f:
        tasks = [task for task in json.load(f) if task["direction"] == "right"]

    convert_and_save_tasks(tasks, output_dir)


def left_right_orig():
    input_file = "sentences/left_right_directions.json"
    output_dir = f"{BASE_OUT_DIR}\\left_right_orig_sentences"

    with open(input_file, "r") as f:
        tasks = [task for task in json.load(f)]

    convert_and_save_tasks(tasks, output_dir)


def left_new():
    input_file = "sentences/all_directions.json"
    output_dir = f"{BASE_OUT_DIR}\\left_new_sentences"

    with open(input_file, "r") as f:
        tasks = [
            task
            for task in json.load(f)
            if task["short_direction"] == 2 and not task["negation"]
        ]

    convert_and_save_tasks(tasks, output_dir)


def right_new():
    input_file = "sentences/all_directions.json"
    output_dir = f"{BASE_OUT_DIR}\\right_new_sentences"

    with open(input_file, "r") as f:
        tasks = [
            task
            for task in json.load(f)
            if task["short_direction"] == 3 and not task["negation"]
        ]

    convert_and_save_tasks(tasks, output_dir)


def left_right_new():
    input_file = "sentences/all_directions.json"
    output_dir = f"{BASE_OUT_DIR}\\left_right_new_sentences"

    with open(input_file, "r") as f:
        tasks = [
            task
            for task in json.load(f)
            if task["short_direction"] in [2, 3] and not task["negation"]
        ]

    convert_and_save_tasks(tasks, output_dir)


def all_new():
    input_file = "sentences/all_directions.json"
    output_dir = f"{BASE_OUT_DIR}\\all_new_sentences"

    with open(input_file, "r") as f:
        tasks = [task for task in json.load(f) if not task["negation"]]

    convert_and_save_tasks(tasks, output_dir)


def all_negation_new():
    input_file = "sentences/all_directions.json"
    output_dir = f"{BASE_OUT_DIR}\\all_negation_new_sentences"

    with open(input_file, "r") as f:
        tasks = [task for task in json.load(f)]

    convert_and_save_tasks(tasks, output_dir)


def all_perfect() -> list[MazeTask]:
    directions = ["up", "down", "left", "right"]
    result = []
    for short_direction in range(4):
        for blocked in [True, False]:
            for negation in [True, False]:
                for long_direction in range(4):
                    if long_direction == short_direction:
                        continue
                    result.append(
                        MazeTask(
                            embedding=torch.Tensor(
                                [
                                    short_direction,
                                    float(blocked ^ negation),
                                    float(negation),
                                ]
                            ),
                            blocked=blocked ^ negation,
                            task_type=short_direction * 4 + int(blocked ^ negation),
                            word="",
                            sentence="",
                            object_type="heavy" if blocked else "light",
                            negation=negation,
                            direction=directions[short_direction],
                            short_direction=short_direction,
                            short_hook_direction=short_direction,
                            long_direction=long_direction,
                            long_hook_direction=long_direction,
                        )
                    )
    for method in ["perfect", "baseline"]:
        all_negation = result
        if method == "baseline":
            for task in all_negation:
                task.embedding = torch.Tensor([])

        save_tasks(result, f"{BASE_OUT_DIR}\\all_negation_{method}")

        all = [task for task in all_negation if not task.negation]
        save_tasks(all, f"{BASE_OUT_DIR}\\all_{method}")

        left_right = [task for task in all if task.short_direction in [2, 3]]
        save_tasks(left_right, f"{BASE_OUT_DIR}\\left_right_{method}")

        left = [task for task in left_right if task.short_direction == 2]
        save_tasks(left, f"{BASE_OUT_DIR}\\left_{method}")

        right = [task for task in left_right if task.short_direction == 3]
        save_tasks(right, f"{BASE_OUT_DIR}\\right_{method}")


def generalization_weight():
    input_file = "sentences/left_direction.json"
    output_dir = f"{BASE_OUT_DIR}\\added_weight_sentences"

    with open(input_file, "r") as f:
        tasks = json.load(f)

    for task in tasks:
        task[
            "sentence"
        ] = f"{task['sentence']}. {task['word'].capitalize()} is {task['object_type']}."
        print(task["sentence"])

    convert_and_save_tasks(tasks, output_dir)


if __name__ == "__main__":
    print("All perfect")
    all_perfect()
    # print("Left orig")
    # left_orig()
    # print("Left LR orig")
    # left_lr_orig()
    # print("Right orig")
    # right_orig()
    # print("Left right orig")
    # left_right_orig()
    # print("Left new")
    # left_new()
    # print("Right new")
    # right_new()
    # print("Left right new")
    # left_right_new()
    # print("All new")
    # all_new()
    # print("All negation new")
    # all_negation_new()
    # print("Generalization weight")
    # generalization_weight()
