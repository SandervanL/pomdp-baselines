import json
from copy import deepcopy

import dill

from embeddings.load_sentences import anti_direction
from embeddings.real import MazeTask


def main(in_file: str, out_file: str):
    with open(in_file, "rb") as file:
        tasks: list[MazeTask] = dill.load(file)

    result: list[MazeTask] = []
    for task in tasks:
        direction = anti_direction(task.short_direction)
        # for direction in range(0, 4):
        #     if direction == task.short_direction:
        #         continue

        # for hook_direction in range(0, 4):
        # for hook_direction in range(direction, direction + 1):
        #     if anti_direction(direction) == hook_direction:
        #         continue
        #
        new_task = deepcopy(task)
        new_task.long_direction = direction
        new_task.long_hook_direction = direction
        result.append(new_task)

    with open(out_file, "wb") as file:
        dill.dump(result, file)


def repair_two_directions(input_file: str, output_file: str):
    with open(input_file, "r") as file:
        tasks: dict = json.load(file)

    for task in tasks:
        if task["blocked"]:
            task_type = 2
            object_type = "heavy "
        else:
            task_type = 0
            object_type = "light "

        if task["short_direction"] == 2:
            task_type += 1
            object_type += "left"
        else:
            task_type += 0
            object_type += "right"

        task["task_type"] = task_type
        task["object_type"] = object_type

    with open(output_file, "w") as file:
        json.dump(tasks, file, indent=4)


if __name__ == "__main__":
    main(
        "embeddings/one_direction/sentences_simcse.dill",
        "embeddings/all_directions/left_allstraight.dill",
    )
    main(
        "embeddings/two_directions/sentences_simcse.dill",
        "embeddings/all_directions/leftright_allstraight.dill",
    )
