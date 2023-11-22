import os

from embeddings.save_tasks import save_tasks
from envs.meta.maze.MazeTask import MazeTask
from envs.meta.maze.MultitaskMaze import load_tasks_dir


def main(base_dir: str):
    input_file = os.path.join(
        base_dir, "all_directions_negation_decoupled\\sentences_simcse"
    )
    all_directions_negation: list[MazeTask] = load_tasks_dir(input_file)

    just_directions = [task for task in all_directions_negation if not task.negation]
    save_tasks(
        just_directions,
        os.path.join(base_dir, "all_directions_decoupled\\sentences_simcse"),
    )

    leftright = [task for task in just_directions if task.short_direction in [2, 3]]
    save_tasks(
        leftright, os.path.join(base_dir, "leftright_decoupled\\sentences_simcse")
    )

    left = [task for task in leftright if task.short_direction == 2]
    save_tasks(left, os.path.join(base_dir, "left_decoupled\\sentences_simcse"))


if __name__ == "__main__":
    main("C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\embeddings2\\")
