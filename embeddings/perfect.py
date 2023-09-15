import dill
from torch import Tensor

from envs.meta.maze.MultitaskMaze import MazeTask


def make_single_tasks_file():
    no_blockage = MazeTask(
        embedding=Tensor([0.0]), blocked=False, right_direction=False, task_type=0
    )
    blockage = MazeTask(
        embedding=Tensor([1.0]),
        blocked=True,
        right_direction=False,
        task_type=1,
    )
    tasks = []
    for i in range(18):
        tasks.append(no_blockage)
        tasks.append(blockage)

    with open("configs/meta/maze/v/single_embeddings.pkl", "wb") as file:
        dill.dump(tasks, file)


def make_double_tasks_file():
    no_blockage_1 = MazeTask(
        embedding=Tensor([0.0, 0.0]),
        blocked=False,
        right_direction=True,
        task_type=1,
    )
    no_blockage_2 = MazeTask(
        embedding=Tensor([1.0, 1.0]),
        blocked=False,
        right_direction=True,
        task_type=2,
    )
    right_blockage_obj1 = MazeTask(
        embedding=Tensor([1.0, 0.0]), blocked=True, right_direction=True, task_type=0
    )
    left_blockage_obj1 = MazeTask(
        embedding=Tensor([-1.0, 0.0]), blocked=True, right_direction=False, task_type=4
    )
    right_blockage_obj2 = MazeTask(
        embedding=Tensor([0.0, 1.0]), blocked=False, right_direction=True, task_type=5
    )
    left_blockage_obj2 = MazeTask(
        embedding=Tensor([0.0, -1.0]),
        blocked=False,
        right_direction=False,
        task_type=6,
    )

    tasks = [
        no_blockage_1,
        no_blockage_2,
        right_blockage_obj1,
        left_blockage_obj1,
        right_blockage_obj2,
        left_blockage_obj2,
    ]

    # Write tasks to a pickle file
    with open("configs/meta/maze/v/simple_embeddings.pkl", "wb") as file:
        dill.dump(tasks, file)

    tasks = [
        right_blockage_obj1,
        left_blockage_obj1,
    ]
    with open("configs/meta/maze/v/double_embeddings.pkl", "wb") as file:
        dill.dump(tasks, file)


if __name__ == "__main__":
    make_single_tasks_file()
