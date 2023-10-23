import dill
from torch import Tensor

from envs.meta.maze.MultitaskMaze import MazeTask, anti_direction


def one_direction_file():
    no_blockage = MazeTask(
        embedding=Tensor([0.0]),
        blocked=False,
        task_type=0,
        sentence="no",
        word="no",
        short_direction=2,
        short_hook_direction=2,
        long_direction=3,
        long_hook_direction=3,
    )
    blockage = MazeTask(
        embedding=Tensor([1.0]),
        blocked=True,
        task_type=1,
        sentence="yes",
        word="yes",
        short_direction=2,
        short_hook_direction=2,
        long_direction=3,
        long_hook_direction=3,
    )
    tasks = [blockage, no_blockage]

    with open("embeddings/one_direction/perfect.dill", "wb") as file:
        dill.dump(tasks, file)


def two_directions_file():
    no_blockage_left = MazeTask(
        embedding=Tensor([0.0, 0.0]),
        blocked=False,
        task_type=0,
        sentence="noleft",
        word="noleft",
        short_direction=2,
        short_hook_direction=2,
        long_direction=3,
        long_hook_direction=3,
    )
    blockage_left = MazeTask(
        embedding=Tensor([1.0, 0.0]),
        blocked=True,
        task_type=1,
        sentence="yesleft",
        word="yesleft",
        short_direction=2,
        short_hook_direction=2,
        long_direction=3,
        long_hook_direction=3,
    )
    no_blockage_right = MazeTask(
        embedding=Tensor([0.0, 1.0]),
        blocked=False,
        task_type=2,
        sentence="noright",
        word="noright",
        short_direction=3,
        short_hook_direction=3,
        long_direction=2,
        long_hook_direction=2,
    )
    blockage_right = MazeTask(
        embedding=Tensor([1.0, 1.0]),
        blocked=True,
        task_type=3,
        sentence="yesright",
        word="yesright",
        short_direction=3,
        short_hook_direction=3,
        long_direction=2,
        long_hook_direction=2,
    )

    tasks = [no_blockage_left, blockage_left, no_blockage_right, blockage_right]

    with open("embeddings/two_directions/perfect.dill", "wb") as file:
        dill.dump(tasks, file)


def all_directions():
    word = 0
    tasks = []
    for long_direction in range(4):
        for long_hook_direction in range(4):
            for short_direction in range(4):
                for short_hook_direction in range(4):
                    if (
                        short_direction == long_direction
                        or short_hook_direction == long_direction
                        or anti_direction(short_direction) == short_hook_direction
                        or anti_direction(long_direction) == long_hook_direction
                    ):
                        continue
                    for blocked in [True, False]:
                        word += 1
                        tasks.append(
                            MazeTask(
                                embedding=Tensor(
                                    [
                                        float(long_direction),
                                        float(long_hook_direction),
                                        float(short_direction),
                                        float(short_hook_direction),
                                        float(blocked),
                                    ]
                                ),
                                blocked=blocked,
                                task_type=int(blocked),
                                word=str(word),
                                sentence=str(word),
                                short_direction=short_direction,
                                short_hook_direction=short_hook_direction,
                                long_direction=long_direction,
                                long_hook_direction=long_hook_direction,
                            )
                        )
    with open("embeddings/all_directions/perfect.dill", "wb") as file:
        dill.dump(tasks, file)


if __name__ == "__main__":
    one_direction_file()
    two_directions_file()
    all_directions()
