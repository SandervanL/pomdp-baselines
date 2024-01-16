import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

from envs.meta.maze.MazeTask import MazeTask
from envs.meta.maze.MultitaskMaze import load_tasks_dir

misclassified_objects = [
    "Iron",
    "Meteorite",
    "Ship",
    "Machine",
    "Spiderweb",
    "Concrete",
    "Quarry",
    "Excavator",
    "Pile",
    "Train",
    "Car",
    "Barbell",
    "Cruise",
    "Lead",
    "Engine",
    "Mammoth",
    "Locomotive",
    "Submarine",
    "Turbine",
    "Pencil",
    "Tadpole",
    "Brick",
    "Truck",
    "Lumber",
    "Tractor",
    "Seed",
    "Straw",
]


def annotate_objects(tsne_result: np.ndarray, tasks: list[MazeTask]):
    # Find the indices of the misclassified objects
    indices = []
    for obj_index, obj in enumerate(misclassified_objects):
        if obj_index > 10:
            break
        indices.append(
            [index // 30 for index, task in enumerate(tasks) if task.word == obj][0]
        )

    annotate_groups(indices, tsne_result, tasks)


def annotate_groups(
    groups: list[int],
    tsne_result: np.ndarray,
    tasks: list[MazeTask],
):
    for _index in groups:
        index = _index * 30
        mean = np.mean(tsne_result[index : index + 30, :], axis=0)
        plt.annotate(
            tasks[index].word,
            # str(index // 30),
            xy=mean,
            horizontalalignment="center",
            xytext=(0, -34),
            textcoords="offset points",
            fontsize=7,
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
        )


def draw_scatter(
    groups: list[int] | None,
    tsne_result: np.ndarray,
    tasks: list[MazeTask],
):
    objects = [task.word for task in tasks]
    labels = [objects.index(task.word) for task in tasks]
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], s=5, c=labels, cmap="bwr")
    legend_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="blue", markersize=5
        ),
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="red", markersize=5
        ),
    ]
    legend_labels = ["Heavy", "Light"]
    plt.legend(legend_handles, legend_labels)

    plt.title("t-SNE Plot of Unidirectional Sentence Embeddings")
    plt.xticks([])
    plt.yticks([])
    if groups is not None:
        annotate_groups(groups, tsne_result, tasks)
    else:
        annotate_objects(tsne_result, tasks)

    file_addition = "-misclassified" if groups is None else ""
    plt.savefig(
        f"C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\Thesis\\figures\\tsne{file_addition}.png"
    )
    plt.show()


def main(tasks_dir: str):
    # Bring the mxn embeddings down to a mx2 matrix using PCA
    tasks: list[MazeTask] = load_tasks_dir(tasks_dir)
    embeddings = torch.stack([task.embedding for task in tasks])

    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(embeddings)
    with open("tsne_result.npy", "wb") as file:
        np.save(file, tsne_result)


def main_draw():
    with open("tsne_result.npy", "rb") as file:
        tsne_result = np.load(file)
    tasks: list[MazeTask] = load_tasks_dir(
        "C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\embeddings\\left_coupled\\sentences_simcse"
    )
    draw_scatter(
        [99, 28, 92, 54, 55, 95, 3, 116, 114, 21, 69, 78, 102, 103, 119],
        tsne_result,
        tasks,
    )
    draw_scatter(
        None,
        tsne_result,
        tasks,
    )
    print("Breakpoint")

    # pca_result = torch.mm(embeddings, v[:, :2])

    # Plot u as a scatter plot using matplotlib
    # plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, s=2)
    # plt.title("TSNE of Unidirectional Sentence Embeddings")
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    # plt.show()


if __name__ == "__main__":
    # main(
    #     "C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\embeddings\\left_coupled\\sentences_simcse"
    # )
    main_draw()
