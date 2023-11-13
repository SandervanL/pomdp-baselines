import torch
from sklearn.linear_model import LinearRegression

from envs.meta.maze.MultitaskMaze import load_tasks_dir


def main(file_path: str):
    tasks = load_tasks_dir(file_path)
    x = torch.stack([task.embedding for task in tasks]).cpu().numpy()
    y = torch.tensor([task.task_type for task in tasks]).cpu().numpy()

    model = LinearRegression()
    model.fit(x, y)
    r_squared = model.score(x, y)
    print(f"R^2: {r_squared:.3f}")
    y_predict = model.predict(x)

    for i in range(0, 101):
        threshold = i / 100
        y_predict_binary = y_predict > threshold
        accuracy = (y_predict_binary == y).sum() / len(y)
        if accuracy >= 0.99999:
            print(f"{threshold:.2f} {accuracy:.7f}")
            break
    else:
        print("No linear decision boundary found")


if __name__ == "__main__":
    main(
        "C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\embeddings\\one_direction\\sentences_simcse"
    )
