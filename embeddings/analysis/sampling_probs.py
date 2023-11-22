import torch
from tqdm import tqdm


def main():
    total_sum = 0
    num_runs = 100000
    for run in tqdm(range(num_runs)):
        shuffled_tasks = torch.randperm(3600)
        chosen_tasks = shuffled_tasks[: int(0.3 * 3600)]
        for i in range(0, 3600, 30):
            valid_tasks = torch.logical_and(chosen_tasks >= i, chosen_tasks < i + 30)
            if valid_tasks.int().sum() == 0:
                total_sum += 1
                print(total_sum / run)
                break

    print(total_sum / num_runs)


if __name__ == "__main__":
    main()
