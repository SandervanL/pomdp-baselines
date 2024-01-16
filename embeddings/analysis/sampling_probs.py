from typing import Callable

import torch
from tqdm import tqdm


def brute_force():
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


import math


def choose(n, r):
    if n < r:
        return 0
    f = math.factorial
    return f(n) // f(r) // f(n - r)


def sampling_probability_ryry(train_set_size: int, min_groups=1) -> float:
    if train_set_size < 120 - min_groups + 1:
        return 0

    total_sum = 0
    for k in range(min_groups, 121):
        total_sum += choose(120, k) * choose(3600 - 30 * k, train_set_size)

    return 1 - total_sum / choose(3600, train_set_size)


def large_num_to_str(num: int) -> str:
    e_power = math.floor(math.log10(num))
    divided_num = num / (10**e_power)
    return f"{divided_num}e{e_power}"


def sampling_probability_gabriel(train_set_size: int, min_groups=1) -> float:
    total_sum = 0
    for i in range(min_groups, 121):
        total_sum += (
            (-1) ** (i + 1) * choose(120, i) * choose((120 - i) * 30, train_set_size)
        )
    divisor = choose(3600, train_set_size)
    # print(f"Total sum: {large_num_to_str(total_sum)}")
    # print(f"Divisor: {large_num_to_str(divisor)}")
    return 1 - total_sum / divisor


def seed_sampling_prob(train_set_size: int) -> float:
    prob = sampling_probability_gabriel(train_set_size)
    return prob**48


def binary_search(
    func: Callable[[int], float], goal: float, start: int, end: int
) -> int:
    while start < end:
        mid = (start + end) // 2
        result = func(mid)
        if result < goal:
            start = mid + 1
        elif result > goal:
            end = mid - 1
        else:
            return mid

    return start


if __name__ == "__main__":
    for i in range(120, 3601, 10):
        print(f"({(i / 3600):.4f}, {seed_sampling_prob(i):.4f})")
    # print(sampling_probability_gabriel(1080))
    # threshold = binary_search(sampling_probability_gabriel, 0.5, 0, 3600)
    # print(threshold)
    # print(sampling_probability_gabriel(threshold))
