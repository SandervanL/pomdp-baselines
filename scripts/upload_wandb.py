import math
import multiprocessing

import pandas as pd
from pandas.errors import ParserError

import wandb

import os

from ruamel.yaml import YAML


def find_csv_and_yaml_pairs(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        csv_file = None
        yaml_file = None
        csv_filled = None

        for filename in filenames:
            if filename.endswith("progress.csv"):
                csv_file = os.path.join(dirpath, filename)
            elif filename.endswith(".yml") or filename.endswith(".yaml"):
                yaml_file = os.path.join(dirpath, filename)
            elif filename.endswith("progress-filled.csv"):
                csv_filled = os.path.join(dirpath, filename)

        if csv_file and yaml_file and csv_filled is None:
            yield csv_file, yaml_file


groups = {
    "sentences_word2vec.dill": "Sentences Word2vec",
    "sentences_simcse.dill": "Sentences SimCSE",
    "words_word2vec.dill": "Words Word2Vec",
    "words_simcse.dill": "Words SimCSE",
    "object_type_word2vec.dill": "Type Word2Vec",
    "object_type_simcse.dill": "Type SimCSE",
}

groups_info = {
    "sentences_simcse.dill": "Sentences",
    "words_simcse.dill": "Words",
    "object_type_simcse.dill": "Object Type",
    "perfect.dill": "Perfect",
}


def get_group(groups: dict[str, str], filename: str) -> str:
    for key, value in groups.items():
        if key in filename:
            return value
    raise f"Could not find group for {filename}"


def init_wandb(yaml_path: str, project: str, is_old: bool, group: str) -> bool:
    yaml = YAML()
    with open(yaml_path, "r") as file:
        config = yaml.load(file)

    # For generalization
    selection = config["env"]["task_selection"]
    percentage = int(round(100 * (1 - config["env"]["train_test_split"])))
    if selection == "random":
        group_name = "Global"
    elif selection == "random-word":
        group_name = "Words"
    elif selection == "random-within-word":
        group_name = "Within Words"
    else:
        raise f"Unknown task selection {selection}"
    group = f"{percentage}% {group_name}"

    # For info dilution
    # group = get_group(groups_info, config["env"]["task_file"])

    # For embedding type tasks
    # group = get_group(groups, config["env"]["task_file"])

    config["old"] = is_old
    config["file"] = yaml_path
    wandb.init(project=project, group=group, config=config)
    return True


def insert_wandb(csv_file: str) -> None:
    print(csv_file)
    try:
        df = pd.read_csv(csv_file)
    except ParserError:
        raise f"Could not read from {csv_file}"

    data_list = df.to_dict(orient="records")
    old_data = data_list[0]
    old_data["metrics/cumulative_regret_blocked"] = 0
    old_data["metrics/cumulative_regret_unblocked"] = 0
    old_data["metrics/cumulative_regret_total"] = 0
    old_data["metrics/cumulative_regret_bayesian"] = 0

    previous_env_steps = 6000
    result_table = None
    for index, data in enumerate(data_list):
        # Update the data with the new values of the current row
        new_data = {
            key: (
                old_data[key] if key not in data or math.isnan(data[key]) else data[key]
            )
            for key in old_data.keys()
        }
        if not math.isnan(data["rl_loss/qf1_loss"]) or index == 0:
            old_data = new_data
            continue

        # Append the old data first (new data is not seen for the previous steps)
        if not math.isnan(old_data["z/env_steps"]):
            if math.isnan(data["z/env_steps"]):
                raise f"Found NaN in {csv_file} at index {index}"
            for time in range(previous_env_steps, int(data["z/env_steps"]), 10):
                old_data["z/env_steps"] = time
                if result_table is None:
                    result_table = pd.DataFrame(old_data, index=[0])
                else:
                    result_table = pd.concat(
                        [result_table, pd.DataFrame([old_data])], ignore_index=True
                    )
                # wandb.log(old_data)
                previous_env_steps = time + 10

        # Calculate the new values
        blocked = data["metrics/total_steps_eval_blocked"]
        unblocked = data["metrics/total_steps_eval_unblocked"]
        blocked_regret = blocked - (
            13 if "rnn-0" in csv_file and "obs-0" in csv_file else 10
        )
        unblocked_regret = unblocked - 4
        new_data["metrics/regret"] = blocked_regret + unblocked_regret
        new_data["metrics/cumulative_regret_blocked"] += blocked_regret
        new_data["metrics/cumulative_regret_unblocked"] += unblocked_regret
        new_data["metrics/cumulative_regret_total"] += blocked_regret + unblocked_regret
        old_data = new_data

    new_filename = csv_file.replace("progress.csv", "progress-filled.csv")
    result_table.to_csv(new_filename, index=False)


def main():
    group = "Baseline"
    root_directory = "C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\Logs\\directions\\directions2-logs"

    project = "Generalization 5"
    is_old = False

    with multiprocessing.Pool(8) as pool:
        csv_yamls = list(find_csv_and_yaml_pairs(root_directory))
        csv_files = [csv_file for csv_file, _ in csv_yamls]

        # Pool map to call insert_wandb
        pool.map(insert_wandb, csv_files)

    # for csv_file, yaml_file in find_csv_and_yaml_pairs(root_directory):
    #     print(f"CSV File: {csv_file}")
    #     print(f"YAML File: {yaml_file}")
    #
    #     success = init_wandb(yaml_file, project, is_old, group)
    #     if not success:
    #         continue
    #     insert_wandb(csv_file)
    #     wandb.finish()


if __name__ == "__main__":
    main()
