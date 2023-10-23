import math

import pandas as pd
from pandas.errors import ParserError

import wandb

import os

from ruamel.yaml import YAML


def find_csv_and_yaml_pairs(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        csv_file = None
        yaml_file = None

        for filename in filenames:
            if filename.endswith("progress.csv"):
                csv_file = os.path.join(dirpath, filename)
            elif filename.endswith(".yml") or filename.endswith(".yaml"):
                yaml_file = os.path.join(dirpath, filename)

        if csv_file and yaml_file:
            yield csv_file, yaml_file


groups = {
    "sentences_word2vec.dill": "Sentences Word2vec",
    "sentences_simcse.dill": "Sentences SimCSE",
    "words_word2vec.dill": "Words Word2Vec",
    "words_simcse.dill": "Words SimCSE",
}

groups_info = {
    "sentences_simcse.dill": "Sentences",
    "words_simcse.dill": "Words",
    "perfect.dill": "Perfect",
}

# sentences_simcse.dill-44 2727340
# sentences_word2vec.dill-45 2727342
# sentences_simcse.dill-45 2727344
# sentences_word2vec.dill-46 2727346
# words_word2vec.dill-46 2727347
# words_simcse.dill-46 2727349
# Everything above 47 failed immediately (except sentences_word2vec.dill-47)


def init_wandb(yaml_path: str, project: str, is_old: bool, group: str) -> bool:
    yaml = YAML()
    with open(yaml_path, "r") as file:
        config = yaml.load(file)

    # For generalization
    # selection = config["env"]["task_selection"]
    # if selection == "random":
    #     group = "Global"
    # elif selection == "random-word":
    #     group = "80% Within Words"
    # else:
    #     raise f"Unknown task selection {selection}"

    # For info dilution
    # task_file = config["env"]["task_file"]
    # task_name = None
    # for key in groups_info.keys():
    #     if key in task_file:
    #         task_name = key
    #         break
    # if task_name is None:
    #     raise f"Could not find group for {task_file}"
    # group = groups_info[task_name]

    # For embedding type tasks
    # if config["env"]["task_file"] not in groups:
    #     raise f"Could not find group for {config['env']['task_file']}"
    # group = groups[config["env"]["task_file"]]

    config["old"] = is_old
    config["file"] = yaml_path
    wandb.init(project=project, group=group, config=config)
    return True


def insert_wandb(csv_file: str) -> None:
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
                wandb.log(old_data)
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
    group = ""
    root_directory = "C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\Logs\\embedding-consumption\\embedding-fifty-logs"
    project = "Embedding Consumption 2"
    is_old = False

    for csv_file, yaml_file in find_csv_and_yaml_pairs(root_directory):
        print(f"CSV File: {csv_file}")
        print(f"YAML File: {yaml_file}")

        success = init_wandb(yaml_file, project, is_old, group)
        if not success:
            continue
        insert_wandb(csv_file)
        wandb.finish()
        # You can process the CSV and YAML files here as needed


if __name__ == "__main__":
    main()
