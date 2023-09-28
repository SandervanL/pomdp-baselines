import math

import pandas as pd
from pandas.errors import ParserError

import wandb

import os
import yaml
import csv

from ruamel.yaml import YAML


def find_csv_and_yaml_pairs(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if len(dirnames) == 1 and dirnames[0] == "save":
            csv_file = None
            yaml_file = None

            for filename in filenames:
                if filename.endswith(".csv"):
                    csv_file = os.path.join(dirpath, filename)
                elif filename.endswith(".yml") or filename.endswith(".yaml"):
                    yaml_file = os.path.join(dirpath, filename)

            if csv_file and yaml_file:
                yield csv_file, yaml_file


def init_wandb(yaml_path: str, project: str, is_old: bool, group: str) -> None:
    yaml = YAML()
    with open(yaml_path, "r") as file:
        config = yaml.load(file)

    config["old"] = is_old
    wandb.init(project=project, group=group, config=config)


def insert_wandb(csv_file: str) -> None:
    try:
        df = pd.read_csv(csv_file)
    except ParserError:
        print(f"Could not read from {csv_file}")
        return

    data_list = df.to_dict(orient="records")
    prev_dict = {
        "z/env_steps": 0,
        "z/rl_steps": 0,
    }
    counter = 0
    for index, data in enumerate(data_list):
        filtered_data = {
            key: (value if not math.isnan(value) else prev_dict[key])
            for key, value in data.items()
            if not math.isnan(value) or key in prev_dict
        }

        # Fill some values with future values
        for key, value in data.items():
            if key not in filtered_data:
                for _, next_data in enumerate(data_list, start=index + 1):
                    if not math.isnan(next_data[key]):
                        filtered_data[key] = next_data[key]
                        break

            if key not in filtered_data or math.isnan(filtered_data[key]):
                print("What is this")
        counter += 1
        prev_dict = filtered_data
        wandb.log(filtered_data)


def main():
    group = "embedding"
    root_directory = f"C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\Random\\embedding-tests\\{group}-logs"
    project = "Language Assistance"
    is_old = False

    for csv_file, yaml_file in find_csv_and_yaml_pairs(root_directory):
        print(f"CSV File: {csv_file}")
        print(f"YAML File: {yaml_file}")
        init_wandb(yaml_file, project, is_old, group)
        insert_wandb(csv_file)
        wandb.finish()
        # You can process the CSV and YAML files here as needed


if __name__ == "__main__":
    main()
