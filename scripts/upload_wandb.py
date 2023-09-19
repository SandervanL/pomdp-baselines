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


def init_wandb(yaml_path: str, project: str, is_old: bool) -> None:
    yaml = YAML()
    with open(yaml_path, "r") as file:
        config = yaml.load(file)

    wandb.init(
        project=project,
        config={
            "env": config["env"]["env_name"],
            "actions": config["env"].get("valid_actions"),
            "old": is_old,
        },
    )


def insert_wandb(csv_file: str) -> None:
    try:
        df = pd.read_csv(csv_file)
    except ParserError:
        print(f"Could not read from {csv_file}")
        return

    data_list = df.to_dict(orient="records")
    for data in data_list:
        filtered_data = {
            key: value for key, value in data.items() if not math.isnan(value)
        }
        wandb.log(filtered_data)


def main():
    root_directory = (
        "C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\"
        "Random\\logs-distance-test-new 15-09\\logs\\pomdp"
    )
    project = "Distance Test"
    is_old = False

    found_csv = False
    for csv_file, yaml_file in find_csv_and_yaml_pairs(root_directory):
        if not found_csv:
            if (
                csv_file
                == "C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\Random\\logs-distance-test-new 15-09\\logs\\pomdp\\double-blocked-4-maze-partial-v0\\sacd_lstm\\len--1\\bs-32\\freq-1\\oar\\09-15_16-44_04_62\\progress.csv"
            ):
                found_csv = True
            continue

        print(f"CSV File: {csv_file}")
        print(f"YAML File: {yaml_file}")
        init_wandb(yaml_file, project, is_old)
        insert_wandb(csv_file)
        wandb.finish()
        # You can process the CSV and YAML files here as needed


if __name__ == "__main__":
    main()
