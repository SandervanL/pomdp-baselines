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
            if filename.endswith(".csv"):
                csv_file = os.path.join(dirpath, filename)
            elif filename.endswith(".yml") or filename.endswith(".yaml"):
                yaml_file = os.path.join(dirpath, filename)

        if csv_file and yaml_file:
            yield csv_file, yaml_file


groups = {
    "/home/sajvanleeuwen/embedding-type/deployment/../embeddings/embeddings/one_direction/sentences_word2vec.dill": "Sentences Word2vec",
    "/home/sajvanleeuwen/embedding-type/deployment/../embeddings/embeddings/one_direction/sentences_simcse.dill": "Sentences SimCSE",
    "/home/sajvanleeuwen/embedding-type/deployment/../embeddings/embeddings/one_direction/words_word2vec.dill": "Words Word2Vec",
    "/home/sajvanleeuwen/embedding-type/deployment/../embeddings/embeddings/one_direction/words_simcse.dill": "Words SimCSE",
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

    seed = config["seed"]
    if config["env"]["task_file"] not in groups:
        print("breakpoint")
    group = groups[config["env"]["task_file"]]

    if not (
        seed in [42, 43]
        or (seed == 44 and group != "Sentences SimCSE")
        or (seed == 45 and "Sentences" not in group)
    ):
        return False

    config["old"] = is_old
    config["file"] = yaml_path
    wandb.init(project=project, group=group, config=config)
    return True


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
    group = "Words Word2vec"
    root_directory = f"C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\Logs\\embedding-type\\embedding-type-logs"
    project = "Embedding Type"
    is_old = False

    for csv_file, yaml_file in find_csv_and_yaml_pairs(root_directory):
        print(f"CSV File: {csv_file}")
        print(f"YAML File: {yaml_file}")
        if "rnn-0" not in csv_file or "obs-1" not in csv_file:
            continue

        success = init_wandb(yaml_file, project, is_old, group)
        if not success:
            continue
        insert_wandb(csv_file)
        wandb.finish()
        # You can process the CSV and YAML files here as needed


if __name__ == "__main__":
    main()
