import os
import sys
import time
from datetime import datetime

t0 = time.time()
import socket
import numpy as np
import torch
from ruamel.yaml import YAML
from absl import flags
from utils import logger
from pathlib import Path
import psutil


def find_yaml_and_model(dir: str):
    # I have a directory structure with a bunch of folders with a yaml file, and a save
    # folder with a model. They correspond to each other.
    # Example of dir: folder1/2021-08-31_15-00-00_123456
    # Example of yaml file: folder1/2021-08-31_15-00-00_123456/variant_123456.yml
    # Example of model file: folder1/2021-08-31_15-00-00_123456/save/model.pt
    # Traverse the directory structure for me, and yield the directory, yaml, and model file paths.
    for dirpath, dirnames, filenames in os.walk(dir):
        if len(dirnames) == 1 and dirnames[0] == "save":
            yaml_file = None
            model_file = None

            for filename in filenames:
                if filename.endswith(".pt"):
                    model_file = os.path.join(dirpath, filename)
                elif filename.endswith(".yml") or filename.endswith(".yaml"):
                    yaml_file = os.path.join(dirpath, filename)

            if yaml_file and model_file:
                yield dirpath, yaml_file, model_file


def main():
    FLAGS = flags.FLAGS
    flags.DEFINE_string("dir", None, "path to directory with model and config")
    flags.DEFINE_integer("cuda", None, "cuda device id")
    flags.DEFINE_string(
        "render_mode", None, "render mode ('null', 'human' or 'rgb_array')"
    )
    flags.FLAGS(sys.argv)

    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    logger.log(f"preload cost {time.time() - t0:.2f}s")

    yaml = YAML()
    for dirpath, yaml_file, model_file in find_yaml_and_model(FLAGS.dir):
        with open(yaml_file, "r") as file:
            config = yaml.load(file)

        current_dir = os.getcwd()
        os.chdir(dirpath)
        from policies.learner import LEARNER_CLASS

        os.chdir(current_dir)

        config["env"]["num_eval_tasks"] = None  # Evaluate on all tasks
        learner = LEARNER_CLASS[config["env"]["env_type"]](
            config["env"], config["train"], config["eval"], config["policy"], seed=0
        )
        learner.agent.state_dict = torch.load(model_file)
        learner.log_evaluate(learner.eval_tasks)


if __name__ == "__main__":
    main()
