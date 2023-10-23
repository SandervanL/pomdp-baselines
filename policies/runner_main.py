import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
import wandb
from absl import flags
import numpy as np
import torch
from ruamel.yaml import YAML

from torchkit.pytorch_utils import set_gpu_mode
from utils import system, logger
from policies.learner import LEARNER_CLASS


def set_seed(v: dict):
    # system: device, threads, seed, pid
    seed = v["seed"]
    system.reproduce(seed)

    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)


def build_folder_name(v: dict):
    pid = str(os.getpid())
    if "SLURM_JOB_ID" in os.environ:
        pid += "_" + str(os.environ["SLURM_JOB_ID"])  # use job id

    # set gpu
    set_gpu_mode(torch.cuda.is_available() and v["cuda"] >= 0, v["cuda"])

    # logs
    if v["debug"]:
        exp_id = "debug/"
    else:
        exp_id = "logs/"

    env_type = v["env"]["env_type"]
    # if len(v["env"]["env_name"].split("-")) == 3:
    #     # pomdp env: name-{F/P/V}-v0
    #     env_name, pomdp_type, _ = v["env"]["env_name"].split("-")
    #     env_name = env_name + "/" + pomdp_type
    # else:
    #     env_name = v["env"]["env_name"]
    exp_id += f"{env_type}/"
    # exp_id += f"f{env_name}/"
    if "oracle" in v["env"] and v["env"]["oracle"] == True:
        oracle = True
    else:
        oracle = False

    seq_model, algo = v["policy"]["seq_model"], v["policy"]["algo_name"]
    if seq_model == "mlp":
        if oracle:
            algo_name = f"oracle_{algo}"
        else:
            algo_name = f"Markovian_{algo}"
        # exp_id += algo_name
    else:  # rnn
        if oracle:
            exp_id += "oracle_"
        if "rnn_num_layers" in v["policy"]:
            rnn_num_layers = v["policy"]["rnn_num_layers"]
            if rnn_num_layers == 1:
                rnn_num_layers = ""
            else:
                rnn_num_layers = str(rnn_num_layers)
        else:
            rnn_num_layers = ""
        # exp_id += f"{algo}_{rnn_num_layers}{seq_model}"
        if "separate" in v["policy"] and v["policy"]["separate"] == False:
            exp_id += "_shared"
    exp_id += f"obs-{v['policy']['embedding_obs_init']}/"
    exp_id += f"rnn-{v['policy']['embedding_rnn_init']}/"
    exp_id += f"updates-{v['train']['num_updates_per_iter']}/"

    # if algo in ["sac", "sacd"]:
    #     if not v["policy"][algo]["automatic_entropy_tuning"]:
    #         exp_id += f"alpha-{v['policy'][algo]['entropy_alpha']}/"
    #     elif "target_entropy" in v["policy"]:
    #         exp_id += f"ent-{v['policy'][algo]['target_entropy']}/"

    # exp_id += f"gamma-{v['policy']['gamma']}/"

    # if seq_model != "mlp":
    # exp_id += f"len-{v['train']['sampled_seq_len']}/"
    # exp_id += f"bs-{v['train']['batch_size']}/"

    # exp_id += f"baseline-{v['train']['sample_weight_baseline']}/"
    # exp_id += f"freq-{v['train']['num_updates_per_iter']}/"
    # assert v["policy"]["observ_embedding_size"] > 0
    # policy_input_str = "o"
    # if v["policy"]["action_embedding_size"] > 0:
    #     policy_input_str += "a"
    # if v["policy"]["reward_embedding_size"] > 0:
    #     policy_input_str += "r"
    # exp_id += policy_input_str + "/"

    if "task_file" in v["env"]:
        file_part = v["env"]["task_file"].split("/")[-1]
        file_part = file_part.split(".")[0]
        exp_id += f"task-{file_part}/"
    if "task_selection" in v["env"]:
        exp_id += f"selection-{v['env']['task_selection']}/"

    exp_id += f"seed-{v['seed']}/"
    os.makedirs(exp_id, exist_ok=True)
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-2]
    log_folder = os.path.join(exp_id, time_str)
    logger_formats = ["stdout", "log", "csv"]
    if v["eval"]["log_tensorboard"]:
        logger_formats.append("tensorboard")
    logger.configure(dir=log_folder, format_strs=logger_formats, precision=4)
    logger.log(f"preload cost {time.time() - v['t0']:.2f}s")

    os.system(f"cp -r policies/ {log_folder}")
    yaml = YAML()
    yaml.dump(v, Path(f"{log_folder}/variant_{pid}.yml"))
    key_flags = flags.FLAGS.get_key_flags_for_module(sys.argv[0])
    logger.log("\n".join(f.serialize() for f in key_flags) + "\n")
    logger.log("pid", pid, socket.gethostname())
    os.makedirs(os.path.join(logger.get_dir(), "save"))
    return log_folder


def runner_main(v: dict):
    set_seed(v)
    log_folder = build_folder_name(v)

    seed = v["seed"]
    # start training
    learner = LEARNER_CLASS[v["env"]["env_type"]](
        env_args=v["env"],
        train_args=v["train"],
        eval_args=v["eval"],
        policy_args=v["policy"],
        seed=seed,
    )

    logger.log(
        f"total RAM usage: {psutil.Process().memory_info().rss / 1024 ** 3 :.2f} GB\n"
    )

    # wandb.init(
    #     project="Test 2",
    #     config=v,
    #     dir=log_folder,
    # )

    learner.train()
    # wandb.finish()
