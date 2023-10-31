import os
import sys
import time
from copy import deepcopy

t0 = time.time()
from torch.multiprocessing import Pool
from ruamel.yaml import YAML
from absl import flags


def define_flags():
    flags.DEFINE_string("cfg", None, "path to configuration file")
    flags.DEFINE_string("env", None, "env_name")
    flags.DEFINE_string("algo", None, '["td3", "sac", "sacd"]')

    flags.DEFINE_boolean("automatic_entropy_tuning", None, "for [sac, sacd]")
    flags.DEFINE_float("target_entropy", None, "for [sac, sacd]")
    flags.DEFINE_float("entropy_alpha", None, "for [sac, sacd]")

    flags.DEFINE_integer("seed", None, "seed")
    flags.DEFINE_integer("cuda", None, "cuda device id")
    flags.DEFINE_boolean(
        "oracle",
        False,
        "whether observe the privileged information of POMDP, reduced to MDP",
    )
    flags.DEFINE_boolean("debug", False, "debug mode")
    flags.DEFINE_float("gamma", None, "discount factor")
    flags.DEFINE_string(
        "render_mode", None, "render mode ('null', 'human' or 'rgb_array')"
    )
    flags.DEFINE_integer(
        "embedding_obs_init", None, "How the embedding is appended to the obs"
    )
    flags.DEFINE_integer(
        "embedding_rnn_init", None, "How the embedding is initialized to the RNN"
    )
    flags.DEFINE_string("task_file", None, "The file to use for the embeddings.")
    flags.DEFINE_string(
        "task_selection", None, "How to select the tasks for train and eval."
    )
    flags.DEFINE_float(
        "num_updates_per_iter_float", None, "How many updates per environment step."
    )
    flags.DEFINE_integer(
        "num_updates_per_iter_int", None, "How many updates per iteration."
    )
    flags.DEFINE_integer("num_cpus", None, "How many cpus to use.")
    flags.DEFINE_float("uncertainty_scale", None, "The scale of the uncertainty.")

    flags.FLAGS(sys.argv)


def load_config() -> dict:
    FLAGS = flags.FLAGS
    yaml = YAML()
    with open(FLAGS.cfg, "r") as file:
        v = yaml.load(file)

    # overwrite config params
    if FLAGS.env is not None:
        v["env"]["env_name"] = FLAGS.env
    if FLAGS.algo is not None:
        v["policy"]["algo_name"] = FLAGS.algo
    if FLAGS.render_mode is not None:
        v["env"]["render_mode"] = (
            None if FLAGS.render_mode == "null" else FLAGS.render_mode
        )
    if FLAGS.task_selection is not None:
        v["env"]["task_selection"] = FLAGS.task_selection

    if (
        FLAGS.num_updates_per_iter_float is not None
        and FLAGS.num_updates_per_iter_int is not None
    ):
        raise ValueError(
            "Cannot set both num_updates_per_iter_float and num_updates_per_iter_int"
        )
    elif FLAGS.num_updates_per_iter_float is not None:
        v["train"]["num_updates_per_iter"] = FLAGS.num_updates_per_iter_float
    elif FLAGS.num_updates_per_iter_int is not None:
        v["train"]["num_updates_per_iter"] = FLAGS.num_updates_per_iter_int

    seq_model, algo = v["policy"]["seq_model"], v["policy"]["algo_name"]
    assert seq_model in ["mlp", "lstm", "gru", "lstm-mlp", "gru-mlp"]
    assert algo in ["td3", "sac", "sacd"]

    if FLAGS.automatic_entropy_tuning is not None:
        v["policy"][algo]["automatic_entropy_tuning"] = FLAGS.automatic_entropy_tuning
    if FLAGS.entropy_alpha is not None:
        v["policy"][algo]["entropy_alpha"] = FLAGS.entropy_alpha
    if FLAGS.target_entropy is not None:
        v["policy"][algo]["target_entropy"] = FLAGS.target_entropy
    if FLAGS.gamma is not None:
        v["policy"]["gamma"] = FLAGS.gamma
    if FLAGS.embedding_obs_init is not None:
        v["policy"]["embedding_obs_init"] = FLAGS.embedding_obs_init
    if FLAGS.embedding_rnn_init is not None:
        v["policy"]["embedding_rnn_init"] = FLAGS.embedding_rnn_init
    if FLAGS.uncertainty_scale is not None:
        v["policy"]["uncertainty"]["scale"] = FLAGS.uncertainty_scale

    if FLAGS.seed is not None:
        v["seed"] = FLAGS.seed
    if FLAGS.cuda is not None:
        v["cuda"] = FLAGS.cuda
    if FLAGS.oracle:
        v["env"]["oracle"] = True
    if FLAGS.task_file is not None:
        v["env"]["task_file"] = FLAGS.task_file

    if FLAGS.num_cpus is not None:
        v["num_cpus"] = FLAGS.num_cpus
    v["debug"] = FLAGS.debug

    return v


def local_runner_main(v: dict):
    from policies.runner_main import runner_main

    runner_main(v)


def main():
    define_flags()
    v = load_config()

    num_cpus = v["num_cpus"] if "num_cpus" in v else os.cpu_count()
    if num_cpus is None:
        raise ValueError("Could not detect the number of cpus")
    print("Using", num_cpus, "CPUs")
    runner_config = [v] * num_cpus
    for index, seed in enumerate(range(num_cpus)):
        new_v = deepcopy(v)
        new_v["seed"] = v["seed"] + index
        new_v["t0"] = t0
        runner_config[index] = new_v

    if num_cpus > 1:
        with Pool(num_cpus) as p:
            p.map(local_runner_main, runner_config)
    else:
        local_runner_main(runner_config[0])


if __name__ == "__main__":
    main()
