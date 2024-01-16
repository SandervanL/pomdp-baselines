import os
import sys
import time
from typing import Callable

import torch

from ruamel.yaml import YAML
from absl import flags
from torch import Tensor

from policies.learner import MetaLearner
import torchkit.pytorch_utils as ptu
import utils.helpers as utl


class DemoLearner(MetaLearner):
    def demo(self, embedder: Callable[[str], Tensor]):
        last_task_blocked = False
        for task in self.eval_tasks:
            if task.blocked == last_task_blocked:
                continue
            last_task_blocked = task.blocked

            running_reward = 0
            obs_numpy, info = self.eval_env.reset(
                seed=self.seed + 1, options={"task": task}
            )
            obs = ptu.from_numpy(obs_numpy)
            obs = obs.reshape(1, obs.shape[-1])

            sentence = input(
                f"Object is {'heavy' if task.blocked else 'light'}. Please describe the object>"
            )
            task_embedding = embedder(sentence).unsqueeze(dim=0)
            action, reward, internal_state = self.agent.get_initial_info(
                task=task_embedding
            )

            step = 0
            while step < 20:
                (action, _, _, _), internal_state = self.agent.act(
                    prev_internal_state=internal_state,
                    prev_action=action,
                    reward=torch.tensor([[reward]]),
                    obs=obs,
                    deterministic=True,
                    task=task_embedding,  # TODO change here
                )

                next_obs, reward, terminated, truncated, info = utl.env_step(
                    self.eval_env, action.squeeze(dim=0)
                )
                step += 1
                running_reward += reward.item()
                done_rollout = (
                    ptu.get_numpy(terminated[0][0]) == 1.0
                    or ptu.get_numpy(truncated[0][0]) == 1.0
                )
                if done_rollout:
                    break

                time.sleep(0.4)
                obs = next_obs

            # print(
            #     f"Finished episode within {step} steps and received {running_reward} reward."
            # )


def define_flags():
    flags.DEFINE_string(
        "model_directory",
        None,
        "path to the directory with the agent weights",
        required=True,
    )
    flags.DEFINE_string("embedder", "simcse", "How to embed the sentences")
    flags.DEFINE_string(
        "task_file", None, "Which file of Maze tasks to use", required=True
    )
    flags.FLAGS(sys.argv)


def get_model_files(directory: str) -> tuple[str, str]:
    yml_file = None
    model_file = None
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pt") and "save" in root:
                assert model_file is None
                model_file = os.path.join(root, file)

            if file.endswith(".yml") or file.endswith(".yaml"):
                assert yml_file is None
                yml_file = os.path.join(root, file)

    return yml_file, model_file


def get_embedder(embedder: str):
    if embedder == "simcse":
        # set_huggingface_cache_dir()
        from simcse import SimCSE

        simcse_model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
        simcse_model.encode("hello world!")
        return simcse_model.encode

    raise NotImplementedError(f"Embedder {embedder} is not supported.")


def load_config() -> dict:
    model_yml_path, model_params_path = get_model_files(flags.FLAGS.model_directory)

    yaml = YAML()
    with open(model_yml_path, "r") as file:
        v = yaml.load(file)

    v["num_cpus"] = 1
    v["env"]["render_mode"] = "human"
    v["model_params_path"] = model_params_path

    # overwrite config params
    v["embedder"] = flags.FLAGS.embedder
    if flags.FLAGS.task_file is not None:
        v["env"]["task_file"] = flags.FLAGS.task_file

    return v


def main():
    define_flags()
    v = load_config()
    embedder = get_embedder(v["embedder"])

    demo_learner = DemoLearner(v["env"], v["train"], v["eval"], v["policy"], 0)
    demo_learner.agent.load_state_dict(torch.load(v["model_params_path"]))
    demo_learner.demo(embedder)


if __name__ == "__main__":
    main()
