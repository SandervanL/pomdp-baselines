""" Recommended Architecture
Separate RNN architecture is inspired by a popular RL repo
https://github.com/quantumiracle/Popular-RL-Algorithms/blob/master/POMDP/common/value_networks.py#L110
which has another branch to encode current state (and action)

Hidden state update functions get_hidden_state() is inspired by varibad encoder 
https://github.com/lmzintgraf/varibad/blob/master/models/encoder.py
"""
from typing import Optional

import numpy as np
import torch
from copy import deepcopy
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Adam

from policies.rl.base import RLAlgorithmBase
from utils import helpers as utl
from policies.rl import RL_ALGORITHMS
import torchkit.pytorch_utils as ptu
from policies.models.recurrent_critic import CriticRnn
from policies.models.recurrent_actor import ActorRnn


class ModelFreeOffPolicy_Separate_RNN(nn.Module):
    """Recommended Architecture
    Recurrent Actor and Recurrent Critic with separate RNNs
    """

    ARCH = "memory"
    Markov_Actor = False
    Markov_Critic = False

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        algo_name: str,
        rnn_num_layers: int = 1,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 5e-3,
        # pixel obs
        image_encoder_fn=lambda: None,
        **kwargs
    ):
        super().__init__()

        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.gamma: float = gamma
        self.tau: float = tau

        algo_kwargs = kwargs[algo_name] if algo_name in kwargs else {}
        self.algo: RLAlgorithmBase = RL_ALGORITHMS[algo_name](
            **algo_kwargs, action_dim=action_dim
        )

        # Critics
        self.critic = CriticRnn(
            obs_dim=obs_dim,
            action_dim=action_dim,
            algo=self.algo,
            rnn_num_layers=rnn_num_layers,
            image_encoder=image_encoder_fn(),  # separate weight,
            **kwargs,
        )
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        # target networks
        self.critic_target = deepcopy(self.critic)

        # Actor
        self.actor = ActorRnn(
            obs_dim=obs_dim,
            action_dim=action_dim,
            algo=self.algo,
            rnn_num_layers=rnn_num_layers,
            image_encoder=image_encoder_fn(),  # separate weight
            **kwargs,
        )
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        # target networks
        self.actor_target: ActorRnn = deepcopy(self.actor)

    @torch.no_grad()
    def get_initial_info(self, *args, **kwargs) -> tuple[Tensor, Tensor, tuple[Tensor]]:
        return self.actor.get_initial_info(*args, **kwargs)

    @torch.no_grad()
    def act(
        self,
        prev_internal_state: tuple[Tensor, Tensor],
        prev_action: Tensor,
        reward: Tensor,
        obs: Tensor,
        task: Optional[Tensor] = None,
        deterministic: bool = False,
        return_log_prob: bool = False,
        valid_actions: Optional[np.ndarray] = None,
    ):
        prev_action = prev_action.unsqueeze(0)  # (1, B, dim)
        reward = reward.unsqueeze(0)  # (1, B, 1)
        obs = obs.unsqueeze(0)  # (1, B, dim)
        task = task.unsqueeze(0)  # (1, B, dim)

        current_action_tuple, current_internal_state = self.actor.act(
            prev_internal_state=prev_internal_state,
            prev_action=prev_action,
            reward=reward,
            obs=obs,
            task=task,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
            valid_actions=valid_actions,
        )

        return current_action_tuple, current_internal_state

    def forward(
        self,
        actions: Tensor,
        rewards: Tensor,
        observations: Tensor,
        dones: Tensor,
        masks: Tensor,
        tasks: Optional[Tensor] = None,
    ) -> dict[str, float]:
        """
        For actions a, rewards r, observations o, dones d: (T+1, B, dim)
                where for each t in [0, T], take action a[t], then receive reward r[t], done d[t], and next obs o[t]
                the hidden state h[t](, c[t]) = RNN(h[t-1](, c[t-1]), a[t], r[t], o[t])
                specially, a[0]=r[0]=d[0]=h[0]=c[0]=0.0, o[0] is the initial obs

        The loss is still on the Q value Q(h[t], a[t]) with real actions taken, i.e. t in [1, T]
                based on Masks (T, B, 1)
        """
        assert (
            actions.dim()
            == rewards.dim()
            == dones.dim()
            == observations.dim()
            == masks.dim()
            == (3 if tasks is None else tasks.dim())
            == 3
        )
        assert (
            actions.shape[0]
            == rewards.shape[0]
            == dones.shape[0]
            == observations.shape[0]
            == (dones.shape[0] if tasks is None else tasks.shape[0])
            == masks.shape[0] + 1
        )
        num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss

        ### 1. Critic loss
        (q1_pred, q2_pred), q_target = self.algo.critic_loss(
            markov_actor=self.Markov_Actor,
            markov_critic=self.Markov_Critic,
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            tasks=tasks,
            gamma=self.gamma,
        )

        # masked Bellman error: masks (T,B,1) ignore the invalid error
        # this is not equal to masks * q1_pred, cuz the denominator in mean()
        # 	should depend on masks > 0.0, not a constant B*T
        q1_pred, q2_pred = q1_pred * masks, q2_pred * masks
        q_target = q_target * masks
        qf1_loss = ((q1_pred - q_target) ** 2).sum() / num_valid  # TD error
        qf2_loss = ((q2_pred - q_target) ** 2).sum() / num_valid  # TD error

        self.critic_optimizer.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.critic_optimizer.step()

        ### 2. Actor loss
        policy_loss, log_probs = self.algo.actor_loss(
            markov_actor=self.Markov_Actor,
            markov_critic=self.Markov_Critic,
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,
            observations=observations,
            actions=actions,
            rewards=rewards,
            tasks=tasks,
        )
        # masked policy_loss
        policy_loss = (policy_loss * masks).sum() / num_valid

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        outputs = {
            "qf1_loss": qf1_loss.item(),
            "qf2_loss": qf2_loss.item(),
            "policy_loss": policy_loss.item(),
        }

        ### 3. soft update
        self.soft_target_update()

        ### 4. update others like alpha
        if log_probs is not None:
            # extract valid log_probs
            with torch.no_grad():
                current_log_probs = (log_probs[:-1] * masks).sum() / num_valid
                current_log_probs = current_log_probs.item()

            other_info = self.algo.update_others(current_log_probs)
            outputs.update(other_info)

        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.critic, self.critic_target, self.tau)
        if self.algo.use_target_actor:
            ptu.soft_update_from_to(self.actor, self.actor_target, self.tau)

    def report_grad_norm(self):
        # may add qf1, policy, etc.
        return {
            "q_grad_norm": utl.get_grad_norm(self.critic),
            "q_rnn_grad_norm": utl.get_grad_norm(self.critic.rnn),
            "pi_grad_norm": utl.get_grad_norm(self.actor),
            "pi_rnn_grad_norm": utl.get_grad_norm(self.actor.rnn),
        }

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        # all are 3D tensor (T,B,dim)
        actions, rewards, dones = batch["act"], batch["rew"], batch["term"]
        _, batch_size, _ = actions.shape
        if not self.algo.continuous_action:
            # for discrete action space, convert to one-hot vectors
            actions = F.one_hot(
                actions.squeeze(-1).long(), num_classes=self.action_dim
            ).float()  # (T, B, A)

        masks = batch["mask"]
        obs, next_obs = batch["obs"], batch["obs2"]  # (T, B, dim)
        tasks = batch["task"] if "task" in batch else None  # (T, B, dim)

        # extend observations, actions, rewards, dones from len = T to len = T+1
        observations = torch.cat((obs[[0]], next_obs), dim=0)  # (T+1, B, dim)
        actions = torch.cat(
            (ptu.zeros((1, batch_size, self.action_dim)).float(), actions), dim=0
        )  # (T+1, B, dim)
        rewards = torch.cat(
            (ptu.zeros((1, batch_size, 1)).float(), rewards), dim=0
        )  # (T+1, B, dim)
        dones = torch.cat(
            (ptu.zeros((1, batch_size, 1)).float(), dones), dim=0
        )  # (T+1, B, dim)
        tasks = torch.cat((tasks[0, :, :].unsqueeze(0), tasks))  # (T+1, B, dim)

        return self.forward(actions, rewards, observations, dones, masks, tasks)
