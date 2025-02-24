"""
PPO Agent implementation for the Gobang game.
Implements Proximal Policy Optimization with clipped objective.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import os
from ppo_model import PolicyNet, ValueNet

class PPOAgent:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, num_heads, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, num_heads).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state, mask):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mask = torch.tensor([mask], dtype=torch.bool).to(self.device)
        probs = self.actor(state, mask)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict, lr):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        masks = torch.tensor(transition_dict['masks'],
                              dtype=torch.bool).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_probs = self.actor(states, masks)
        # old_phi_probs = old_probs.gather(1, actions).detach()
        old_log_probs = torch.log(old_probs.gather(1, actions)).detach()

        for _ in range(self.epochs):
            probs = self.actor(states, masks)
            log_probs = torch.log(probs.gather(1, actions))
            # phi_probs = probs.gather(1, actions)
            ratio = torch.exp(log_probs - old_log_probs)
            # ratio = torch.div(phi_probs, old_phi_probs)
            # assert torch.equal(ratio, ratio_log), f"New calculation of ratio is not equivalent"
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            # 梯度裁剪，TODO：原理？
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            for name, param in self.actor.transformer_encoder.named_parameters():
                assert torch.isfinite(param.data).any(), f"transformer_encoder params not finite!\n Parameter: {name}\n Weights/Biases:\n{param.data}\n"
                if param.grad is not None:
                    assert torch.isfinite(param.grad).any(), f"transformer_encoder grad not finite!\n Parameter: {name}\n grad:\n{param.grad}\n"
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = lr
            self.actor_optimizer.step()
            self.critic_optimizer.step()


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)