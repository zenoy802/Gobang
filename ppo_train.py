"""
Training script for the Gobang AI agent using PPO algorithm.
Runs in parallel with DQN for comparison.
"""

from ppo_agent import PPOAgent
from environment import GobangEnv
import numpy as np
import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from collections import deque
import os

def train_on_policy_agent(env, agent, num_episodes, num_loops):
    return_list = []
    for i in range(num_loops):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device:", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # To start new training:
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500
    num_loops = 10
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2

    env = GobangEnv()
    state_dim = len(env.get_state())
    # action_dim is the same as state_dim because it chooses an action from all spots
    action_dim =len(env.get_state())
    torch.manual_seed(0)
    agent = PPOAgent(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)

    return_list = train_on_policy_agent(env, agent, num_episodes, num_loops)
    
    # To resume training:
    # train_ppo_agent(
    #     resume_from="ppo_model_r82.2_e18280.pth",
    #     start_episode=18281,  # Start from next episode
    #     episodes=25000  # Train for more episodes
    # ) 