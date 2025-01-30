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

def plot_ppo_metrics(rewards, losses, avg_window=100):
    """Plot training metrics for PPO"""
    if not rewards:
        return
        
    plt.figure(figsize=(12, 5))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, label='Raw Rewards')
    if len(rewards) >= avg_window:
        moving_avg = np.convolve(rewards, np.ones(avg_window)/avg_window, mode='valid')
        plt.plot(range(avg_window-1, len(rewards)), moving_avg, label=f'Moving Average ({avg_window})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('PPO Training Rewards')
    plt.legend()
    plt.grid(True)
    
    # Plot losses
    plt.subplot(1, 2, 2)
    if losses:
        plt.plot(losses, alpha=0.3, label='Raw Losses')
        if len(losses) >= avg_window:
            moving_avg = np.convolve(losses, np.ones(avg_window)/avg_window, mode='valid')
            plt.plot(range(avg_window-1, len(losses)), moving_avg, label=f'Moving Average ({avg_window})')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('PPO Training Losses')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ppo_training_metrics.png')
    plt.close()

def train_ppo_agent(episodes=20000, board_size=15, batch_size=512, save_every=1000, num_best_models=5):
    env = GobangEnv(board_size)
    agent = PPOAgent(
        board_size=board_size,
        batch_size=batch_size,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        c1=0.5,  # Reduced value loss coefficient
        c2=0.01,
        n_epochs=4  # Reduced number of epochs
    )
    
    # Metrics tracking
    episode_rewards = []
    training_losses = []
    recent_rewards = deque(maxlen=100)
    best_models = []
    
    pbar = tqdm(range(episodes), desc="PPO Training")
    
    for episode in pbar:
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            valid_moves = env.get_valid_moves()
            action, log_prob, value = agent.act(state, valid_moves)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, log_prob, value, done)
            total_reward += reward
            state = next_state
            
            # Train if enough samples
            if len(agent.memory) >= batch_size:
                loss = agent.train()
                if loss is not None:
                    training_losses.append(loss)
        
        # Record metrics
        episode_rewards.append(total_reward)
        recent_rewards.append(total_reward)
        avg_reward = np.mean(recent_rewards)
        
        # Update progress bar
        pbar.set_postfix({
            'reward': f'{total_reward:.2f}',
            'avg_reward': f'{avg_reward:.2f}',
            'best_reward': f'{max([r for r, _, _ in best_models], default=0):.2f}'
        })
        
        # Update best models list
        if not best_models or total_reward > best_models[-1][0] or len(best_models) < num_best_models:
            model_filename = f"ppo_model_r{total_reward:.1f}_e{episode}.pth"
            agent.save(model_filename)
            best_models.append((total_reward, episode, model_filename))
            best_models.sort(reverse=True)
            
            if len(best_models) > num_best_models:
                os.remove(os.path.join("checkpoint", best_models[-1][2]))
                best_models = best_models[:num_best_models]
        
        # Regular saving and plotting
        if episode % save_every == 0:
            agent.save(f"ppo_model_episode_{episode}.pth")
            plot_ppo_metrics(episode_rewards, training_losses)
    
    # Print final best models
    print("\nBest PPO models:")
    for reward, episode, filename in best_models:
        print(f"Episode {episode}: Reward = {reward:.1f}, File = {filename}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(4)
    train_ppo_agent() 