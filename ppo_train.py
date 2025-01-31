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

def train_ppo_agent(episodes=20000, board_size=15, batch_size=256, save_every=1000, num_best_models=5, 
                    resume_from=None, start_episode=0):
    """
    Train or continue training a PPO agent.
    Args:
        episodes (int): Number of episodes to train
        board_size (int): Size of the game board
        batch_size (int): Size of training batches
        save_every (int): Episodes between saves
        num_best_models (int): Number of best models to keep
        resume_from (str): Path to checkpoint to resume from
        start_episode (int): Episode number to start from when resuming
    """
    env = GobangEnv(board_size)
    agent = PPOAgent(
        board_size=board_size,
        batch_size=batch_size,
        learning_rate=3e-4,  # Slightly higher learning rate
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.3,  # Higher initial clip epsilon
        c1=0.5,  # Value loss coefficient
        c2=0.02,  # Higher entropy coefficient for better exploration
        n_epochs=8  # More epochs per update
    )
    
    # Load checkpoint if resuming
    if resume_from:
        print(f"Resuming training from {resume_from}")
        agent.load(os.path.join("checkpoint", resume_from))
        
        # Load training metrics if they exist
        metrics_path = os.path.join("checkpoint", "ppo_training_metrics.npz")
        if os.path.exists(metrics_path):
            metrics = np.load(metrics_path)
            episode_rewards = list(metrics['rewards'])
            training_losses = list(metrics['losses'])
            best_models = [(r, e, f) for r, e, f in zip(metrics['best_rewards'], 
                                                       metrics['best_episodes'], 
                                                       metrics['best_filenames'])]
            print(f"Loaded {len(episode_rewards)} previous episodes of metrics")
        else:
            episode_rewards = []
            training_losses = []
            best_models = []
    else:
        episode_rewards = []
        training_losses = []
        best_models = []
    
    # Metrics tracking
    recent_rewards = deque(maxlen=100)
    if episode_rewards:
        recent_rewards.extend(episode_rewards[-100:])
    
    pbar = tqdm(range(start_episode, episodes), desc="PPO Training")
    
    try:
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
                best_models.sort(reverse=True)  # Sort by reward
                
                # Keep only top N models
                if len(best_models) > num_best_models:
                    old_filename = os.path.join("checkpoint", best_models[-1][2])
                    try:
                        if os.path.exists(old_filename):
                            os.remove(old_filename)
                    except Exception as e:
                        print(f"Warning: Could not remove old model file {old_filename}: {e}")
                    best_models = best_models[:num_best_models]
            
            # Regular saving and plotting
            if episode % save_every == 0:
                # Save current model
                agent.save(f"ppo_model_episode_{episode}.pth")
                
                # Save metrics
                try:
                    np.savez(
                        os.path.join("checkpoint", "ppo_training_metrics.npz"),
                        rewards=np.array(episode_rewards),
                        losses=np.array(training_losses),
                        best_rewards=np.array([r for r, _, _ in best_models]),
                        best_episodes=np.array([e for _, e, _ in best_models]),
                        best_filenames=np.array([f for _, _, f in best_models])
                    )
                except Exception as e:
                    print(f"Warning: Could not save metrics: {e}")
                
                # Plot metrics
                try:
                    plot_ppo_metrics(episode_rewards, training_losses)
                except Exception as e:
                    print(f"Warning: Could not plot metrics: {e}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        agent.save("ppo_interrupted.pth")
        print("Checkpoint saved as 'ppo_interrupted.pth'")
    
    # Print final best models
    print("\nBest PPO models:")
    for reward, episode, filename in best_models:
        print(f"Episode {episode}: Reward = {reward:.1f}, File = {filename}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(4)
    
    # To start new training:
    # train_ppo_agent()
    
    # To resume training:
    train_ppo_agent(
        resume_from="ppo_model_r82.2_e18280.pth",
        start_episode=18281,  # Start from next episode
        episodes=25000  # Train for more episodes
    ) 