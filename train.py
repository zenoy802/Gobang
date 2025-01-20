"""
Training script for the Gobang AI agent.
Implements the main training loop and optimization settings.
"""

from agent import DQNAgent
from environment import GobangEnv
import numpy as np
import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from collections import deque
import os

def plot_metrics(rewards, losses, avg_window=100):
    """
    Plot training metrics.
    Args:
        rewards (list): Episode rewards
        losses (list): Training losses
        avg_window (int): Window size for moving average
    """
    if not rewards:  # Skip if no data
        return
        
    plt.figure(figsize=(12, 5))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, label='Raw Rewards')
    # Calculate and plot moving average if enough data
    if len(rewards) >= avg_window:
        moving_avg = np.convolve(rewards, np.ones(avg_window)/avg_window, mode='valid')
        plt.plot(moving_avg, label=f'Moving Average ({avg_window})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    
    # Plot losses only if we have loss data
    plt.subplot(1, 2, 2)
    if losses:
        plt.plot(losses, alpha=0.3, label='Raw Losses')
        # Calculate and plot moving average if enough data
        if len(losses) >= avg_window:
            moving_avg = np.convolve(losses, np.ones(avg_window)/avg_window, mode='valid')
            plt.plot(moving_avg, label=f'Moving Average ({avg_window})')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def train_agent(episodes=20000, board_size=15, batch_size=256, update_target_every=100, 
                save_model_every=1000, num_best_models=5):
    """
    Train the DQN agent through self-play.
    Args:
        episodes (int): Number of episodes to train
        board_size (int): Size of the game board
        batch_size (int): Size of training batches
        update_target_every (int): Episodes between target network updates
        save_model_every (int): Episodes between regular model saves
        num_best_models (int): Number of best models to keep
    """
    env = GobangEnv(board_size)
    agent = DQNAgent(board_size, 
                     memory_size=100000,
                     batch_size=batch_size,
                     gamma=0.99,
                     epsilon=1.0,
                     epsilon_min=0.01,
                     epsilon_decay=0.9995,
                     learning_rate=0.0001)
    
    # Enable training mode for faster training
    agent.model.train()
    agent.target_model.eval()
    
    # Metrics tracking
    episode_rewards = []
    training_losses = []
    recent_rewards = deque(maxlen=100)
    
    # Keep track of best models
    best_models = []  # List of tuples (reward, episode)
    
    # Use tqdm for progress tracking
    pbar = tqdm(range(episodes), desc="Training")
    
    for episode in pbar:
        state = env.reset()
        total_reward = 0
        moves = 0
        episode_losses = []
        
        # Pre-allocate tensors for batch processing
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []
        
        while True:
            valid_moves = env.get_valid_moves()
            action = agent.act(state, valid_moves)
            next_state, reward, done = env.step(action)
            
            # Append to batch
            states_batch.append(state)
            actions_batch.append(action)
            rewards_batch.append(reward)
            next_states_batch.append(next_state)
            dones_batch.append(done)
            
            total_reward += reward
            moves += 1
            
            # Process batch when it reaches batch_size or episode ends
            if len(states_batch) >= batch_size or done:
                agent.remember_batch(
                    states_batch, 
                    actions_batch, 
                    rewards_batch, 
                    next_states_batch, 
                    dones_batch
                )
                loss = agent.train()
                if loss is not None:
                    episode_losses.append(loss)
                
                # Clear batches
                states_batch = []
                actions_batch = []
                rewards_batch = []
                next_states_batch = []
                dones_batch = []
            
            if done:
                break
                
            state = next_state
        
        # Record metrics
        episode_rewards.append(total_reward)
        recent_rewards.append(total_reward)
        if episode_losses:
            training_losses.extend(episode_losses)
        
        # Calculate average reward
        avg_reward = np.mean(recent_rewards)
        
        # Update progress bar with useful metrics
        pbar.set_postfix({
            'reward': f'{total_reward:.2f}',
            'avg_reward': f'{avg_reward:.2f}',
            'epsilon': f'{agent.epsilon:.2f}',
            'moves': moves,
            'best_reward': f'{max([r for r, _, _ in best_models], default=0):.2f}'
        })
        
        # Update best models list
        if not best_models or total_reward > best_models[-1][0] or len(best_models) < num_best_models:
            # Save the current model
            model_filename = f"best_model_r{total_reward:.1f}_e{episode}.pth"
            agent.save(model_filename)
            
            # Add to best models list
            best_models.append((total_reward, episode, model_filename))
            best_models.sort(reverse=True)  # Sort by reward
            
            # Keep only top N models
            if len(best_models) > num_best_models:
                # Remove the old model file
                old_filename = best_models[-1][2]
                try:
                    os.remove(old_filename)
                except:
                    pass
                best_models = best_models[:num_best_models]
        
        # Save average reward model
        if avg_reward > max([r for r, _, _ in best_models], default=0):
            agent.save(f"best_avg_model_r{avg_reward:.1f}_e{episode}.pth")
            
        if episode % update_target_every == 0:
            agent.update_target_model()
            if episode_losses:  # Only plot if we have loss data
                plot_metrics(episode_rewards, training_losses)
        
        if episode % save_model_every == 0:
            agent.save(f"model_episode_{episode}.pth")
    
    # Print final best models
    print("\nBest models:")
    for reward, episode, filename in best_models:
        print(f"Episode {episode}: Reward = {reward:.1f}, File = {filename}")

if __name__ == "__main__":
    # Enable PyTorch optimizations for faster training
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Set high precision for matrix operations
    torch.set_float32_matmul_precision('high')
    
    # Optimize CPU thread usage
    torch.set_num_threads(4)
    
    # Start training
    train_agent() 