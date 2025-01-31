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
    """Train PPO agents using self-play"""
    env = GobangEnv(board_size)
    
    # Create two agents
    agent1 = PPOAgent(  # Current best agent
        board_size=board_size,
        batch_size=batch_size,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.3,
        c1=0.5,
        c2=0.02,
        n_epochs=8
    )
    
    agent2 = PPOAgent(  # Training agent
        board_size=board_size,
        batch_size=batch_size,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.3,
        c1=0.5,
        c2=0.02,
        n_epochs=8
    )
    
    # Load checkpoint if resuming
    if resume_from:
        print(f"Resuming training from {resume_from}")
        agent1.load(os.path.join("checkpoint", resume_from))
        agent2.load(os.path.join("checkpoint", resume_from))
        
        # Load training metrics if they exist
        metrics_path = os.path.join("checkpoint", "ppo_training_metrics.npz")
        if os.path.exists(metrics_path):
            metrics = np.load(metrics_path)
            episode_rewards = list(metrics['rewards'])
            training_losses = list(metrics['losses'])
            # Add win rate to loaded best models
            best_models = [(r, e, f, 0.0) for r, e, f in zip(metrics['best_rewards'], 
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
    
    # Track win rates for model updates
    evaluation_window = 100
    win_rates = deque(maxlen=evaluation_window)
    
    pbar = tqdm(range(start_episode, episodes), desc="PPO Training")
    
    try:
        for episode in pbar:
            state = env.reset()
            total_reward = 0
            done = False
            
            # Keep track of experiences for both agents
            experiences1 = []
            experiences2 = []
            
            while not done:
                valid_moves = env.get_valid_moves()
                current_player = env.current_player
                
                if current_player == 1:
                    action, log_prob, value = agent2.act(state, valid_moves)  # Training agent plays
                    experiences2.append((state, action, log_prob, value))
                else:
                    action, log_prob, value = agent1.act(state, valid_moves)  # Best agent plays
                    experiences1.append((state, action, log_prob, value))
                
                next_state, reward, done = env.step(action)
                
                # Store experiences with rewards
                if done:
                    # Assign rewards based on game outcome
                    if reward > 0:  # Current player won
                        if current_player == 1:
                            win_rates.append(1)  # Training agent won
                            for exp in experiences2:
                                agent2.remember(exp[0], exp[1], reward, next_state, exp[2], exp[3], done)
                            for exp in experiences1:
                                agent1.remember(exp[0], exp[1], -reward, next_state, exp[2], exp[3], done)
                        else:
                            win_rates.append(0)  # Training agent lost
                            for exp in experiences2:
                                agent2.remember(exp[0], exp[1], -reward, next_state, exp[2], exp[3], done)
                            for exp in experiences1:
                                agent1.remember(exp[0], exp[1], reward, next_state, exp[2], exp[3], done)
                    else:  # Draw
                        win_rates.append(0.5)
                        for exp in experiences2:
                            agent2.remember(exp[0], exp[1], 0, next_state, exp[2], exp[3], done)
                        for exp in experiences1:
                            agent1.remember(exp[0], exp[1], 0, next_state, exp[2], exp[3], done)
                
                total_reward += reward
                state = next_state
                
                # Train if enough samples
                if len(agent2.memory) >= batch_size:
                    loss = agent2.train()
                    if loss is not None:
                        training_losses.append(loss)
            
            # Update best agent if training agent is performing well
            if len(win_rates) == evaluation_window:
                win_rate = sum(win_rates) / len(win_rates)
                if win_rate > 0.55:  # Training agent is winning more than 55% of games
                    print(f"\nUpdating best agent (win rate: {win_rate:.2f})")
                    # Copy the entire state from agent2 to agent1
                    agent1.load_state_dict(agent2.state_dict())
                    win_rates.clear()  # Reset win rate tracking
                    
                    # Save as one of the best models
                    model_filename = f"ppo_model_r{total_reward:.1f}_e{episode}_wr{win_rate:.2f}.pth"
                    agent2.save(model_filename)
                    best_models.append((total_reward, episode, model_filename, win_rate))
                    best_models.sort(key=lambda x: (x[3], x[0]), reverse=True)  # Sort by win rate, then reward
                    
                    if len(best_models) > num_best_models:
                        old_filename = os.path.join("checkpoint", best_models[-1][2])
                        try:
                            if os.path.exists(old_filename):
                                os.remove(old_filename)
                        except Exception as e:
                            print(f"Warning: Could not remove old model file {old_filename}: {e}")
                        best_models = best_models[:num_best_models]
            
            # Record metrics
            episode_rewards.append(total_reward)
            recent_rewards.append(total_reward)
            avg_reward = np.mean(recent_rewards)
            
            # Update progress bar with win rate info
            current_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0
            pbar.set_postfix({
                'reward': f'{total_reward:.2f}',
                'avg_reward': f'{avg_reward:.2f}',
                'win_rate': f'{current_win_rate:.2f}'
            })
            
            # Regular saving and plotting
            if episode % save_every == 0:
                agent2.save(f"ppo_model_episode_{episode}.pth")
                
                # Save metrics
                try:
                    np.savez(
                        os.path.join("checkpoint", "ppo_training_metrics.npz"),
                        rewards=np.array(episode_rewards),
                        losses=np.array(training_losses),
                        best_rewards=np.array([r for r, _, _, _ in best_models]),
                        best_episodes=np.array([e for _, e, _, _ in best_models]),
                        best_filenames=np.array([f for _, _, f, _ in best_models])
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
        agent2.save("ppo_interrupted.pth")
        print("Checkpoint saved as 'ppo_interrupted.pth'")
    
    # Print final best models
    print("\nBest PPO models:")
    for reward, episode, filename, win_rate in best_models:
        print(f"Episode {episode}: Reward = {reward:.1f}, Win Rate = {win_rate:.2f}, File = {filename}")

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