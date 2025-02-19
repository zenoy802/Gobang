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
import json
from datetime import datetime

def save_training_results(return_list, agent, episode, save_dir="training_results"):
    """Save training results, metrics, and model checkpoints"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"run_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    
    # Save metrics
    metrics = {
        'returns': return_list,
        'mean_return': float(np.mean(return_list[-10:])),
        'episode': episode,
        'timestamp': timestamp
    }
    
    with open(os.path.join(save_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save model checkpoint
    checkpoint_path = os.path.join(save_path, f'model_episode_{episode}.pth')
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
        'return_list': return_list,
        'episode': episode
    }, checkpoint_path)
    
    # Plot and save learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(return_list)
    plt.title(f'Learning Curve (Episode {episode})')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'learning_curve.png'))
    plt.close()
    
    print(f"\nSaved training results to {save_path}")
    return save_path

def load_training_checkpoint(agent, checkpoint_path):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=agent.device)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    return checkpoint['return_list'], checkpoint['episode']

def train_on_policy_agent(env, agent, num_episodes, num_loops, save_interval=100, 
                         resume_from=None, save_dir="training_results"):
    """
    Train PPO agent with periodic saving of results
    Args:
        env: Game environment
        agent: PPO agent
        num_episodes: Number of episodes per loop
        num_loops: Number of training loops
        save_interval: Episodes between saves
        resume_from: Path to checkpoint to resume from
        save_dir: Directory to save results
    """
    return_list = []
    start_loop = 0
    
    # Resume from checkpoint if specified
    if resume_from:
        print(f"Resuming training from {resume_from}")
        loaded_returns, loaded_episode = load_training_checkpoint(agent, resume_from)
        return_list.extend(loaded_returns)
        start_loop = loaded_episode // (num_episodes // 10)
        print(f"Resumed from episode {loaded_episode}")
    
    for i in range(start_loop, num_loops):
        with tqdm(total=int(num_episodes/10), desc=f'Loop {i+1}/{num_loops}') as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'masks': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                
                while not done:
                    mask = state == 0
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['masks'].append(mask)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                
                return_list.append(episode_return)
                agent.update(transition_dict)
                
                current_episode = num_episodes//10 * i + i_episode + 1
                if current_episode % save_interval == 0:
                    save_training_results(return_list, agent, current_episode, save_dir)
                
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': f'{current_episode}',
                        'return': f'{np.mean(return_list[-10:]):.3f}'
                    })
                pbar.update(1)
    
    # Save final results
    save_training_results(return_list, agent, num_episodes * num_loops, save_dir)
    return return_list

if __name__ == "__main__":
    # Setup device
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() 
                         else "cpu")
    print(f"Using device: {device}")
    
    # Training parameters
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 1000
    num_loops = 100
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    save_interval = 1000  # Save every 100 episodes
    
    # Initialize environment and agent
    env = GobangEnv()
    state_dim = len(env.get_state())
    action_dim = len(env.get_state())
    
    torch.manual_seed(0)
    agent = PPOAgent(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, 
                     lmbda, epochs, eps, gamma, device)
    
    # Train agent
    return_list = train_on_policy_agent(
        env, 
        agent, 
        num_episodes, 
        num_loops,
        save_interval=save_interval,
        # resume_from="training_results/run_20250218_225307/model_episode_1100.pth"  # Uncomment to resume
    )
    
    # To resume training:
    # train_ppo_agent(
    #     resume_from="ppo_model_r82.2_e18280.pth",
    #     start_episode=18281,  # Start from next episode
    #     episodes=25000  # Train for more episodes
    # ) 