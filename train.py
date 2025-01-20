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

def train_agent(episodes=20000, board_size=15, batch_size=128, update_target_every=100):
    """
    Train the DQN agent through self-play.
    Args:
        episodes (int): Number of episodes to train
        board_size (int): Size of the game board
        batch_size (int): Size of training batches
        update_target_every (int): Episodes between target network updates
    """
    env = GobangEnv(board_size)
    agent = DQNAgent(board_size, 
                     memory_size=100000,  # Increased memory size
                     batch_size=batch_size,
                     gamma=0.99,
                     epsilon=1.0,
                     epsilon_min=0.01,
                     epsilon_decay=0.9995,  # Slower epsilon decay
                     learning_rate=0.0001)  # Lower learning rate
    
    # Enable training mode for faster training
    agent.model.train()
    agent.target_model.eval()
    
    # Use tqdm for progress tracking
    pbar = tqdm(range(episodes), desc="Training")
    
    best_reward = float('-inf')
    episode_rewards = []
    
    for episode in pbar:
        state = env.reset()
        total_reward = 0
        moves = 0
        
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
                agent.train()
                
                # Clear batches
                states_batch = []
                actions_batch = []
                rewards_batch = []
                next_states_batch = []
                dones_batch = []
            
            if done:
                break
                
            state = next_state
        
        episode_rewards.append(total_reward)
        
        # Update progress bar with useful metrics
        pbar.set_postfix({
            'reward': f'{total_reward:.2f}',
            'epsilon': f'{agent.epsilon:.2f}',
            'moves': moves,
            'avg_reward': f'{np.mean(episode_rewards[-100:]):.2f}'
        })
        
        # Save best model and update target network
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save("best_model.pth")
            
        if episode % update_target_every == 0:
            agent.update_target_model()
            agent.save(f"model_episode_{episode}.pth")

if __name__ == "__main__":
    # Enable PyTorch optimizations for faster training
    torch.backends.cudnn.benchmark = True  # Optimize CUDA operations
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic optimizations
    
    # Set high precision for matrix operations
    torch.set_float32_matmul_precision('high')
    
    # Optimize CPU thread usage
    torch.set_num_threads(4)
    
    # Start training with increased batch size for better GPU utilization
    train_agent(batch_size=64) 