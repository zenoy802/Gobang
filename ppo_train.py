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

class ModelEvaluator:
    """Evaluates model performance through tournaments and benchmark testing"""
    def __init__(self, board_size=15, n_evaluation_games=50):
        self.board_size = board_size
        self.n_evaluation_games = n_evaluation_games
        self.env = GobangEnv(board_size)
        
        # Create a benchmark agent (could be loaded from a known strong model)
        self.benchmark_agent = PPOAgent(board_size)
        benchmark_path = "benchmark_model.pth"  # A stable, strong model
        if os.path.exists(os.path.join("checkpoint", benchmark_path)):
            self.benchmark_agent.load(os.path.join("checkpoint", benchmark_path))
    
    def evaluate_against_benchmark(self, candidate_agent):
        """Evaluate agent against benchmark agent"""
        wins = 0
        draws = 0
        
        for game in range(self.n_evaluation_games):
            # Play both as black and white alternately
            if game % 2 == 0:
                result = self._play_game(candidate_agent, self.benchmark_agent)
            else:
                result = self._play_game(self.benchmark_agent, candidate_agent)
                # Invert result since candidate played as white
                result = -result if result != 0 else 0
            
            if result > 0:
                wins += 1
            elif result == 0:
                draws += 0.5
        
        return (wins + draws) / self.n_evaluation_games
    
    def evaluate_against_previous_best(self, candidate_agent, previous_best_models):
        """Evaluate agent against previous best models"""
        if not previous_best_models:
            return 0.0
        
        total_score = 0
        
        for prev_model_data in previous_best_models:
            # Load previous best model
            prev_agent = PPOAgent(self.board_size)
            prev_agent.load(os.path.join("checkpoint", prev_model_data[2]))  # filename is at index 2
            
            # Play against previous best
            score = 0
            for game in range(self.n_evaluation_games // len(previous_best_models)):
                if game % 2 == 0:
                    result = self._play_game(candidate_agent, prev_agent)
                else:
                    result = self._play_game(prev_agent, candidate_agent)
                    result = -result if result != 0 else 0
                
                if result > 0:
                    score += 1
                elif result == 0:
                    score += 0.5
            
            total_score += score / (self.n_evaluation_games // len(previous_best_models))
        
        return total_score / len(previous_best_models)
    
    def _play_game(self, black_agent, white_agent):
        """Play a single game between two agents"""
        state = self.env.reset()
        done = False
        
        while not done:
            valid_moves = self.env.get_valid_moves()
            current_agent = black_agent if self.env.current_player == 1 else white_agent
            
            with torch.no_grad():
                action, _, _ = current_agent.act(state, valid_moves)
            
            state, reward, done = self.env.step(action)
        
        return reward

def train_ppo_agent(episodes=20000, board_size=15, batch_size=256, save_every=1000, num_best_models=5, 
                    resume_from=None, start_episode=0):
    """Train PPO agents using self-play with improved model selection"""
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
            
            # Load best models and verify they exist
            best_models = []
            for r, e, f in zip(metrics['best_rewards'], metrics['best_episodes'], metrics['best_filenames']):
                model_path = os.path.join("checkpoint", str(f))
                if os.path.exists(model_path):
                    best_models.append((r, e, str(f), 0.0))
                else:
                    print(f"Warning: Best model file not found: {f}")
            
            if not best_models:
                print("No existing best models found, starting fresh best models list")
                # Add the resume model as the first best model
                best_models = [(0.0, start_episode, resume_from, 0.0)]
            
            print(f"Loaded {len(episode_rewards)} previous episodes of metrics")
            print(f"Loaded {len(best_models)} previous best models")
        else:
            episode_rewards = []
            training_losses = []
            best_models = [(0.0, start_episode, resume_from, 0.0)]  # Start with resume model
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
    
    # Initialize model evaluator
    evaluator = ModelEvaluator(board_size)
    
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
                if win_rate > 0.55:  # Initial filter to save evaluation time
                    # Evaluate the candidate model
                    # benchmark_score = evaluator.evaluate_against_benchmark(agent2)
                    benchmark_score = 0
                    historical_score = evaluator.evaluate_against_previous_best(agent2, best_models)
                    
                    # Combined score weighing both benchmark and historical performance
                    total_score = 0.6 * benchmark_score + 0.4 * historical_score
                    
                    print(f"\nEvaluation Results:")
                    print(f"Benchmark Score: {benchmark_score:.2f}")
                    print(f"Historical Score: {historical_score:.2f}")
                    print(f"Total Score: {total_score:.2f}")
                    
                    # Update if the model shows strong performance
                    if total_score > 0.55:  # Threshold for accepting new best model
                        print(f"Updating best agent (total score: {total_score:.2f})")
                        agent1.load_state_dict(agent2.state_dict())
                        win_rates.clear()
                        
                        # Save as one of the best models
                        model_filename = f"ppo_model_r{total_reward:.1f}_e{episode}_s{total_score:.2f}.pth"
                        agent2.save(model_filename)
                        
                        # Verify the file was saved before adding to best_models
                        if os.path.exists(os.path.join("checkpoint", model_filename)):
                            best_models.append((total_reward, episode, model_filename, total_score))
                            best_models.sort(key=lambda x: x[3], reverse=True)  # Sort by total score
                            
                            # Remove old model file only after verifying new one exists
                            if len(best_models) > num_best_models:
                                old_filename = os.path.join("checkpoint", best_models[-1][2])
                                try:
                                    if os.path.exists(old_filename):
                                        os.remove(old_filename)
                                        best_models = best_models[:num_best_models]
                                except Exception as e:
                                    print(f"Warning: Could not remove old model file {old_filename}: {e}")
                        else:
                            print(f"Warning: Failed to save new best model {model_filename}")
            
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
    train_ppo_agent()
    
    # To resume training:
    # train_ppo_agent(
    #     resume_from="ppo_model_r82.2_e18280.pth",
    #     start_episode=18281,  # Start from next episode
    #     episodes=25000  # Train for more episodes
    # ) 