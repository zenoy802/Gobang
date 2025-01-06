# Gobang AI with Deep Q-Learning

This project implements an AI agent that learns to play Gobang (Five in a Row) using Deep Q-Learning.

## Project Structure

- `model.py` - Contains the neural network architecture (GobangNet)
- `train.py` - Training script for the DQN agent
- `agent.py` - Implementation of the DQN agent
- `environment.py` - Gobang game environment

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- tqdm

## Installation

```bash
pip install torch torchvision torchaudio numpy tqdm
```

## Training

To train the agent, run the following command:

```bash
python train.py
```

This will start the training process and save the model after each episode.