# Standard vs. Curriculum Reinforcement Learning in Ant Locomotion

This repository contains the code for the research paper "Standard vs. Curriculum Reinforcement Learning in Ant Locomotion: An Empirical Investigation of When Curriculum Learning Fails to Deliver".

## Overview

We compare standard reinforcement learning with two curriculum learning approaches (basic adaptive curriculum and optimized stage-based curriculum) on the Ant-v5 locomotion task from the Gymnasium suite.

## Requirements

- Python 3.8+
- Gymnasium with MuJoCo
- Stable-Baselines3
- PyTorch
- NumPy
- Matplotlib
- FFmpeg (for video generation)
- PyVirtualDisplay (for headless rendering)

## Installation

```bash
# Install system dependencies
apt-get install -y xvfb python-opengl ffmpeg

# Install Python dependencies
pip install gymnasium[mujoco] stable-baselines3 matplotlib pyvirtualdisplay
```

## Project Structure

```
.
├── README.md
├── main.py              # Main experiment code
├── models/              # Saved model files
│   ├── ant_standard.zip
│   ├── ant_curriculum.zip
│   └── ant_optimized.zip
├── vec_normalize/       # Saved environment normalization states
│   ├── ant_standard.pkl
│   ├── ant_curriculum.pkl
│   └── ant_optimized.pkl
├── videos/              # Generated agent videos
│   ├── standard.mp4
│   ├── basic_curriculum.mp4
│   └── optimized_curriculum.mp4
└── plots/               # Generated performance plots
    └── comparison_matrix.png
```

## Usage

### Running the Full Experiment

To reproduce all results from the paper, run:

```bash
python main.py
```

This will:
1. Train a standard RL agent
2. Train a basic curriculum learning agent
3. Train an optimized curriculum learning agent
4. Generate videos of agent behavior
5. Evaluate and compare performance across difficulty levels
6. Generate comparison plots

### Training Individual Models

You can train models separately:

```python
from main import train_ant, train_optimized_curriculum

# Train standard RL model
standard_model = train_ant(use_curriculum=False)
standard_model.save("./models/ant_standard.zip")

# Train basic curriculum model
curriculum_model = train_ant(use_curriculum=True)
curriculum_model.save("./models/ant_curriculum.zip")

# Train optimized curriculum model
optimized_model = train_optimized_curriculum()
optimized_model.save("./models/ant_optimized.zip")
```

### Generating Videos

To generate videos from saved models:

```python
import gymnasium as gym
from stable_baselines3 import PPO
from main import generate_ant_video, FairAntCurriculum

# Load models
standard_model = PPO.load("./models/ant_standard.zip")
curriculum_model = PPO.load("./models/ant_curriculum.zip")
optimized_model = PPO.load("./models/ant_optimized.zip")

# Generate videos
generate_ant_video(standard_model, difficulty=0.5, filename="standard.mp4")
generate_ant_video(curriculum_model, difficulty=0.5, filename="basic_curriculum.mp4")
generate_ant_video(optimized_model, difficulty=1.0, filename="optimized_curriculum.mp4")
```

### Evaluating Performance

To evaluate models across difficulty levels:

```python
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from main import FairAntCurriculum
import gymnasium as gym

# Load models
standard_model = PPO.load("./models/ant_standard.zip")
curriculum_model = PPO.load("./models/ant_curriculum.zip")
optimized_model = PPO.load("./models/ant_optimized.zip")

# Evaluate across difficulty levels
difficulties = [0.2, 0.5, 0.8, 1.0]
results = {d: {'standard': 0, 'basic': 0, 'optimized': 0} for d in difficulties}

for diff in difficulties:
    # Create evaluation environment
    env = FairAntCurriculum(gym.make('Ant-v5'))
    env.difficulty = diff
    
    # Evaluate models
    std_mean, _ = evaluate_policy(standard_model, env, n_eval_episodes=5)
    basic_mean, _ = evaluate_policy(curriculum_model, env, n_eval_episodes=5)
    opt_mean, _ = evaluate_policy(optimized_model, env, n_eval_episodes=5)
    
    # Store results
    results[diff]['standard'] = std_mean
    results[diff]['basic'] = basic_mean
    results[diff]['optimized'] = opt_mean

# Plot results
plt.figure(figsize=(12, 6))
x = np.arange(len(difficulties))
width = 0.25

plt.bar(x - width, [results[d]['standard'] for d in difficulties], width, label='Standard')
plt.bar(x, [results[d]['basic'] for d in difficulties], width, label='Basic Curriculum')
plt.bar(x + width, [results[d]['optimized'] for d in difficulties], width, label='Optimized Curriculum')

plt.xticks(x, [f"{d:.1f}" for d in difficulties])
plt.xlabel('Difficulty Level')
plt.ylabel('Average Reward')
plt.title('Performance Comparison Across Difficulties')
plt.legend()
plt.grid(True)
plt.savefig('./plots/comparison_matrix.png')
plt.show()
```

## Curriculum Implementation

The curriculum learning approaches use the `FairAntCurriculum` wrapper, which modifies:
- Initial conditions (higher initial torso height, reduced velocities)
- Reward scaling (proportional to difficulty)
- Environmental perturbations (random forces applied with probability proportional to difficulty)

The basic adaptive curriculum adjusts difficulty based on recent performance:
- Increases difficulty by 0.05 when success rate exceeds 70%
- Decreases difficulty by 0.025 when success rate falls below 70%

The optimized stage-based curriculum uses predefined stages:
- Stage 0: Difficulty 0.2, advances when mean reward exceeds 50
- Stage 1: Difficulty 0.5, advances when mean reward exceeds 200
- Stage 2: Difficulty 0.8, advances when mean reward exceeds 500
- Stage 3: Difficulty 1.0 (full task complexity)

## Troubleshooting

### Video Generation Issues

If you encounter issues with video generation:

```python
# Test environment rendering
test_env = gym.make('Ant-v5', render_mode='rgb_array')
test_env.reset()  # Required before rendering
frame = test_env.render()
print(f"Frame shape: {frame.shape if frame is not None else 'None'}")
test_env.close()
```

### FFmpeg Issues

Ensure FFmpeg is properly installed:

```bash
apt-get install -y ffmpeg
```

## Citation

If you use this code in your research, please cite our paper:

```
@article{standard_vs_curriculum_2025,
  title={Standard vs. Curriculum Reinforcement Learning in Ant Locomotion: An Empirical Investigation of When Curriculum Learning Fails to Deliver},
  author={Aastha kataria,Suyash Mundhe,Sarika Dharangaonkar},
  journal={Your Journal/Conference},
  year={2025}
}
```

