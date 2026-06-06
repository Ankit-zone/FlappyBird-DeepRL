# 🐦 Flappy Bird AI using Deep Q-Network (DQN)

An autonomous Flappy Bird agent built using **Deep Reinforcement Learning (DRL)** with **PyTorch** and **Gymnasium**. The agent learns to play the game by interacting with the environment, collecting rewards, and improving its policy through **Deep Q-Learning (DQN)**.

---

## 🚀 Project Overview

This project implements a Deep Q-Network (DQN) agent capable of learning how to play Flappy Bird without any hardcoded rules.

The agent starts with random actions and gradually learns optimal gameplay through trial and error using reinforcement learning techniques such as:

- Deep Q-Networks (DQN)
- Experience Replay
- Target Networks
- Epsilon-Greedy Exploration
- Bellman Equation Optimization

The trained model learns when to flap and when not to flap in order to maximize survival time and achieve higher scores.

---

## 🎯 Objectives

- Understand Deep Reinforcement Learning fundamentals.
- Implement Deep Q-Learning from scratch using PyTorch.
- Train an autonomous game-playing AI agent.
- Explore neural network-based decision making.
- Learn experience replay and target network synchronization techniques.

---

## 🧠 Reinforcement Learning Workflow

```text
Environment State
        ↓
   DQN Network
        ↓
   Choose Action
        ↓
 Execute Action
        ↓
 Receive Reward
        ↓
 Store Experience
        ↓
 Sample Mini-Batch
        ↓
 Update Network
        ↓
 Repeat
```

---

## 🛠 Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| PyTorch | Deep Learning Framework |
| Gymnasium | Reinforcement Learning Environment |
| Flappy Bird Gymnasium | Flappy Bird Environment |
| NumPy | Numerical Computations |
| PyYAML | Configuration Management |

---

## 📂 Project Structure

```text
FLAPPY_BIRD_RL/
│
├── agent.py                # Training and testing pipeline
├── dqn.py                  # Deep Q-Network architecture
├── experience_replay.py    # Replay Memory implementation
├── parameters.yaml         # Hyperparameter configuration
├── game_flappy_bird.py     # Manual gameplay version
│
├── runs/
│   ├── flappybirdv0.pt     # Trained model
│   └── flappybirdv0.log    # Training logs
│
└── README.md
```

---

## 🏗 DQN Architecture

The Deep Q-Network approximates the Q-function:

```math
Q(State, Action)
```

Current architecture:

```python
Linear(State_Dim, 256)
ReLU
Linear(256, Action_Dim)
```

Input:
- Bird position
- Bird velocity
- Pipe information

Output:
- Q-value for No Flap
- Q-value for Flap

The action with the highest Q-value is selected.

---

## 🔄 Experience Replay

The agent stores experiences in a replay buffer:

```text
(State,
 Action,
 Next State,
 Reward,
 Done)
```

Example:

```text
(
 [state],
 1,
 [next_state],
 +1,
 False
)
```

Benefits:
- Breaks correlation between experiences
- Improves training stability
- Increases sample efficiency

---

## 🎯 Target Network

A separate target network is maintained to stabilize learning.

```text
Policy Network
      ↓
 Learn Continuously

Target Network
      ↓
 Periodically Updated
```

This reduces oscillations and divergence during training.

---

## 🔍 Epsilon-Greedy Exploration

The agent balances:

### Exploration

Random actions:

```text
Choose random move
```

### Exploitation

Best learned action:

```text
Choose action with highest Q-value
```

Epsilon decays over time:

```yaml
epsilon_init: 1.0
epsilon_min: 0.05
epsilon_decay: 0.9995
```

---

## 📈 Bellman Equation

The agent learns using:

```math
Q(s,a) =
R + \gamma \max Q(s',a')
```

Where:

- R = Immediate reward
- γ = Discount factor
- Q(s',a') = Future reward estimate

---

## ⚙ Hyperparameters

```yaml
epsilon_init: 1.0
epsilon_min: 0.05
epsilon_decay: 0.9995

replay_memory_size: 100000
mini_batch_size: 32

network_sync_rate: 10

alpha: 0.001
gamma: 0.99

reward_threshold: 1000
```

---

## 💻 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/flappy-bird-dqn.git

cd flappy-bird-dqn
```

### Create Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install torch
pip install gymnasium
pip install flappy-bird-gymnasium
pip install pygame
pip install pyyaml
```

---

## 🏋 Training the Agent

Run:

```bash
python agent.py flappybirdv0 --train
```

During training:

- Agent explores environment
- Experiences are stored
- DQN learns from replay memory
- Best model is automatically saved

Model file:

```text
runs/flappybirdv0.pt
```

---

## 🎮 Testing the Trained Agent

Run:

```bash
python agent.py flappybirdv0
```

The trained agent will automatically play Flappy Bird.

---

## 📊 Training Output

Example:

```text
episode=1 with total reward=-7.5 & epsilon=1.0

episode=200 with total reward=45.3 & epsilon=0.45

episode=1000 with total reward=120.5 & epsilon=0.05
```

As training progresses:

- Reward increases
- Exploration decreases
- Agent survives longer

---

## 📸 Results

### Before Training

- Random actions
- Frequent collisions
- Low rewards

### After Training

- Intelligent flapping behavior
- Improved obstacle avoidance
- Higher average rewards
- Longer survival time

---

## 📚 Concepts Learned

This project demonstrates:

- Reinforcement Learning
- Deep Reinforcement Learning
- Deep Q-Networks (DQN)
- Experience Replay
- Target Networks
- Bellman Equation
- Epsilon-Greedy Exploration
- Neural Network Optimization
- PyTorch Model Training
- Game AI Development

---

## 🔮 Future Improvements

Potential enhancements:

- Double DQN (DDQN)
- Dueling DQN
- Prioritized Experience Replay
- PPO (Proximal Policy Optimization)
- TensorBoard Visualization
- Reward Curve Plotting
- Model Checkpointing
- Hyperparameter Optimization

---

## 👨‍💻 Author

**Ankit Yadav**

Data Science Student | Machine Learning Enthusiast | Aspiring AI Engineer

### Connect With Me

- LinkedIn: https://linkedin.com/in/iankityadav03
- GitHub: https://github.com/Ankit-zone

---

## ⭐ If you found this project useful

Consider giving this repository a star and sharing it with others interested in Reinforcement Learning and Game AI.
