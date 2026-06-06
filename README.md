# 🐦 Flappy Bird AI using Deep Q-Network (DQN)

An autonomous Flappy Bird agent built using **Deep Reinforcement Learning (DRL)** with **PyTorch** and **Gymnasium**. The agent learns to play Flappy Bird through trial and error using rewards and penalties, eventually discovering an optimal strategy to survive longer and achieve higher scores.

---

# 🎥 Demo

![Flappy Bird Demo](assets/flappy_bird_demo.gif)

---

# 🎮 Gameplay

![Gameplay](assets/gameplay.png)

---

# 📈 Training Logs

![Training Logs](assets/training_logs.png)

---

# 🚀 Project Overview

This project implements a **Deep Q-Network (DQN)** from scratch to train an AI agent capable of playing Flappy Bird autonomously.

Instead of hardcoding rules, the agent learns through interaction with the environment:

1. Observe the game state.
2. Choose an action (Flap / No Flap).
3. Receive a reward.
4. Store the experience.
5. Learn from previous experiences.
6. Improve future decisions.

Over thousands of episodes, the agent gradually learns how to navigate pipes efficiently and maximize rewards.

---

# ✨ Features

- Deep Q-Network (DQN) implementation using PyTorch
- Experience Replay Buffer
- Target Network Synchronization
- Epsilon-Greedy Exploration Strategy
- Automatic Model Checkpoint Saving
- Training & Evaluation Modes
- Configurable Hyperparameters via YAML
- GPU Support (CUDA/MPS)
- Gymnasium-based Environment
- Reinforcement Learning from Scratch

---

# 🧠 Reinforcement Learning Workflow

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

# 🏗️ Project Architecture

```text
FLAPPY_BIRD_RL/
│
├── agent.py
├── dqn.py
├── experience_replay.py
├── parameters.yaml
├── game_flappy_bird.py
│
├── assets/
│   ├── flappy_bird_demo.gif
│   ├── gameplay.png
│   └── training_logs.png
│
├── runs/
│   ├── flappybirdv0.pt
│   └── flappybirdv0.log
│
└── README.md
```

---

# 🛠️ Tech Stack

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| PyTorch | Deep Learning Framework |
| Gymnasium | Reinforcement Learning Environment |
| Flappy Bird Gymnasium | Game Environment |
| PyYAML | Hyperparameter Management |
| NumPy | Numerical Operations |
| Pygame | Game Rendering |

---

# 🧩 Deep Q-Network (DQN)

The DQN acts as the agent's brain.

Input:

```text
Game State
```

Output:

```text
Q(No Flap)
Q(Flap)
```

The network predicts the expected future reward of each action.

Example:

```text
Q(No Flap) = 1.2
Q(Flap)    = 3.8
```

The agent chooses:

```text
Flap
```

because:

```text
3.8 > 1.2
```

---

# 🔄 Experience Replay

Instead of learning from only the latest move, experiences are stored in memory.

Each experience contains:

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

- Breaks data correlation
- Improves training stability
- Increases learning efficiency

---

# 🎯 Target Network

The project uses two networks:

### Policy Network

Used for selecting actions.

```text
Current Learning Network
```

### Target Network

Used for generating stable learning targets.

```text
Periodically Updated Copy
```

This significantly improves DQN stability.

---

# 🎲 Epsilon-Greedy Exploration

The agent balances:

### Exploration

Random actions:

```text
Choose Random Action
```

### Exploitation

Learned actions:

```text
Choose Best Action
```

Training starts with:

```yaml
epsilon_init: 1.0
```

Meaning:

```text
100% Random Exploration
```

Gradually decays to:

```yaml
epsilon_min: 0.05
```

Meaning:

```text
Mostly Learned Decisions
```

---

# 📚 Bellman Equation

The agent learns using the Bellman Equation:

```math
Q(s,a)=R+\gamma \max Q(s',a')
```

Where:

- R = Immediate reward
- γ = Discount factor
- Q(s',a') = Future reward estimate

This allows the agent to consider future consequences of actions.

---

# ⚙️ Hyperparameters

```yaml
flappybirdv0:
  env_id: FlappyBird-v0

  epsilon_init: 1
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

# 💻 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/flappy-bird-dqn.git
cd flappy-bird-dqn
```

---

## Create Virtual Environment (Optional)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install torch
pip install gymnasium
pip install flappy-bird-gymnasium
pip install pygame
pip install pyyaml
```

Or:

```bash
pip install -r requirements.txt
```

---

# 🏋️ Training the Agent

Start training:

```bash
python agent.py flappybirdv0 --train
```

During training:

- Experiences are collected
- Replay Memory grows
- DQN learns optimal actions
- Best model is automatically saved

Saved model:

```text
runs/flappybirdv0.pt
```

---

# 🎮 Testing the Trained Agent

Run:

```bash
python agent.py flappybirdv0
```

The trained AI agent will automatically play Flappy Bird.

---

# 📊 Sample Training Output

```text
episode=1 with total reward=-7.5 & epsilon=1.0

episode=100 with total reward=18.4 & epsilon=0.78

episode=500 with total reward=65.2 & epsilon=0.32

episode=1000 with total reward=130.8 & epsilon=0.05
```

As training progresses:

✅ Higher rewards

✅ Better pipe navigation

✅ Improved survival time

✅ Reduced exploration

---

# 📈 Results

### Before Training

- Random actions
- Frequent collisions
- Very low scores

### After Training

- Strategic flapping
- Better obstacle avoidance
- Longer survival
- Improved average rewards

---

# 🎓 Learning Outcomes

This project demonstrates practical implementation of:

- Reinforcement Learning
- Deep Reinforcement Learning
- Deep Q-Networks (DQN)
- Bellman Equation
- Experience Replay
- Target Networks
- Epsilon-Greedy Exploration
- Neural Network Optimization
- PyTorch Model Training
- AI Game Development

---

# 🔮 Future Improvements

Potential enhancements include:

- Double DQN (DDQN)
- Dueling DQN
- Prioritized Experience Replay
- PPO (Proximal Policy Optimization)
- TensorBoard Integration
- Reward Visualization Dashboard
- Hyperparameter Optimization
- Multi-Agent Reinforcement Learning

---

# 👨‍💻 Author

**Ankit Yadav**

Data Science Student | Machine Learning Enthusiast | Aspiring AI Engineer

### Connect With Me

- GitHub: https://github.com/YOUR_USERNAME
- LinkedIn: https://linkedin.com/in/YOUR_LINKEDIN_PROFILE

---

# ⭐ Support

If you found this project interesting, consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates future improvements.
