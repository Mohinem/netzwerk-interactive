# 🧠 Wordle RL Environment

This repository contains a **custom Wordle environment** built using [Gymnasium](https://gymnasium.farama.org/) and two types of agents for solving the game:

1. **Pure Strategy Agent** — A rule-based solver using action masking, letter constraints, and heuristics (**~93% win rate** on a 542-word dataset).
2. **Reinforcement Learning (RL) Agents** — Implemented with [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (DQN and others).

The environment supports **human-mode rendering**, so you can watch the agent play Wordle in real-time.


---

## 🚀 Features

- **Custom `gymnasium` environment for Wordle**:
  - Implements true Wordle rules and letter feedback
  - Supports rendering (`render_mode="human"`)
  - Step-by-step feedback after each guess

- **Two solving approaches**:
  - **Pure Strategy Agent** (no training required)  
    - High-information opening words (`ARISE`, `SOARE`, etc.)
    - Action masking to only select guesses that match discovered constraints
    - Letter frequency and pattern-fitting heuristics  
  - **DQN RL Agent**  
    - Trains using Stable-Baselines3  
    - Uses enhanced state representation including letter constraints  
    - Custom reward shaping to speed up learning  

- **Statistics tracking**:
  - Win rate, recent win rate, average guesses, average reward  
  - Win distribution by attempt number  
  - Rolling performance plots using Matplotlib  

---

## 📦 Installation

**Python 3.10+ recommended.**

### Install core dependencies:

pip install gymnasium numpy matplotlib

### For RL training:

pip install stable-baselines3 torch


> **Note:** Some RL libraries may require `numpy<2.0`. If you encounter compatibility errors:

pip install "numpy<2.0"


---

## 🕹 Usage

### 1️⃣ Run the Pure Strategy Agent (Recommended starter)
Runs instantly without training. Achieves ~93% on a 542-word dataset.

- Plays multiple episodes with **human-mode view**  
- Displays game board after each guess  
- Tracks performance statistics and generates plots  

To adjust:
- Change `num_episodes` in `test_strategy_agent()`  
- Modify dataset in `wordle_env.py` for different difficulty  

---

### 2️⃣ Train & Test DQN RL Agent

cd examples
python3 train_wordle_agent.py


