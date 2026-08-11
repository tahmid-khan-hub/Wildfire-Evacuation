# Wildfire Evacuation RL Agent
 
An autonomous drone learns to navigate a procedurally generated forest grid and rescue survivors before a dynamically spreading wildfire reaches them. Built as a custom [`gymnasium`](https://gymnasium.farama.org/) environment, with tabular Q-learning, DQN and PPO trained and compared on identical setups and a sprite-based Pygame visualization of trained agent behavior.
 
## Overview
 
Fire in this environment isn't a static or randomly-moving obstacle — it spreads and intensifies over time based on terrain type (dense forest, dry brush, water, rock), forcing the agent to learn risk-aware, time-sensitive decisions rather than simple shortest-path navigation. The project's core focus is the RL experimentation itself: comparing algorithms, diagnosing *why* they behave the way they do and reporting honest findings — not just producing a working demo.
 
## Environment
 
- **Grid:** 12×10, procedurally generated each episode (`grid_generator.py`)
- **Terrain types:** forest, brush, water, rock — each with distinct fire-spread properties. Rock is fire-proof and **passable** by design (a safe shortcut through fire-prone areas); water is impassable. This is an intentional design choice, not a bug.
- **Fire:** cellular-automaton-style spread, four intensity states (none → low → mid → high → ash)
- **Actions:** 4 discrete movements (up/down/left/right)
- **Survivors:** 3 per episode, rescued by moving onto their cell before fire reaches them
- **Reward shaping:** step penalty, rescue bonus, survivor-burned penalty, agent-burned penalty, distance-shaping bonus (toward the most fire-threatened survivor), full-clear bonus, timeout penalty
- **Termination:** all survivors resolved (rescued or burned) or max step limit reached

## Algorithms Compared
 
| Algorithm | Approach | Training |
|---|---|---|
| Tabular Q-learning | Hand-engineered discretized state, epsilon-greedy, Bellman updates | From scratch, fixed map (seed=42) |
| DQN | Stable-Baselines3, `MlpPolicy` | Fixed map (seed=42), 300k timesteps — trained on **3 separate fixed maps** (seeds 42 and two additional seeds) to demonstrate the approach generalizes across individually-trained maps |
| PPO | Stable-Baselines3, `MlpPolicy`, entropy regularization (`ent_coef=0.01`) | Fixed map (seed=42), 1,000,000 timesteps (increased from an initial 300k budget to rule out under-training as the cause of weaker performance) |
 
Each algorithm was also evaluated on 50 unseen random maps to test generalization beyond the maps it was trained on.
 
## Results
 
**Fixed-map performance (seed=42, 100 evaluation episodes):**
 
| Model | Avg. Rescued | Avg. Burned | Full 3/3 Clears | Agent Deaths |
|---|---|---|---|---|
| Q-learning | 2/3 | 1/3 | 0/100 | 0/100 |
| DQN | 2/3 | 1/3 | 0/100 | 0/100 |
| PPO (300k steps) | 1/3 | 2/3 | 0/100 | 0/100 |
| PPO (1M steps) | 1/3 | 2/3 | 0/100 | 0/100 |
 
**Generalization to 50 unseen random maps:**
 
| Model | Avg. Rescued | Avg. Burned | Full 3/3 Clears |
|---|---|---|---|
| DQN (trained on one fixed map) | 0.16/3 | — | 0/50 |
| DQN (trained on randomized maps, 1.5M steps) | 0.32/3 | 1.04/3 | 0/50 |
| PPO (trained on one fixed map) | 0.20/3 | — | 0/50 |
 
## Key Findings
 
### 1. The greedy-trap local optimum
All three algorithms converge on the same limitation: **2/3 rescued (Q-learning, DQN) or 1/3 rescued (PPO), never 3/3**, on the fixed evaluation map. This was root-caused, not assumed:
 
- A analyze_map_solvability (`training/analyze_map_solvability.py`) exhaustively tested all 6 possible rescue orders and proved 3/3 is achievable — but *only* via one specific order (rescue the two fire-threatened survivors first, save the nearby "easy" one for last). The naive order (grab the nearby survivor first) caps out at 1/3.
- A step-by-step behavioral trace confirmed both Q-learning and DQN immediately grab the nearby "free" rescue on step one, even though the environment's own fire-priority signal correctly flags a different survivor as more urgent from the start.
- Three independent fixes were tested to break the trap: realigning the distance-shaping reward to target the fire-priority survivor instead of the nearest one, adding a large bonus (`REWARD_FULL_CLEAR_BONUS`) exclusively for a clean 3/3 finish and retraining under both changes. **All three produced the identical result.** This is a genuine, reproducible structural local optimum in the reward landscape — not a bug and not something straightforward reward engineering could resolve within this project's scope.
### 2. Why the agent can't rescue all 3/3 survivors
In short: the immediate reward of rescuing the nearby survivor is learned faster and more reliably than the longer-horizon strategy of ignoring it temporarily to save the two fire-threatened survivors first. Once the "easy" survivor is rescued out of order, the map's fire spread makes 3/3 mathematically unreachable for the remaining two — confirmed by the analyze solvability. This held even after reward shaping was specifically redesigned to counteract it, indicating the greedy behavior is a stable local optimum rather than an artifact of poor reward tuning.
 
### 3. Generalization does not transfer from fixed-map training
A model trained to reliably solve one fixed map does not perform well on unseen maps (0.16/3 average). Training directly on randomized maps improves this (0.32/3) but does not solve it and never achieved a full 3/3 clear across 50 test maps. The most likely explanation: the observation space encodes absolute grid coordinates, so the network partially memorizes spatial patterns tied to specific layouts rather than learning fully transferable spatial reasoning. A local, agent-centered observation window (instead of absolute position) is the most promising direction for closing this gap, but was out of scope for this project's timeline.
 
### 4. PPO underperforms DQN even after matching training budget
PPO was retrained at over 3x the original timestep budget (300k → 1,000,000) specifically to rule out under-training as the cause of its weaker result. The outcome was unchanged (1/3 rescued, 2/3 burned). This suggests PPO's on-policy learning is more susceptible to this environment's greedy local optimum than DQN's off-policy approach, rather than simply needing more data. PPO does use entropy regularization (`ent_coef=0.01`) to encourage continued exploration during training — but this discourages the policy from converging too early, it does not help it escape a local optimum once found, which is a meaningful distinction reflected in the unchanged result.
 
## Limitations
 
- **No algorithm achieves a full 3/3 rescue rate on the fixed evaluation map.** This is the project's central, honestly-reported finding rather than a hidden weakness — see "Key Findings" above.
- **Generalization to unseen maps is weak across all models tested.** Best result is 0.32/3 average on 50 held-out random maps, with zero full clears.
- **Tabular Q-learning does not generalize by construction** — its state representation is keyed to a specific map layout, so it was trained and evaluated on a single fixed map only, as expected for the method.
- **No traditional labeled dataset** — as a reinforcement learning project, there is no fixed dataset in the supervised-learning sense. The environment procedurally generates training scenarios each episode, and agents learn from direct interaction rather than static labeled data.
- **Regularization is limited to PPO's entropy coefficient.** No dropout or weight decay was applied to the DQN/PPO networks; this was a deliberate scope decision to prioritize the core algorithm comparison and diagnostic work within the project timeline.
- **No dedicated dashboard/web UI.** The Pygame renderer with a live HUD (step count, rescued/burned counters, current reward) serves as the project's visualization layer; the optional Streamlit results dashboard (Stage 10) was not built.
- **PPO was trained on a single map only**, unlike DQN which was trained on three separate fixed maps to demonstrate consistency of the approach across different layouts.
## Project Structure
 
```
├── agents/          # Q-learning, DQN, PPO training scripts
├── env/             # gymnasium environment, fire spread, grid generation, constants
├── render/          # Pygame rendering, asset management, HUD, diagnostics
├── training/        # evaluation, comparison, oracle, tracing/debugging scripts
├── demo/            # trained-episode GIF recording
├── models/          # saved trained weights
├── tests/           # environment sanity/unit tests
└── requirements.txt
```
 
## Setup
 
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
 
## Running
 
```powershell
# Train
python -m agents.q_learning
python -m agents.dqn_train
python -m agents.ppo_train
 
# Evaluate
python -m training.evaluate_dqn
python -m training.evaluate_ppo
python -m training.compare_all
 
# Watch a trained agent
python -m render.diagnostics.watch_dqn
python -m render.diagnostics.watch_ppo
 
# Record a demo GIF
python -m demo.record_episode
```
 
## Tech Stack
 
Python, NumPy, Gymnasium, Pygame, Stable-Baselines3, PyTorch, TensorBoard, matplotlib.
 
