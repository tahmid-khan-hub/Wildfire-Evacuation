import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO
from agents.q_learning import QLearningAgent, discretize_state
from env.emberpath_env import EmberPathEnv

SEED = 42
NUM_EPISODES = 100

def eval_qlearning():
    agent = QLearningAgent()
    agent.load("models/q_table.pkl")
    agent.epsilon = 0.0
    return _run(lambda env: agent.select_action(discretize_state(env), greedy=True))

def eval_dqn():
    model = DQN.load("models/dqn_emberpath.zip")
    return _run(lambda env: int(model.predict(env._get_obs(), deterministic=True)[0]))

def eval_ppo():
    model = PPO.load("models/ppo_emberpath.zip")
    return _run(lambda env: int(model.predict(env._get_obs(), deterministic=True)[0]))

def _run(policy_fn):
    env = EmberPathEnv(seed=SEED)
    rescued_list, burned_list, steps_list = [], [], []
    for _ in range(NUM_EPISODES):
        env.reset(seed=SEED)
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy_fn(env)
            _, _, terminated, truncated, info = env.step(action)
        rescued_list.append(info["rescued"])
        burned_list.append(info["burned"])
        steps_list.append(info["step_count"])
    return {
        "rescued": np.mean(rescued_list),
        "burned": np.mean(burned_list),
        "steps": np.mean(steps_list),
    }

def main():
    results = {
        "Q-Learning": eval_qlearning(),
        "DQN": eval_dqn(),
        "PPO": eval_ppo(),
    }

    for name, r in results.items():
        print(f"{name:12s} rescued={r['rescued']:.2f}/3  burned={r['burned']:.2f}/3  steps={r['steps']:.1f}")

    # comparison bar chart
    names = list(results.keys())
    rescued_vals = [results[n]["rescued"] for n in names]
    burned_vals = [results[n]["burned"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, rescued_vals, width, label="Rescued", color="#2a9d8f")
    ax.bar(x + width/2, burned_vals, width, label="Burned", color="#e76f51")
    ax.set_ylabel("Average count (out of 3)")
    ax.set_title("Algorithm comparison on fixed map (seed=42)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.set_ylim(0, 3)
    plt.tight_layout()
    plt.savefig("training/logs/algorithm_comparison.png")
    print("Saved chart to training/logs/algorithm_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()