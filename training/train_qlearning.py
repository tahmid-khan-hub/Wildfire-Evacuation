import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.emberpath_env import EmberPathEnv
from agents.q_learning import QLearningAgent, discretize_state

TRAIN_SEED = 42  # fixed map — same layout every episode, on purpose
NUM_EPISODES = 6000
LOG_EVERY = 100

def train():
    env = EmberPathEnv(seed=TRAIN_SEED)
    agent = QLearningAgent()

    episode_rewards = []

    for episode in range(1, NUM_EPISODES + 1):
        # reseed with the SAME seed every episode -> same map, same fire origin,
        # same survivor placement. Only the agent's actions/epsilon change.
        obs, info = env.reset(seed=TRAIN_SEED)
        state = discretize_state(env)

        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.select_action(state)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_state(env)

            agent.update(state, action, reward, next_state, terminated or truncated)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if episode % LOG_EVERY == 0:
            avg_recent = np.mean(episode_rewards[-LOG_EVERY:])
            print(f"Episode {episode:5d} | avg reward (last {LOG_EVERY}): "
                  f"{avg_recent:7.2f} | epsilon: {agent.epsilon:.3f} | "
                  f"rescued: {info['rescued']}/3")

    # save trained Q-table
    os.makedirs("models", exist_ok=True)
    agent.save("models/q_table.pkl")
    print("Saved Q-table to models/q_table.pkl")

    # plot reward curve
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.3, label="per-episode reward")
    # rolling average for a readable trend line
    window = 50
    if len(episode_rewards) >= window:
        rolling = np.convolve(episode_rewards, np.ones(window)/window, mode="valid")
        plt.plot(range(window - 1, len(episode_rewards)), rolling, label=f"{window}-ep rolling avg")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Q-learning training reward (fixed map)")
    plt.legend()
    plt.tight_layout()
    os.makedirs("training/logs", exist_ok=True)
    plt.savefig("training/logs/qlearning_reward_curve.png")
    print("Saved reward curve to training/logs/qlearning_reward_curve.png")
    plt.show()


if __name__ == "__main__":
    train()