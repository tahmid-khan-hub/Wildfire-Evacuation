import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.emberpath_env import EmberPathEnv
from agents.q_learning import QLearningAgent, discretize_state

TRAIN_SEED = 42
NUM_EVAL_EPISODES = 100

def evaluate():
    env = EmberPathEnv(seed=TRAIN_SEED)
    agent = QLearningAgent()
    agent.load("models/q_table.pkl")

    successes = 0
    total_rescued = 0
    total_burned = 0
    total_steps = 0
    burned_deaths = 0

    for _ in range(NUM_EVAL_EPISODES):
        obs, info = env.reset(seed=TRAIN_SEED)
        state = discretize_state(env)
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.select_action(state, greedy=True)  # no exploration
            obs, reward, terminated, truncated, info = env.step(action)
            state = discretize_state(env)

        total_rescued += info["rescued"]
        total_burned += info["burned"]
        total_steps += info["step_count"]
        if info["rescued"] == 3:
            successes += 1
        if reward <= -100:
            burned_deaths += 1

    print(f"Success rate (3/3 rescued): {successes}/{NUM_EVAL_EPISODES} "
          f"({100*successes/NUM_EVAL_EPISODES:.1f}%)")
    print(f"Avg survivors rescued: {total_rescued/NUM_EVAL_EPISODES:.2f}/3")
    print(f"Avg survivors burned: {total_burned/NUM_EVAL_EPISODES:.2f}/3")
    print(f"Avg steps per episode: {total_steps/NUM_EVAL_EPISODES:.1f}")
    print(f"Agent-burned deaths: {burned_deaths}/{NUM_EVAL_EPISODES}")

if __name__ == "__main__":
    evaluate()