from stable_baselines3 import DQN
from env.emberpath_env import EmberPathEnv

MODEL_PATH = "models/dqn_emberpath.zip"
FIXED_SEED = 42
NUM_EPISODES = 100

def main():
    env = EmberPathEnv(seed=FIXED_SEED)
    model = DQN.load(MODEL_PATH)

    rescued_counts = []
    burned_counts = []
    agent_deaths = 0
    full_rescues = 0

    for ep in range(NUM_EPISODES):
        obs, info = env.reset(seed=FIXED_SEED)  # force same map every episode
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)  # greedy, no exploration
            obs, reward, terminated, truncated, info = env.step(int(action))

        rescued_counts.append(info["rescued"])
        burned_counts.append(info["burned"])
        if info["rescued"] == 3:
            full_rescues += 1
        if reward <= -100:  # agent-burned penalty triggered on the final step
            agent_deaths += 1

    avg_rescued = sum(rescued_counts) / NUM_EPISODES
    avg_burned = sum(burned_counts) / NUM_EPISODES

    print(f"--- DQN evaluation over {NUM_EPISODES} episodes (seed={FIXED_SEED}) ---")
    print(f"Average rescued: {avg_rescued:.2f} / 3")
    print(f"Average burned:  {avg_burned:.2f} / 3")
    print(f"Full 3/3 rescues: {full_rescues}/{NUM_EPISODES}")
    print(f"Agent deaths: {agent_deaths}/{NUM_EPISODES}")


if __name__ == "__main__":
    main()