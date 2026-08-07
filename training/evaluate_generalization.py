import numpy as np
from stable_baselines3 import DQN, PPO
from env.emberpath_env import EmberPathEnv

NUM_TEST_MAPS = 50
TEST_SEEDS = range(1000, 1000 + NUM_TEST_MAPS)  # never used during training (training only ever saw seed=42)


def evaluate_model(model, model_name):
    rescued_counts = []
    burned_counts = []
    full_rescues = 0

    for seed in TEST_SEEDS:
        env = EmberPathEnv(seed=seed)
        obs, info = env.reset(seed=seed)
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))

        rescued_counts.append(info["rescued"])
        burned_counts.append(info["burned"])
        if info["rescued"] == 3:
            full_rescues += 1

    avg_rescued = np.mean(rescued_counts)
    avg_burned = np.mean(burned_counts)

    print(f"--- {model_name}: {NUM_TEST_MAPS} unseen random maps ---")
    print(f"Average rescued: {avg_rescued:.2f} / 3")
    print(f"Average burned:  {avg_burned:.2f} / 3")
    print(f"Full 3/3 rescues: {full_rescues}/{NUM_TEST_MAPS}")
    print()


def main():
    dqn_model = DQN.load("models/dqn_emberpath.zip")
    evaluate_model(dqn_model, "DQN (trained on fixed map)")

    ppo_model = PPO.load("models/ppo_emberpath.zip")
    evaluate_model(ppo_model, "PPO (trained on fixed map)")


if __name__ == "__main__":
    main()