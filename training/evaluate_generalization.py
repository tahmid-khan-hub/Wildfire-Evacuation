import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from env.emberpath_env import EmberPathEnv

NUM_MAPS = 50
UNSEEN_SEEDS = range(1000, 1000 + NUM_MAPS)

MODELS = {
    "DQN (fixed-map trained)": ("dqn", "models/dqn_emberpath.zip"),
    "DQN (randomized-map trained)": ("dqn", "models/dqn_random.zip"),
    "PPO (fixed-map trained)": ("ppo", "models/ppo_emberpath.zip"),
}


def load_model(algo, path):
    if algo == "dqn":
        return DQN.load(path)
    elif algo == "ppo":
        return PPO.load(path)
    raise ValueError(f"Unknown algo: {algo}")


def evaluate_on_unseen_maps(algo, path):
    model = load_model(algo, path)

    rescued_counts = []
    burned_counts = []
    full_rescues = 0

    for seed in UNSEEN_SEEDS:
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

    avg_rescued = sum(rescued_counts) / NUM_MAPS
    avg_burned = sum(burned_counts) / NUM_MAPS

    return avg_rescued, avg_burned, full_rescues


def plot_results(results):
    names = list(results.keys())
    rescued_vals = [results[n][0] for n in names]
    burned_vals = [results[n][1] for n in names]

    x = range(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    bars1 = ax.bar([i - width/2 for i in x], rescued_vals, width, label="Avg. Rescued", color="#4C9A2A")
    bars2 = ax.bar([i + width/2 for i in x], burned_vals, width, label="Avg. Burned", color="#D9534F")

    ax.set_ylabel("Average Survivors (out of 3)")
    ax.set_title(f"Generalization to {NUM_MAPS} Unseen Maps")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 3)
    ax.legend()

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    output_path = "training/logs/generalization_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved chart to {output_path}")


def main():
    print(f"--- Generalization evaluation over {NUM_MAPS} unseen maps ---\n")

    results = {}
    for name, (algo, path) in MODELS.items():
        avg_rescued, avg_burned, full_rescues = evaluate_on_unseen_maps(algo, path)
        results[name] = (avg_rescued, avg_burned, full_rescues)
        print(f"{name}: rescued={avg_rescued:.2f}/3  burned={avg_burned:.2f}/3  full_clears={full_rescues}/{NUM_MAPS}")

    plot_results(results)


if __name__ == "__main__":
    main()