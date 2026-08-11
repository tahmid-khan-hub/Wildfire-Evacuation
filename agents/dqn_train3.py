import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from env.emberpath_env import EmberPathEnv

FIXED_SEED = 37
TOTAL_TIMESTEPS = 300_000 
MODEL_SAVE_PATH = "models/dqn_emberpath3"
LOG_DIR = "training/logs/dqn_map3"

class FixedSeedWrapper(gym.Wrapper):
    """
    Forces every episode to regenerate the exact same map (seed=42),
    regardless of whether the caller (e.g. SB3 internals) passes a seed.
    This keeps DQN's comparison against the Q-learning baseline fair —
    same map, not a moving target.
    """
    def __init__(self, env, fixed_seed):
        super().__init__(env)
        self.fixed_seed = fixed_seed

    def reset(self, **kwargs):
        kwargs["seed"] = self.fixed_seed
        return self.env.reset(**kwargs)


def make_env():
    env = EmberPathEnv(seed=FIXED_SEED)
    env = FixedSeedWrapper(env, FIXED_SEED)
    env = Monitor(env)  # tracks episode reward/length for TensorBoard
    return env

def main():
    env = make_env()

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=64,
        gamma=0.95,  # matching the reward horizon assumptions
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Saved model to {MODEL_SAVE_PATH}.zip")


if __name__ == "__main__":
    main()