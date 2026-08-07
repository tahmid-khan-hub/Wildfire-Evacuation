from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env.emberpath_env import EmberPathEnv
from agents.dqn_train import FixedSeedWrapper  

FIXED_SEED = 42
TOTAL_TIMESTEPS = 300_000  # same budget as DQN
MODEL_SAVE_PATH = "models/ppo_emberpath"
LOG_DIR = "training/logs/ppo"

def make_env():
    env = EmberPathEnv(seed=FIXED_SEED)
    env = FixedSeedWrapper(env, FIXED_SEED)
    env = Monitor(env)
    return env

def main():
    env = make_env()

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.95, # matches Q-learning/DQN gamma for a fair comparison
        gae_lambda=0.95,
        ent_coef=0.01, # small entropy bonus encourages exploring alternate paths,
                        
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Saved model to {MODEL_SAVE_PATH}.zip")

if __name__ == "__main__":
    main()