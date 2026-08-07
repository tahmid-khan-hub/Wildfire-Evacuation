import argparse
import numpy as np
from env.emberpath_env import EmberPathEnv
from env.constants import ACTION_NAMES, DANGEROUS_FIRE_STATES

SEED = 42


def get_priority_target(env):
    # replicates the env's internal fire-priority target selection,
    # but also returns *which* survivor index it picked (for tracing)
    agent_x, agent_y = env.agent_pos
    best_i = None
    best_fire_dist = None
    best_agent_dist = None

    for i in range(len(env.survivor_positions)):
        if env.survivor_rescued[i] or env.survivor_burned[i]:
            continue
        sx, sy = env.survivor_positions[i]
        fire_dist = env._nearest_fire_distance_to(sx, sy)
        fire_dist = fire_dist if fire_dist is not None else 10_000
        agent_dist = abs(sx - agent_x) + abs(sy - agent_y)

        is_better = (
            best_i is None
            or fire_dist < best_fire_dist
            or (fire_dist == best_fire_dist and agent_dist < best_agent_dist)
        )
        if is_better:
            best_i, best_fire_dist, best_agent_dist = i, fire_dist, agent_dist

    return best_i


def trace_qlearning():
    from agents.q_learning import QLearningAgent, discretize_state
    agent = QLearningAgent()
    agent.load("models/q_table.pkl")
    agent.epsilon = 0.0

    def policy(env):
        state = discretize_state(env)
        return agent.select_action(state, greedy=True)

    return policy


def trace_dqn():
    from stable_baselines3 import DQN
    model = DQN.load("models/dqn_emberpath.zip")

    def policy(env):
        obs = env._get_obs()
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["qlearning", "dqn"], required=True)
    args = parser.parse_args()

    policy = trace_qlearning() if args.algo == "qlearning" else trace_dqn()

    env = EmberPathEnv(seed=SEED)
    env.reset(seed=SEED)

    print(f"--- Tracing {args.algo} on seed={SEED} ---")
    print(f"Survivor positions: {env.survivor_positions}")
    print(f"Oracle-proven best order: (1, 2, 0) -> 3/3 rescued\n")

    terminated = truncated = False
    step = 0
    rescue_order = []
    prev_rescued = [False, False, False]

    while not (terminated or truncated):
        step += 1
        target_i = get_priority_target(env)
        action = policy(env)

        _, reward, terminated, truncated, info = env.step(action)

        for i in range(3):
            if env.survivor_rescued[i] and not prev_rescued[i]:
                rescue_order.append(("rescued", i, step))
            if env.survivor_burned[i] and not prev_rescued[i]:
                rescue_order.append(("burned", i, step))
        prev_rescued = list(env.survivor_rescued)

        print(f"step={step:3d}  agent={env.agent_pos}  "
              f"priority_target=survivor_{target_i}  "
              f"action={ACTION_NAMES[action]:5s}  "
              f"reward={reward:6.2f}  "
              f"rescued={info['rescued']}  burned={info['burned']}")

    print(f"\nFinal: rescued={info['rescued']}/3, burned={info['burned']}/3")
    print(f"Actual resolve order: {rescue_order}")


if __name__ == "__main__":
    main()