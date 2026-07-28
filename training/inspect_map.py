import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.emberpath_env import EmberPathEnv
from env.constants import TERRAIN_NAMES, FIRE_STATE_NAMES, DANGEROUS_FIRE_STATES

TRAIN_SEED = 42

def inspect():
    env = EmberPathEnv(seed=TRAIN_SEED)
    obs, info = env.reset(seed=TRAIN_SEED)

    print("Agent start:", env.agent_pos)
    print("Survivor positions:", env.survivor_positions)

    for i, (sx, sy) in enumerate(env.survivor_positions):
        dist_from_agent = abs(sx - env.agent_pos[0]) + abs(sy - env.agent_pos[1])
        print(f"  Survivor {i}: pos=({sx},{sy}), "
              f"manhattan_dist_from_agent={dist_from_agent}, "
              f"terrain={TERRAIN_NAMES[env.terrain[sy, sx]]}")

    # find initial fire ignition point(s)
    ys, xs = (env.fire_state != 0).nonzero()
    print("Initial ignition point(s):", list(zip(xs.tolist(), ys.tolist())))

    # simulate fire spreading alone, no agent movement, to see how fast
    # it reaches each survivor's cell
    for i, (sx, sy) in enumerate(env.survivor_positions):
        env2 = EmberPathEnv(seed=TRAIN_SEED)
        env2.reset(seed=TRAIN_SEED)
        for step in range(50):
            env2.fire_state, env2.burn_timer = __import__("env.fire_spread", fromlist=["spread_fire"]).spread_fire(
                env2.terrain, env2.fire_state, env2.burn_timer, env2.rng
            )
            if env2.fire_state[sy, sx] in DANGEROUS_FIRE_STATES:
                print(f"  Survivor {i} at ({sx},{sy}) catches fire at step {step+1} "
                      f"(agent needs to arrive and rescue before then)")
                break
        else:
            print(f"  Survivor {i} at ({sx},{sy}) never caught fire in 50 steps")

if __name__ == "__main__":
    inspect()