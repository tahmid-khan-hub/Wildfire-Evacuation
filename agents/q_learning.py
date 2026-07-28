import numpy as np
from env.constants import ( GRID_WIDTH, GRID_HEIGHT, ACTION_DELTAS, IMPASSBLE_TERRAIN, DANGEROUS_FIRE_STATES, )

def _distance_bucket(dist):
    # using it so the Q-table doesn't explode with exact distances
    if dist is None:
        return "none"
    elif dist <= 1:
        return "adjacent"
    if dist <= 3:
        return "near"
    if dist <= 6:
        return "mid"
    return "far"

def _sign(v):
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0

def discretize_state(env):
    # turns the env's internal state into a small, hashable tuple
    # suitable for a dict-based Q-table. reads env internals directly rather than using _get_obs(), since that vector is sized for NN input.
    
    ax, ay = env.agent_pos

    # find nearest unresolved survivor
    best_dist = None
    best_dx, best_dy = 0, 0
    for i in range(len(env.survivor_positions)):
        sx, sy = env.survivor_positions[i]

        if env.survivor_rescued[i] or env.survivor_burned[i]:
            continue # already burned or rescued then skip
        
        dist = abs(sx - ax) + abs(sy - ay)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_dx, best_dy = sx - ax, sy - ay

        dir_x = _sign(best_dx)
        dir_y = _sign(best_dy)
        dist_bucket = _distance_bucket(best_dist)

        # danger flags for the 4 adjacent cells (order matches ACTION_DELTAS)
        danger_flags = []
        for action in sorted(ACTION_DELTAS.keys()):
            dx, dy = ACTION_DELTAS[action]
            nx, ny = dx+ax, dy+ay

            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                danger = env.fire_state[ny, nx] in DANGEROUS_FIRE_STATES
            else:
                danger = True
            danger_flags.append(int(danger))

        survivors_remaining = 0
        for i in range(len(env.survivor_positions)):
            if not env.survivor_rescued[i] and not env.survivor_burned[i]:
                survivors_remaining += 1

        return (ax, ay, dir_x, dir_y, dist_bucket, tuple(danger_flags), survivors_remaining)

