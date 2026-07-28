import numpy as np
from env.constants import ( GRID_WIDTH, GRID_HEIGHT, ACTION_DELTAS, IMPASSBLE_TERRAIN, DANGEROUS_FIRE_STATES, )
import random
import pickle
from collections import defaultdict

# Convert the environment into a simple state

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
    

# Learn which action is best in each state
class QLearningAgent:
    def __init__(self, n_actions=4, alpha=0.1, gamma=0.95, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        # here, all the parameters value are set as default value
        # n_actions = 4 possible actions, alpha = how much the agent changes its old knowledge, gamma = how much the agent does care about the future rewards, epsilon_start = Initial exploration rate.
        # epsilon_end = The minimum exploration rate

        self.n_actions = n_actions
        self.alpha = alpha          
        self.gamma = gamma         
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # defaultdict creates a default value automatically when a missing key is accessed.
        # ex: array([0., 0., 0., 0.], dtype=float32) - instead of rising an error it stores value automatically
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32))

