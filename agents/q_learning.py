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

    def select_action(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions) # agent ignores the Q-table and chooses a random action

        q_values = self.q_table[state]
        # random tie-break instead of always picking action 0 when values are equal
        max_q = np.max(q_values)
        best_actions = np.flatnonzero(q_values == max_q)
        return int(random.choice(best_actions)) # randomly choose the best actions

    # where the agent learns from its experience
    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay) # to reduce exploration over time

    def save(self, path):
        # save the trained Q-table to a file so the agent can remember what it learned
        # convert defaultdict to a plain dict before pickling

        with open(path, "wb") as f: # wb means write binary format
            pickle.dump(dict(self.q_table), f) # pickle converts the Python object into bytes and writes it into the file.
            # Python's pickle may have trouble saving that lambda function - so we convert it to plain dict

    def load(self, path):
        # to save the training as plain dict but for training it works with defaultDict
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32))
        self.q_table.update(loaded)
