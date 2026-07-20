import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env.constants import ( GRID_WIDTH, GRID_HEIGHT, ACTION_DELTAS, IMPASSBLE_TERRAIN, DANGEROUS_FIRE_STATES, NUM_SURVIVORS, REWARD_STEP_PENALTY, REWARD_AGENT_BURNED, REWARD_RESCUE, REWARD_SURVIVOR_BURNED, REWARD_DISTANCE_SHAPING, MAX_EPISODE_STEPS, REWARD_TIMEOUT)
from env.grid_generator import generate_grid
from env.fire_spread import spread_fire, init_burn_timer

class EmberPathEnv(gym.Env):
    metadata = {"render_modes": []}

    # called once when the environment is created.
    def __init__(self, seed=None):
        super().__init__()

        self.action_space = spaces.Discrete(4)

        obs_size = (GRID_HEIGHT*GRID_WIDTH*2) + 2 + (NUM_SURVIVORS*3) # *2 because terrain and fire grid, +2 because agent position takes 2 value, *3 because survivor contributes 3 value
        self.observation_space = spaces.Box(low=-1, high=max(GRID_WIDTH, GRID_HEIGHT), shape=(obs_size,), dtype=np.float32) # shape will be taken as tuple not int because of gym

        self.rng = np.random.default_rng(seed)

        # value will be initialized in reset()
        self.terrain = None
        self.fire_state = None
        self.burn_timer = None
        self.agent_pos = None
        self.survivor_positions = None
        self.survivor_rescued = None
        self.survivor_burned = None
        self.step_count = 0

    # it creates the actual env
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        terrain, fire_state, agent_start, survivor_positions = generate_grid(self.rng)

        self.terrain = terrain
        self.fire_state = fire_state
        self.burn_timer = init_burn_timer()
        self.agent_pos = agent_start
        self.survivor_positions = survivor_positions
        self.survivor_rescued = [False] * NUM_SURVIVORS
        self.survivor_burned = [False] * NUM_SURVIVORS
        self.step_count = 0

        return self._get_obs(), self._get_info()

    # if the agent perform action then what would happen
    def step(self, action):
        self.step_count += 1 # agent move one cell means +1
        prev_dist = self._nearest_unresolved_distance() # we will compare it with after agent pos move so that agent get nearest result
    
        self._move_agent(action) # now agent moved
    
        self.fire_state, self.burn_timer = spread_fire(self.terrain, self.fire_state, self.burn_timer, self.rng) # fire advances whether the agent move was valid
    
        reward = REWARD_STEP_PENALTY
        terminated = False # initially episode can not be true
    
        # if the agent is in fire state then finish the episode with reward
        agent_x, agent_y = self.agent_pos
        if self.fire_state[agent_y, agent_x] in DANGEROUS_FIRE_STATES:
            reward += REWARD_AGENT_BURNED
            terminated = True
    
        # if the agent not burnt then calculate the reward how many it rescued and burned
        reward += self._check_rescues() * REWARD_RESCUE
        reward += self._check_survivors_burned() * REWARD_SURVIVOR_BURNED
    
        new_dist = self._nearest_unresolved_distance()
        if prev_dist is not None and new_dist is not None and new_dist < prev_dist:
            reward += REWARD_DISTANCE_SHAPING
    
        # nothing left, episode over
        resolved_count = sum(self.survivor_burned) + sum(self.survivor_rescued)
        if resolved_count == NUM_SURVIVORS:
            terminated = True
    
        # still survivors left but curr step is greater than max step so ended the episode of external limit
        truncated = False
        if self.step_count >= MAX_EPISODE_STEPS and not terminated:
            truncated = True
            reward += REWARD_TIMEOUT
    
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _move_agent(self, action):
        dx, dy = ACTION_DELTAS[action]
        nx = self.agent_pos[0] + dx
        ny = self.agent_pos[1] + dy
        if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
            if self.terrain[ny, nx] not in IMPASSBLE_TERRAIN:
                self.agent_pos = (nx, ny)

    def _check_rescues(self):
        # agent standing on an unresolved survivor's cell rescues them
        count = 0
        for i in range(len(self.survivor_positions)):
            sx, sy = self.survivor_positions[i]

            if self.survivor_rescued[i] or self.survivor_burned[i]:
                continue
            if self.agent_pos == (sx, sy):
                self.survivor_rescued[i] = True
                count += 1
        return count

    def _check_survivors_burned(self):
        # unresolved survivor standing on a cell that's now on fire is lost
        count = 0
        for i in range(len(self.survivor_positions)):
            sx, sy = self.survivor_positions[i]

            if self.survivor_rescued[i] or self.survivor_burned[i]:
                continue
            if self.fire_state[sy, sx] in DANGEROUS_FIRE_STATES:
                self.survivor_burned[i] = True
                count += 1
        return count

    def _nearest_unresolved_distance(self):
        agent_x, agent_y = self.agent_pos
        distances = []
        for i in range (len(self.survivor_positions)):
            sx, sy = self.survivor_positions[i]

            if self.survivor_rescued[i] or self.survivor_burned[i]:
                continue

            distances.append(abs(sx - agent_x) + abs(sy - agent_y)) # calculate manhattan distance as it is suitable because our agent can not move diagonally

        if distances:
            return min(distances)
        else:
            return None

    def _get_obs(self):
        # neural networks don't naturally work with multiple separate variables, it expect a fixed-size numerical input. Here, we transforms the environment's internal state into a single vector
        terrain_flat = self.terrain.flatten().astype(np.float32) # flatten converts 2D arr into 1D arr and also converting them into float as ML algorithm can understand/work with numbers
        fire_flat = self.fire_state.flatten().astype(np.float32)
        agent_xy = np.array(self.agent_pos, dtype=np.float32)

        survivor_info = []
        for i in range(len(self.survivor_positions)):
            sx, sy = self.survivor_positions[i]

            if self.survivor_rescued[i] or self.survivor_burned[i]:
                survivor_info.extend([-1.0, -1.0, 1.0])  # resolved -> flagged, position hidden
            else:
                survivor_info.extend([float(sx), float(sy), 0.0]) # 0.0 means active

        survivor_info = np.array(survivor_info, dtype=np.float32)

        return np.concatenate([terrain_flat, fire_flat, agent_xy, survivor_info])

    def _get_info(self):
        return {
            "step_count": self.step_count,
            "rescued": sum(self.survivor_rescued),
            "burned": sum(self.survivor_burned),
            "agent_pos": self.agent_pos,
        }
