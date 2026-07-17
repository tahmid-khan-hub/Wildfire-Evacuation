import numpy as np
from env.constants import ( GRID_WIDTH, GRID_HEIGHT, ACTION_DELTAS, IMPASSABLE_TERRAIN, DANGEROUS_FIRE_STATES )

def _move_agent(self, action):
    dx, dy = ACTION_DELTAS[action]
    nx = self.agent_pos[0] + dx
    ny = self.agent_pos[1] + dy
    if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
        if self.terrain[ny, nx] not in IMPASSABLE_TERRAIN:
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
        if self.fire_states[sy, sx] in DANGEROUS_FIRE_STATES:
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
