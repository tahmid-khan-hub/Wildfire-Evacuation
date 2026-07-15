import numpy as np

from env.constants import ( GRID_WIDTH, GRID_HEIGHT, DANGEROUS_FIRE_STATES, FIREPROOF_TERRAIN, FIRE_LOW, IGNITION_PROB )

# 8-directional neighborhood
NEIGHBOR_OFFSETS = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
]

def init_burn_timer():
    # tracks how many consecutive steps each cell has been burning for
    return np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int16)

def _try_ignite_neighbors(terrain, fire_state, new_fire_state, new_burn_timer, rng):
    burning_cells = np.argwhere(np.isin(fire_state, list(DANGEROUS_FIRE_STATES))) # get the all burning cells cordinate

    for (y, x) in burning_cells:
        for dx, dy in NEIGHBOR_OFFSETS:
            nx, ny = x + dx, y + dy
            if not 0 <= nx < GRID_WIDTH and 0 <= ny <GRID_HEIGHT:
                continue # checking whether the neighbor cells is inside the grid or not

            neighbor_terrain = terrain[ny, nx]
            if neighbor_terrain in FIREPROOF_TERRAIN:
                continue # if the neighbor cell is fire proof then skip
            if fire_state[ny, nx] != FIRE_LOW:
                continue # already burning or ash

            prob = IGNITION_PROB[neighbor_terrain]
            if rng.random() < prob: # if the random num is less than the neighbor probability then it will caught fire
                new_fire_state[ny, nx] = FIRE_LOW
                new_burn_timer[ny, nx] = 0 # for fire_low the value is set as 0



