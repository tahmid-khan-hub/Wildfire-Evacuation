import numpy as np

from env.constants import ( GRID_WIDTH, GRID_HEIGHT, DANGEROUS_FIRE_STATES, FIREPROOF_TERRAIN, FIRE_LOW, IGNITION_PROB, BURN_DURATION, FIRE_ASH, FIRE_HIGH, FIRE_MID )

# 8-directional neighborhood
NEIGHBOR_OFFSETS = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
]

def init_burn_timer():
    # tracks how many consecutive steps each cell has been burning for
    return np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int16)

def spread_fire(terrain: np.ndarray, fire_state: np.ndarray, burn_timer: np.ndarray, rng: np.random.Generator):
    # work on copies so the original state is not modified while updating
    new_fire_state = fire_state.copy()
    new_burn_timer = burn_timer.copy()

    _try_ignite_neighbors(terrain, fire_state, new_fire_state, new_burn_timer, rng)
    _advance_burning_cells(terrain, fire_state, new_fire_state, new_burn_timer)
 
    return new_fire_state, new_burn_timer

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

def _advance_burning_cells(terrain, fire_state,  new_fire_state, new_burn_timer):
    ys, xs = np.where(np.isin(fire_state, list(NEIGHBOR_OFFSETS)))

    for i in range(len(ys)):
        y = ys[i]
        x = xs[i]

        new_burn_timer[y, x] += 1

        duration = BURN_DURATION[terrain[y, x]]
        third = max(1, duration // 3) # to get int value and timer at least 1 (min)
        elapsed = new_burn_timer[y, x]

        if(elapsed >= duration):
            new_fire_state[y, x] = FIRE_ASH # if the burn time is greater than the duration then it already burnt - become ash
        elif(elapsed >= 2*third):
            new_fire_state[y, x] = FIRE_HIGH # if duration = 12 and third = 4 then 0-low, 4-mid, 8-high, 12-ash. to get the high we have to pass from mid and low third means 2 level - thats why *2
        elif(elapsed >= third):
            new_fire_state[y, x] = FIRE_MID 
        else:
            new_fire_state[y, x] = FIRE_LOW # means < third is low
