import numpy as np

from env.constants import (GRID_WIDTH, GRID_HEIGHT, IMPASSBLE_TERRAIN, TERRAIN_FOREST, TERRAIN_BRUSH, TERRAIN_ROCK, TERRAIN_WATER, FIRE_NONE, FIRE_LOW, NUM_INITIAL_IGNITIONS, NUM_SURVIVORS)

def generate_grid(rng: np.random.Generator = None):
    # generate a fresh grid, fire state grid, agent start position and survivor positions for a new episode
    if rng is None:
        rng = np.random.default_rng()

    terrain = _generate_terrain(rng)
    fire_state = np.full((GRID_HEIGHT, GRID_WIDTH), FIRE_NONE, dtype=np.int8)

    passable_cells = _get_passable_cells(terrain)

    # the location where fire initially starts
    ignition_points = _pick_random_cells(passable_cells, NUM_INITIAL_IGNITIONS, rng)
    for (x,y) in ignition_points:
        fire_state[y, x] = FIRE_LOW

    used = set(ignition_points)

    # place survivors
    available_for_survivors = []
    for c in passable_cells:
        if c not in used:
            available_for_survivors.append(c)
    survivor_positions = _pick_random_cells( available_for_survivors, NUM_SURVIVORS, rng )

    used.update(survivor_positions)

    # place agent
    available_for_agents = []
    for a in passable_cells:
        if a not in used:
            available_for_agents.append(a)
    agent_start = _pick_random_cells( available_for_agents, 1, rng )[0]

    return terrain, fire_state, agent_start, survivor_positions

def _generate_terrain(rng: np.random.Generator) -> np.ndarray:
    # build a static terrain layer - mostly forest, brush patches, a rock scattering and one water body
    terrain = np.full((GRID_HEIGHT, GRID_WIDTH), TERRAIN_FOREST, dtype=np.int8)

    _scatter_patches(terrain, TERRAIN_BRUSH, patch_count=3, patch_size=4, rng=rng)
    _scatter_patches(terrain, TERRAIN_ROCK, patch_count=2, patch_size=2, rng=rng)
    _carve_water_body(terrain, TERRAIN_WATER, length=6, rng=rng)

    return terrain


def _scatter_patches(terrain, terrain_type, patch_count, patch_size, rng):
    # drop a few small random-walk blobs of a given terrain type onto the grid
    for _ in range(patch_count):
        x = rng.integers(0, GRID_WIDTH)
        y = rng.integers(0, GRID_HEIGHT)
        for _ in range(patch_size):
            if 0 <= x < GRID_WIDTH and 0 <= y <GRID_HEIGHT:
                terrain[y, x] = terrain_type
            # now random walk to next cell in the patch
            dx, dy = rng.integers(-1, 2), rng.integers(-1, 2) # range (-1,0,1)
            x = np.clip(x+dx, 0, GRID_WIDTH-1) # clip(value, min, max)
            y = np.clip(y+dy, 0, GRID_HEIGHT-1)

def _carve_water_body(terrain, terrain_type, length, rng):
    # carve a single winding line of water cells
    x = rng.integers(0, GRID_WIDTH)
    y = rng.integers(0, GRID_HEIGHT)
    for _ in range(length):
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            terrain[y,x] = terrain_type
        dx, dy = rng.integers(-1, 2) , rng.integers(-1, 2)
        x = np.clip(x+dx, 0, GRID_WIDTH-1)
        y = np.clip(y+dy, 0, GRID_HEIGHT-1)

def _get_passable_cells(terrain) -> list:
    # return all cells where agent could stand on
    cells = []
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if terrain[y,x] not in IMPASSBLE_TERRAIN:
                cells.append((x,y))
    return cells

def _pick_random_cells(candidates: list, count: int, rng: np.random.Generator) -> list:
    # pick count unique random cells from a candidate list without replacement
    if len(candidates) < count:
        raise ValueError(
            f"not enough passable cells ({len(candidates)}) to place {count} items."
        )
    indices = rng.choice(len(candidates), size=count, replace=False)
    return [candidates[i] for i in indices]
