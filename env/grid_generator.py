import numpy as np

from env.constants import (GRID_WIDTH, GRID_HEIGHT, IMPASSBLE_TERRAIN, )

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
