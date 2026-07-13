import numpy as np

from env.constants import (GRID_WIDTH, GRID_HEIGHT, IMPASSBLE_TERRAIN, )

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
