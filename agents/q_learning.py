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

