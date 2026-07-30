# training/oracle_bruteforce.py
import os
import sys
import copy
import itertools
import numpy as np
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.emberpath_env import EmberPathEnv
from env.constants import GRID_WIDTH, GRID_HEIGHT, ACTION_DELTAS, IMPASSBLE_TERRAIN, DANGEROUS_FIRE_STATES

TRAIN_SEED = 42


def bfs_next_step(terrain, fire_state, start, goal):
    if start == goal:
        return None
    visited = {start}
    queue = deque([(start, None)])
    while queue:
        (x, y), first_action = queue.popleft()
        for action, (dx, dy) in ACTION_DELTAS.items():
            nx, ny = x + dx, y + dy
            if not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT):
                continue
            if terrain[ny, nx] in IMPASSBLE_TERRAIN:
                continue
            if fire_state[ny, nx] in DANGEROUS_FIRE_STATES:
                continue
            if (nx, ny) in visited:
                continue
            next_first_action = first_action if first_action is not None else action
            if (nx, ny) == goal:
                return next_first_action
            visited.add((nx, ny))
            queue.append(((nx, ny), next_first_action))
    return None


def try_order(order):
    env = EmberPathEnv(seed=TRAIN_SEED)
    env.reset(seed=TRAIN_SEED)

    terminated = False
    truncated = False
    order_idx = 0

    while not (terminated or truncated) and order_idx < len(order):
        target_i = order[order_idx]

        if env.survivor_rescued[target_i] or env.survivor_burned[target_i]:
            order_idx += 1
            continue

        goal = env.survivor_positions[target_i]
        action = bfs_next_step(env.terrain, env.fire_state, env.agent_pos, goal)

        if action is None:
            return None  # stuck, can't proceed on this order right now

        obs, reward, terminated, truncated, info = env.step(action)

        if env.survivor_rescued[target_i] or env.survivor_burned[target_i]:
            order_idx += 1

    return info["rescued"], info["burned"]


def search_all_orders():
    best_result = None
    best_order = None
    for order in itertools.permutations([0, 1, 2]):
        result = try_order(order)
        if result is None:
            print(f"Order {order}: got stuck (no path available)")
            continue
        rescued, burned = result
        print(f"Order {order}: rescued={rescued}/3, burned={burned}/3")
        if best_result is None or rescued > best_result[0]:
            best_result = result
            best_order = order

    print("-" * 60)
    print(f"Best possible: order={best_order}, result={best_result}")


if __name__ == "__main__":
    search_all_orders()