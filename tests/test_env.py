import numpy as np
import pytest
from env.emberpath_env import EmberPathEnv
from env.constants import (IMPASSBLE_TERRAIN ,ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT )

@pytest.mark.parametrize(
    "action, dx, dy",
    [
        (ACTION_UP, 0, -1),
        (ACTION_DOWN, 0, 1),
        (ACTION_LEFT, -1, 0),
        (ACTION_RIGHT, 1, 0),
    ],
)

# to check reset() works correctly
def test_reset_returns_valid_obs():
    env = EmberPathEnv(seed=1)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape # if it is false then it will return assertion error - using it for testing
    assert not np.any(np.isnan(obs)) # checking whether a Nan is present or not - if yes then assertion error 
    # (not used because if it is false then not will reverse it to true and assertion error will not be thrown if it is true - here false means no Nan present in obs)
    assert info["step_count"] == 0
    assert info["rescued"] == 0
    assert info["burned"] == 0
    assert info["agent_pos"] == env.agent_pos

# after reset(), the agent and all survivors must be placed on passable terrain
def test_reset_agent_and_survivors_on_passable_terrain():
    env = EmberPathEnv(seed=2)
    env.reset()
    ax, ay = env.agent_pos
    assert env.terrain[ay, ax] not in IMPASSBLE_TERRAIN # checking whether the agent is in the impassble position or not
    for(sx, sy) in env.survivor_positions:
        assert env.terrain[sy, sx] not in IMPASSBLE_TERRAIN # checking whether the survivors are in the impassble positions or not

# calling step() with random actions, the environment should not crash and should always return valid data
def test_step_does_not_crash():
    env = EmberPathEnv(seed=3)
    env.reset() # new episode
    for _ in range(50): # upto 50 times
        action = env.action_space.sample() # here, sample choose any random action value
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape # if the shape changes then assertion error
        assert isinstance(reward, float) # checking whether the reward is in float or not
        if terminated or truncated:
            break # if terminated or truncated happen then stop the loop

def test_agent_moves_correctly(action, dx, dy):
    env = EmberPathEnv(seed=4)
    env.reset()

    sx, sy = env.agent_pos
    env._move_agent(action)
    nx, ny = env.agent_pos

    assert (nx, ny) in [
        (sx, sy), # movement blocked
        (sx + dx, sy + dy), # movement succeed
    ]

def test_same_seed_generates_same_world():
    env1 = EmberPathEnv(seed=42)
    env2 = EmberPathEnv(seed=42)

    env1.reset()
    env2.reset()

    assert np.array_equal(env1.terrain, env2.terrain)
    assert np.array_equal(env1.fire_state, env2.fire_state)
    assert env1.agent_pos == env2.agent_pos
    assert env1.survivor_positions == env2.survivor_positions

