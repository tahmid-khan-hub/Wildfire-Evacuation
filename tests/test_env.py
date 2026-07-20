import numpy as np
import pytest
from env.emberpath_env import EmberPathEnv

# to check reset() works correctly
def test_reset_returns_valid_obs():
    env = EmberPathEnv(seed=1)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape # if it is false then it will return assertion error - using it for testing
    assert not np.any(np.isnan(obs)) # checking whether a Nan is present or not - if yes then assertion error 
    # (not used because if it is false then not will reverse it to true and assertion error will not be thrown if it is true - here false means no Nan present in obs)

