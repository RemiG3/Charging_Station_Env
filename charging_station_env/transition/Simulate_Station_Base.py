from typing import Any, Dict, Optional, Tuple, Union, List, Type
import numpy as np
import gymnasium as gym

"""
This is the base class for all the station simulators. It is used to simulate the station of the environment.
"""
class Simulate_Station_Base:
    def __init__(self):
        pass
    
    """
    This function is used to get the observation size of the environment.

    Parameters:
        env : The environment in which the agent is acting

    Returns:
        Size of the observation
    """
    def get_observation_size(self, env: Type[gym.Env]) -> int:
        return 8+5*env.number_of_chargers
    
    """
    This function is used to simulate the station of the environment.

    Parameters:
        env : The environment in which the agent is acting

    Returns:
        A numpy array containing the state of the station
    """
    def __call__(self, env: Type[gym.Env]) -> np.ndarray:
        raise NotImplementedError('Call to Base class is not implemented')
