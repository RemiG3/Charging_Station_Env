from typing import Any, Dict, Optional, Tuple, Union, List, Type
import numpy as np
import gymnasium as gym

"""
This is the base class for all the action simulators. It is used to simulate the actions taken by the agent in the environment.
"""
class Simulate_Actions_Base:
    """
    Initializes the class with the given parameters.

    Parameters:
        alpha_reject : The penalty for rejecting a demand
        alpha_grid_limit : The penalty for exceeding the grid limit
        alpha_price : The penalty for the price of charging
        alpha_charging_penalty : The penalty for the remaining SOC at departure
    """
    def __init__(self):
        pass
    
    """
    This function is used to get the action space of the environment.

    Parameters:
        env : The environment in which the agent is acting

    Returns:
        Gymnasium action space
    """
    def get_action_space(self, env: Type[gym.Env]) -> int:
        return NotImplementedError('get_action_space function at Base class is not implemented')

    """
    This function is used to simulate the actions taken by the agent in the environment.

    Parameters:
        env : The environment in which the agent is acting
        actions : The actions taken by the agent

    Returns:
        A tuple containing the reward and the metrics
    """
    def __call__(self, env: Type[gym.Env], actions: np.ndarray) -> Tuple[float, Dict]:
        raise NotImplementedError('__call__ function at Base class is not implemented')
    
    """
    This function is used to get the metrics of the environment.

    Parameters:
        env : The environment in which the agent is acting

    Returns:
        A dictionary containing the metrics
    """
    def get_metrics(self) -> Dict:
        return {}
