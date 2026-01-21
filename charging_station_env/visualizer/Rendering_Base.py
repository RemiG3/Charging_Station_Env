from typing import Any, Dict, Optional, Tuple, Union, List, Type
import numpy as np
import gymnasium as gym

"""
This is the base class for all rendering methods. It is used to visualize the states of the environment.
"""
class Rendering_Base:
    def __init__(self):
        pass
    
    """
    This function is used to visualize the state of the environment.

    Parameters:
        env : The environment in which the agent is acting
    """
    def __call__(self, env: Type[gym.Env]):
        raise NotImplementedError('Call to Base class is not implemented')
