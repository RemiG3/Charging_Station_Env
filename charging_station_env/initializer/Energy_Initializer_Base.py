from typing import Any, Dict, Optional, Tuple, Union, List, Type
import numpy as np
import gymnasium as gym

"""
This is the base class for all the energy initializers. It is used to initialize the energy computation of solar production and electricity price of the environment.
"""
class Energy_Initializer_Base:
    """
    Initializes the class.
    """
    def __init__(self):
        pass
    
    """
    This function is used to initialize the energy computation of solar production and electricity price of the environment.
    
    Parameters:
        env : The environment in which the agent is acting
    
    Returns:
        A dictionary containing the solar production and electricity price
    """
    def __call__(self, env: Type[gym.Env]) -> Dict:
        raise NotImplementedError('Call to Base class is not implemented')

