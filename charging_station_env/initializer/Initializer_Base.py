from typing import Any, Dict, Optional, Tuple, Union, List, Type
from charging_station_env.initializer import Energy_Initializer_Base
import gymnasium as gym
import pickle
import os

"""
This is the base class for all the initializers. It is used to initialize the energy computation and setup the scenario of the environment.
"""
class Initializer_Base:
    """
    Initializes the class with the given parameters.
    
    Parameters:
        energy_initializer : The energy initializer to use
    """
    def __init__(self, energy_initializer: Energy_Initializer_Base):
        self.energy_initializer = energy_initializer

    """
    This function is used to initialize the energy computation and setup the scenario of the environment.

    Parameters:
        env : The environment in which the agent is acting
        reset_flag : The flag indicating if the environment is being reset

    Returns:
        A tuple containing the scenario and the energy
    """
    def __call__(self, env: Type[gym.Env], reset_flag: int) -> Tuple[Dict, Dict]:
        if (reset_flag == 0):
            energy = self._energy_init(env)
            scenario = self._scenario_init(env)
            self._save(env, scenario, energy)
            return scenario, energy
        else:
            scenario_energy = self._load(env)
            return scenario_energy['scenario'], scenario_energy['energy']
    
    """
    This function is used to setup the scenario of the environment.

    Parameters:
        env : The environment in which the agent is acting
    
    Returns:
        A dictionary containing the scenario
    """
    def _scenario_init(self, env: Type[gym.Env]) -> Dict:
        raise NotImplementedError('_scenario_init function at Base class is not implemented')

    """
    This function is used to initialize the energy computation of solar production and electricity price of the environment.

    Parameters:
        env : The environment in which the agent is acting

    Returns:
        A dictionary containing the solar production and electricity price
    """
    def _energy_init(self, env: type) -> Dict:
        return self.energy_initializer(env)

    """
    This function is used to save the scenario of the environment.

    Parameters:
        env : The environment in which the agent is acting
        scenario : The scenario to save
    """
    def _save(self, env: Type[gym.Env], scenario: Dict, energy: Dict) -> None:
        data = {'scenario': scenario, 'energy': energy}
        if(env.current_folder is not None):
            if (env.id_save is not None):
                with open(os.path.join(env.current_folder, f'Initial_Values-{env.id_save}.pickle'), 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(os.path.join(env.current_folder, 'Initial_Values.pickle'), 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    """
    This function is used to load the scenario of the environment.

    Parameters:
        env : The environment in which the agent is acting

    Returns:
        A dictionary containing the scenario
    """
    def _load(self, env: Type[gym.Env]) -> Dict:
        if (env.id_save is not None):
            with open(os.path.join(env.current_folder, f'Initial_Values-{env.id_save}.pickle'), 'rb') as f:
                data = pickle.load(f)
        else:
            with open(os.path.join(env.current_folder, f'Initial_Values.pickle'), 'rb') as f:
                data = pickle.load(f)
        return data
