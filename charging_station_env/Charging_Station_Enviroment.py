from typing import Any, Dict, Optional, Tuple, List, Type
import charging_station_env
from charging_station_env.transition.Constants import Status, ChargingStatus

import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import random
import numpy as np
import torch
import pickle
import json


class ChargingStationEnv(gym.Env):
    """
    Initialize the Charging Station Environment.

    Arguments:
        schema: the path to the schema file to config the environment/charging station
        initializer: the initializer instance
        simulation_controller: the simulation controller instance
        action_controller: the action controller instance
        current_folder: the path to the current folder
        results_folder: the path to the results folder
    """
    def __init__(self, schema: str, initializer: Type[charging_station_env.Initializer_Base], simulation_controller: Type[charging_station_env.Simulate_Station_Base], action_controller: Type[charging_station_env.Simulate_Actions_Base], visualizer: Type[charging_station_env.Rendering_Base]=None, current_folder: Optional[str]=None, results_folder: Optional[str]=None):
        super().__init__()
        with open(schema, 'r', encoding='utf-8') as f:
            self.schema = json.load(f)
        
        self.initializer = initializer
        self.simulation_controller = simulation_controller
        self.action_controller = action_controller
        self.visualizer = visualizer
        self.number_of_days = self.schema['number_of_days']
        self.price_flag = self.schema['price_flag']
        self.solar_flag = self.schema['solar_flag']
        self.schema_name = self.schema['schema_name']
        self.grid_limit = self.schema['grid_limit'] # -1 for unlimited power
        self.has_grid_limit = (self.grid_limit > 0.)
        self.step_time = self.schema['step_time'] # in minutes
        self.TIMESTEP_MAX = 24*60
        self.algo_name = None
        self.id_save = None
        self.done = False
        
        # EV parameters
        self.ev_types = self.schema['Types_of_EV']
        self.ev_config = self.schema['EV_config']
        
        # Charger types
        self.chargers_types = self.schema['Types_of_chargers']
        self.chargers_config = self.schema['Chargers_config']
        self.number_of_chargers = len(self.chargers_config['list_chargers'])
        self.preemptive_charging = bool(self.chargers_config['preemptive'])
        self.use_V2G = self.chargers_config['use_V2G']

        self.results_folder = results_folder
        self.current_folder = current_folder

        if (self.chargers_config['charging_mode'] == 'variable'):
            # Actions: Accept or reject EV (shape: number_of_chargers) + Charging rate (shape: number_of_chargers)
            self.action_space = spaces.Box(
                low=-1 if self.use_V2G else 0,
                high=1, shape=(self.number_of_chargers*2,),
                dtype=np.float32
            )
        elif (self.chargers_config['charging_mode'] == 'constant'):
            if self.use_V2G:
                print('WARNING: V2G will be ignored since the charging mode is constant!')
            # Note: No V2G possible for constant charging mode
            self.action_space = spaces.MultiBinary(self.number_of_chargers*2)
        elif (self.chargers_config['charging_mode'] == 'discrete'):
            list_len_charging = []
            self.discrete_charging_rate = [[] for _ in range(self.number_of_chargers)]
            self.discrete_charging_eff = [[] for _ in range(self.number_of_chargers)]
            for charger in range(self.number_of_chargers):
                charging_rate = self.chargers_types[self.chargers_config['list_chargers'][charger]]['charging_rate']
                if isinstance(charging_rate, list):
                    charging_rate = charging_rate.copy()
                    list_len_charging.append( len(charging_rate) )
                else:
                    charging_rate = [charging_rate]
                    list_len_charging.append( 1 )
                if 0 not in charging_rate:
                    charging_rate.append( 0 )
                    list_len_charging[-1] += 1
                self.discrete_charging_rate[charger] = charging_rate

                charging_eff = self.chargers_types[self.chargers_config['list_chargers'][charger]]['charging_efficiency']
                if not isinstance(charging_eff, list):
                    charging_eff = [charging_eff for _ in range(len(charging_rate))]
                else:
                    charging_eff = charging_eff.copy()
                self.discrete_charging_eff[charger] = charging_eff
                
                if self.use_V2G:
                    discharging_rate = self.chargers_types[self.chargers_config['list_chargers'][charger]]['discharging_rate']
                    if isinstance(discharging_rate, list):
                        discharging_rate = discharging_rate.copy()
                        list_len_charging[-1] += len(discharging_rate)
                    else:
                        discharging_rate = [discharging_rate]
                        list_len_charging[-1] += 1
                    self.discrete_charging_rate[charger] = self.discrete_charging_rate[charger] + [-rate for rate in discharging_rate]

                    discharging_eff = self.chargers_types[self.chargers_config['list_chargers'][charger]]['discharging_efficiency']
                    if not isinstance(discharging_eff, list):
                        discharging_eff = [discharging_eff for _ in range(len(discharging_rate))]
                    else:
                        discharging_eff = discharging_eff.copy()
                    self.discrete_charging_eff[charger] = self.discrete_charging_eff[charger] + [eff for eff in discharging_eff]
                
                assert len(self.discrete_charging_rate[charger]) == len(self.discrete_charging_eff[charger])

            # Actions: Accept or reject EV (shape: number_of_chargers) + Charging rate (shape: number_of_chargers)
            self.action_space = spaces.MultiDiscrete([2 for _ in range(self.number_of_chargers)] + list_len_charging)
        else:
            raise Exception(f'charging_mode: "{self.chargers_config["charging_mode"]}" not recognized')
        
        low = np.array(-np.ones(self.simulation_controller.get_observation_size(self)), dtype=np.float32)
        high = np.array(np.ones(self.simulation_controller.get_observation_size(self)), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float64
            #dtype=np.float32 # PATCH: Error with check_env(env): AssertionError: The observation returned by the `reset()` method does not match the given observation space
        )

        self.seed(seed=self.schema['seed'])

    def action_masks(self):
        return self.action_controller.action_masks(self)

    """
    Update the environment according to an action.

    Arguments:
        actions: The action to be performed on the environment
    Returns:
        Tuple[np.ndarray, float, bool, dict]: The next state, the reward, whether the episode is done, whether the episode is troncated, and additional informations
    """
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        reward, info = self.action_controller(self, actions)

        self.timestep = self.timestep + self.step_time
        next_state = self.get_obs()
        if self.timestep >= self.TIMESTEP_MAX:
            self.done = True
            self.timestep = 0

            if (self.results_folder is not None):
                self.scenario.append(self.action_controller.get_metrics())
                if (self.algo_name is not None):
                    if (self.id_save is not None):
                        with open(os.path.join(self.results_folder, f'{self.algo_name}-Results-{self.id_save}.pickle'), 'wb') as f:
                            pickle.dump(self.scenario, f, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        with open(os.path.join(self.results_folder, f'{self.algo_name}-Results.pickle'), 'wb') as f:
                            pickle.dump(self.scenario, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open(os.path.join(self.results_folder, f'Results.pickle'), 'wb') as f:
                            pickle.dump(self.scenario, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.action_controller.reset_metrics()

        return next_state, reward, self.done, False, info

    """
    Reset the environment.

    Arguments:
        reset_flag: 0 for a new scenario, 1 for the same scenario
        id_save: The id of the scenario to be loaded
        algo_name: The name of the algorithm used
    
    Returns:
        Tuple[np.ndarray, dict]: The initial state of the environment and additional informations
    """
    def reset(self, reset_flag: int=0, id_save: Optional[str]=None, algo_name: Optional[str]=None, seed: Optional[int]=None, options: Dict[str, Any]=None) -> Tuple[np.ndarray, Dict]:
        self.timestep = 0
        self.done = False
        self.algo_name = algo_name
        self.id_save = id_save

        charging_requests, self.energy = self.initializer(self, reset_flag)
        array_size = self.TIMESTEP_MAX//self.step_time+1
        self.scenario = [{} for _ in range(array_size)]
        n = 0
        for time_requests in charging_requests:
            for req in time_requests:
                if(len(self.scenario[req['arr']].keys()) < self.number_of_chargers):
                    self.scenario[req['arr']][n] = {'num': n, 'arrival': req['arr'], 'departure': req['dep'], 'initial_soc': req['soc'], 'current_soc': req['soc'], 'status': Status.ARRIVED, 'charging_status': {ChargingStatus.UNKNOWN}, 'charger': -1, 'type': req['type']}
                    n += 1
        
        return self.get_obs(), {'energy': self.energy,
                                'scenario': self.scenario}

    """
    Render the environment.
    """
    def render(self):
        self.visualizer(self, self.get_obs(update=False))

    """
    Get the observation of the environment.

    Arguments:
        update: Whether to update the observation
    Returns:
        np.ndarray: The observation of the environment
    """
    def get_obs(self, update: Optional[bool]=True) -> np.ndarray:
        if((not update) and (self.observations is not None)):
            return self.observations
        self.observations = self.simulation_controller(self)
        return self.observations

    """
    Seed the environment.

    Arguments:
        seed: The seed to be used
    Returns:
        List[int]: The seed used
    """
    def seed(self, seed: Optional[int]=None) -> List[int]:
        _, seed = seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return seed

    """
    Close the environment.

    Arguments:
        None
    Returns:
        int: 0
    """
    def close(self) -> int:
        return 0
        
