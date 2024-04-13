import sys
from typing import Dict, Type
from charging_station_env.initializer import Initializer_Base, Energy_Initializer_Base, Energy_Initializer
from charging_station_env.initializer.synthetic_data_generator.generate_sample import generate_sample
import charging_station_env.initializer.synthetic_data_generator.modeling as modeling
import charging_station_env.initializer.synthetic_data_generator.handles as handles
sys.modules['modeling'] = modeling
sys.modules['handles'] = handles
import numpy as np
import random
import gymnasium as gym
import time
import datetime
import pickle
import os


"""
This class initializes the scenario of the environment.
Data is generated from the EV-SDG dataset (from https://github.com/mlahariya/EV-SDG).
"""
class Initializer_FIFO(Initializer_Base):
    MODELS = {"ACpoisson_fit": "SDG Model (AC,poisson_fit)",
              "AVneg_bio_reg": "SDG Model (AC,neg_bio_reg)",
              "IATloess": "SDG Model (IAT,poly)",
              "IATmean": "SDG Model (IAT,mean)"}
    
    """
    Initializes the class with hyperparameters for scenarios generation.
    
    Parameters:
        number_of_vehicles : The number of vehicles to use in the scenario
        datafile : The file containing the data to generate the scenario
        energy_initializer : The energy initializer to use
    """
    def __init__(self,
                 number_of_vehicles: int=100,
                 nb_ev_min_range: int=None,
                 nb_ev_max_range: int=None,
                 energy_initializer: Energy_Initializer_Base=Energy_Initializer(),
                 alpha_param: float=1.5,
                 beta_param: float=2.5,
                 coeff: float=2.,
                 bias: float=1.,
                 min_soc: float=.1,
                 max_soc: float=.6,
                 b_max: float=65.):
        super().__init__(energy_initializer=energy_initializer)
        self.number_of_vehicles = number_of_vehicles if(nb_ev_min_range is None or nb_ev_max_range is None) else None
        self.nb_ev_min_range = nb_ev_min_range
        self.nb_ev_max_range = nb_ev_max_range
        self.alpha_param = alpha_param
        self.beta_param = beta_param
        self.coeff = coeff
        self.bias = bias
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.b_max = b_max
        self.epsilon = 1e-6


    """
    Setup the scenario of the environment.

    Parameters:
        env : The environment in which the agent is acting

    Returns:
        A dictionary containing the scenario
    
    Raises:
        AssertionError: If the charging station has different chargers
    """
    def _scenario_init(self, env: Type[gym.Env]) -> Dict:
        assert len(set(env.chargers_config['list_chargers'])) == 1, 'The charging station should have identical chargers'
        timestep_max = env.TIMESTEP_MAX
        step_time = env.step_time
        number_of_vehicles = random.randint(self.nb_ev_min_range, self.nb_ev_max_range) if (self.number_of_vehicles is None) else self.number_of_vehicles
        
        chargers_ev_compatibilities = {}
        for charger_type in env.chargers_types:
            chargers_ev_compatibilities[charger_type] = []
            for ev_num, ev_type in enumerate(env.ev_config['considered_ev']):
                if charger_type in env.ev_types[ev_type]['chargers_type_compatibilities']:
                    chargers_ev_compatibilities[charger_type].append(ev_num)
        
        model_name = self.MODELS[env.schema['data']['sdg_model_name']]
        file = os.path.join(env.schema['data']['sdg_path'], model_name)
        with open(file, 'rb') as f:
            x = pickle.load(f)
        AM, MMc, MMe = x[0], x[1], x[2]

        restart = True
        df = None
        while restart:
            try: # Sometimes sampling fails, so we need to restart
                df = generate_sample(AM=AM, MMc=MMc, MMe=MMe,
                                     horizon_start=horizon_start, horizon_end=horizon_end, n=number_of_vehicles)
                restart = False
            except:
                restart = True
                horizon_start = datetime.datetime.strptime('01/01/2015', '%d/%m/%Y') + datetime.timedelta(days=random.randint(1, 364))
                horizon_end = horizon_start + datetime.timedelta(days=1)
                time.sleep(.5)

        # Retrieve the data to keep
        columns_to_keep = ['Arrival', 'Connected_time', 'Energy_required']
        list_cr = df[columns_to_keep].to_numpy()
        np.random.shuffle(list_cr)
        list_cr = list_cr[list_cr[:, 0] <= 23.5]
        list_cr = list_cr[:number_of_vehicles]
        if(number_of_vehicles > len(list_cr)):
            initial_list_cr = np.array(list_cr.tolist().copy())
            idx = [random.randint(0, len(initial_list_cr)-1) for _ in range(number_of_vehicles-len(list_cr))]
            list_cr = list_cr.tolist().copy() + initial_list_cr[idx].tolist().copy()
            list_cr = np.array(list_cr)
        
        # Arrival times generation
        arrivals = list_cr[:, 0]
        arrivals = np.round(arrivals.astype(float) * (60 // step_time), 0).astype(int)
        
        # Initial SOC generation
        requested_energy = list_cr[:, 2]
        norm_energy = 1. - ( requested_energy / requested_energy.max() )
        initial_socs = self.min_soc + norm_energy * (self.max_soc - self.min_soc)

        # Parking duration generation
        charger_type = env.chargers_types[env.chargers_config['list_chargers'][0]]
        p = charger_type['charging_rate'] * charger_type['charging_efficiency']
        min_dur = (self.b_max * (.8 - initial_socs)) / p * (60 // step_time)
        rvs = np.random.beta(self.alpha_param, self.beta_param, len(min_dur))
        rvs = rvs * self.coeff + self.bias
        durations = min_dur * rvs
        durations = np.ceil(durations).astype(int)
        
        # Departures times calculation
        departures = []
        for arrival, duration in zip(arrivals, durations):
            dep = arrival+duration
            dep = dep if (dep < timestep_max//step_time) else timestep_max//step_time
            departures.append(dep)

        departures_requests = np.array(departures)
        arrivals_requests = np.array(arrivals)

        # Future consideration: To weight the sum for random choice with EV compatibilities
        ev_types_requests = [random.choice(list(range(len(env.ev_config['considered_ev'])))) for _ in range(len(arrivals))]
        
        array_size = timestep_max//env.step_time+1
        charging_requests = [[] for _ in range(array_size)]
        for soc, arr, dep, type in zip(initial_socs, arrivals_requests, departures_requests, ev_types_requests):
            charging_requests[arr].append({'soc': soc, 'arr': arr, 'dep': dep, 'type': type})
        
        return charging_requests
