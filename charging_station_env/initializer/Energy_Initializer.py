from typing import Dict, Type
from charging_station_env.initializer import Energy_Initializer_Base
import numpy as np
import gymnasium as gym
import pandas as pd
import datetime
import random


class Energy_Initializer(Energy_Initializer_Base):
    """
    Initializes the class with the given parameters.
    
    Parameters:
        nb_predict_timestep : The number of timesteps to predict the solar production and electricity price
    """
    def __init__(self, nb_predict_timestep: int=3, pv_peak_production: float=75.900): # 75,9 kW
        self.nb_predict_timestep = nb_predict_timestep
        self.renewable_normalizer = pv_peak_production
    
    """
    Initialize the energy computation of solar production and electricity price of the environment.
    
    Parameters:
        env : The environment in which the agent is acting
    
    Returns:
        A dictionary containing the solar production and electricity price
    """
    def __call__(self, env: Type[gym.Env]) -> Dict:
        days_of_experiment = env.number_of_days
        assert days_of_experiment == 1, "The number of days should be 1"
        price_flag = env.price_flag

        # Load the solar production data
        df = pd.read_csv(env.schema['data']['pv_data_path'])
        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')

        step_time = env.step_time
        timestep = 60/step_time
        nb_timesteps = int(round(timestep*24))
        df['daytime'] = df['time'].apply(lambda x: datetime.datetime.strptime(f'{x.day}/{x.month}/{x.year}', '%d/%m/%Y'))
        renewable = np.zeros((days_of_experiment*(nb_timesteps+self.nb_predict_timestep)))
        timestep = int(round(timestep)) if (timestep > 1) else 1
        for d in range(0, int(days_of_experiment)):
            idx = random.randint(0, len(df['daytime'])-1) # Randomly select a day
            df_day_selected = df[df['daytime'] == df['daytime'][idx]].copy()
            df_day_selected = df_day_selected.reset_index(drop=True)
            df_day_selected.loc[len(df_day_selected)] = [df_day_selected['time'][len(df_day_selected)-1] + datetime.timedelta(minutes=60), 0, 0, 0, 0, 0, 0, 0]
            df_resampled = df_day_selected.resample(f'{step_time}T', on='time').mean()
            renewable[d*(nb_timesteps+self.nb_predict_timestep):(d+1)*(nb_timesteps+self.nb_predict_timestep)] = (df_resampled['P'].interpolate().to_numpy() / 1000).tolist()[:-1] + [-self.renewable_normalizer for _ in range(self.nb_predict_timestep)]
        consumed = np.zeros(np.shape(renewable))
        consumed_normalizer = self.renewable_normalizer

        # Load the electricity price data
        price_day = []
        if price_flag == 'random':
            price_flag = random.randint(0, 4)
        else:
            price_flag = int(price_flag)

        if price_flag == 0:
            price_day = np.ones(24) * 0.1
        elif price_flag == 1:
            price_day = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
        elif price_flag == 2:
            price_day = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07, 0.08 ,0.09, 0.1, 0.1, 0.1, 0.08, 0.06, 0.05, 0.05, 0.05, 0.06, 0.06 ,0.06 ,0.06, 0.05, 0.05, 0.05])
        elif price_flag == 3:
            price_day = np.array([0.071, 0.060, 0.056, 0.056, 0.056, 0.060, 0.060, 0.060, 0.066, 0.066, 0.076, 0.080, 0.080, 0.1, 0.1, 0.076, 0.076, 0.1, 0.082, 0.080, 0.085, 0.079, 0.086, 0.070])
        elif price_flag == 4:
            price_day = np.array([0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.06, 0.06, 0.1, 0.1, 0.1, 0.1])
        else:
            raise ValueError('The price flag is not valid')

        price = np.zeros((days_of_experiment*(nb_timesteps+self.nb_predict_timestep)))
        price_normalizer = price_day.max()
        timestep = int(round(timestep)) if (timestep > 1) else 1
        for d in range(days_of_experiment):
            for n in range(0, nb_timesteps, timestep):
                price[(d+1)*n:(d+1)*(n+timestep)] = price_day[n//timestep]
            price[(d+1)*nb_timesteps:(d+1)*nb_timesteps+self.nb_predict_timestep] = -price_normalizer
        
        return {'consumed': consumed, 'renewable': renewable,
                'price': price,
                'normalizers': {
                    'consumed': consumed_normalizer,
                    'renewable': self.renewable_normalizer,
                    'price': price_normalizer,
                    }
            }

