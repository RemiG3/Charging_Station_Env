from typing import Dict, Type
from charging_station_env.initializer.Energy_Initializer_Base import Energy_Initializer_Base
import numpy as np
import gymnasium as gym
import pandas as pd
import datetime
import random



def resample_discard_or_interp_np(src: np.ndarray, newN: int) -> np.ndarray:
    """
    Resample:
      - If newN < oldN: discard via floor mapping
      - If newN > oldN: linear interpolation over evenly spaced samples
      - If newN == oldN: return src
      - If newN == 1: return src[0]
    Assumes src covers a full day uniformly.
    """
    src = np.asarray(src, dtype=float)
    oldN = int(src.size)
    if newN <= 0 or oldN == 0:
        return np.array([], dtype=float)
    if newN == oldN:
        return src.copy()
    if newN == 1:
        return np.array([src[0]], dtype=float)

    dst = np.empty(newN, dtype=float)

    if newN < oldN:
        # Downsample: discard (floor mapping)
        # pos = t * oldN / newN  in [0, oldN)
        t = np.arange(newN, dtype=float)
        pos = t * oldN / newN
        idx = np.floor(pos).astype(int)
        idx = np.clip(idx, 0, oldN - 1)
        dst[:] = src[idx]
    else:
        # Upsample: linear interpolation across [0, oldN-1]
        t = np.arange(newN, dtype=float)
        pos = t * (oldN - 1) / (newN - 1)    # [0, oldN-1]
        i0 = np.floor(pos).astype(int)
        i1 = np.clip(i0 + 1, 0, oldN - 1)
        a = pos - i0
        dst[:] = (1.0 - a) * src[i0] + a * src[i1]

    return dst



class Energy_Initializer(Energy_Initializer_Base):
    """
    Initializes the class with the given parameters.
    
    Parameters:
        pv_peak_production : The peak production of the photovoltaic panels
        select_day_randomly : A flag to select the day randomly or not
    """
    def __init__(self, pv_peak_production: float=75.900, # 75,9 kW
                       select_day_randomly: bool=True):
        self.renewable_normalizer = pv_peak_production
        if(isinstance(select_day_randomly, str)):
            select_day_randomly = (select_day_randomly.strip().lower() == 'true')
        self.select_day_randomly = select_day_randomly
        self.day = 0 # The day of the year to get the PV production
    
    """
    Initialize the energy computation of solar production and electricity price of the environment.
    
    Parameters:
        env : The environment in which the agent is acting
    
    Returns:
        A dictionary containing the solar production and electricity price
    """
    def __call__(self, env: Type[gym.Env]) -> Dict:
        days_of_experiment = env.schema['number_of_days']
        assert days_of_experiment == 1, "The number of days should be 1"
        price_flag = env.schema['price_flag']

        # Step size and total steps
        step_time = int(round(env.step_time))       # minutes per step
        nb_timesteps = int(round(24 * 60 / step_time))

        # ---------- PV ----------
        df = pd.read_csv(env.schema['data']['pv_data_path'])
        # Expect at least columns: ['time','P'] with uniform sampling over the day
        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
        df['day'] = df['time'].dt.normalize()

        # pick a day
        if self.select_day_randomly:
            chosen_day = df['day'].sample(1).iloc[0]
        else:
            uniq = df['day'].drop_duplicates().sort_values().to_list()
            chosen_day = uniq[self.day % len(uniq)]

        df_day = df[df['day'] == chosen_day].sort_values('time').reset_index(drop=True)

        # Source series assumed uniformly spaced across the day (like 96×15min)
        pv_src = df_day['P'].to_numpy(dtype=float) / 1000.0  # W -> kW if needed

        renewable = resample_discard_or_interp_np(pv_src, nb_timesteps)

        # ---------- PRICE ----------
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

        price = resample_discard_or_interp_np(price_day.astype(float), nb_timesteps)
        price_normalizer = float(price_day.max())

        # ---------- consumed ----------
        consumed = np.zeros_like(renewable, dtype=float)
        consumed_normalizer = self.renewable_normalizer

        if not self.select_day_randomly:
            self.day += 1

        return {
            'consumed': consumed,
            'renewable': renewable,
            'price': price,
            'normalizers': {
                'consumed': consumed_normalizer,
                'renewable': self.renewable_normalizer,
                'price': price_normalizer,
            }
        }
