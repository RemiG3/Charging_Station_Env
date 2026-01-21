from typing import Type
import numpy as np
import gymnasium as gym
from charging_station_env.transition import Simulate_Station_Base
from charging_station_env.transition.Constants import Status, ChargingStatus
import xgboost as xgb


class Simulate_Station_FIFO(Simulate_Station_Base):
    """
    Initializes the class.
    """
    def __init__(self, nb_predict_timestep: int=3,
                 pv_model_path: str=None,
                 price_model_path: str=None,
                 input_window_pv: int=0,
                 input_window_price: int=0):
        super().__init__()
        self.nb_predict_timestep = nb_predict_timestep
        self.epsilon = 1e-6
        self.scenario_keys = None
        self.pv_model_path = pv_model_path
        self.price_model_path = price_model_path
        self.input_window_pv = input_window_pv
        self.input_window_price = input_window_price
        
        if pv_model_path is not None:
            self.pv_model = xgb.Booster()
            self.pv_model.load_model(pv_model_path)
        else:
            self.pv_model = None
        if price_model_path is not None:
            self.price_model = xgb.Booster()
            self.price_model.load_model(price_model_path)
        else:
            self.price_model = None

    """
    This function is used to get the observation size of the environment.

    Parameters:
        env : The environment in which the agent is acting

    Returns:
        Size of the observation
    """
    def get_observation_size(self, env: Type[gym.Env]) -> int:
        # Each timestep has both price and pv_production, so we need 2 * abs(nb_predict_timestep)
        size = 2 + 2*abs(self.nb_predict_timestep) + 6 * env.number_of_chargers
        if not env.preemptive_charging:
            size += 2 * env.number_of_chargers
            if env.use_V2G:
                size += 2 * env.number_of_chargers
        return size
    

    """
    This function is used to generate the observation array for the environment.

    Parameters:
        env : The environment in which the agent is acting

    Returns:
        The observation array
    """
    def __call__(self, env: Type[gym.Env]) -> np.ndarray:
        ts = env.timestep // env.step_time
        array_size = env.TIMESTEP_MAX / env.step_time
        
        # Initialize scenario_keys only once at the beginning of the simulation
        if self.scenario_keys is None:
            for time_requests in env.scenario:
                for req in time_requests.values():
                    self.scenario_keys = list(req.keys())
                    break
                if self.scenario_keys is not None:
                    break
        
        nb_supp_predict = 0
        if(self.nb_predict_timestep > 0):
            end_index = int(self.nb_predict_timestep+1-max(0, (ts+self.nb_predict_timestep+1 - array_size)))
            pv_production = np.array(env.energy["renewable"][ts:ts+self.nb_predict_timestep+1]) / env.energy["normalizers"]["renewable"]
            price = np.array(env.energy["price"][ts:ts+self.nb_predict_timestep+1]) / env.energy["normalizers"]["price"]
            nb_supp_predict = self.nb_predict_timestep+1 - end_index
            if end_index > 1:
                if(self.pv_model is not None):
                    hist_size = self.input_window_pv
                    start_index = max(0, int(ts+1-hist_size))
                    X_test = np.array(env.energy["renewable"][start_index:ts+1]) / env.energy["normalizers"]["renewable"]
                    nb_supp = hist_size - len(X_test)
                    if nb_supp > 0:
                        X_test = np.concatenate((np.array([-1. for _ in range(nb_supp)]), X_test), axis=-1)
                    if(ts == len(env.energy["renewable"])):
                        X_test = np.concatenate((X_test, np.array([-1.])), axis=-1)
                    pv_predictions = np.zeros((self.nb_predict_timestep,))
                    assert len(X_test) == hist_size, f"X_test size mismatch: {len(X_test)} != {hist_size}"
                    for i in range(self.nb_predict_timestep):
                        Ypred_test = np.clip(self.pv_model.predict(xgb.DMatrix(X_test.reshape(1, -1))), 0., 1.)
                        X_test = np.concatenate((X_test[1:], Ypred_test), axis=-1)
                        pv_predictions[i] = Ypred_test[0]
                    pv_production[1:end_index] = pv_predictions[:end_index-1]
                else:
                    # Add noise to predictions with error_range = 30% and z_value_95_confidence = 1.96
                    pv_production[1:end_index] = np.clip( np.random.normal(pv_production[1:end_index], (.3 * pv_production[1:end_index]) / 1.96, end_index-1), 0., 1.)
                if(self.price_model is not None):
                    # Use historical price data for prediction (similar to PV model approach)
                    hist_size = self.input_window_price
                    start_index = max(0, int(ts+1-hist_size))
                    X_price_test = np.array(env.energy["price"][start_index:ts+1]) / env.energy["normalizers"]["price"]
                    nb_supp = hist_size - len(X_price_test)
                    if nb_supp > 0:
                        X_price_test = np.concatenate((np.array([-1. for _ in range(nb_supp)]), X_price_test), axis=-1)
                    if(ts == len(env.energy["price"])):
                        X_price_test = np.concatenate((X_price_test, np.array([-1.])), axis=-1)
                    price_predictions = np.zeros((self.nb_predict_timestep,))
                    assert len(X_price_test) == hist_size, f"X_price_test size mismatch: {len(X_price_test)} != {hist_size}"
                    for i in range(self.nb_predict_timestep):
                        Ypred_price_test = np.clip(self.price_model.predict(xgb.DMatrix(X_price_test.reshape(1, -1))), 0., 1.)
                        X_price_test = np.concatenate((X_price_test[1:], Ypred_price_test), axis=-1)
                        price_predictions[i] = Ypred_price_test[0]
                    price[1:end_index] = price_predictions[:end_index-1]
                else:
                    # Add noise to predictions with error_range = 30% and z_value_95_confidence = 1.96
                    price[1:end_index] = np.clip( np.random.normal(price[1:end_index], (.3 * price[1:end_index]) / 1.96, end_index-1), 0., 1.)
            if nb_supp_predict > 0:
                pv_production = np.concatenate((pv_production, np.array([-1. for _ in range(nb_supp_predict)])), axis=-1)
                price = np.concatenate((price, np.array([-1. for _ in range(nb_supp_predict)])), axis=-1)
        elif(self.nb_predict_timestep < 0):
            start_index = max(0, int(ts+self.nb_predict_timestep))
            pv_production = np.array(env.energy["renewable"][start_index:ts+1]) / env.energy["normalizers"]["renewable"]
            price = np.array(env.energy["price"][start_index:ts+1]) / env.energy["normalizers"]["price"]
            nb_supp_predict = -int(ts+self.nb_predict_timestep) if int(ts+self.nb_predict_timestep) < 0 else 0
            if nb_supp_predict > 0:
                pv_production = np.concatenate((np.array([-1. for _ in range(nb_supp_predict)]), pv_production), axis=-1)
                price = np.concatenate((np.array([-1. for _ in range(nb_supp_predict)]), price), axis=-1)
            if(ts == len(env.energy["price"])):
                pv_production = np.concatenate((pv_production, np.array([-1.])), axis=-1)
                price = np.concatenate((price, np.array([-1.])), axis=-1)
        else:
            pv_production = np.array(env.energy["renewable"][ts:ts+1]) / env.energy["normalizers"]["renewable"]
            price = np.array(env.energy["price"][ts:ts+1]) / env.energy["normalizers"]["price"]
            if(ts == len(env.energy["price"])):
                pv_production = np.concatenate((pv_production, np.array([-1.])), axis=-1)
                price = np.concatenate((price, np.array([-1.])), axis=-1)
        
        
        preemptive_in_charging = np.array([])
        preemptive_has_charged = np.array([])
        preemptive_in_discharging = np.array([])
        preemptive_has_discharged = np.array([])
        
        if self.scenario_keys is None:
            plugged_ev_soc = np.array([0 for _ in range(env.number_of_chargers)])
            plugged_ev_duration = np.array([0 for _ in range(env.number_of_chargers)]) / array_size
            waiting_ev_soc = np.array([0 for _ in range(env.number_of_chargers)])
            waiting_ev_duration = np.array([0 for _ in range(env.number_of_chargers)]) / array_size
            request_ev_soc = np.array([0 for _ in range(env.number_of_chargers)])
            request_ev_duration = np.array([0 for _ in range(env.number_of_chargers)]) / array_size

            if not env.preemptive_charging:
                preemptive_in_charging = np.array([0 for _ in range(env.number_of_chargers)])
                preemptive_has_charged = np.array([0 for _ in range(env.number_of_chargers)])
                preemptive_in_discharging = np.array([0 for _ in range(env.number_of_chargers)])
                preemptive_has_discharged = np.array([0 for _ in range(env.number_of_chargers)])
        else:
            current_plugged_list = sorted([list(req.values()) for req in env.scenario[ts].values() if(req['status'] == Status.ACCEPTED)], key=lambda e: (e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('arrival')], e[self.scenario_keys.index('current_soc')]))
            current_waiting_list = sorted([list(req.values()) for req in env.scenario[ts].values() if(req['status'] == Status.WAITING)], key=lambda e: (e[self.scenario_keys.index('arrival')], e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('current_soc')]))
            plugged_ev_soc = np.array([req_tuple[self.scenario_keys.index('current_soc')] for req_tuple in current_plugged_list] + [0 for _ in range(env.number_of_chargers - len(current_plugged_list))])
            plugged_ev_duration = np.array([req_tuple[self.scenario_keys.index('departure')]-ts for req_tuple in current_plugged_list] + [0 for _ in range(env.number_of_chargers - len(current_plugged_list))]) / array_size
            waiting_ev_soc = np.array([req_tuple[self.scenario_keys.index('current_soc')] for req_tuple in current_waiting_list] + [0 for _ in range(env.number_of_chargers - len(current_waiting_list))])
            waiting_ev_duration = np.array([req_tuple[self.scenario_keys.index('departure')]-ts for req_tuple in current_waiting_list] + [0 for _ in range(env.number_of_chargers - len(current_waiting_list))]) / array_size

            current_requests_list = sorted([list(req.values()) for req in env.scenario[ts].values() if((req['status'] == Status.ARRIVED) or (req['status'] == Status.REJECTED))], key=lambda e: (e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('current_soc')]))
            request_ev_soc = np.array([req_tuple[self.scenario_keys.index('current_soc')] for req_tuple in current_requests_list] + [0 for _ in range(env.number_of_chargers - len(current_requests_list))])
            request_ev_duration = np.array([req_tuple[self.scenario_keys.index('departure')]-ts for req_tuple in current_requests_list] + [0 for _ in range(env.number_of_chargers - len(current_requests_list))]) / array_size
            
            if not env.preemptive_charging:
                preemptive_in_charging = np.array([int(ChargingStatus.PREEMPTIVE_IN_CHARGING in req_tuple[self.scenario_keys.index('charging_status')]) for req_tuple in current_plugged_list] + [0 for _ in range(env.number_of_chargers - len(current_plugged_list))])
                preemptive_has_charged = np.array([int(ChargingStatus.PREEMPTIVE_HAS_CHARGED in req_tuple[self.scenario_keys.index('charging_status')]) for req_tuple in current_plugged_list] + [0 for _ in range(env.number_of_chargers - len(current_plugged_list))])
                if env.use_V2G:
                    preemptive_in_discharging = np.array([int(ChargingStatus.PREEMPTIVE_IN_DISCHARGING in req_tuple[self.scenario_keys.index('charging_status')]) for req_tuple in current_plugged_list] + [0 for _ in range(env.number_of_chargers - len(current_plugged_list))])
                    preemptive_has_discharged = np.array([int(ChargingStatus.PREEMPTIVE_HAS_DISCHARGED in req_tuple[self.scenario_keys.index('charging_status')]) for req_tuple in current_plugged_list] + [0 for _ in range(env.number_of_chargers - len(current_plugged_list))])

        observations = np.concatenate((price, pv_production, plugged_ev_soc, preemptive_in_charging, preemptive_has_charged, preemptive_in_discharging, preemptive_has_discharged,
                                       plugged_ev_duration, waiting_ev_soc, waiting_ev_duration, request_ev_soc, request_ev_duration), axis=None)
        
        assert len(observations) == self.get_observation_size(env), f"Observation size mismatch: {len(observations)} != {self.get_observation_size(env)}"
        assert not ((observations < -1.).any() or (observations > 1.).any()), "Observation values out of range"

        return observations