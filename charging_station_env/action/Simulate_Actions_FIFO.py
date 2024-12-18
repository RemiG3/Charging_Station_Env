from typing import Any, Dict, Optional, Tuple, Union, List, Type
from charging_station_env.transition.Constants import Status, ChargingStatus
from charging_station_env.action import Simulate_Actions_Base
import numpy as np
import gymnasium as gym
from gymnasium import spaces



class Simulate_Actions_FIFO(Simulate_Actions_Base):
    """
    Initializes the class with the given reward weights.
    
    Parameters:
        alpha_reject : The penalty for rejecting a demand
        alpha_grid_limit : The penalty for exceeding the grid limit
        alpha_price : The penalty for the price of charging
        alpha_charging_reward : The penalty for the remaining SOC at departure
    """
    def __init__(self, alpha_accept: float = 10, alpha_grid_limit: float = 0, alpha_price: float = 1, alpha_charging_reward: float = 100):
        super().__init__()
        self.alpha_accept = alpha_accept
        self.alpha_grid_limit = alpha_grid_limit
        self.alpha_price = alpha_price
        self.alpha_charging_reward = alpha_charging_reward
        self.reset_metrics()
    
    """
    Reset the metrics of the environment.
    """
    def reset_metrics(self):
        self.history = {
            "grid_history": [],
            "pv_wasted_history": [],
            "energy_consumed_history": [],
            "reward_history": [],
            "charge_reward_history": [],
            "accept_reward_history": [],
            "grid_penalty_history": [],
            "price_cost_history": [],
            "rejection_history": [],
            "auto_rejection_history": [],
            "pv_energy_history": [],
            "waiting_time_ev": {}
        }
        self.scenario_keys = None
    
    
    """
    This function is used to get the action space of the environment.

    Parameters:
        env : The environment in which the agent is acting

    Returns:
        Gymnasium action space
    """
    def get_action_space(self, env: Type[gym.Env]) -> int:
        if env.use_V2G:
            print('WARNING: V2G will be ignored since the action space is binary!')
        return spaces.MultiBinary(env.number_of_chargers*2)


    """
    This function is used to get the metrics of the environment.

    Parameters:
        env : The environment in which the agent is acting

    Returns:
        A dictionary containing the metrics
    """
    def get_metrics(self) -> Dict:
        return self.history
    

    def _get_next_available_charger(self, m, chargers_list):
        for charger in range(m):
            if charger not in chargers_list:
                return charger
        return None
    
    """
    Get the current invalid actions through masks.
    
    Parameters:
        env : The environment in which the agent is acting
        
    Returns:
        The invalid actions masks
    """
    def action_masks(self, env: Type[gym.Env]) -> np.ndarray:
        ts = env.timestep // env.step_time
        current_requests_list = [req['current_soc'] for req in sorted([req for req in env.scenario[ts].values() if((req['status'] == Status.ARRIVED) or (req['status'] == Status.REJECTED))], key=lambda e: (e['departure'], e['current_soc']))]
        nb_requests = min(env.number_of_chargers, len(current_requests_list))
        current_plugged_list = [req['current_soc'] for req in sorted([req for req in env.scenario[ts].values() if(req['status'] == Status.ACCEPTED)], key=lambda e: (e['departure'], e['arrival'], e['current_soc']))]
        nb_plugged = len(current_plugged_list)
        current_waiting_list = [req['current_soc'] for req in sorted([req for req in env.scenario[ts].values() if(req['status'] == Status.WAITING)], key=lambda e: (e['arrival'], e['departure'], e['current_soc']))]
        nb_waiting = len(current_waiting_list)

        request_mask = [1 for _ in range(nb_requests)] + [0 for _ in range(env.number_of_chargers-nb_requests)]
        power_allocation_mask = [1 if soc < 1. else 0 for soc in current_plugged_list] + [1 for _ in range(min(env.number_of_chargers-nb_plugged, nb_waiting+nb_requests))] + [0 for _ in range(max(0, env.number_of_chargers-nb_plugged-nb_waiting-nb_requests))]
        action_mask = [[1, 1] if mask else [1, 0] for mask in request_mask+power_allocation_mask]
        return np.array( action_mask )
    

    def _process_chargers_and_requests(self, env, actions, ts):
        ts_chargers_used_list = [sorted([req['charger'] for req in time_requests.values() if(req['charger'] != -1)]) for time_requests in env.scenario]
        current_requests_list = sorted([list(req.values()) for req in env.scenario[ts].values() if((req['status'] == Status.ARRIVED) or (req['status'] == Status.REJECTED))], key=lambda e: (e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('current_soc')]))
        if (env.schema['Chargers_config']['charging_mode'] == 'discrete') or (env.schema['Chargers_config']['charging_mode'] == 'constant'):
            actions[env.number_of_chargers:] = list(map(int, map(round, actions[env.number_of_chargers:])))

        for idx in range(env.number_of_chargers):
            if idx < len(current_requests_list):
                num = current_requests_list[idx][self.scenario_keys.index('num')]
                req = env.scenario[ts][num]

                if (int(round(actions[idx])) == 1) and (req['status'] == Status.ARRIVED):
                    charger = None
                    for t in range(ts, req['departure']):
                        env.scenario[t][num] = {
                                k: v for k, v in req.items()
                        }
                        if len(ts_chargers_used_list[t]) >= env.number_of_chargers:
                            env.scenario[t][num].update({
                                'status': Status.WAITING,
                                'charging_status': {ChargingStatus.UNPLUGGED},
                                'charger': -1,
                            })
                        else:
                            if charger is None:
                                charger = self._get_next_available_charger(env.number_of_chargers, ts_chargers_used_list[t])
                            env.scenario[t][num].update({
                                'status': Status.ACCEPTED,
                                'charging_status': {ChargingStatus.UNKNOWN},
                                'charger': charger,
                            })
                            if num not in self.history["waiting_time_ev"]:
                                self.history["waiting_time_ev"][num] = t - ts
                            ts_chargers_used_list[t].append(charger)

                    if (charger is None):
                        env.scenario[ts][num].update({
                            'status': Status.REJECTED,
                            'charging_status': {ChargingStatus.UNPLUGGED},
                            'charger': -1,
                        })
                        for t in range(ts+1, req['departure']):
                            del env.scenario[t][num]
                    else:
                        env.scenario[req['departure']][num] = {
                                k: v for k, v in req.items()
                        }
                        env.scenario[req['departure']][num].update({
                            'status': Status.FINISHED,
                            'charging_status': {ChargingStatus.UNPLUGGED},
                            'charger': -1,
                        })
                else:
                    req['status'] = Status.REJECTED
                    req['charging_status'] = {ChargingStatus.UNPLUGGED}
        
        nb_total_rejected = 0
        for req in env.scenario[ts].values():
            if(req['status'] == Status.REJECTED):
                nb_total_rejected += 1

        self.history["rejection_history"].append(nb_total_rejected)


    def _process_charging_actions(self, env, actions, ts):
        current_plugged_list = sorted([list(req.values()) for req in env.scenario[ts].values() if(req['status'] == Status.ACCEPTED)], key=lambda e: (e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('arrival')], e[self.scenario_keys.index('current_soc')]))

        list_ev_types = list(env.schema['EV_config']['considered_ev'])
        p_charging = np.zeros(env.number_of_chargers)
        consumed_energies = np.zeros(env.number_of_chargers)
        offset = env.number_of_chargers
        for idx, req_tuple in enumerate(current_plugged_list):
            num = req_tuple[self.scenario_keys.index('num')]
            req = env.scenario[ts][num]
            selected_charger = env.schema['Charger_types'][env.schema['Chargers_config']["list_chargers"][req['charger']]]
            car_type = env.schema['EV_types'][list_ev_types[req['type']]]
            
            if ChargingStatus.UNKNOWN in req['charging_status']:
                req['charging_status'].remove(ChargingStatus.UNKNOWN)
            
            if not env.preemptive_charging:
                if (actions[offset+idx] > 0) and (ChargingStatus.PREEMPTIVE_HAS_CHARGED not in req['charging_status']):
                    max_charging_energy = min(actions[offset+idx]*selected_charger['charging_rate']*selected_charger['charging_efficiency'] * (env.step_time / 60), (1-req['current_soc'])*car_type['capacity'])
                    consumed_energy = max_charging_energy/selected_charger['charging_efficiency']
                    req['charging_status'].add(ChargingStatus.IN_CHARGING)
                elif env.use_V2G and (actions[offset+idx] < 0) and (ChargingStatus.PREEMPTIVE_HAS_DISCHARGED not in req['charging_status']):
                    max_charging_energy = -min(-actions[offset+idx]*selected_charger['discharging_rate'] * (env.step_time / 60), req['current_soc']*car_type['capacity'])
                    consumed_energy = max_charging_energy*selected_charger['discharging_efficiency']
                    req['charging_status'].add(ChargingStatus.IN_DISCHARGING)
                else:
                    max_charging_energy = 0
                    consumed_energy = 0
                p_charging[idx] = max_charging_energy
                consumed_energies[idx] = consumed_energy

                if (round(p_charging[idx]) == 0.):
                    p_charging[idx] = 0.
                    consumed_energies[idx] = 0.
                
                if num in env.scenario[ts-1]:
                    prev_req = env.scenario[ts-1][num]
                else:
                    prev_req = {'charging_status': {}}
                if (p_charging[idx] > 0):
                    req['charging_status'].add(ChargingStatus.PREEMPTIVE_IN_CHARGING)
                    if env.use_V2G and (ts > 0) and (ChargingStatus.PREEMPTIVE_IN_DISCHARGING in prev_req['charging_status']):
                        for t in range(ts, req['departure']):
                            next_req = env.scenario[t][num]
                            next_req['charging_status'].add(ChargingStatus.PREEMPTIVE_HAS_DISCHARGED)
                elif env.use_V2G and (p_charging[idx] < 0):
                    req['charging_status'].add(ChargingStatus.PREEMPTIVE_IN_DISCHARGING)
                    if (ts > 0) and (ChargingStatus.PREEMPTIVE_IN_CHARGING in prev_req['charging_status']):
                        for t in range(ts, req['departure']):
                            next_req = env.scenario[t][num]
                            next_req['charging_status'].add(ChargingStatus.PREEMPTIVE_HAS_CHARGED)
                else:
                    if (ts > 0):
                        if env.use_V2G and (ChargingStatus.PREEMPTIVE_IN_DISCHARGING in prev_req['charging_status']):
                            for t in range(ts, req['departure']):
                                next_req = env.scenario[t][num]
                                next_req['charging_status'].add(ChargingStatus.PREEMPTIVE_HAS_DISCHARGED)
                        if (ChargingStatus.PREEMPTIVE_IN_CHARGING in prev_req['charging_status']):
                            for t in range(ts, req['departure']):
                                next_req = env.scenario[t][num]
                                next_req['charging_status'].add(ChargingStatus.PREEMPTIVE_HAS_CHARGED)
            else:
                if (actions[offset+idx] >= 0):
                    max_charging_energy = min(actions[offset+idx]*selected_charger['charging_rate']*selected_charger['charging_efficiency'] * (env.step_time / 60), (1-req['current_soc'])*car_type['capacity'])
                    consumed_energy = max_charging_energy/selected_charger['charging_efficiency']
                    req['charging_status'].add(ChargingStatus.IN_CHARGING)
                elif env.use_V2G:
                    max_charging_energy = -min(-actions[offset+idx]*selected_charger['discharging_rate'] * (env.step_time / 60), req['current_soc']*car_type['capacity'])
                    consumed_energy = max_charging_energy*selected_charger['discharging_efficiency']
                    req['charging_status'].add(ChargingStatus.IN_DISCHARGING)
                else:
                    max_charging_energy = 0
                    consumed_energy = 0
                    req['charging_status'].add(ChargingStatus.IN_CHARGING)
                p_charging[idx] = max_charging_energy
                consumed_energies[idx] = consumed_energy
            
            if (len(req['charging_status']) == 0):
                req['charging_status'].add(ChargingStatus.UNKNOWN)
        
        total_consumed_energy = sum(consumed_energies)
        renewable_energy = env.energy['renewable'][ts] * (env.step_time / 60)
        grid_final = max(total_consumed_energy - renewable_energy, 0.)
        energy_wasted = max(renewable_energy - total_consumed_energy, 0.)
        price = self.alpha_price * grid_final * env.energy["price"][ts]

        # Cost when total power reach the grid limit
        grid_penalty = 0
        if (env.schema['grid_limit'] > 0.) and grid_final > env.schema['grid_limit'] * (env.step_time / 60):
            grid_penalty = self.alpha_grid_limit
            grid_final = env.schema['grid_limit'] * (env.step_time / 60)
            scale_factor = (grid_final + renewable_energy) / np.sum(p_charging)
            p_charging *= scale_factor
            price = self.alpha_price * grid_final * env.energy["price"][ts]
            total_consumed_energy = grid_final + renewable_energy
            energy_wasted = max(renewable_energy - total_consumed_energy, 0)
        
        # Calculation of next state of Battery based on actions
        for idx, req_tuple in enumerate(current_plugged_list):
            num = req_tuple[self.scenario_keys.index('num')]
            req = env.scenario[ts][num]
            car_type = env.schema['EV_types'][list(env.schema['EV_config']['considered_ev'])[req['type']]]
            additional_soc = p_charging[idx] / car_type['capacity']
            new_soc = min(1., req['current_soc'] + additional_soc)
            env.scenario[ts+1][num]['current_soc'] = new_soc
        
        self.history["pv_energy_history"].append(renewable_energy)
        self.history["grid_history"].append(grid_final)
        self.history["energy_consumed_history"].append(total_consumed_energy)
        self.history["pv_wasted_history"].append(energy_wasted)
        self.history["price_cost_history"].append(price)
        self.history["grid_penalty_history"].append(grid_penalty)

        return price, grid_penalty


    def _calculate_charge_reward(self, env, ts):
        charge_reward = 0
        for req in env.scenario[ts+1].values():
            if (req['status'] == Status.FINISHED) and (float(f"{req['current_soc']:.6f}") >= .8):
                charge_reward += self.alpha_charging_reward
        self.history["charge_reward_history"].append(charge_reward)
        return charge_reward


    def _next_rejection_computation(self, env, ts):
        next_requests_list = sorted([list(req.values()) for req in env.scenario[ts+1].values() if(req['status'] == Status.ARRIVED)], key=lambda e: (e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('current_soc')]))
        next_waiting_list = sorted([list(req.values()) for req in env.scenario[ts+1].values() if(req['status'] == Status.WAITING)], key=lambda e: (e[self.scenario_keys.index('arrival')], e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('current_soc')]))
        next_plugged_list = sorted([list(req.values()) for req in env.scenario[ts+1].values() if(req['status'] == Status.ACCEPTED)], key=lambda e: (e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('arrival')], e[self.scenario_keys.index('current_soc')]))
        next_ts_chargers_used_list = [[req['charger'] for req in time_requests.values() if(req['charger'] != -1)] for time_requests in env.scenario]
        nb_max_next_requests = max(min(len(next_requests_list), 2*env.number_of_chargers-len(next_plugged_list)-len(next_waiting_list)), 0)
        nb_requests_to_delete = len(next_requests_list) - nb_max_next_requests
        for req_tuple in next_requests_list:
            num = req_tuple[self.scenario_keys.index('num')]
            req = env.scenario[ts+1][num]
            if(len(next_ts_chargers_used_list[req['departure']-1]) >= env.number_of_chargers):
                req['status'] = Status.REJECTED
                req['charging_status'] = {ChargingStatus.UNPLUGGED}
                nb_requests_to_delete -= 1
        next_requests_list = sorted([list(req.values()) for req in env.scenario[ts+1].values() if(req['status'] == Status.ARRIVED)], key=lambda e: (e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('current_soc')]))
        nb_auto_next_rejected = 0
        for i in range(nb_requests_to_delete):
            num = next_requests_list[i][self.scenario_keys.index('num')]
            env.scenario[ts+1][num]['status'] = Status.REJECTED
            env.scenario[ts+1][num]['charging_status'] = {ChargingStatus.UNPLUGGED}
            nb_auto_next_rejected += 1
        
        self.history["auto_rejection_history"].append(nb_auto_next_rejected)


    """
    This function is used to simulate the actions taken by the agent in the environment.

    Parameters:
        env : The environment in which the agent is acting
        actions : The actions taken by the agent

    Returns:
        A tuple containing the reward and the metrics
    """
    def __call__(self, env: Type[gym.Env], actions: np.ndarray) -> Tuple[float, Dict]:
        ts = env.timestep // env.step_time
        
        if self.scenario_keys is None:
            for time_requests in env.scenario:
                for req in time_requests.values():
                    self.scenario_keys = list(req.keys())
                    break
                if self.scenario_keys is not None:
                    break

        if self.scenario_keys is None:
            price, charge_reward, accept_reward, grid_penalty, reward = 0, 0, 0, 0, 0
        else:
            self._process_chargers_and_requests(env, actions, ts)
            accept_reward = sum(self.alpha_accept for req in env.scenario[ts].values() if(req['status'] in [Status.ACCEPTED, Status.WAITING]) and (req['arrival'] == ts))
            self.history["accept_reward_history"].append(accept_reward)
            
            price, grid_penalty = self._process_charging_actions(env, actions, ts)
            charge_reward = self._calculate_charge_reward(env, ts)
        
        reward = -price - grid_penalty + charge_reward + accept_reward
        self.history["reward_history"].append(reward)
        self._next_rejection_computation(env, ts)

        return reward, {key: self.history[key][-1] for key in self.history.keys() if(key != "waiting_time_ev") and (len(self.history[key]) > 0)}


