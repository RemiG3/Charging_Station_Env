from charging_station_env import Status


class PrioritySocCentricCharging:
    def __init__(self):
        self.epsilon = 1e-3
        self.scenario_keys = None

    def select_action(self, env, states):
        action = [0 for _ in range(2*env.number_of_chargers)]
        step_time = env.schema['step_time']
        ts = env.timestep // step_time
        list_ev_types = list(env.schema['EV_config']['considered_ev'])

        if(self.scenario_keys is None):
            for time_requests in env.scenario:
                for req in time_requests.values():
                    self.scenario_keys = list(req.keys())
                    break
                if self.scenario_keys is not None:
                    break


        # Charging request acceptance
        chargers_release_dict = {f'charger {charger}': 0 for charger in range(env.number_of_chargers)}
        chargers_release_dict.update({'next_charger_available': 0})
        for req in env.scenario[ts].values():
            if(req['status'] == Status.ACCEPTED):
                chargers_release_dict[f"charger {req['charger']}"] = req['departure']
        chargers_departure = [chargers_release_dict[f'charger {charger}'] for charger in range(env.number_of_chargers)]
        chargers_release_dict['next_charger_available'] = chargers_departure.index(min(chargers_departure))
        current_waiting_list = sorted([list(req.values()) for req in env.scenario[ts].values() if(req['status'] == Status.WAITING)], key=lambda e: (e[self.scenario_keys.index('arrival')], e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('initial_soc')]))
        for req_tuple in current_waiting_list:
            dep = req_tuple[self.scenario_keys.index('departure')]
            chargers_release_dict[f"charger {chargers_release_dict['next_charger_available']}"] = dep
            chargers_departure = [chargers_release_dict[f'charger {charger}'] for charger in range(env.number_of_chargers)]
            chargers_release_dict['next_charger_available'] = chargers_departure.index(min(chargers_departure))
        
        current_requests_list = sorted([list(req.values()) for req in env.scenario[ts].values() if(req['status'] == Status.ARRIVED)], key=lambda e: (e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('initial_soc')]))
        for idx, req_tuple in enumerate(current_requests_list):
            req_soc = req_tuple[self.scenario_keys.index('current_soc')]
            remaining_duration = req_tuple[self.scenario_keys.index('departure')] - max(chargers_release_dict[f"charger {chargers_release_dict['next_charger_available']}"], req_tuple[self.scenario_keys.index('arrival')])
            selected_charger = env.schema['Charger_types'][env.schema['Chargers_config']["list_chargers"][0]]              # Note: We assume that all chargers have the same charging rate and efficiency
            charger_power = selected_charger['charging_rate']*selected_charger['charging_efficiency']
            if (remaining_duration >= ((0.8 - req_soc) * env.schema['EV_types'][list_ev_types[req_tuple[self.scenario_keys.index('type')]]]['capacity']) / charger_power):
                action[idx] = 1
                chargers_release_dict[f"charger {chargers_release_dict['next_charger_available']}"] = req_tuple[self.scenario_keys.index('departure')]
                chargers_departure = [chargers_release_dict[f'charger {charger}'] for charger in range(env.number_of_chargers)]
                chargers_release_dict['next_charger_available'] = chargers_departure.index(min(chargers_departure))
            else:
                action[idx] = 0


        # Power allocation
        current_plugged_list = sorted([list(req.values()) for req in env.scenario[ts].values() if(req['status'] == Status.ACCEPTED)], key=lambda e: (e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('arrival')], e[self.scenario_keys.index('initial_soc')]))
        cumul_power_demand = -env.energy['renewable'][ts]
        offset = env.number_of_chargers
        for idx, req_tuple in enumerate(current_plugged_list):
            if (req_tuple[self.scenario_keys.index('current_soc')] < .8):
                car_type = env.schema['EV_types'][list_ev_types[req_tuple[self.scenario_keys.index('type')]]]
                selected_charger = env.schema['Charger_types'][env.schema['Chargers_config']["list_chargers"][req_tuple[self.scenario_keys.index('charger')]]]
                power_demand = min(selected_charger['charging_rate']*selected_charger['charging_efficiency'], (1-req_tuple[self.scenario_keys.index('current_soc')])*car_type['capacity'] / (step_time / 60))
                power_demand = power_demand/selected_charger['charging_efficiency']
                if (power_demand > self.epsilon) and (cumul_power_demand+power_demand < env.schema['grid_limit']):
                    action[offset+idx] = 1
                    cumul_power_demand += power_demand
                else:
                    action[offset+idx] = 0
            else:
                action[offset+idx] = 0
        
        if (cumul_power_demand < 0):
            for idx, req_tuple in enumerate(current_plugged_list):
                car_type = env.schema['EV_types'][list_ev_types[req_tuple[self.scenario_keys.index('type')]]]
                selected_charger = env.schema['Charger_types'][env.schema['Chargers_config']["list_chargers"][req_tuple[self.scenario_keys.index('charger')]]]
                power_demand = min(selected_charger['charging_rate']*selected_charger['charging_efficiency'], (1-req_tuple[self.scenario_keys.index('current_soc')])*car_type['capacity'] / (step_time / 60))
                power_demand = power_demand/selected_charger['charging_efficiency']
                if (power_demand > self.epsilon) and (cumul_power_demand+power_demand < 0) and (action[offset+idx] == 0):
                    action[offset+idx] = 1
                    cumul_power_demand += power_demand

        return action


