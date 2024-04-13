from charging_station_env import Status


class Baseline:
    def __init__(self):
        self.epsilon = 1e-6
        self.scenario_keys = None

    def select_action(self, env, states):
        action = [0 for _ in range(2*env.number_of_chargers)]
        ts = env.timestep // env.step_time
        list_ev_types = list(env.ev_config['considered_ev'])

        if(self.scenario_keys is None):
            for time_requests in env.scenario:
                for req in time_requests.values():
                    self.scenario_keys = list(req.keys())
                    break
                if self.scenario_keys is not None:
                    break


        # Charging request acceptance
        # Accept while it's possible and let the environment manage the rejection automatically
        nb_requests = min(len([req for req in env.scenario[ts].values() if(req['status'] == Status.ARRIVED)]), env.number_of_chargers)
        action[:env.number_of_chargers] = [1 for _ in range(nb_requests)] + [0 for _ in range(env.number_of_chargers-nb_requests)]


        # Power allocation
        current_plugged_list = sorted([list(req.values()) for req in env.scenario[ts].values() if(req['status'] == Status.ACCEPTED)], key=lambda e: (e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('arrival')], e[self.scenario_keys.index('current_soc')]))
        cumul_power_demand = -env.energy['renewable'][ts]
        offset = env.number_of_chargers
        for idx, req_tuple in enumerate(current_plugged_list):
            dep = req_tuple[self.scenario_keys.index('departure')]

            car_type = env.ev_types[list_ev_types[req_tuple[self.scenario_keys.index('type')]]]
            selected_charger = env.chargers_types[env.chargers_config["list_chargers"][req_tuple[self.scenario_keys.index('charger')]]]
            power_demand = min(selected_charger['charging_rate']*selected_charger['charging_efficiency'], (1-req_tuple[self.scenario_keys.index('current_soc')])*car_type['capacity'] / (env.step_time / 60))
            power_demand = power_demand/selected_charger['charging_efficiency']
            if (power_demand > self.epsilon) and (cumul_power_demand+power_demand < env.grid_limit):
                if ((dep - ts) * env.step_time/60 <= 3):
                    action[offset+idx] = 1
                    cumul_power_demand += power_demand
                else:
                    T = env.TIMESTEP_MAX//env.step_time
                    sp_t = [sp * env.energy["normalizers"]["renewable"] for sp in states[4:4+min(T-1, ts+4) - ts]]
                    if(power_demand/2 <= (sp_t[0] + sp_t[-1])/2):
                        action[offset+idx] = 1
                        cumul_power_demand += power_demand
                    # else:
                    #     action[offset+idx] = 0
            # else:
            #     action[offset+idx] = 0
            
        
        #print('states', states)
        #print('action', action)

        return action



# Code from Chargym (https://github.com/georkara/Chargym-Charging-Station)
#
# class Rule_Based_Controller:
#     def __init__(self):
#         pass

#     def select_action(self, env, states):
#         action = [0 for _ in range(env.number_of_chargers*2)]

#         # Charging request
#         for charger in range(env.number_of_chargers):
#             present_car = env.present_cars[charger, (env.timestep // env.step_time)]
#             if present_car == -1:
#                 action[charger] = 1
        
#         # Power allocation
#         offset = env.number_of_chargers
#         for charger in range(env.number_of_chargers):
#             #the departure hour for every spot is placed on the last 10 positions in states vector(10 spots)
#             #have in mind that departure time is normalized in [0,1] so if T_leave is within the next 3 hours then
#             #action[charger]=1, else action[charger]=solar_radiation or action[charger]={mean value of solar radiation and the predicted one hour radiation}
#             if (states[18+charger] == 0):
#                 action[offset+charger] = 0
#             elif (states[18+charger] > 0) and (states[18+charger] < 0.16667):# (< 4h/24h equivalent to <= 3h/24h for 1h of timestep)
#                 action[offset+charger] = 1
#             else:
#                 #solar ratiation is states[0] and the predictions on ratiation are states[2],states[3],states[4]

#                 #this case describes that if T_leave> 3 hours, then scenario 1: action is equal to the radiation
#                 #scenario 2: action is equal to the mean value of current radiation and its next hour prediction

#                 # scenario 1, current value of radiation
#                 #action[charger]=states[0]

#                 # scenario 2, mean value of current radiation and one hour ahead
#                 action[offset+charger] = (states[0] + states[2]) / 2

#         return action
