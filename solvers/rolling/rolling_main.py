import sys
sys.path.append("../../")
from utils import parse_dic_args
import gymnasium as gym
import argparse
import os

from charging_station_env import Status
import ctypes
from ctypes import c_int, c_float, c_bool, POINTER
import numpy as np


class Result(ctypes.Structure):
    _fields_ = [("soc_jf", POINTER(c_float)),
                ("e_t", POINTER(c_float)),
                ("x_ijt", POINTER(POINTER(POINTER(c_bool)))),
                ("y_ij", POINTER(POINTER(c_bool))),
                ("u_j", POINTER(c_bool)),
                ("solved", c_bool)]


class Rolling:
    def __init__(self, env, T, m, n, w, eta, w_G, tau, b, preemptive_charging, simulation_controller_class_name: str = 'Simulate_Station_FIFO'):
        self.env = env
        self.T = T
        self.m = m              # Number of chargers
        self.N = n              # Total number of EVs
        self.n = 0              # Current number of EVs
        self.w = w
        self.eta = eta
        self.w_G = w_G
        self.tau = tau
        self.b = b
        self.preemptive_charging = preemptive_charging
        self.simulation_controller_class_name = simulation_controller_class_name
        self.scenario_keys = None
        self.reset()
    
    def __call_solving(self):
        nb = len(self.ev_list)

        BatteryArray = c_float * nb
        b_ptr = ctypes.cast(BatteryArray(*[self.b[0] for _ in range(nb)]), POINTER(c_float))
        
        DepartureArray = c_int * nb
        departure_ptr = ctypes.cast(DepartureArray(*[self.d_j[req['num']] for req in self.ev_list]), POINTER(c_int))
        
        ArrivalArray = c_int * nb
        arrival_ptr = ctypes.cast(ArrivalArray(*[self.r_j[req['num']] for req in self.ev_list]), POINTER(c_int))
        
        Soc0Array = c_float * nb
        soc0_ptr = ctypes.cast(Soc0Array(*[self.soc_0[req['num']] for req in self.ev_list]), POINTER(c_float))
        
        PvArray = c_float * self.T
        pv_ptr = ctypes.cast(PvArray(*self.pv_t), POINTER(c_float))
        
        PrArray = c_float * self.T
        pr_ptr = ctypes.cast(PrArray(*self.pr_t), POINTER(c_float))
        
        XijtArray = c_bool * self.T
        XijArray = POINTER(c_bool) * nb
        XiArray = POINTER(POINTER(c_bool)) * self.m
        xijt_ptr = ctypes.cast(
            XiArray(
                *[XijArray(
                    *[XijtArray(*self.x_ijt[i][req['num']]) for req in self.ev_list]
                ) for i in range(self.m)]
            ), 
            POINTER(POINTER(POINTER(c_bool)))
        )
        
        SocjfArray = c_float * nb
        socjf_ptr = ctypes.cast(SocjfArray(*[self.soc_jf[req['num']] for req in self.ev_list]), POINTER(c_float))
        
        EtArray = c_float * self.T
        et_ptr = ctypes.cast(EtArray(*self.e_t), POINTER(c_float))
        
        YijArray = c_bool * nb
        YiArray = POINTER(c_bool) * self.m
        yij_ptr = ctypes.cast(
            YiArray(
                *[YijArray(*[self.y_ij[i][req['num']] for req in self.ev_list]) for i in range(self.m)]
            ),
            POINTER(POINTER(c_bool))
        )
        
        UjArray = c_bool * nb
        uj_ptr = ctypes.cast(UjArray(*[self.u_j[req['num']] for req in self.ev_list]), POINTER(c_bool))

        assignedEVArray = c_bool * nb
        asgnev_ptr = ctypes.cast(assignedEVArray(*[req['assigned'] for req in self.ev_list]), POINTER(c_bool))

        pastEVArray = c_bool * nb
        pastev_ptr = ctypes.cast(pastEVArray(*[req['past'] for req in self.ev_list]), POINTER(c_bool))
        
        result_ptr = libmodel.solve(1, 10, 100, self.ts, self.T, self.m, 0, 0, nb, self.w, self.eta, self.w_G, self.tau, b_ptr, departure_ptr, arrival_ptr, soc0_ptr, pv_ptr, pr_ptr, xijt_ptr, socjf_ptr, yij_ptr, et_ptr, uj_ptr, asgnev_ptr, pastev_ptr)
        res = result_ptr.contents
        new_soc_jf = [res.soc_jf[j] for j in range(nb)]
        new_e_t = [res.e_t[t] for t in range(T)]
        new_x_ijt = [[[res.x_ijt[i][j][t] for t in range(T)] for j in range(nb)] for i in range(self.m)]
        new_y_ij = [[res.y_ij[i][j] for j in range(nb)] for i in range(self.m)]
        new_u_j = [res.u_j[j] for j in range(nb)]
        
        assert res.solved, "No solution found by the solver!"
        libmodel.destroy_result(result_ptr, self.m, nb, self.T)

        return new_y_ij, new_x_ijt, new_soc_jf, new_u_j, new_e_t


    def predict(self, state):
        if(self.scenario_keys is None):
            for time_requests in self.env.scenario:
                for req in time_requests.values():
                    self.scenario_keys = list(req.keys())
                    break
                if self.scenario_keys is not None:
                    break
        
        current_rejected_list = sorted([list(req.values()) for req in env.scenario[self.ts].values() if(req['status'] == Status.REJECTED)], key=lambda e: (e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('initial_soc')]))
        current_requests_list = sorted([list(req.values()) for req in env.scenario[self.ts].values() if(req['status'] == Status.ARRIVED) or (req['status'] == Status.REJECTED)], key=lambda e: (e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('initial_soc')]))
        current_waiting_list = sorted([list(req.values()) for req in env.scenario[self.ts].values() if(req['status'] == Status.WAITING)], key=lambda e: (e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('arrival')], e[self.scenario_keys.index('initial_soc')]))
        current_plugged_list = sorted([list(req.values()) for req in env.scenario[self.ts].values() if(req['status'] == Status.ACCEPTED)], key=lambda e: (e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('arrival')], e[self.scenario_keys.index('initial_soc')]))
        nb_predict = self.env.simulation_controller.nb_predict_timestep

        self.pr_t[self.ts] = state[0] * self.env.energy["normalizers"]["price"]
        self.pv_t[self.ts] = state[1] * self.env.energy["normalizers"]["renewable"]
        if(self.pr_t[self.ts] > self.max_pr_known):
            self.max_pr_known = self.pr_t[self.ts]
        #self.pv_t[self.ts:min(self.T-1, self.ts+nb_predict+1)] = [sp * self.env.energy["normalizers"]["renewable"] for sp in state[1:1+min(self.T-1, self.ts+nb_predict+1)-self.ts]]
        
        for req_tuple in current_requests_list:#+current_rejected_list:
            num = req_tuple[self.scenario_keys.index('num')]
            self.soc_0[num] = req_tuple[self.scenario_keys.index('initial_soc')]
            self.r_j[num] = req_tuple[self.scenario_keys.index('arrival')]
            self.d_j[num] = req_tuple[self.scenario_keys.index('departure')]
            
            self.soc_jf[num] = self.soc_0[num]
            self.u_j[num] = False
            self.ev_list.append({'num': num, 'charger': -1, 'arrival': self.r_j[num], 'departure': self.d_j[num], 'initial_soc': self.soc_0[num], 'past': False, 'assigned': False})
            if(req_tuple[self.scenario_keys.index('status')] == Status.REJECTED):
                self.ev_list[-1].update({'past': True, 'assigned': True})
            
            for i in range(self.m):
                self.x_ijt[i][num] = [False for _ in range(self.T)]
                self.y_ij[i][num] = False
        
        for req_tuple in current_rejected_list:
            self.past_rejected_ev[req_tuple[self.scenario_keys.index('num')]] = {'charger': -1, 'arrival': req_tuple[self.scenario_keys.index('arrival')], 'departure': req_tuple[self.scenario_keys.index('departure')]}

        plugged_ev = {}
        for req_tuple in current_plugged_list:
            plugged_ev[req_tuple[self.scenario_keys.index('num')]] = {'charger': req_tuple[self.scenario_keys.index('charger')], 'arrival': req_tuple[self.scenario_keys.index('arrival')], 'departure': req_tuple[self.scenario_keys.index('departure')]}
        
        chargers_departure = [0 for _ in range(self.m)]
        for req_tuple in current_plugged_list:
            num = req_tuple[self.scenario_keys.index('num')]
            charger = req_tuple[self.scenario_keys.index('charger')]
            chargers_departure[charger] = self.d_j[num]

        waiting_ev = {}
        for req_tuple in current_waiting_list:
            num = req_tuple[self.scenario_keys.index('num')]
            charger = env.scenario[self.d_j[num]-1][num]['charger']
            chargers_departure[charger] = self.d_j[num]
            waiting_ev[num] = {'charger': charger, 'arrival': self.r_j[num], 'departure': self.d_j[num]}

        request_ev = {}
        for req_tuple in current_requests_list:
            if((req_tuple[self.scenario_keys.index('status')] == Status.REJECTED)):
                continue
            num = req_tuple[self.scenario_keys.index('num')]
            request_ev[num] = {'charger': -1, 'arrival': self.r_j[num], 'departure': self.d_j[num]}

        nb = len(self.ev_list)
        new_y_ij, new_x_ijt, new_soc_jf, new_u_j, new_e_t = self.__call_solving()
        
        self.e_t.append(new_e_t[self.ts])
        current_requests_num = [req[self.scenario_keys.index('num')] for req in current_requests_list]
        charging_requests = [0 for _ in range(self.m)]
        power_allocations = [0 for _ in range(self.m)]
        idx_not_found = 0
        for j in range(nb):
            num = self.ev_list[j]['num']
            charger = self.ev_list[j]['charger']
            if (self.ts == self.r_j[num]):
                y_j = (sum([new_y_ij[i][j] for i in range(self.m)]) == 1)
                if(y_j):
                    charger = chargers_departure.index(min(chargers_departure))
                    chargers_departure[charger] = self.d_j[num]
                    self.y_ij[charger][num] = y_j
                    request_ev[num] = {'charger': charger, 'arrival': self.r_j[num], 'departure': self.d_j[num]}
                    self.ev_list[j].update({'assigned': True, 'past': False, 'charger': request_ev[num]['charger']})
                else:
                    request_ev[num] = {'charger': -1, 'arrival': self.r_j[num], 'departure': self.d_j[num]}
                    self.ev_list[j].update({'assigned': True, 'past': True, 'charger': -1})
            
            if(self.ts == self.r_j[num]):
                y_j = sum([self.y_ij[i][num] for i in range(self.m)])
                if (y_j == 1):
                    if num in current_requests_num:
                        idx = current_requests_num.index(num)
                        charging_requests[idx] = 1
                    else:
                        idx_not_found = 1
                else:
                    self.ev_list[j].update({'past': True, 'charger': -1})
            
            if(charger != -1):
                self.x_ijt[charger][num][self.ts] = (sum([new_x_ijt[i][j][self.ts] for i in range(self.m) if(new_y_ij[i][j])]) == 1)
                assert (sum([new_x_ijt[i][j][self.ts] for i in range(self.m)]) == 0) or (sum([new_x_ijt[i][j][self.ts] for i in range(self.m)]) == 1), f"sum([new_x_ijt[i][j][self.ts] for i in range(self.m)])={sum([new_x_ijt[i][j][self.ts] for i in range(self.m)])}"
            else:
                assert (sum([new_x_ijt[i][j][self.ts] for i in range(self.m)]) == 0), str(num) + " - " + str([new_x_ijt[i][j][self.ts] for i in range(self.m)]) + " - " + str(self.soc_0[num])

        accepted_current_requests_list = [req for req in current_requests_list for i in range(self.m) if(self.y_ij[i][req[self.scenario_keys.index('num')]])]
        num_list = [req[self.scenario_keys.index('num')] for req in current_plugged_list+current_waiting_list+accepted_current_requests_list]
        next_num_list = sorted(num_list[:env.number_of_chargers], key=lambda num: (self.d_j[num], self.r_j[num], self.soc_0[num]))
        
        redo_optimization = 0
        re_optimization_done = False
        while(redo_optimization < 2):
            for j in range(nb):
                num = self.ev_list[j]['num']
                if(self.r_j[num] <= self.ts < self.d_j[num]):
                    y_j = sum([self.y_ij[i][num] for i in range(self.m)])
                    assert (y_j == 0) or (y_j == 1), f"y_j={y_j}"
                    if (y_j == 1):
                        if num in next_num_list:
                            idx = next_num_list.index(num)
                            power_allocations[idx] = sum([int(self.x_ijt[i][num][self.ts]) for i in range(self.m)])
                            assert (power_allocations[idx] == 0) or (power_allocations[idx] == 1), f"power_allocations[idx]={power_allocations[idx]}"
                        else:
                            if(redo_optimization == 1):
                                redo_optimization += 1
                                break
                            if(sum([int(self.x_ijt[i][num][self.ts]) for i in range(self.m)]) > 0):
                                for i in range(self.m):
                                    for req in self.ev_list:
                                        self.x_ijt[i][req['num']][self.ts] = 0
                                new_y_ij, new_x_ijt, new_soc_jf, new_u_j, new_e_t = self.__call_solving()
                                for i in range(self.m):
                                    for j in range(nb):
                                        self.x_ijt[i][self.ev_list[j]['num']][self.ts] = new_x_ijt[i][j][self.ts]
                                self.e_t[-1] = new_e_t[self.ts]
                                redo_optimization += 1
                                re_optimization_done = True
                                break
                    else:
                        assert (sum([int(self.x_ijt[i][num][self.ts]) for i in range(self.m)]) == 0), str(num) + " - " + str([self.x_ijt[i][num][self.ts] for i in range(self.m)]) + " - " + str(self.soc_0[num])
                        assert (sum([sum(self.x_ijt[i][num]) for i in range(self.m)]) == 0), str(num) + " - " + str([self.x_ijt[i][num][self.ts] for i in range(self.m)]) + " - " + str(self.soc_0[num])
            redo_optimization = 2
        
        for j in range(nb):
            num = self.ev_list[j]['num']
            charger = self.ev_list[j]['charger']
            if (self.ts == self.d_j[num]-1) and (num not in self.past_rejected_ev):
                self.u_j[num] = new_u_j[j]
                self.soc_jf[num] = new_soc_jf[j]
                self.ev_list[j].update({'past': True})
        
        assert sum([int(self.x_ijt[i][req['num']][self.ts]) for i in range(self.m) for req in self.ev_list]) == sum([int(new_x_ijt[i][j][self.ts]) for i in range(self.m) for j in range(nb)]), f"sum([sum([int(self.x_ijt[i][req['num']][self.ts]) for i in range(self.m)]) for req in self.ev_list])={sum([sum([int(self.x_ijt[i][req['num']][self.ts]) for i in range(self.m)]) for req in self.ev_list])} != sum([int(new_x_ijt[i][j][self.ts]) for i in range(self.m) for j in range(nb)]={sum([int(new_x_ijt[i][j][self.ts]) for i in range(self.m) for j in range(nb)])}"
        assert (idx_not_found == 0), f"Index not found! (idx_not_found={idx_not_found})"
        
        self.ts += 1

        return charging_requests + power_allocations

    def reset(self):
        self.x_ijt = [{} for _ in range(self.m)]
        self.soc_jf = {}
        self.e_t = []
        self.y_ij = [{} for _ in range(self.m)]
        self.u_j = {}
        self.d_j = {}
        self.r_j = {}
        self.soc_0 = {}
        self.pv_t = [0. for _ in range(self.T)]
        self.pr_t = [0. for _ in range(self.T)]
        self.max_pr_known = 0
        self.past_rejected_ev = {}
        self.ev_list = []
        self.ts = 0
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="ChargingStationEnv-v0")
    parser.add_argument("--episodes", default=100, type=int)
    parser.add_argument("--schema", default="../../schema.json")
    parser.add_argument("--current_folder", default="../../dataset/ev_scenario-50/", type=str)
    parser.add_argument("--results_folder", default="../../results/rolling/ev_scenario-50/", type=str)
    parser.add_argument("--initializer", default="Initializer")
    parser.add_argument("--simulation", default="Simulate_Station_FIFO")
    parser.add_argument("--action", default="Simulate_Actions_FIFO")
    parser.add_argument("--energy", default="Energy_Initializer")
    parser.add_argument("--visualizer", default=None)
    parser.add_argument("--initializer_args", default={}, type=parse_dic_args)
    parser.add_argument("--simulation_args", default={}, type=parse_dic_args)
    parser.add_argument("--action_args", default={}, type=parse_dic_args)
    parser.add_argument("--energy_args", default={}, type=parse_dic_args)
    parser.add_argument("--libmodel_name", default="lib_online_rolling.so", type=str)
    parser.add_argument("--reset_flag", default=1, type=int) # reset_flag=0 for new generation and reset_flag=1 for same day
    args = parser.parse_args()

    libmodel = ctypes.CDLL(args.libmodel_name)
    libmodel.solve.restype = POINTER(Result)
    libmodel.solve.argtypes = [
        c_float,  # alpha
        c_float,  # beta
        c_float,  # gamma
        c_int,    # current_ts
        c_int,    # T
        c_int,    # m
        c_int,    # nb_past
        c_int,    # nb_assigned
        c_int,    # nb_requests
        c_float,  # w
        c_float,  # eta
        c_float,  # w_G
        c_float,  # tau
        POINTER(c_float),  # b_ptr
        POINTER(c_int),    # d_j_ptr
        POINTER(c_int),    # r_j_ptr
        POINTER(c_float),  # soc_0_ptr
        POINTER(c_float),  # pv_ptr
        POINTER(c_float),  # pr_ptr
        POINTER(POINTER(POINTER(c_bool))),  # past_x_ijt_ptr_ptr
        POINTER(c_float),  # past_soc_jf_ptr
        POINTER(POINTER(c_bool)),  # past_y_ij_ptr
        POINTER(c_float),  # past_e_t_ptr
        POINTER(c_bool),   # past_u_j_ptr
        POINTER(c_bool),   # assigned_ev_ptr
        POINTER(c_bool)    # past_ev_ptr
    ]
    libmodel.destroy_result.argtypes = [POINTER(Result), c_int, c_int, c_int]

    algo_name = 'ROLLING'
    current_folder = args.current_folder
    results_folder = args.results_folder
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    env = gym.make(args.env,
                   schema=args.schema,
                   current_folder=current_folder,
                   results_folder=results_folder,
                   initializer=getattr(getattr(__import__('charging_station_env'), 'initializer'), args.initializer)(**args.initializer_args, energy_initializer=getattr(getattr(__import__('charging_station_env'), 'initializer'), args.energy)(**args.energy_args)),
                   simulation_controller=getattr(getattr(__import__('charging_station_env'), 'transition'), args.simulation)(**args.simulation_args),
                   action_controller=getattr(getattr(__import__('charging_station_env'), 'action'), args.action)(**args.action_args),
                   visualizer=getattr(getattr(__import__('charging_station_env'), 'visualizer'), args.visualizer)() if(args.visualizer is not None) else None,
                   )

    episodes = args.episodes
    for ep in range(episodes):
        done = False
        state, info = env.reset(reset_flag=args.reset_flag, id_save=ep+1, algo_name=algo_name)
        T = env.TIMESTEP_MAX // env.step_time
        m = env.number_of_chargers
        n = sum([1 for time_req in env.scenario for _ in time_req])
        charger_config = env.schema['Charger_types'][env.schema['Chargers_config']['list_chargers'][0]] # We assume that all the chargers are identical
        w = charger_config['charging_rate']
        eta = charger_config['charging_efficiency']
        w_G = env.schema['grid_limit']
        tau = (env.step_time / 60)
        b = env.schema['EV_types'][env.schema['EV_config']['considered_ev'][0]]['capacity'] # We assume that there is only one type of EV
        preemptive_charging = env.preemptive_charging
        print(f'T: {T}, m: {m}, n: {n}, w: {w}, eta: {eta}, w_G: {w_G}, tau: {tau}, b: {b}, preemptive_charging: {preemptive_charging}')
        simulation_controller_class_name = env.simulation_controller.__class__.__name__

        algo = Rolling(env, T=T, m=m, n=n, w=w, eta=eta, w_G=w_G, tau=tau, b=[b for _ in range(n)], preemptive_charging=preemptive_charging, simulation_controller_class_name=simulation_controller_class_name)
        rewards_list = []
        
        t = 0
        while not done:
            if (args.visualizer is not None):
                env.render()
            actions = algo.predict(state)
            next_state, rewards, done, _, info = env.step(actions)
            state = next_state
            rewards_list.append(rewards)
            t += 1
        print(f'episode: {ep+1}/{args.episodes}', end='\r', flush=True)
    print(flush=True)
    env.close()



