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
                ("x_jt", POINTER(POINTER(c_bool))),
                ("y_j", POINTER(c_bool)),
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

    def predict(self, state):
        if(self.scenario_keys is None):
            for time_requests in self.env.scenario:
                for req in time_requests.values():
                    self.scenario_keys = list(req.keys())
                    break
                if self.scenario_keys is not None:
                    break
        
        current_requests_list = sorted([list(req.values()) for req in env.scenario[self.ts].values() if(req['status'] == Status.ARRIVED)], key=lambda e: (e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('current_soc')]))
        current_rejected_list = sorted([list(req.values()) for req in env.scenario[self.ts].values() if(req['status'] == Status.REJECTED)], key=lambda e: (e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('current_soc')]))
        current_waiting_list = sorted([list(req.values()) for req in env.scenario[self.ts].values() if(req['status'] == Status.WAITING)], key=lambda e: (e[self.scenario_keys.index('arrival')], e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('current_soc')]))
        current_plugged_list = sorted([list(req.values()) for req in env.scenario[self.ts].values() if(req['status'] == Status.ACCEPTED)], key=lambda e: (e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('arrival')], e[self.scenario_keys.index('current_soc')]))

        ######
        # For Simulate_Station_FIFO:
        # States: [pr_t, pr_t+1, pr_t+2, pr_t+3, sp_t, sp_t+1, sp_t+2, sp_t+3, soc¹, soc², soc³, soc⁴, soc⁵, soc⁶, soc⁷, soc⁸, soc⁹, soc¹⁰, dur¹, dur², dur³, dur⁴, dur⁵, dur⁶, dur⁷, dur⁸, dur⁹, dur¹⁰, soc_wq¹, soc_wq², soc_wq³, soc_wq⁴, soc_wq⁵, soc_wq⁶, soc_wq⁷, soc_wq⁸, soc_wq⁹, soc_wq¹⁰, dur_wq¹, dur_wq², dur_wq³, dur_wq⁴, dur_wq⁵, dur_wq⁶, dur_wq⁷, dur_wq⁸, dur_wq⁹, dur_wq¹⁰, soc_cr¹, soc_cr², soc_cr³, soc_cr⁴, soc_cr⁵, soc_cr⁶, soc_cr⁷, soc_cr⁸, soc_cr⁹, soc_cr¹⁰, dur_cr¹, dur_cr², dur_cr³, dur_cr⁴, dur_cr⁵, dur_cr⁶, dur_cr⁷, dur_cr⁸, dur_cr⁹, dur_cr¹⁰]
        # Index:  [ 0  ,   1   ,   2   ,   3   ,  4  ,   5   ,   6   ,   7   ,  8  ,  9  ,  10 ,  11 ,  12 ,  13 ,  14 ,  15 ,  16 ,  17  ,  18 ,  19 ,  20 ,  21 ,  22 ,  23 ,  24 ,  25 ,  26 ,  27  ,    28  ,    29  ,    30  ,    31  ,    32  ,    33  ,    34  ,    35  ,    36  ,    37   ,   38   ,   39   ,   40   ,   41   ,   42   ,   43   ,   44   ,   45   ,   46   ,    47   ,   48   ,   49   ,   50   ,   51   ,   52   ,   53   ,   54   ,   55   ,   56   ,    57   ,   58   ,   59   ,   60   ,   61   ,   62   ,   63   ,   64   ,   65   ,   66   ,    67   ]
        ######
        assert (self.simulation_controller_class_name == 'Simulate_Station_FIFO')
        self.pr_t[self.ts:min(self.T-1, self.ts+4)] = [pr * self.env.energy["normalizers"]["price"] for pr in state[0:min(self.T-1, self.ts+4)-self.ts]]
        if(self.pr_t[0] > self.max_pr_known):
            self.max_pr_known = self.pr_t[0]
        self.pr_t[self.ts+4:] = [self.max_pr_known for _ in range(self.T-(self.ts+4))]

        self.pv_t[self.ts:min(self.T-1, self.ts+4)] = [sp * self.env.energy["normalizers"]["renewable"] for sp in state[4:4+min(self.T-1, self.ts+4)-self.ts]]
        self.pv_t[self.ts+4:] = [0 for _ in range(self.T-(self.ts+4))]

        for req_tuple in current_requests_list+current_rejected_list:
            num = req_tuple[self.scenario_keys.index('num')]
            self.soc_0[num] = req_tuple[self.scenario_keys.index('initial_soc')]
            self.r_j[num] = req_tuple[self.scenario_keys.index('arrival')]
            self.d_j[num] = req_tuple[self.scenario_keys.index('departure')]
            
            self.x_jt[num] = [False for _ in range(self.T)]
            self.soc_jf[num] = self.soc_0[num]
            self.y_j[num] = False
            self.u_j[num] = False
        
        for req_tuple in current_rejected_list:
            self.past_rejected_ev[req_tuple[self.scenario_keys.index('num')]] = {'charger': -1, 'arrival': req_tuple[self.scenario_keys.index('arrival')], 'departure': req_tuple[self.scenario_keys.index('departure')]}

        plugged_ev = {}
        for req_tuple in current_plugged_list:
            plugged_ev[req_tuple[self.scenario_keys.index('num')]] = {'charger': req_tuple[self.scenario_keys.index('charger')], 'arrival': req_tuple[self.scenario_keys.index('arrival')], 'departure': req_tuple[self.scenario_keys.index('departure')]}
        
        all_ev = {}
        all_ev.update(self.past_plugged_ev.copy())
        all_ev.update(self.past_rejected_ev.copy())
        all_ev.update(plugged_ev)
        all_ev_num_list = [num for num in self.past_plugged_ev] +\
                          [num for num in self.past_rejected_ev] +\
                          [req_tuple[self.scenario_keys.index('num')] for req_tuple in current_plugged_list] #sorted([list(req.values()) for req in env.scenario[self.ts].values() if(req['status'] == Status.ACCEPTED)], key=lambda e: (e[self.scenario_keys.index('departure')], e[self.scenario_keys.index('arrival')], e[self.scenario_keys.index('current_soc')]))]

        chargers_departure = [0 for _ in range(self.m)]
        for num in all_ev_num_list:
            if(all_ev[num]['charger'] != -1) and (all_ev[num]['departure'] > chargers_departure[all_ev[num]['charger']]):
                chargers_departure[all_ev[num]['charger']] = all_ev[num]['departure']

        waiting_ev = {}
        for req_tuple in current_waiting_list:
            num = req_tuple[self.scenario_keys.index('num')]
            charger = chargers_departure.index(min(chargers_departure))
            chargers_departure[charger] = self.d_j[num]
            waiting_ev[num] = {'charger': charger, 'arrival': self.r_j[num], 'departure': self.d_j[num]}
        
        request_ev = {}
        for req_tuple in current_requests_list:
            num = req_tuple[self.scenario_keys.index('num')]
            charger = chargers_departure.index(min(chargers_departure))
            chargers_departure[charger] = self.d_j[num]
            request_ev[num] = {'charger': charger, 'arrival': self.r_j[num], 'departure': self.d_j[num]}

        all_ev.update(waiting_ev)
        all_ev.update(request_ev)
        all_ev_num_list = all_ev_num_list +\
                          [req_tuple[self.scenario_keys.index('num')] for req_tuple in current_waiting_list] +\
                          [req_tuple[self.scenario_keys.index('num')] for req_tuple in current_requests_list]
        
        nb_past = len(self.past_plugged_ev) + len(self.past_rejected_ev)
        nb_ev_assigned = len(plugged_ev) + len(waiting_ev)
        nb_new_requests = len(request_ev)
        nb = nb_past + nb_ev_assigned + nb_new_requests

        assert nb == len(all_ev_num_list), f'nb={nb} != len(all_ev_num_list)={len(all_ev_num_list)}'

        self.z_jk = {num1: {num2: False for num2 in all_ev_num_list} for num1 in all_ev_num_list}
        for num1 in all_ev_num_list:
            for num2 in all_ev_num_list:
                if(all_ev[num1]['charger'] != -1) and (all_ev[num2]['charger'] != -1) and (num1 != num2) and (all_ev[num1]['charger'] == all_ev[num2]['charger']) and (all_ev[num2]['departure'] <= all_ev[num1]['departure']) and (all_ev[num1]['arrival'] < all_ev[num2]['departure']):
                    self.z_jk[num1][num2] = True
        
        
        BatteryArray = c_float * nb
        b_ptr = ctypes.cast(BatteryArray(*[self.b[0] for _ in range(nb)]), POINTER(c_float))
        
        DepartureArray = c_int * nb
        departure_ptr = ctypes.cast(DepartureArray(*[self.d_j[num] for num in all_ev_num_list]), POINTER(c_int))
        
        ArrivalArray = c_int * nb
        arrival_ptr = ctypes.cast(ArrivalArray(*[self.r_j[num] for num in all_ev_num_list]), POINTER(c_int))
        
        Soc0Array = c_float * nb
        soc0_ptr = ctypes.cast(Soc0Array(*[self.soc_0[num] for num in all_ev_num_list]), POINTER(c_float))
        
        PvArray = c_float * self.T
        pv_ptr = ctypes.cast(PvArray(*self.pv_t), POINTER(c_float))
        
        PrArray = c_float * self.T
        pr_ptr = ctypes.cast(PrArray(*self.pr_t), POINTER(c_float))
        
        XjtArray = c_bool * self.T
        XjArray = POINTER(ctypes.c_bool) * nb
        xjt_ptr = ctypes.cast(XjArray(*[XjtArray(*self.x_jt[num]) for num in all_ev_num_list]), POINTER(POINTER(c_bool)))
        
        ZjkArray = c_bool * nb
        ZjArray = POINTER(ctypes.c_bool) * nb
        zjk_ptr = ctypes.cast(ZjArray(*[ZjkArray(*[self.z_jk[num1][num2] for num2 in all_ev_num_list]) for num1 in all_ev_num_list]), POINTER(POINTER(c_bool)))
        
        SocjfArray = c_float * nb
        socjf_ptr = ctypes.cast(SocjfArray(*[self.soc_jf[num] for num in all_ev_num_list]), POINTER(c_float))
        
        EtArray = c_float * self.T
        et_ptr = ctypes.cast(EtArray(*self.e_t), POINTER(c_float))
        
        YjArray = c_bool * nb
        yj_ptr = ctypes.cast(YjArray(*[self.y_j[num] for num in all_ev_num_list]), POINTER(c_bool))
        
        UjArray = c_bool * nb
        uj_ptr = ctypes.cast(UjArray(*[self.u_j[num] for num in all_ev_num_list]), POINTER(c_bool))
        
        result_ptr = libmodel.solve(1, 10, 100, self.ts, self.T, self.m, nb_past, nb_ev_assigned, nb_new_requests, self.w, self.eta, self.w_G, self.tau, b_ptr, departure_ptr, arrival_ptr, soc0_ptr, pv_ptr, pr_ptr, xjt_ptr, zjk_ptr, socjf_ptr, yj_ptr, et_ptr, uj_ptr)
        res = result_ptr.contents
        new_soc_jf = [res.soc_jf[j] for j in range(nb)]
        new_e_t = [res.e_t[t] for t in range(T)]
        new_x_jt = np.array([[res.x_jt[j][t] for t in range(T)] for j in range(nb)])
        new_y_j = [res.y_j[j] for j in range(nb)]
        new_u_j = [res.u_j[j] for j in range(nb)]

        libmodel.destroy_result(result_ptr, nb)
        assert res.solved, "No solution found by CPLEX solver!"
        
        self.e_t.append(new_e_t[self.ts])
        
        current_requests_num = [req[self.scenario_keys.index('num')] for req in current_requests_list]
        charging_requests = [0 for _ in range(self.m)]
        power_allocations = [0 for _ in range(self.m)]
        idx_not_found = 0
        for j, num in enumerate(all_ev_num_list):
            if (self.ts == self.r_j[num]):
                self.y_j[num] = new_y_j[j]
            elif (self.ts == self.d_j[num]-1) and (num not in self.past_rejected_ev):
                self.u_j[num] = new_u_j[j]
                self.soc_jf[num] = new_soc_jf[j]
                self.past_plugged_ev[num] = {'charger': all_ev[num]['charger'], 'arrival': self.r_j[num], 'departure': self.d_j[num]}
            
            if(self.ts == self.r_j[num]):
                if self.y_j[num]:
                    if num in current_requests_num:
                        idx = current_requests_num.index(num)
                        charging_requests[idx] = 1
                    else:
                        idx_not_found = 1
                else:
                    self.past_rejected_ev[num] = {'charger': -1, 'arrival': self.r_j[num], 'departure': self.d_j[num]}
            
            self.x_jt[num][:] = new_x_jt[j, :]

        accepted_current_requests_list = [req for req in current_requests_list if(self.y_j[req[self.scenario_keys.index('num')]])]
        num_list = [req[self.scenario_keys.index('num')] for req in current_plugged_list+current_waiting_list+accepted_current_requests_list]
        next_num_list = sorted(num_list[:env.number_of_chargers], key=lambda num: (self.d_j[num], self.r_j[num], self.soc_0[num]))
        for j, num in enumerate(all_ev_num_list):
            if(self.r_j[num] <= self.ts < self.d_j[num]):
                if self.y_j[num]:
                    if num in next_num_list:
                        idx = next_num_list.index(num)
                        power_allocations[idx] = int(self.x_jt[num][self.ts])
                    else:
                        assert (int(self.x_jt[num][self.ts]) == 0), str(num) + " - " + str(self.x_jt[num][self.ts]) + " - " + str(self.soc_0[num])
                else:
                    assert (int(self.x_jt[num][self.ts]) == 0), str(num) + " - " + str(self.x_jt[num][self.ts]) + " - " + str(self.soc_0[num])
                    assert (sum(self.x_jt[num]) == 0), str(num) + " - " + str(self.x_jt[num][self.ts]) + " - " + str(self.soc_0[num])
        
        assert sum([int(self.x_jt[num][self.ts]) for num in all_ev_num_list]) == sum([int(new_x_jt[j, self.ts]) for j in range(nb)])
        assert (idx_not_found == 0), f"Index not found! (idx_not_found={idx_not_found})"
        
        self.ts += 1

        return charging_requests + power_allocations

    def reset(self):
        self.x_jt = {}
        self.z_jk = {}
        self.soc_jf = {}
        self.e_t = []
        self.y_j = {}
        self.u_j = {}
        self.d_j = {}
        self.r_j = {}
        self.soc_0 = {}
        self.pv_t = [0. for _ in range(self.T)]
        self.pr_t = [0. for _ in range(self.T)]
        self.max_pr_known = 0
        self.past_plugged_ev = {}
        self.past_rejected_ev = {}
        self.ts = 0
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="ChargingStationEnv-v0")
    parser.add_argument("--episodes", default=100, type=int)
    parser.add_argument("--schema", default="../../schema.json")
    parser.add_argument("--current_folder", default="../../dataset/ev_scenario-50/", type=str)
    parser.add_argument("--results_folder", default="../../results/rolling/ev_scenario-50/", type=str)
    parser.add_argument("--initializer", default="Initializer_FIFO")
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
    libmodel.solve.argtypes = [c_float, c_float, c_float, c_int, c_int, c_int, c_int, c_int, c_int, c_float, c_float, c_float, c_float, POINTER(c_float), POINTER(c_int), POINTER(c_int), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(POINTER(c_bool)), POINTER(POINTER(c_bool)), POINTER(c_float), POINTER(c_bool), POINTER(c_float), POINTER(c_bool)]
    libmodel.destroy_result.argtypes = [POINTER(Result), c_int]

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
        charger_config = env.chargers_types[env.chargers_config['list_chargers'][0]] # We assume that all the chargers are identical
        w = charger_config['charging_rate']
        eta = charger_config['charging_efficiency']
        w_G = env.grid_limit
        tau = (env.step_time / 60)
        b = env.ev_types[env.ev_config['considered_ev'][0]]['capacity'] # We assume that there is only one type of EV
        preemptive_charging = env.preemptive_charging
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



