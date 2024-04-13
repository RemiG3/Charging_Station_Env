from charging_station_env import Status, ChargingStatus
import os
import pickle
import numpy as np



def get_variables(filename):
    result = {}
    with open(filename, 'r') as f:
        current_var_type = ''
        current_decision_var = ''
        current_charger_num = ''
        possible_var_types = ['Parameters', 'Decision variables', 'Additional infos']
        decision_variables = ['X_ijt', 'y_ij', 'SOC_f', 'e_t']
        line = f.readline()
        while line != '':
            is_var_type_def = False
            for var_type in possible_var_types:
                if line.startswith(var_type):
                    result[var_type] = {}
                    current_var_type = var_type
                    is_var_type_def = True
                    current_decision_var = ''
                    break
            
            if(not is_var_type_def):
                if('=' in line):
                    var_name, var_val = line.split('=')
                    if(',' in var_val):
                        try:
                            var_val = list(map(eval, var_val.split(',')))
                        except:
                            var_val = [0. for _ in range(len(var_val.split(',')))]
                    else:
                        var_val = eval(var_val)
                    result[current_var_type][var_name] = var_val
                    current_decision_var = ''
                elif(':' in line):
                    var_name, var_val = line.split(':')
                    if (var_val == '\n') and (current_decision_var == 'X_ijt') and (var_name not in decision_variables):
                        current_charger_num = eval(var_name)
                        result[current_var_type][current_decision_var][current_charger_num] = {}
                    elif (var_val == '\n') and (var_name in decision_variables):
                        current_decision_var = var_name
                        result[current_var_type][current_decision_var] = {}
                        current_charger_num = ''
                    else:
                        if(',' in var_val) or (current_decision_var in decision_variables):
                            if (var_val == '\n'):
                                var_val = []
                            else:
                                var_val = list(map(eval, var_val.split(',')))
                        else:
                            var_val = eval(var_val)
                        if(current_charger_num == ''):
                            result[current_var_type][current_decision_var][eval(var_name) if var_name.isdigit() else var_name] = var_val
                        else:
                            result[current_var_type][current_decision_var][current_charger_num][eval(var_name)] = var_val
            line = f.readline()
    return result

def convert_result_to_online_format(result, env, T=97):
    accepted_requests = set()
    charger_assoication = {}
    for charger_num, acc_ev in enumerate(result['Decision variables']['y_ij'].values()):
        accepted_requests.update(acc_ev)
        for ev in acc_ev:
            charger_assoication[ev] = charger_num

    online_result = {}
    n_set_treated = set()
    for t in range(T):
        for req in env.scenario[t].values():
            if(req['status'] == Status.ARRIVED):
                found = False
                for n, (soc_0, arr, dep) in enumerate(zip(result['Parameters']['SOC_0'], result['Parameters']['arrivals'], result['Parameters']['departures'])):
                    offline_soc_0_5f_str = f'{soc_0:.5f}'
                    online_soc_0_5f_str = f'{req["initial_soc"]:.5f}'
                    online_soc_0_5rnd_str = f'{round(req["initial_soc"], 6):.5f}'
                    online_soc_0_5rnd_sup_str = float(f'{round(req["initial_soc"], 6):.5f}') - 1e-5
                    online_soc_0_5rnd_sup_str = f'{online_soc_0_5rnd_sup_str:.5f}'
                    online_soc_0_5rnd_inf_str = float(f'{round(req["initial_soc"], 6):.5f}') + 1e-5
                    online_soc_0_5rnd_inf_str = f'{online_soc_0_5rnd_inf_str:.5f}'
                    if (n not in n_set_treated) and (req['arrival'] == arr) and (req['departure'] == dep) and ((online_soc_0_5f_str == offline_soc_0_5f_str) or (online_soc_0_5rnd_str == offline_soc_0_5f_str) or (online_soc_0_5rnd_sup_str == offline_soc_0_5f_str) or (online_soc_0_5rnd_inf_str == offline_soc_0_5f_str)):
                        found = True
                        if n in accepted_requests:
                            online_result[req['num']] = {'PA': result['Decision variables']['X_ijt'][charger_assoication[n]][n], 'charger': charger_assoication[n], 'arrival': arr, 'departure': dep, 'initial_soc': soc_0}#, 'current_soc': soc_0, 'type': 0}
                            for ts, is_charging in enumerate(online_result[req['num']]['PA']):
                                if is_charging == 1:
                                    online_result[req['num']]['start_charging'] = ts
                                    break
                        else:
                            online_result[req['num']] = {'PA': None, 'start_charging': -1, 'charger': -1, 'arrival': arr, 'departure': dep, 'initial_soc': soc_0}#, 'current_soc': soc_0, 'type': 0}
                        n_set_treated.add( n )
                        break
                assert found, f'Error: EV {req["num"]}, arr={req["arrival"]}, dep={req["departure"]}, and soc0={req["initial_soc"]} not found in offline result dict'
            
    return online_result


def get_ev_num_from_soc0(ev_dmd, soc0):
    for ev_num, ev in ev_dmd.items():
        if ev['initial_soc'] == soc0:
            return ev_num
    return None


def is_charger_busy(online_res, ev_dmd, ts, num_ref, charger):
    for ev_num, ev in ev_dmd.items():
        if (ev_num != num_ref) and ((ev['arrival'] != ts) or ((ev['arrival'] == ts) and (online_res[ev_num]['charger'] == ev['charger']))) and (ev['charger'] == charger) and (ev['status'] == Status.ACCEPTED):
            return True
    return False


def process_offline_result(filename, env, scenario_keys):
    offline_sol = get_variables(filename)
    online_res = convert_result_to_online_format(offline_sol , env)
    
    rewards_list = []
    t = 0
    done = False
    while not done:
        charging_request_actions = []
        auto_rejected_num = set()
        for num, req in env.scenario[t].items():
            if((req['status'] == Status.REJECTED) and (req['arrival'] == t)):
                env.scenario[t][num]['status'] = Status.ARRIVED
                auto_rejected_num.add( num )
        current_requests_list = sorted([list(req.values()) for req in env.scenario[t].values() if(req['status'] == Status.ARRIVED)], key=lambda e: (e[scenario_keys.index('departure')], e[scenario_keys.index('current_soc')]))
        for cur_req in current_requests_list:
            ev_num = cur_req[scenario_keys.index("num")]
            charging_request_actions.append( int(online_res[ev_num]['PA'] is not None) )
        for _ in range(len(current_requests_list), env.number_of_chargers):
            charging_request_actions.append( 0 )
        
        for cur_req in current_requests_list:
            ev_num = cur_req[scenario_keys.index("num")]
            dep = cur_req[scenario_keys.index('departure')]
            arr = cur_req[scenario_keys.index('arrival')]
            charger = online_res[ev_num]['charger']
            if charger >= 0:
                if ev_num in auto_rejected_num:
                    env.action_controller.history['auto_rejection_history'][-1] -= 1
                for ts in range(t, dep):
                    if(online_res[ev_num]['start_charging'] > ts):
                        env.scenario[ts][ev_num] = {'num': ev_num, 'arrival': arr, 'departure': dep, 'initial_soc': cur_req[scenario_keys.index('initial_soc')], 'current_soc': cur_req[scenario_keys.index('current_soc')], 'status': Status.WAITING, 'charging_status': {ChargingStatus.UNPLUGGED}, 'charger': -1, 'type': cur_req[scenario_keys.index('type')]}
                    else:
                        env.scenario[ts][ev_num] = {'num': ev_num, 'arrival': arr, 'departure': dep, 'initial_soc': cur_req[scenario_keys.index('initial_soc')], 'current_soc': cur_req[scenario_keys.index('current_soc')], 'status': Status.ACCEPTED, 'charging_status': {ChargingStatus.UNKNOWN}, 'charger': charger, 'type': cur_req[scenario_keys.index('type')]}
                        if(num not in env.action_controller.history['waiting_time_ev']):
                            env.action_controller.history['waiting_time_ev'][num] = ts - t
                env.scenario[dep][ev_num] = {'num': ev_num, 'arrival': arr, 'departure': dep, 'initial_soc': cur_req[scenario_keys.index('initial_soc')], 'current_soc': cur_req[scenario_keys.index('current_soc')], 'status': Status.FINISHED, 'charging_status': {ChargingStatus.UNPLUGGED}, 'charger': -1, 'type': cur_req[scenario_keys.index('type')]}
            else:
                env.scenario[t][ev_num]['status'] = Status.REJECTED
                env.scenario[t][ev_num]['charging_status'] = {ChargingStatus.UNPLUGGED}

        power_allocation_actions = []
        current_plugged_list = sorted([list(req.values()) for req in env.scenario[t].values() if(req['status'] == Status.ACCEPTED)], key=lambda e: (e[scenario_keys.index('departure')], e[scenario_keys.index('arrival')], e[scenario_keys.index('current_soc')]))
        current_waiting_list = sorted([list(req.values()) for req in env.scenario[t].values() if(req['status'] == Status.WAITING)], key=lambda e: (e[scenario_keys.index('arrival')], e[scenario_keys.index('departure')], e[scenario_keys.index('current_soc')]))
        current_charging_list = current_plugged_list+current_waiting_list
        for cur_req in current_charging_list:
            ev_num = cur_req[scenario_keys.index("num")]
            power_allocation_actions.append( online_res[ev_num]['PA'][t] if(online_res[ev_num]['PA'] is not None) else 0 )
        for _ in range(len(current_charging_list), env.number_of_chargers):
            power_allocation_actions.append( 0 )

        assert (sum(power_allocation_actions) <= env.number_of_chargers)
        assert(sum(power_allocation_actions[env.number_of_chargers:]) == 0), power_allocation_actions[env.number_of_chargers:]
        
        action = charging_request_actions + power_allocation_actions
        next_state, rewards, done, _, info = env.step(action)

        current_plugged_list = sorted([list(req.values()) for req in env.scenario[t].values() if(req['status'] == Status.ACCEPTED)], key=lambda e: (e[scenario_keys.index('departure')], e[scenario_keys.index('arrival')], e[scenario_keys.index('current_soc')]))
        current_waiting_list = sorted([list(req.values()) for req in env.scenario[t].values() if(req['status'] == Status.WAITING)], key=lambda e: (e[scenario_keys.index('arrival')], e[scenario_keys.index('departure')], e[scenario_keys.index('current_soc')]))
        current_charging_list = current_plugged_list+current_waiting_list
        for ev_num in range(len(online_res)):
            if((online_res[ev_num]['PA'] is not None) and (online_res[ev_num]['PA'][t] == 1)):
                assert (env.scenario[t][ev_num]["current_soc"] < env.scenario[t+1][ev_num]["current_soc"]), f'Power allocation failed for EV {ev_num} on charger {cur_req[scenario_keys.index("charger")]} with current soc {env.scenario[t][ev_num]["current_soc"]} and next soc {env.scenario[t+1][ev_num]["current_soc"]}'

        for ts in range(t, 97):
            charger_list = []
            ev_num_list = []
            for ev in env.scenario[ts].values():
                if(ev['status'] == Status.ACCEPTED):
                    charger_list.append(ev['charger'])
                    ev_num_list.append(ev['num'])
            for charger in charger_list:
                assert len([c for c in charger_list if(c == charger)]) <= 1, f'{ts}, {charger_list}'
        
        rewards_list.append(rewards)
        t += 1

    accepted_set = set([ev['num'] for ts in range(97) for ev in env.scenario[ts].values() if(ev['status'] == Status.ACCEPTED)])
    nb_excepted_acceptation = sum([int(ev['PA'] is not None) for ev in online_res.values()])
    if nb_excepted_acceptation > len(accepted_set):
        for num, ev in online_res.items():
            if (ev['PA'] is not None) and (num not in accepted_set):
                print('nb_excepted_acceptation > len(accepted_set)', ev)
    elif nb_excepted_acceptation < len(accepted_set):
        for num in accepted_set:
            if (online_res[num]['PA'] is None):
                print('nb_excepted_acceptation < len(accepted_set)', online_res[num])
    
    assert sum([int(ev['PA'] is not None) for ev in online_res.values()]) == len(set([ev['num'] for ts in range(97) for ev in env.scenario[ts].values() if(ev['status'] == Status.ACCEPTED)])), f'online_res: {[num for num, ev in online_res.items() if(ev["PA"] is not None)]}, simulation: {set([ev["num"] for ts in range(97) for ev in env.scenario[ts].values() if(ev["status"] == Status.ACCEPTED)])}'
    assert sum(env.action_controller.history['grid_history']) == 0, 'At least one grid excess detected!'



def process_metrics(TO, filename, algo_name_dict=None, dir_res_name=None):
    metric = {}
    result = get_variables(filename)
    n = len(result['Parameters']['SOC_0'])
    metric['filename'] = filename.split('/')[-1]
    metric['computation_time'] = result['Additional infos']['computation_time']
    metric['timeout'] = int((result['Additional infos']['computation_time'] > TO) and (result['Additional infos']['solved'] == 0))
    acceptance = np.zeros((result['Parameters']['n'],))
    for charger in range(result['Parameters']['m']):
        for ev_acc in result['Decision variables']['y_ij'][charger]:
            acceptance[ev_acc] = 1.
    metric['charging_satisfaction'] = {}
    metric['charging_satisfaction']['acceptance'] = acceptance.sum() / n
    metric['charging_satisfaction']['< 80%'] = (acceptance * (np.array(result['Decision variables']['SOC_f']) < 0.8)).sum() / n
    metric['charging_satisfaction']['>= 80%'] = (acceptance * (np.array(result['Decision variables']['SOC_f']) >= 0.8)).sum() / n
    waiting_time = []
    chargers_occupation = []
    for charger in range(result['Parameters']['m']):
        chargers_occupation.append(0.)
        for i in range(0, len(result['Decision variables']['y_ij'][charger])-1):
            ev_num = result['Decision variables']['y_ij'][charger][i]
            next_ev_num = result['Decision variables']['y_ij'][charger][i+1]
            waiting_time.append( max(0, result['Parameters']['departures'][ev_num]-result['Parameters']['arrivals'][next_ev_num]) )
            chargers_occupation[charger] += (result['Parameters']['departures'][ev_num] - result['Parameters']['arrivals'][ev_num]) - max(0, result['Parameters']['departures'][ev_num]-result['Parameters']['arrivals'][next_ev_num])
        if(len(result['Decision variables']['y_ij'][charger]) > 0):
            ev_num = result['Decision variables']['y_ij'][charger][len(result['Decision variables']['y_ij'][charger])-1] # Take the last EV
            chargers_occupation[charger] += (result['Parameters']['departures'][ev_num] - result['Parameters']['arrivals'][ev_num])
    T = len(result['Decision variables']['e_t'])
    m = result['Parameters']['m']
    X_ijt = np.array([ [ [ result['Decision variables']['X_ijt'][charger][ev][t] for t in range(T)] for ev in range(n)] for charger in range(m)])
    metric['chargers_usage'] = np.array([sum([X_ijt[charger,:,t].sum() for t in range(T)]) / T for charger in range(m)])
    metric['chargers_occupation'] = np.array([chargers_occupation[charger] / T for charger in range(m)])
    metric['waiting_time'] = np.array(waiting_time)
    metric['electricity_cost_evolution'] = np.array([gt*pr for gt, pr in zip(result['Decision variables']['e_t'], result['Parameters']['price'])])
    metric['res_wasted_evolution'] = np.array([max(0, pv-gt) for gt, pv in zip(result['Decision variables']['e_t'], result['Parameters']['pv'])])
    metric['soc_f_acceptance'] = np.array([result['Decision variables']['SOC_f'][j] for j, acc in enumerate(acceptance) if(acc == 1.)])
    metric['soc_0_acceptance'] = np.array([result['Parameters']['SOC_0'][j] for j, acc in enumerate(acceptance) if(acc == 1.)])
    metric['soc_0_total'] = np.array([result['Parameters']['SOC_0'][j] for j in range(n)])
    metric['parking_duration_acceptance'] = np.array([result['Parameters']['departures'][j] - result['Parameters']['arrivals'][j] for j, acc in enumerate(acceptance) if(acc == 1.)])
    metric['parking_duration_total'] = np.array([result['Parameters']['departures'][j] - result['Parameters']['arrivals'][j] for j in range(n)])
    metric['pv_energy_generation_evolution'] = np.array(result['Parameters']['pv'])
    metric['grid_energy_consumption_evolution'] = np.array(result['Decision variables']['e_t'])
    metric['energy_provided'] = ((np.array(result['Decision variables']['SOC_f']) - np.array(result['Parameters']['SOC_0'])) * np.array(result['Parameters']['Bmax']))
    diff = (np.ones((result['Parameters']['n'],)) * .8) - np.array(result['Parameters']['SOC_0'])
    metric['min_energy_requested'] = (diff[diff >= 0.] * np.array(result['Parameters']['Bmax']))
    metric['max_energy_requested'] = ((np.ones((result['Parameters']['n'],)) - np.array(result['Parameters']['SOC_0'])) * np.array(result['Parameters']['Bmax']))
    metric['solved'] = result['Additional infos']['solved']
    metric['obj_val'] = result['Additional infos']['obj_val']
    if('GAP' in result['Additional infos']):
        metric['GAP_P1'] = result['Additional infos']['GAP']
    elif('GAP_P1' in result['Additional infos']):
        metric['GAP_P1'] = result['Additional infos']['GAP_P1']
    else:
        metric['GAP_P1'] = None
    if('GAP_P2' in result['Additional infos']):
        metric['GAP_P2'] = result['Additional infos']['GAP_P2']
    else:
        metric['GAP_P2'] = None
    if('GAP_P3' in result['Additional infos']):
        metric['GAP_P3'] = result['Additional infos']['GAP_P3']
    else:
        metric['GAP_P3'] = None
    
    if (dir_res_name is not None) and (algo_name_dict is not None):
        algo_name = algo_name_dict[metric['filename'].split('-')[0]+'-']
        try:
            n = metric['filename'].split('-')[-1].split('.')[0]
            with open(os.path.join(dir_res_name, f"{algo_name}-Results-{n}.pickle"), 'rb') as f:
                results = pickle.load(f)
            metric['reward value'] = sum(results[-1]['reward_history'])
        except:
            print(f"WARNING: Can't open {os.path.join(dir_res_name, f'{algo_name}-Results-{n}.pickle')}")
            metric['reward value'] = None
    
    return metric