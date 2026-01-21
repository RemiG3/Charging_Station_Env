from charging_station_env import Status, ChargingStatus
import os
import pickle
import numpy as np


def compute_T_MAX(env):
    """Compute T_MAX based on environment step_time.
    
    Args:
        env: Environment object with schema containing step_time
    
    Returns:
        int: Maximum timesteps (T_MAX = int(24/tau) + 1 where tau = step_time/3)
    """
    tau = env.schema['step_time'] / 60.0 # Convert step_time to tau
    return int(24 / tau)


def _pa_at(pa_seq, ts_env, T_MAX):
    """Return PA value (0/1) at env timestep.
    If PA already matches env resolution (len == T_MAX), index directly.
    Otherwise, map by nearest (coarse) index.
    """
    if pa_seq is None or len(pa_seq) == 0:
        return 0
    off_len = len(pa_seq)
    if off_len == T_MAX:
        return pa_seq[ts_env]
    # Fallback mapping when resolutions differ
    k = max(1, int(round(T_MAX / off_len)))
    idx = min(off_len - 1, ts_env // k)
    return pa_seq[idx]


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


def check_pv_alignment(offline_sol, env, specific_ts=None, tol_kw=1e-5):
    """Compare PV production from the offline log (Parameters.pv) vs the online env (env.energy['renewable']).

    Prints summary stats and, if specific_ts is provided, the values at that timestep.

    - Units: both sources are in kW. Step energy is kWh = kW * (env.step_time/60).
    - Returns a small dict of stats.
    """
    pv_off = list(offline_sol.get('Parameters', {}).get('pv', []))
    pv_env = list(env.energy.get('renewable', [])) if hasattr(env, 'energy') and ('renewable' in env.energy) else []
    dt_h = (env.step_time / 60.0) if hasattr(env, 'step_time') else 0.05

    n_off = len(pv_off)
    n_env = len(pv_env)
    n_cmp = min(n_off, n_env)
    if n_cmp == 0:
        print("[PV] Cannot compare: missing PV series in offline or env.")
        return {'ok': False, 'reason': 'missing_series'}

    # Compute diffs (kW)
    diffs = np.array(pv_env[:n_cmp], dtype=float) - np.array(pv_off[:n_cmp], dtype=float)
    abs_diffs = np.abs(diffs)
    max_idx = int(np.argmax(abs_diffs))
    max_abs_kw = float(abs_diffs[max_idx])
    rmse_kw = float(np.sqrt(np.mean(diffs**2)))
    sum_energy_diff_kwh = float(np.sum(diffs) * dt_h)

    if specific_ts is not None and 0 <= specific_ts < n_cmp:
        p_env = float(pv_env[specific_ts])
        p_off = float(pv_off[specific_ts])
        print(f"[PV] t={specific_ts}: env={p_env:.6f} kW ({p_env*dt_h:.6f} kWh) vs offline={p_off:.6f} kW ({p_off*dt_h:.6f} kWh) Δ={p_env-p_off:.6f} kW")

    # List first few mismatches above tolerance
    above = np.where(abs_diffs > tol_kw)[0]
    if above.size > 0:
        kshow = min(50, above.size)
        print(f"[PV] {above.size} steps differ by more than {tol_kw} kW. First {kshow}:")
        for i in above[:kshow]:
            print(f"   t={int(i)} env={pv_env[i]:.6f} kW off={pv_off[i]:.6f} kW Δ={pv_env[i]-pv_off[i]:.6f} kW")
        assert False, "The PV data differ between the environment and the PV data in the offline reference."

    return {
        'ok': True,
        'n_compare': n_cmp,
        'max_abs_kw': max_abs_kw,
        'max_abs_idx': max_idx,
        'rmse_kw': rmse_kw,
        'sum_energy_diff_kwh': sum_energy_diff_kwh,
    }

def convert_result_to_online_format(result, env, T_MAX):
    accepted_requests = set()
    charger_association = {}
    for charger_num, acc_ev in enumerate(result['Decision variables']['y_ij'].values()):
        accepted_requests.update(acc_ev)
        for ev in acc_ev:
            charger_association[ev] = charger_num

    online_result = {}
    n_set_treated = set()
    env_len = len(env.energy["price"]) if ("price" in env.energy) else T_MAX
    offline_T = len(result['Decision variables'].get('e_t', []))
    assert offline_T == env_len, f"Offline horizon T ({offline_T}) != env horizon ({env_len}). Check step resolution/log."
    for t in range(T_MAX):
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
                            pa_seq = result['Decision variables']['X_ijt'][charger_association[n]][n]
                            # Basic integrity checks on PA
                            assert isinstance(pa_seq, list) and len(pa_seq) == offline_T, f"PA length mismatch for offline EV {n}: {len(pa_seq)} != {offline_T}"
                            for tt, bit in enumerate(pa_seq):
                                assert bit in (0, 1), f"PA value not binary for EV {n} at t={tt}: {bit}"
                                # Enforce zero outside [arr, dep)
                                if not (arr <= tt < dep):
                                    assert bit == 0, f"PA has charge outside window for EV {n} at t={tt} (arr={arr}, dep={dep})"
                            # compute first env ts where PA=1
                            start_env_ts = None
                            for ts_env in range(env_len):
                                if _pa_at(pa_seq, ts_env, T_MAX) == 1:
                                    start_env_ts = ts_env
                                    break
                            assert start_env_ts is not None, f"No charging timestep found for accepted EV offline idx={n} (arr={arr}, dep={dep})"
                            online_result[req['num']] = {'PA': pa_seq, 'charger': charger_association[n], 'arrival': arr, 'departure': dep, 'initial_soc': soc_0, 'start_charging': (start_env_ts if start_env_ts is not None else -1), 'offline_idx': n}
                        else:
                            online_result[req['num']] = {'PA': None, 'start_charging': -1, 'charger': -1, 'arrival': arr, 'departure': dep, 'initial_soc': soc_0, 'offline_idx': n}#, 'current_soc': soc_0, 'type': 0}
                        n_set_treated.add( n )
                        break
                assert found, f'Error: EV {req["num"]}, arr={req["arrival"]}, dep={req["departure"]}, and soc0={req["initial_soc"]} not found in offline result dict'
            
    # Assert charger exclusivity per timestep from offline X_ijt
    for c, x_c in result['Decision variables']['X_ijt'].items():
        for t in range(offline_T):
            sum_t = 0
            for ev in x_c:
                sum_t += x_c[ev][t]
            assert sum_t <= 1, f"Offline X_ijt violates charger exclusivity: charger={c}, t={t}, sum={sum_t}"

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
    T_MAX = compute_T_MAX(env)
    offline_sol = get_variables(filename)
    assert offline_sol['Parameters']['n'] == sum([int((req['status'] == Status.REJECTED) or (req['status'] == Status.ARRIVED) or (req['status'] == Status.FINISHED) or (req['status'] == Status.ACCEPTED)) for t in range(T_MAX) for req in env.scenario[t].values()]), f"{sum([int((req['status'] == Status.REJECTED) or (req['status'] == Status.ARRIVED) or (req['status'] == Status.FINISHED) or (req['status'] == Status.ACCEPTED)) for t in range(T_MAX) for req in env.scenario[t].values()])} != {offline_sol['Parameters']['n']} \t Filename: {filename}"

    online_res = convert_result_to_online_format(offline_sol, env, T_MAX)

    # Ensure that the offline log PV production is equal to the online PV production
    check_pv_alignment(offline_sol, env, specific_ts=None)
    
    # Ensure that the offline log PR price is equal to the online PR price
    price_off = offline_sol['Parameters'].get('price', [])
    price_env = env.energy.get('price', [])
    assert len(price_off) == len(price_env), f"Price length mismatch: offline={len(price_off)} vs env={len(price_env)}"
    assert len(price_off) == T_MAX, f"Price length mismatch: offline={len(price_off)} vs T_MAX={T_MAX}"
    assert len(price_env) == T_MAX, f"Price length mismatch: env={len(price_env)} vs T_MAX={T_MAX}"
    assert len(env.scenario) == T_MAX+1, f"Scenario length mismatch: env={len(env.scenario)} vs T_MAX={T_MAX}"
    for i in range(T_MAX):
        assert abs(price_off[i] - price_env[i]) < 1e-6, f"Price mismatch at t={i}: offline={price_off[i]} vs env={price_env[i]}"
    
    # Ensure that the offline station data is equal to the online station data
    assert offline_sol['Parameters']['m'] == env.number_of_chargers, f"Charger count mismatch: offline={offline_sol['Parameters']['m']} vs env={env.number_of_chargers}"
    
    # Ensure that the offline EV information (arrival, departure, initial_soc, battery capacity) is equal to the online EV information
    # This is already verified in convert_result_to_online_format function

    # Build immutable EV metadata from the original scenario
    ev_meta = {}
    for ts in range(T_MAX+1):
        for num, req in env.scenario[ts].items():
            if num not in ev_meta:
                ev_meta[num] = {
                    'arrival': req.get('arrival'),
                    'departure': req.get('departure'),
                    'initial_soc': req.get('initial_soc'),
                    'type': req.get('type', 0)
                }
    
    # Process each EV based on their PA schedule
    for ev_num, plan in online_res.items():
        if plan['PA'] is None:
            # If sum(online_res[j]['PA']) == 0, then reject request j (REJECTED status instead of ARRIVED)
            arr = plan.get('arrival')
            if arr is not None and ev_num in env.scenario[arr]:
                env.scenario[arr][ev_num]['status'] = Status.REJECTED
                env.scenario[arr][ev_num]['charging_status'] = {ChargingStatus.UNPLUGGED}
                env.scenario[arr][ev_num]['charger'] = -1
        else:
            # EV has a charging schedule
            arr = plan.get('arrival')
            dep = plan.get('departure')
            charger = plan.get('charger')
            
            if arr is not None and dep is not None and charger is not None:
                # Find max_k d_k (maximum departure of EVs on the same charger that depart before this EV)
                max_k_dk = arr  # default to arrival time
                for other_num, other_plan in online_res.items():
                    if (other_num != ev_num and 
                        other_plan.get('charger') == charger and 
                        other_plan.get('departure', 0) <= dep and
                        other_plan.get('departure', 0) > max_k_dk):
                        max_k_dk = other_plan['departure']
                
                # For t in {d_j-1, ..., max(r_j, max_k d_k)} [decreasing loop]
                # Mark EV j as ACCEPTED for timestep t
                start_accepted = max(arr, max_k_dk)
                # Record per-EV waiting time (first accepted ts minus arrival),
                # using the same dictionary as Simulate_Actions_FIFO
                if (ev_num not in env.action_controller.history.get("waiting_time_ev", {})) and (online_res[ev_num]['PA'] is not None):
                    env.action_controller.history["waiting_time_ev"][ev_num] = max(0, start_accepted - arr)
                for t in range(dep - 1, start_accepted - 1, -1):
                    if t < T_MAX:
                        if ev_num not in env.scenario[t]:
                            prev_soc = ev_meta[ev_num]['initial_soc']
                            if t > arr and ev_num in env.scenario[t-1]:
                                prev_soc = env.scenario[t-1][ev_num]['current_soc']
                            
                            env.scenario[t][ev_num] = {
                                'num': ev_num,
                                'arrival': ev_meta[ev_num]['arrival'],
                                'departure': ev_meta[ev_num]['departure'],
                                'initial_soc': ev_meta[ev_num]['initial_soc'],
                                'current_soc': prev_soc,
                                'status': Status.ACCEPTED,
                                'charging_status': {ChargingStatus.UNKNOWN},
                                'charger': charger,
                                'type': ev_meta[ev_num]['type']
                            }
                        else:
                            env.scenario[t][ev_num]['status'] = Status.ACCEPTED
                            env.scenario[t][ev_num]['charger'] = charger
                            env.scenario[t][ev_num]['charging_status'] = {ChargingStatus.UNKNOWN}
                
                # For t in {max(r_j, max_k d_k)-1, ..., r_j} [decreasing loop]
                # Mark EV j as WAITING for timestep t
                for t in range(start_accepted - 1, arr - 1, -1):
                    if t < T_MAX and t >= 0:
                        if ev_num not in env.scenario[t]:
                            env.scenario[t][ev_num] = {
                                'num': ev_num,
                                'arrival': ev_meta[ev_num]['arrival'],
                                'departure': ev_meta[ev_num]['departure'],
                                'initial_soc': ev_meta[ev_num]['initial_soc'],
                                'current_soc': ev_meta[ev_num]['initial_soc'],
                                'status': Status.WAITING,
                                'charging_status': {ChargingStatus.UNPLUGGED},
                                'charger': -1,
                                'type': ev_meta[ev_num]['type']
                            }
                        else:
                            env.scenario[t][ev_num]['status'] = Status.WAITING
                            env.scenario[t][ev_num]['charging_status'] = {ChargingStatus.UNPLUGGED}
                            env.scenario[t][ev_num]['charger'] = -1
                # Mark EV j as FINISHED for timestep d_j
                if dep <= T_MAX:
                    if ev_num not in env.scenario[dep]:
                        env.scenario[dep][ev_num] = {
                            'num': ev_num,
                            'arrival': ev_meta[ev_num]['arrival'],
                            'departure': ev_meta[ev_num]['departure'],
                            'initial_soc': ev_meta[ev_num]['initial_soc'],
                            'current_soc': ev_meta[ev_num]['initial_soc'],  # Will be updated during simulation
                            'status': Status.FINISHED,
                            'charging_status': {ChargingStatus.UNPLUGGED},
                            'charger': -1,
                            'type': ev_meta[ev_num]['type']
                        }
                    else:
                        env.scenario[dep][ev_num]['status'] = Status.FINISHED
                        env.scenario[dep][ev_num]['charging_status'] = {ChargingStatus.UNPLUGGED}
                        env.scenario[dep][ev_num]['charger'] = -1
    
    # Assertion: Check that the sum of WAITING and ACCEPTED EVs doesn't exceed 2 * number_of_chargers
    for t in range(T_MAX):
        waiting_count = 0
        accepted_count = 0
        for ev_num, req in env.scenario[t].items():
            if req['status'] == Status.WAITING:
                waiting_count += 1
            elif req['status'] == Status.ACCEPTED:
                accepted_count += 1
        
        total_count = waiting_count + accepted_count
        max_allowed = 2 * env.number_of_chargers
        
        assert total_count <= max_allowed, (
            f"At timestep {t}, there are {total_count} EVs (WAITING: {waiting_count}, ACCEPTED: {accepted_count}), "
            f"which exceeds 2 * number_of_chargers ({max_allowed}). "
            f"WAITING EVs: {[num for num, req in env.scenario[t].items() if req['status'] == Status.WAITING]}. "
            f"ACCEPTED EVs: {[num for num, req in env.scenario[t].items() if req['status'] == Status.ACCEPTED]}"
        )
    
    # Loop over env.step:
    t = 0
    done = False
    rewards_list = []
    
    while not done:
        # For each timestep t:
        # Set array of actions to [0] * env.number_of_chargers for accepting requests
        charging_request_actions = [0] * env.number_of_chargers
        
        # Get current requests (ARRIVED status)
        current_requests_list = sorted(
            [req for req in env.scenario[t].values() if req['status'] == Status.ARRIVED],
            key=lambda r: (r['departure'], r['current_soc'])
        )
        
        # Build request decisions based on offline acceptance
        for idx, cur_req in enumerate(current_requests_list):
            if idx < env.number_of_chargers:
                ev_num = cur_req['num']
                if ev_num in online_res and online_res[ev_num]['PA'] is not None:
                    charging_request_actions[idx] = 1
        
        # For each current EV plugged in at t:
        power_allocation_actions = [0] * env.number_of_chargers
        current_plugged_list = sorted(
            [req for req in env.scenario[t].values() if req['status'] == Status.ACCEPTED],
            key=lambda r: (r['departure'], r['arrival'], r['current_soc'])
        )
        
        for idx, cur_req in enumerate(current_plugged_list):
            if idx < env.number_of_chargers:
                ev_num = cur_req['num']
                # If online_res[j]['PA'] is not None:
                if ev_num in online_res and online_res[ev_num]['PA'] is not None:
                    # If _pa_at(online_res[j]['PA'], t, T_MAX) == 1:
                    if _pa_at(online_res[ev_num]['PA'], t, T_MAX) == 1:
                        # Mark EV j as charging for timestep t
                        power_allocation_actions[idx] = 1
                        # Assertion on the fact that this EV should be marked as ACCEPTED for timestep t
                        assert cur_req['status'] == Status.ACCEPTED, f"EV {ev_num} should be ACCEPTED at t={t} but has status {cur_req['status']}"
                    # Else: Mark EV j as not charging for timestep t (already 0)
        
        # Assertion on the size of action array (should be env.number_of_chargers*2)
        action = charging_request_actions + power_allocation_actions
        assert len(action) == env.number_of_chargers * 2, f"Action array size mismatch: {len(action)} != {env.number_of_chargers * 2}"
        
        # Store SOC before step for verification
        pre_step_socs = {}
        for num, req in env.scenario[t].items():
            pre_step_socs[num] = req.get('current_soc')
        
        next_state, rewards, done, _, info = env.step(action)
        rewards_list.append(rewards)
        
        # Verification (assertions) at each timestep t:
        if (t + 1) < T_MAX:
            for ev_num in online_res:
                if ev_num in env.scenario[t] and ev_num in env.scenario[t+1]:
                    pre_soc = pre_step_socs.get(ev_num)
                    post_soc = env.scenario[t+1][ev_num].get('current_soc')
                    
                    if pre_soc is not None and post_soc is not None:
                        # If online_res[j]['PA'] == 1, should have charged (current_soc < next_soc)
                        if (online_res[ev_num]['PA'] is not None and 
                            _pa_at(online_res[ev_num]['PA'], t, T_MAX) == 1):
                            assert post_soc > pre_soc + 1e-6, f"EV {ev_num} should have charged at t={t}: {pre_soc} -> {post_soc}"
                        
                        # If online_res[j]['PA'] == 0, should have not charged (current_soc == next_soc)
                        elif (online_res[ev_num]['PA'] is not None and 
                              _pa_at(online_res[ev_num]['PA'], t, T_MAX) == 0):
                            assert abs(post_soc - pre_soc) < 1e-6, f"EV {ev_num} should not have charged at t={t}: {pre_soc} -> {post_soc}"
                        
                        # Ensure that the SOC is non-decreasing
                        assert post_soc >= pre_soc - 1e-6, f"SOC decreased for EV {ev_num} at t={t}: {pre_soc} -> {post_soc}"
        
        # Ensure that the grid limit isn't exceeded
        if env.schema.get('grid_limit', -1) and env.schema['grid_limit'] > 0:
            dt_h = env.step_time / 60.0
            grid_limit_kwh = env.schema['grid_limit'] * dt_h
            
            # Calculate total consumed energy
            total_consumed_energy = 0.0
            list_ev_types = list(env.schema['EV_config']['considered_ev'])
            
            for idx, req in enumerate(current_plugged_list):
                if idx < len(power_allocation_actions):
                    a = float(power_allocation_actions[idx])
                    if a > 0.0:
                        ch_key = env.schema['Chargers_config']["list_chargers"][req['charger']]
                        charger = env.schema['Charger_types'][ch_key]
                        rate = charger['charging_rate'] if not isinstance(charger['charging_rate'], list) else max(charger['charging_rate'])
                        eff = charger['charging_efficiency'] if not isinstance(charger['charging_efficiency'], list) else max(charger['charging_efficiency'])
                        ev_type = env.schema['EV_types'][list_ev_types[req['type']]]
                        cap = ev_type['capacity']
                        
                        max_delivered = a * rate * eff * dt_h
                        headroom = max(0.0, (1.0 - float(req['current_soc'])) * float(cap))
                        delivered = min(max_delivered, headroom)
                        consumed = delivered / eff if eff > 0 else 0.0
                        total_consumed_energy += consumed
            
            renewable_energy = float(env.energy['renewable'][t]) * dt_h if 'renewable' in env.energy else 0.0
            raw_grid = max(total_consumed_energy - renewable_energy, 0.0)
            assert raw_grid <= grid_limit_kwh + 1e-6, f"Grid limit exceeded at t={t}: {raw_grid} kWh > {grid_limit_kwh} kWh"
        
        t += 1
    
    # Final verification (assertions):
    # Ensure that for each EV j: The final SOC is equal to the offline SOC_f
    soc_f = offline_sol['Decision variables']['SOC_f']
    for num, meta in online_res.items():
        off_idx = meta.get('offline_idx')
        dep = meta.get('departure')
        if off_idx is not None and dep is not None and dep < len(env.scenario):
            if num in env.scenario[dep]:
                env_final_soc = env.scenario[dep][num]['current_soc']
                off_final_soc = soc_f[off_idx]
                assert abs(env_final_soc - off_final_soc) < 1e-4, f"Final SOC mismatch for EV {num}: env={env_final_soc} vs offline={off_final_soc}"



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
        try:
            algo_name = algo_name_dict[metric['filename'].split('-')[0]+'-']
        except:
            print(f"WARNING: \"{metric['filename'].split('-')[0]+'-'}\" not found in the algorithm dictionary", end='\r')
            return
        try:
            n = metric['filename'].split('-')[-1].split('.')[0]
            with open(os.path.join(dir_res_name, f"{algo_name}-Results-{n}.pickle"), 'rb') as f:
                results = pickle.load(f)
            metric['reward value'] = sum(results[-1]['reward_history'])
        except:
            print(f"WARNING: Can't open {os.path.join(dir_res_name, f'{algo_name}-Results-{n}.pickle')}")
            metric['reward value'] = None
    
    return metric
