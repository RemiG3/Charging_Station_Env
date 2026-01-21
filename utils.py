import numpy as np
from charging_station_env import Status

def get_statistics(initial_values, results):
    results[-1]['pv_energy_history'] = np.array(results[-1]['pv_energy_history'])
    finished_requests = [req for result in results[:-1] for req in result.values() if req['status'] == Status.FINISHED]
    avg_waiting_time = list(results[-1]['waiting_time_ev'].values()) if('waiting_time_ev' in results[-1]) else []
    statistics = {
        'nb_accepted_demands': len(finished_requests),
        'nb_total_demands': sum(len(time_requests) for time_requests in initial_values['scenario']),
        'nb_rejected_demands': sum(results[-1]['rejection_history']),
        'nb_auto_rejected': sum(results[-1]['auto_rejection_history']),
        'avg_soc_charged': np.mean([req['current_soc'] - req['initial_soc'] for req in finished_requests]) if finished_requests else 0,
        'avg_final_soc': np.mean([req['current_soc'] for req in finished_requests]) if finished_requests else 0,
        'nb_charging_below_80': sum(req['current_soc'] < 0.8 for req in finished_requests),
        'nb_charging_80_or_above': sum(req['current_soc'] >= 0.8 for req in finished_requests),
        'nb_grid_limit_overtaking': np.count_nonzero(results[-1]['grid_penalty_history']),
        'electricity_price': sum(results[-1]['price_cost_history']),
        'total_energy': sum(results[-1]['energy_consumed_history']) if('energy_consumed_history' in results[-1]) else (sum(results[-1]['grid_history']) + (sum(results[-1]['pv_energy_history']) - sum(results[-1]['pv_wasted_history']))),
        'total_renewable': sum(results[-1]['pv_energy_history']),
        'renewable_used': sum(results[-1]['pv_energy_history'] - results[-1]['pv_wasted_history']),
        'grid_used': sum(results[-1]['grid_history']),
        'avg_waiting_time': np.mean(avg_waiting_time) if (len(avg_waiting_time) > 0) else 0.,
        'objective_cost': sum(results[-1]['reward_history'])
    }

    return statistics



def parse_list_float(values):
    values = str(values).replace("'", '').replace('"', '')
    if (',' in values):
        return [float(v) for v in values.split(',')]
    return float(values)

def parse_bool(value):
    value = str(value).replace("'", '').replace('"', '').lower()
    if value in ['true', '1']:
        return True
    elif value in ['false', '0']:
        return False
    else:
        raise Exception(f'Boolean value {value} not recognized')

def parse_list_bool(values):
    values = str(values).replace("'", '').replace('"', '')
    if (',' in values):
        return [parse_bool(v) for v in values.split(',')]
    return parse_bool(values)

def parse_dic_element(value):
    value = str(value).replace("'", '').replace('"', '')
    k, v = value.split('=', 1)  # Split only on first '=' to handle values with '='
    
    # Handle None case
    if v.lower() == 'none':
        return {k: None}
    
    # Try to parse as different types
    try:
        # Try integer first
        if v.isdigit() or (v.startswith('-') and v[1:].isdigit()):
            return {k: int(v)}
    except:
        pass
    
    try:
        # Try float
        return {k: float(v)}
    except ValueError:
        pass
    
    try:
        # Try boolean
        if v.lower() in ['true', 'false']:
            return {k: v.lower() == 'true'}
    except:
        pass
    
    # If all else fails, return as string (preserve original case)
    return {k: v}

def parse_dic_args(values):
    values = str(values).replace("'", '').replace('"', '')
    if(values.lower() == 'none'):
        return {}
    dic_args = {}
    for value in values.split(','):
        dic_args.update(parse_dic_element(value))
    return dic_args

def parse_list_int_with_None(value):
    value = str(value).replace("'", '').replace('"', '')
    if (value == 'None' or value == ''):
        return []
    if (',' in value):
        return [int(v) for v in value.split(',')]
    return [int(value)]

def parse_str_with_None(value):
    value = str(value).strip().replace("'", "").replace('"', '')
    if value.lower() == 'none':
        return None
    return value
