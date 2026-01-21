from utils_offline import get_variables
import numpy as np
import argparse
import os




def check_right_to_charge(result):
    charger_result = np.array([[result['Decision variables']['X_ijt'][charger][ev_num] for ev_num in result['Decision variables']['X_ijt'][charger]] for charger in sorted(list(result['Decision variables']['X_ijt'].keys()))])
    number_of_chargers = len(result['Decision variables']['X_ijt'])
    number_of_ev = len(result['Decision variables']['X_ijt'][0])
    T = len(result['Decision variables']['X_ijt'][0][0])
    arrivals = result['Parameters']['arrivals']
    departures = result['Parameters']['departures']
    for i in range(number_of_chargers):
        for j in range(number_of_ev):
            for t in range(T):
                if((t < arrivals[j]) or (t >= departures[j])):
                    assert (charger_result[i][j][t] == 0), f'Unauthorized charging for EV {j} on charger {i} at timestep {t} (where timestep is in {t} < {arrivals[j]} (arrival) OR {t} >= {departures[j]} (departure)).'


def check_charging_overlap(result):
    charger_result = np.array([[result['Decision variables']['X_ijt'][charger][ev_num] for ev_num in result['Decision variables']['X_ijt'][charger]] for charger in sorted(list(result['Decision variables']['X_ijt'].keys()))])
    number_of_chargers = len(result['Decision variables']['X_ijt'])
    number_of_ev = len(result['Decision variables']['X_ijt'][0])
    T = len(result['Decision variables']['X_ijt'][0][0])
    arrivals = result['Parameters']['arrivals']
    departures = result['Parameters']['departures']
    for i in range(number_of_chargers):
        for j1 in range(number_of_ev):
            for t1 in range(T):
                if((t1 >= arrivals[j1]) and (t1 < departures[j1])):
                    tmp_sum = 0
                    list_j2 = []
                    ts = t1
                    for j2 in range(number_of_ev):
                        if(j1 != j2) and (departures[j1] <= departures[j2]):
                            for t2 in range(max(arrivals[j2], arrivals[j1]), min(departures[j2], departures[j1])):
                                tmp_sum += charger_result[i][j2][t2]
                                if(charger_result[i][j2][t2] > 0):
                                    ts = t2
                                    list_j2.append(str(j2))
                    assert (charger_result[i][j1][t1] + 1/((number_of_ev+1)*T) * tmp_sum) <= 1, f'Overlap found on charger {i} with EV {j1} and EVs {",".join(set(list_j2))} at timestep {ts}.'


def check_one_ev_per_charger(result):
    number_of_chargers = len(result['Decision variables']['X_ijt'])
    number_of_ev = len(result['Decision variables']['X_ijt'][0])
    T = len(result['Decision variables']['X_ijt'][0][0])
    intersect_accepted_ev = set(list(range(number_of_ev)))
    for i in range(number_of_chargers):
        intersect_accepted_ev = intersect_accepted_ev & set(result['Decision variables']['y_ij'][i])
    assert intersect_accepted_ev == set(), f'Intersection of accepted EVs: {intersect_accepted_ev}.'


def check_accepted_ev_charged_at_least_once(result):
    charger_result = np.array([[result['Decision variables']['X_ijt'][charger][ev_num] for ev_num in result['Decision variables']['X_ijt'][charger]] for charger in sorted(list(result['Decision variables']['X_ijt'].keys()))])
    number_of_chargers = len(result['Decision variables']['X_ijt'])
    number_of_ev = len(result['Decision variables']['X_ijt'][0])
    T = len(result['Decision variables']['X_ijt'][0][0])
    union_accepted_ev = set()
    union_charged_ev = set()
    for i in range(number_of_chargers):
        union_accepted_ev = union_accepted_ev | set(result['Decision variables']['y_ij'][i])
        for j in range(number_of_ev):
            for t in range(T):
                if(charger_result[i][j][t] == 1):
                    union_charged_ev = union_charged_ev | set([j])
    assert len(union_accepted_ev) == len(union_charged_ev), f'\nAccepted EVs: {union_accepted_ev}\nCharged EVs: {union_charged_ev}.'


def check_accepted_ev_socf_equal_to_soc0(result):
    union_accepted_ev = set()
    number_of_chargers = len(result['Decision variables']['X_ijt'])
    number_of_ev = len(result['Decision variables']['X_ijt'][0])
    for i in range(number_of_chargers):
        union_accepted_ev = union_accepted_ev | set(result['Decision variables']['y_ij'][i])
    for j in union_accepted_ev:
        assert (result['Parameters']['SOC_0'][j] < result['Decision variables']['SOC_f'][j]), f'EV {j} is accepted, its SOC_f should more than SOC_0 ({result["Parameters"]["SOC_0"][j]} >= {result["Decision variables"]["SOC_f"][j]}).'
    for j in range(number_of_ev):
        if j not in union_accepted_ev:
            assert (result['Parameters']['SOC_0'][j] == result['Decision variables']['SOC_f'][j]), f'EV {j} is rejected, but SOC_0 != SOC_f ({result["Parameters"]["SOC_0"][j]} != {result["Decision variables"]["SOC_f"][j]}).'


def check_ev_acc_and_charged_on_same_charger(result):
    charger_result = np.array([[result['Decision variables']['X_ijt'][charger][ev_num] for ev_num in result['Decision variables']['X_ijt'][charger]] for charger in sorted(list(result['Decision variables']['X_ijt'].keys()))])
    number_of_chargers = len(result['Decision variables']['X_ijt'])
    number_of_ev = len(result['Decision variables']['X_ijt'][0])
    T = len(result['Decision variables']['X_ijt'][0][0])
    for i in range(number_of_chargers):
        union_accepted_ev_per_charger = set()
        union_charged_ev_per_charger = set()
        union_accepted_ev_per_charger = union_accepted_ev_per_charger | set(result['Decision variables']['y_ij'][i])
        for j in range(number_of_ev):
            for t in range(T):
                if(charger_result[i][j][t] == 1):
                    union_charged_ev_per_charger = union_charged_ev_per_charger | set([j])
        assert len(union_accepted_ev_per_charger) == len(union_charged_ev_per_charger), f'\nAccepted EVs: {union_accepted_ev_per_charger}\nCharged EVs: {union_charged_ev_per_charger}\nOn charger {i}.'


def check_grid_limit_exceeding(result):
    charger_result = np.array([[result['Decision variables']['X_ijt'][charger][ev_num] for ev_num in result['Decision variables']['X_ijt'][charger]] for charger in sorted(list(result['Decision variables']['X_ijt'].keys()))])
    T = len(result['Decision variables']['X_ijt'][0][0])
    for t in range(T):
        assert result['Parameters']['P']*charger_result[:,:,t].sum()-result['Parameters']['pv'][t] <= result['Parameters']['w_G'], f"Grid limit ({result['Parameters']['w_G']}) exceeded with {result['Parameters']['P']*charger_result[:,:,t].sum()-result['Parameters']['pv'][t]} at timestep {t}."


def check_socf_calculation(result):
    charger_result = np.array([[result['Decision variables']['X_ijt'][charger][ev_num] for ev_num in result['Decision variables']['X_ijt'][charger]] for charger in sorted(list(result['Decision variables']['X_ijt'].keys()))])
    number_of_ev = len(result['Decision variables']['X_ijt'][0])
    for j in range(number_of_ev):
        assert (f"{result['Decision variables']['SOC_f'][j]:.5f}" == f"{round(result['Parameters']['SOC_0'][j] + result['Parameters']['P']*result['Parameters']['eta']*result['Parameters']['tau']*charger_result[:,j,:].sum() / result['Parameters']['Bmax'][j], 6):.5f}"), f"Wrong SOC_f computation: Excpected {result['Decision variables']['SOC_f'][j]:.5f}, got {round(result['Parameters']['SOC_0'][j] + result['Parameters']['P']*result['Parameters']['eta']*result['Parameters']['tau']*charger_result[:,j,:].sum() / result['Parameters']['Bmax'][j], 6):.5f} for EV {j}."


def check_gt_calculation(result):
    charger_result = np.array([[result['Decision variables']['X_ijt'][charger][ev_num] for ev_num in result['Decision variables']['X_ijt'][charger]] for charger in sorted(list(result['Decision variables']['X_ijt'].keys()))])
    T = len(result['Decision variables']['X_ijt'][0][0])
    for t in range(T):
        assert (f"{result['Decision variables']['e_t'][t]:.5f}" == f"{max(0, round(result['Parameters']['tau']*(result['Parameters']['P']*charger_result[:,:,t].sum()-result['Parameters']['pv'][t]), 6)):.5f}"), f"Wrong e_t computation: Excpected {result['Decision variables']['e_t'][t]:.5f}, got {max(0, round(result['Parameters']['tau']*(result['Parameters']['P']*charger_result[:,:,t].sum()-result['Parameters']['pv'][t]), 6)):.5f} at timestep {t}."


def check(result):
    check_right_to_charge(result)
    check_charging_overlap(result)
    check_one_ev_per_charger(result)
    check_ev_acc_and_charged_on_same_charger(result)
    check_grid_limit_exceeding(result)
    check_accepted_ev_charged_at_least_once(result)
    check_accepted_ev_socf_equal_to_soc0(result)
    #check_socf_calculation(result)
    #check_gt_calculation(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=None, type=str)
    parser.add_argument("--file", default=None, type=str)
    args = parser.parse_args()

    assert (args.dir is not None) or (args.file is not None)

    if args.dir is not None:
        for filename in sorted(os.listdir(args.dir)):
            if (filename.endswith('.log')):
                try:
                    filepath = os.path.join(args.dir, filename)
                    result = get_variables(filepath)
                    print(f'Proceeding {filename}...', end='\t', flush=True)
                    check(result)
                    print('DONE!', flush=True)
                except AssertionError as e:
                    print('FAILED!')
                    print(f'ASSERTION: {e}', flush=True)
                except Exception as e:
                    print('FAILED!')
                    print(f'Unexpected Error: {e}', flush=True)
    else:
        try:
            result = get_variables(args.file)
            check(result)
        except AssertionError as e:
            print('FAILED!')
            print(f'ASSERTION: {e}', flush=True)
        except Exception as e:
            print('FAILED!')
            print(f'Unexpected Error: {e}', flush=True)
    
    print('Done')

