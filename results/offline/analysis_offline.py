import sys
sys.path.append("../../")
from utils import get_statistics, parse_dic_args
from utils_offline import process_offline_result, process_metrics
from charging_station_env.visualizer import Console_Rendering
import gymnasium as gym
import numpy as np
import pandas as pd
import argparse
import pickle
import os

ALGO_NAME_DICT = {k: v for k, v in zip(['complete_milp_gurobi-'],
                                       ['COMPLETE_MODEL'])}
NO_GRID_LIMIT_ALGO_NAMES = set(['COMPLETE_NO_GRID_MODEL'])

TO = 30*60 # Timeout out constant in models (in seconds)


# for n in 50 60 70 90 120 150; do python analysis_offline.py --num "$n" --current_folder ../../dataset/ev_scenario-"$n" --results_folder ./ev_scenario-"$n" --save metrics_ev_scenario-"$n".pickle; done
# To process only a single instance: python analysis_offline.py --num 50 --current_folder ../../dataset/ev_scenario-50 --results_folder ./ev_scenario-50 --index "5"
# To process multiple instances: python analysis_offline.py --num 50 --current_folder ../../dataset/ev_scenario-50 --results_folder ./ev_scenario-50 --index "1,3,5,7"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="ChargingStationEnv-v0")
    parser.add_argument("--num", default=1, type=int, help="Number of the experiment")
    parser.add_argument("--episodes", default=100, type=int, help="Number of scenarios")
    parser.add_argument("--schema", default="../../schema.json")
    parser.add_argument("--current_folder", default="../../dataset/ev_scenario-50")
    parser.add_argument("--results_folder", default="./ev_scenario-50")
    parser.add_argument("--initializer", default="Initializer")
    parser.add_argument("--simulation", default="Simulate_Station_FIFO")
    parser.add_argument("--action", default="Simulate_Actions_FIFO")
    parser.add_argument("--energy", default="Energy_Initializer")
    parser.add_argument("--initializer_args", default={}, type=parse_dic_args)
    parser.add_argument("--simulation_args", default={}, type=parse_dic_args)
    parser.add_argument("--action_args", default={}, type=parse_dic_args)
    parser.add_argument("--energy_args", default={}, type=parse_dic_args)
    parser.add_argument("--res_dir", default='./', help="Results directory")
    parser.add_argument("--save", default='metrics_ev_scenario-50.pickle', type=str)
    parser.add_argument("--index", default='', type=str, help="Comma-separated list of scenario indices to process; empty means all. For single instance, use '5' instead of 5")
    args = parser.parse_args()

    # Print information about instance filtering
    if args.index:
        print(f"Processing only instances: {args.index}")

    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)


    #################### Replay Simulation ####################

    for start_result_pattern, algo_name in ALGO_NAME_DICT.items():
        env = gym.make(args.env,
                       schema=args.schema,
                       current_folder=args.current_folder,
                       results_folder=args.results_folder,
                       initializer=getattr(getattr(__import__('charging_station_env'), 'initializer'), args.initializer)(**args.initializer_args, energy_initializer=getattr(getattr(__import__('charging_station_env'), 'initializer'), args.energy)(**args.energy_args)),
                       simulation_controller=getattr(getattr(__import__('charging_station_env'), 'transition'), args.simulation)(**args.simulation_args),
                       action_controller=getattr(getattr(__import__('charging_station_env'), 'action'), args.action)(**args.action_args),
                       visualizer=Console_Rendering()
                       )
        if algo_name in NO_GRID_LIMIT_ALGO_NAMES:
            env.schema['grid_limit'] = -1
        
        scenario_keys = ['num', 'arrival', 'departure', 'initial_soc', 'current_soc', 'status', 'charging_status', 'charger', 'type']

        n = 0
        # Build filter set of indices to process from --index
        only_idx = set()
        if args.index:
            try:
                only_idx |= {int(x.strip()) for x in args.index.split(',') if x.strip()}
            except Exception:
                pass
        for file in sorted(os.listdir(args.results_folder)):
            if file.startswith(start_result_pattern) and file.endswith('.log') and not file.startswith('LAUNCHER'):
                num = int(file.split('-')[-1].split('.')[0])
                if only_idx and (num not in only_idx):
                    continue
                env.reset(reset_flag=1, id_save=num, algo_name=algo_name)

                process_offline_result(os.path.join(args.results_folder, file), env, scenario_keys)
                
                n += 1
                print(f'Solution file "{file}" n°{n} proceeded!\t', end='\r')
        print(flush=True)
    

    #################### Extract Metrics ####################

    metrics = []
    # Build filter set again for the metrics extraction
    only_idx = set()
    if args.index:
        try:
            only_idx |= {int(x.strip()) for x in args.index.split(',') if x.strip()}
        except Exception:
            pass

    for filename in sorted(os.listdir(args.results_folder)):
        if filename.endswith('.log'):
            try:
                scen_idx = int(filename.split('-')[-1].split('.')[0])
            except Exception:
                scen_idx = None
            if only_idx and (scen_idx is not None) and (scen_idx not in only_idx):
                continue
            print(f'Extract metric from {os.path.join(args.results_folder, filename)}\t', end='\r')
            metric = process_metrics(TO, os.path.join(args.results_folder, filename), algo_name_dict=ALGO_NAME_DICT, dir_res_name=args.results_folder)
            if metric is not None:
                metrics.append( metric )
    print(flush=True)

    with open(args.save, 'wb') as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    #################### Analyze Results ####################
    
    current_root_results_folder = os.path.join(os.getcwd(), args.results_folder)
    
    list_df = []
    for algo_name in ALGO_NAME_DICT.values():
        statistics_lists = {key: [] for key in ['nb_total_demands', 'nb_accepted_demands', 'nb_rejected_demands', 'nb_auto_rejected', 'nb_charging_below_80', 'nb_charging_80_or_above', 'nb_grid_limit_overtaking', 'electricity_price', 'total_energy', 'total_renewable', 'renewable_used', 'grid_used', 'energy_per_ev', 'avg_waiting_time', 'avg_soc_charged', 'avg_final_soc', 'objective_cost']}

        # Build scenario list based on filter (reuse the filter set from metrics extraction)
        scenario_list = []
        if only_idx:
            scenario_list = sorted(only_idx)
        else:
            scenario_list = list(range(1, args.episodes + 1))

        for scenario_number in scenario_list:
            try:
                with open(os.path.join(args.current_folder, f'Initial_Values-{scenario_number}.pickle'), 'rb') as f:
                    initial_values = pickle.load(f)
                with open(os.path.join(current_root_results_folder, f"{algo_name}-Results-{scenario_number}.pickle"), 'rb') as f:
                    results = pickle.load(f)
                stats = get_statistics(initial_values, results)
                for key, value in stats.items():
                    statistics_lists[key].append(value)
            except FileNotFoundError as e:
                print(f"WARNING: {e}")
                continue

        for key in statistics_lists:
            statistics_lists[key] = np.array(statistics_lists[key])

        pv_used_over_total_energy = np.nan_to_num( statistics_lists["renewable_used"]/statistics_lists["total_energy"], nan=0. )
        pv_used_over_total_energy = np.where(pv_used_over_total_energy > 1., 1., pv_used_over_total_energy)
        pv_used_over_total_energy = np.where(pv_used_over_total_energy < 0., 0., pv_used_over_total_energy)
        energy_per_ev = np.nan_to_num( statistics_lists["total_energy"]/statistics_lists["nb_accepted_demands"] )
        data_for_df = {
            'Algorithm': algo_name,
            'Charging satisf. <80%': f'{np.mean(statistics_lists["nb_charging_below_80"]/statistics_lists["nb_total_demands"])*100.:.2f}±{np.std(statistics_lists["nb_charging_below_80"]/statistics_lists["nb_total_demands"])*100.:.2f}%',
            'Charging satisf. >=80%': f'{np.mean(statistics_lists["nb_charging_80_or_above"]/statistics_lists["nb_total_demands"])*100.:.2f}±{np.std(statistics_lists["nb_charging_80_or_above"]/statistics_lists["nb_total_demands"])*100.:.2f}%',
            'Overall charging satisf.': f'{np.mean((statistics_lists["nb_charging_80_or_above"]+statistics_lists["nb_charging_below_80"])/statistics_lists["nb_total_demands"])*100.:.2f}±{np.std((statistics_lists["nb_charging_80_or_above"]+statistics_lists["nb_charging_below_80"])/statistics_lists["nb_total_demands"])*100.:.2f}%',
            'Overall charging unsatisf.': f'{np.mean(statistics_lists["nb_rejected_demands"]/statistics_lists["nb_total_demands"])*100.:.2f}±{np.std(statistics_lists["nb_rejected_demands"]/statistics_lists["nb_total_demands"]*100.):.2f}%',
            'Charging auto rejection': f'{np.mean(statistics_lists["nb_auto_rejected"]/statistics_lists["nb_total_demands"])*100.:.2f}±{np.std(statistics_lists["nb_auto_rejected"]/statistics_lists["nb_total_demands"]*100.):.2f}%',
            'Price': f'{np.mean(statistics_lists["electricity_price"]):.2f}±{np.std(statistics_lists["electricity_price"]):.2f}',
            'Grid limit': f'{np.sum(statistics_lists["nb_grid_limit_overtaking"])}',
            'Total Energy': f'{np.mean(statistics_lists["total_energy"]):.2f}±{np.std(statistics_lists["total_energy"]):.2f}',
            'Total PV': f'{np.mean(statistics_lists["total_renewable"]):.2f}±{np.std(statistics_lists["total_renewable"]):.2f}',
            'PV used': f'{np.mean(statistics_lists["renewable_used"]):.2f}±{np.std(statistics_lists["renewable_used"]):.2f}',
            'Grid used': f'{np.mean(statistics_lists["grid_used"]):.2f}±{np.std(statistics_lists["grid_used"]):.2f}',
            'PV used/Total PV': f'{np.mean(statistics_lists["renewable_used"]/statistics_lists["total_renewable"])*100.:.2f}±{np.std(statistics_lists["renewable_used"]/statistics_lists["total_renewable"])*100.:.2f}%',
            'PV used/Total Energy': f'{np.mean(pv_used_over_total_energy)*100.:.2f}±{np.std(pv_used_over_total_energy)*100:.2f}%',
            'Energy/EV': f'{np.mean(energy_per_ev):.2f}±{np.std(energy_per_ev):.2f}',
            'SoC charged': f'{np.mean(statistics_lists["avg_soc_charged"])*100.:.2f}±{np.std(statistics_lists["avg_soc_charged"])*100.:.2f}%',
            'Final SoC': f'{np.mean(statistics_lists["avg_final_soc"])*100.:.2f}±{np.std(statistics_lists["avg_final_soc"])*100.:.2f}%',
            'Waiting Time': f'{np.mean(statistics_lists["avg_waiting_time"]):.2f}±{np.std(statistics_lists["avg_waiting_time"]):.2f}',
            'Obj. cost': f'{np.mean(statistics_lists["objective_cost"]):.2f}±{np.std(statistics_lists["objective_cost"]):.2f}',
            'Detailed obj. cost': [statistics_lists['objective_cost'].copy()],
            'Detailed PV used': [statistics_lists['renewable_used'].copy()],
            'Detailed Total Energy': [statistics_lists['total_energy'].copy()],
            'Detailed Total PV': [statistics_lists['total_renewable'].copy()],
        }

        list_df.append( pd.DataFrame(data_for_df) )
    
    results_df = pd.concat(list_df)
    results_filename = f'analysis_offline_{args.num}.csv'
    results_filepath = os.path.join(args.res_dir, results_filename)
    results_df.to_csv(results_filepath, sep='\t', encoding='utf-8', index=False)
    print(f'Analysis saved to {results_filepath}!')
