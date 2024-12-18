import sys
sys.path.append("../../")
from utils import get_statistics

import numpy as np
import pandas as pd
import argparse
import pickle
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", default=1, type=int, help="Number of the experiment")
    parser.add_argument("--episodes", default=100, type=int, help="Number of scenarios")
    parser.add_argument("--eval_dir", default='./ev_scenario-50', help="Evaluation directory")
    parser.add_argument("--res_dir", default='./', help="Results directory")
    parser.add_argument("--current_folder", default='../../dataset/ev_scenario-50', help="Current folder for scenarios")
    args = parser.parse_args()

    current_root_results_folder = os.path.join(os.getcwd(), args.eval_dir)
    list_algorithms_name = ['ROLLING']

    list_df = []
    for algo_name in list_algorithms_name:
        statistics_lists = {key: [] for key in ['nb_total_demands', 'nb_accepted_demands', 'nb_rejected_demands', 'nb_auto_rejected', 'nb_charging_below_80', 'nb_charging_80_or_above', 'nb_grid_limit_overtaking', 'electricity_price', 'total_energy', 'total_renewable', 'renewable_used', 'grid_used', 'energy_per_ev', 'avg_waiting_time', 'avg_soc_charged', 'avg_final_soc', 'objective_cost']}

        for scenario_number in range(1, args.episodes + 1):
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
    results_filename = f'analysis_rolling_{args.num}.csv'
    results_filepath = os.path.join(args.res_dir, results_filename)
    results_df.to_csv(results_filepath, sep='\t', encoding='utf-8', index=False)
    print(f'Analysis saved to {results_filepath}!')


