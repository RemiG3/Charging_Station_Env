import sys
sys.path.append("../../")
from utils import parse_dic_args, get_statistics
sys.path.append("../../solvers/reinforcement_learning/")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import gymnasium as gym
import numpy as np
import pandas as pd
import argparse
import pickle
import os


def get_logs_dataframe(logs_dir):
    i = 0
    df = pd.DataFrame(columns=['Directory_name', 'Algorithm', 'Training history value', 'Training history step', 'Training history walltime'])
    for path, dir, files in sorted(list(os.walk(logs_dir))):
        if len(files) > 0:
            event = EventAccumulator(path=path)
            event.Reload()
            if (len(event.Tags()['scalars']) > 0):
                path = path.replace('\\', '/')
                dir_name = path.split('/')[-2]
                exp_name = '-'.join(dir_name.split('-')[:-1])
                algo_name = exp_name.split('-')[0].split('_')[0]
                training_hist_val = [scalar.value for scalar in event.Scalars('rollout/ep_rew_mean')]
                training_hist_step = [scalar.step for scalar in event.Scalars('rollout/ep_rew_mean')]
                training_hist_time = [scalar.wall_time for scalar in event.Scalars('rollout/ep_rew_mean')]
                df.loc[i] = [dir_name, algo_name, training_hist_val, training_hist_step, training_hist_time]
                i += 1
                print(f'Process log n°{i}', end='\r')
            else:
                print(f'Events not found in {path}')
    print('Logs dataframe created!')
    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", default=1, type=int)
    parser.add_argument("--episodes", default=100, type=int)
    parser.add_argument("--env", default="ChargingStationEnv-v0")
    parser.add_argument("--schema", default="../../schema.json")
    parser.add_argument("--current_folder", default='../../dataset/ev_scenario-50/', type=str)
    parser.add_argument("--eval_dir", default='./ev_scenario-50', type=str)
    parser.add_argument("--analysis_dir", default='./', type=str)
    parser.add_argument("--module_aglorithm", default="mppo_customized_train_with_evaluation", type=str)
    parser.add_argument("--algorithm", default="CustomMPPO", type=str)
    parser.add_argument("--initializer", default="Initializer_FIFO")
    parser.add_argument("--simulation", default="Simulate_Station_FIFO")
    parser.add_argument("--action", default="Simulate_Actions_FIFO")
    parser.add_argument("--energy", default="Energy_Initializer")
    parser.add_argument("--initializer_args", default={}, type=parse_dic_args)
    parser.add_argument("--simulation_args", default={}, type=parse_dic_args)
    parser.add_argument("--action_args", default={}, type=parse_dic_args)
    parser.add_argument("--energy_args", default={}, type=parse_dic_args)
    parser.add_argument("--logs_dir", default='../../solvers/reinforcement_learning/logs', type=str)
    parser.add_argument("--model_folder", default='../../solvers/reinforcement_learning/models', type=str)
    args = parser.parse_args()

    exp_num = args.num
    df = get_logs_dataframe(args.logs_dir)
    nb = len(df)
    for idx, df_row in df.iterrows():
        results_folder = os.path.join(args.eval_dir, df_row["Directory_name"])
        model_module_name = df_row["Algorithm"]
        model_path = f'./{args.model_folder}/{df_row["Directory_name"]}'
        list_dir = os.listdir(model_path)
        if(len(list_dir) > 0):
            if('best_model.zip' in list_dir):
                model_filepath = os.path.join(model_path, 'best_model.zip')
            else:
                model_filepath = os.path.join(model_path, list_dir[-1])

            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
            
            env = gym.make(args.env,
                           schema=args.schema,
                           current_folder=args.current_folder,
                           results_folder=results_folder,
                           initializer=getattr(getattr(__import__('charging_station_env'), 'initializer'), args.initializer)(**args.initializer_args, energy_initializer=getattr(getattr(__import__('charging_station_env'), 'initializer'), args.energy)(**args.energy_args)),
                           simulation_controller=getattr(getattr(__import__('charging_station_env'), 'transition'), args.simulation)(**args.simulation_args),
                           action_controller=getattr(getattr(__import__('charging_station_env'), 'action'), args.action)(**args.action_args),
                           )

            Model_Class = getattr(__import__(args.module_aglorithm), args.algorithm)
            model = Model_Class.load(model_filepath, env=env)

            for ep in range(args.episodes):
                obs, info = env.reset(reset_flag=1, id_save=ep+1, algo_name=df_row["Algorithm"])
                done = False
                while not done:
                    action, states = model.predict(env, obs)
                    obs, reward, done, _, info = env.step(action)
            env.close()
        
        print(f'Evaluate model {idx+1}/{nb}', end='\r')
    

    list_df = []
    for i, (_, df_row) in enumerate(df.iterrows()):
        print(f'Analysis of the model n°{i+1}/{len(df)}', end='\r')
        dir_name = df_row['Directory_name']
        algo_name = df_row['Algorithm']
        statistics_lists = {key: [] for key in ['nb_total_demands', 'nb_accepted_demands', 'nb_rejected_demands', 'nb_auto_rejected', 'nb_charging_below_80', 'nb_charging_80_or_above', 'nb_grid_limit_overtaking', 'electricity_price', 'total_energy', 'total_renewable', 'renewable_used', 'grid_used', 'energy_per_ev', 'avg_waiting_time', 'avg_soc_charged', 'avg_final_soc', 'objective_cost']}

        for scenario_number in range(1, args.episodes + 1):
            try:
                with open(os.path.join(args.current_folder, f'Initial_Values-{scenario_number}.pickle'), 'rb') as f:
                    initial_values = pickle.load(f)
                try:
                    current_root_results_folder = os.path.join(os.getcwd(), args.eval_dir)
                    with open(os.path.join(current_root_results_folder, dir_name, f"{algo_name}-Results-{scenario_number}.pickle"), 'rb') as f:
                        results = pickle.load(f)
                except FileNotFoundError as e:
                    continue
                stats = get_statistics(initial_values, results)
                for key, value in stats.items():
                    statistics_lists[key].append(value)
            except FileNotFoundError as e:
                print(f"WARNING: {e}")
                continue

        if(len(statistics_lists['objective_cost']) == 0):
            continue

        for key in statistics_lists:
            statistics_lists[key] = np.array(statistics_lists[key])

        pv_used_over_total_energy = np.nan_to_num( statistics_lists["renewable_used"]/statistics_lists["total_energy"], nan=0. )
        pv_used_over_total_energy = np.where(pv_used_over_total_energy > 1., 1., pv_used_over_total_energy)
        pv_used_over_total_energy = np.where(pv_used_over_total_energy < 0., 0., pv_used_over_total_energy)
        energy_per_ev = np.nan_to_num( statistics_lists["total_energy"]/statistics_lists["nb_accepted_demands"] )
        data_for_df = {
            'Algorithm': dir_name.split('-')[0],
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
    results_filename = f'analysis_{args.algorithm}_{args.num}.csv'
    results_filepath = os.path.join(args.analysis_dir, results_filename)
    results_df.to_csv(results_filepath, sep='\t', encoding='utf-8', index=False)
    print(f'Analysis {results_filepath} saved!')


