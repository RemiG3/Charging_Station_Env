import sys
sys.path.append("../../")
from utils import parse_dic_args
import gymnasium as gym
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="ChargingStationEnv-v0")
    parser.add_argument("--episodes", default=100, type=int)
    parser.add_argument("--schema", default="../../schema.json")
    parser.add_argument("--current_folder", default="../../dataset/ev_scenario-50/")
    parser.add_argument("--results_folder", default="../../results/rule_based/ev_scenario-50/")
    parser.add_argument("--module_algorithm", default="baseline", type=str)
    parser.add_argument("--algorithm", default="Baseline", type=str)
    parser.add_argument("--initializer", default="Initializer_FIFO")
    parser.add_argument("--simulation", default="Simulate_Station_FIFO")
    parser.add_argument("--action", default="Simulate_Actions_FIFO")
    parser.add_argument("--energy", default="Energy_Initializer")
    parser.add_argument("--visualizer", default=None)
    parser.add_argument("--initializer_args", default={}, type=parse_dic_args)
    parser.add_argument("--simulation_args", default={}, type=parse_dic_args)
    parser.add_argument("--action_args", default={}, type=parse_dic_args)
    parser.add_argument("--energy_args", default={}, type=parse_dic_args)
    parser.add_argument("--reset_flag", default=1, type=int) # reset_flag=0 for new generation and reset_flag=1 for same day
    args = parser.parse_args()

    current_folder = args.current_folder
    results_folder = args.results_folder
    algo_name = args.module_algorithm.upper()

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

    rewards_list = []
    for ep in range(args.episodes):
        done = False
        algo = getattr(__import__(args.module_algorithm), args.algorithm)()
        state = env.reset(reset_flag=args.reset_flag, id_save=ep+1, algo_name=algo_name)

        while not done:
            if (args.visualizer is not None):
                env.render()
            action = algo.select_action(env.env, state)
            next_state, rewards, done, _, info = env.step(action)
            state = next_state
            rewards_list.append(rewards)

    print('Total rewards:', sum(rewards_list))
