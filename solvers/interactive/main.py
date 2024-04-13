import sys
sys.path.append("../../")
from utils import parse_dic_args
import gymnasium as gym
import argparse


def get_next_action():
        charging_requests = None
        while charging_requests is None:
            try:
                charging_requests = list(map(float, input('Charging requests: ').split(' ')))
            except KeyboardInterrupt:
                exit(1)
            except:
                charging_requests = None
        
        power_allocations = None
        while power_allocations is None:
            try:
                power_allocations = list(map(float, input('Power allocations: ').split(' ')))
            except KeyboardInterrupt:
                exit(1)
            except:
                power_allocations = None
        
        return charging_requests+power_allocations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="ChargingStationEnv-v0")
    parser.add_argument("--episode", default=1, type=int)
    parser.add_argument("--schema", default="../../schema.json")
    parser.add_argument("--current_folder", default=None)
    parser.add_argument("--initializer", default="Initializer_FIFO")
    parser.add_argument("--simulation", default="Simulate_Station_FIFO")
    parser.add_argument("--action", default="Simulate_Actions_FIFO")
    parser.add_argument("--energy", default="Energy_Initializer")
    parser.add_argument("--visualizer", default="Console_Rendering")
    parser.add_argument("--initializer_args", default={}, type=parse_dic_args)
    parser.add_argument("--simulation_args", default={}, type=parse_dic_args)
    parser.add_argument("--action_args", default={}, type=parse_dic_args)
    parser.add_argument("--energy_args", default={}, type=parse_dic_args)
    parser.add_argument("--reset_flag", default=0, type=int)
    args = parser.parse_args()

    env = gym.make(args.env,
                   schema=args.schema,
                   current_folder=args.current_folder,
                   results_folder=None,
                   initializer=getattr(getattr(__import__('charging_station_env'), 'initializer'), args.initializer)(**args.initializer_args, energy_initializer=getattr(getattr(__import__('charging_station_env'), 'initializer'), args.energy)(**args.energy_args)),
                   simulation_controller=getattr(getattr(__import__('charging_station_env'), 'transition'), args.simulation)(**args.simulation_args),
                   action_controller=getattr(getattr(__import__('charging_station_env'), 'action'), args.action)(**args.action_args),
                   visualizer=getattr(getattr(__import__('charging_station_env'), 'visualizer'), args.visualizer)(),
                   )
    
    done = False
    state, _ = env.reset(reset_flag=args.reset_flag, id_save=args.episode, algo_name='Interactive')
    rewards_list = []
    ts = 0
    while not done:
        env.render()
        action = get_next_action()
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        rewards_list.append(reward)
        print('\nREWARD:', reward, end='\n\n')
        ts += 1

    print('\nFINISED\n')
    final_reward = sum(rewards_list)
    print(final_reward)

    env.close()
