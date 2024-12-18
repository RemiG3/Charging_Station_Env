from utils_rl import parse_dic_args
import gymnasium as gym
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="ChargingStationEnv-v0")
    parser.add_argument("--module_algorithm", default="mppo_customized_train_with_evaluation", type=str)
    parser.add_argument("--algorithm", default="CustomMPPO", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--episodes", default=1000, type=int)
    parser.add_argument("--schema", default="../../schema.json")
    parser.add_argument("--initializer", default="Initializer")
    parser.add_argument("--simulation", default="Simulate_Station_FIFO")
    parser.add_argument("--action", default="Simulate_Actions_FIFO")
    parser.add_argument("--energy", default="Energy_Initializer")
    parser.add_argument("--visualizer", default="Matplotlib_Rendering")
    parser.add_argument("--initializer_args", default={}, type=parse_dic_args)
    parser.add_argument("--simulation_args", default={}, type=parse_dic_args)
    parser.add_argument("--action_args", default={}, type=parse_dic_args)
    parser.add_argument("--energy_args", default={}, type=parse_dic_args)
    args = parser.parse_args()

    assert args.model_path is not None

    env = gym.make(args.env,
                   schema=args.schema,
                   initializer=getattr(getattr(__import__('charging_station_env'), 'initializer'), args.initializer)(**args.initializer_args, energy_initializer=getattr(getattr(__import__('charging_station_env'), 'initializer'), args.energy)(**args.energy_args)),
                   simulation_controller=getattr(getattr(__import__('charging_station_env'), 'transition'), args.simulation)(**args.simulation_args),
                   action_controller=getattr(getattr(__import__('charging_station_env'), 'action'), args.action)(**args.action_args),
                   visualizer=getattr(getattr(__import__('charging_station_env'), 'visualizer'), args.visualizer)(),
                   )

    Model_Class = getattr(__import__(args.module_algorithm), args.algorithm)
    model = Model_Class.load(args.model_path, env=env)

    for ep in range(args.episodes):
        obs, info = env.reset(reset_flag=0, id_save=ep+1, algo_name=args.algorithm.upper())
        done = False
        while not done:
            action, states = model.predict( env, obs )
            obs, reward, done, _, info = env.step( action )
            env.render()

    env.close()


