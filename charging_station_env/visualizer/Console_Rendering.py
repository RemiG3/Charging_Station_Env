from charging_station_env.visualizer import Rendering_Base


class Console_Rendering(Rendering_Base):
    def __init__(self):
        super().__init__()
    
    def __call__(self, env, obs):
        n = env.number_of_chargers
        ts = env.timestep//env.step_time
        print(f'\n############### Timestep: {ts} ###############\n')
        
        print('pr_t', end='')
        for i in range(env.simulation_controller.nb_predict_timestep):
            print(', ', end='')
            print(f'pr_t+{i}', end='')
        print(' = \t', obs[0:env.simulation_controller.nb_predict_timestep+1])
        offset = env.simulation_controller.nb_predict_timestep+1
        print('pv_t', end='')
        for i in range(env.simulation_controller.nb_predict_timestep):
            print(', ', end='')
            print(f'pv_t+{i}', end='')
        print(' = \t', obs[offset:offset+env.simulation_controller.nb_predict_timestep+1])
        offset = offset+env.simulation_controller.nb_predict_timestep+1
        print('Current SOC = \t', obs[offset:offset+n])
        if not env.preemptive_charging:
            print('Are charging = \t', obs[offset+n:offset+2*n])
            print('Have charged = \t', obs[offset+2*n:offset+3*n])
            offset += 2*n
            if env.use_V2G:
                print('Are discharging = \t', obs[offset+n:offset+2*n])
                print('Have discharged = \t', obs[offset+2*n:offset+3*n])
                offset += 2*n
        print('Remaining time = \t', obs[offset+n:offset+2*n])
        print('SOC of waiting EVs = \t', obs[offset+2*n:offset+3*n])
        print('Remaining time of waiting EVs = \t', obs[offset+3*n:offset+4*n])
        print('SOC of new charging requests = \t', obs[offset+4*n:offset+5*n])
        print('Remaining time of new charging requests = \t', obs[offset+5*n:])

