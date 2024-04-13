from charging_station_env.visualizer import Rendering_Base


class Console_Rendering(Rendering_Base):
    def __init__(self):
        super().__init__()
    
    def __call__(self, env, obs):
        n = env.number_of_chargers
        ts = env.timestep//env.step_time
        print(f'\n############### Timestep: {ts} ###############\n')

        ######
        # States: [sp_t, pr_t, sp_t+1, sp_t+2, sp_t+3, pr_t+1, pr_t+2, pr_t+3, soc¹, soc², soc³, soc⁴, soc⁵, soc⁶, soc⁷, soc⁸, soc⁹, soc¹⁰, tl¹, tl², tl³, tl⁴, tl⁵, tl⁶, tl⁷, tl⁸, tl⁹, tl¹⁰, soc_cr¹, soc_cr², soc_cr³, soc_cr⁴, soc_cr⁵, soc_cr⁶, soc_cr⁷, soc_cr⁸, soc_cr⁹, soc_cr¹⁰, tcr¹, tcr², tcr³, tcr⁴, tcr⁵, tcr⁶, tcr⁷, tcr⁸, tcr⁹, tcr¹⁰, a_cr¹, a_cr², a_cr³, a_cr⁴, a_cr⁵, a_cr⁶, a_cr⁷, a_cr⁸, a_cr⁹, a_cr¹⁰]
        # Index:  [ 0  ,  1  ,   2   ,   3   ,   4   ,   5   ,   6   ,   7   ,  8  ,  9  ,  10 ,  11 ,  12 ,  13 ,  14 ,  15 ,  16 ,  17  , 18 , 19 , 20 , 21 , 22 , 23 , 24 , 25 , 26 ,  27 ,    28  ,    29  ,    30  ,    31  ,    32  ,    33  ,    34  ,    35  ,    36  ,    37   ,  38 ,  39 ,  40 ,  41 ,  42 ,  43 ,  44 ,  45 ,  46 ,   47 ,  48  ,  49  ,  50  ,  51  ,  52  ,  53  ,  54  ,  55  ,  56  ,   57  ]
        ######
        print('pr_t, pr_t+1, pr_t+2, pr_t+3 = \t', obs[0:4])
        print('sp_t, sp_t+1, sp_t+2, sp_t+3 = \t', obs[4:8])
        offset = 8
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

