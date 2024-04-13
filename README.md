# Charging Station Environment
Charging_Station_Env is an open-source OpenAI Gymnasium environment to simulate the Electric Vehicle Charging Scheduling problem with admission decision control and photovoltaic panels production.
This environment is design for benchmarking online algorithms, such as reinforcement learning algorithms, rolling horizon methods, and rule-based approaches.
This work also includes comparison of online methods against the (optimal) offline solutions using mixed-integer linear programming models.


## Installation

Command for installation (with pip):
```console
cd Charging_Station_Env
pip install -r requirements.txt
pip install -e .
```

Command for installation (with conda):
```console
cd Charging_Station_Env
conda env create -f env.yml
pip install -e .
```


## Launch

Command to test the installation:
```console
cd solvers/interactive
python main.py
```

Change the parameter to visualize charging behaviour of methods:
```console
cd solvers/rule_based
python rb_main.py --visualizer Matplotlib_Rendering
```


## Files

    Charging_Station_Env

        ├── setup.py

        ├── env.yml

        ├── requirements.txt

        ├── schema.json

        ├── Readme.md

        ├── utils.py

        ├── charging_station_env
            ├── __init__.py
            ├── Charging_Station_Enviroment.py
            ├── visualizer
                ├── __init__.py
                ├── Matplotlib_Rendering.py
                ├── Rendering_Base.py
                └── Console_Rendering.py
            ├── action
                ├── Simulate_Actions_FIFO.py
                ├── Simulate_Actions_Base.py
                └── __init__.py
            └── initializer
                ├── Initializer_FIFO.py
                ├── Initializer_Base.py
                ├── Energy_Initializer.py
                ├── __init__.py
                ├── Energy_Initializer_Base.py
                └── synthetic_data_generator
                    ├── SDG Model (AC,poisson_fit)
                    ├── generate_sample.py
                    ├── SDG_sample_generate.py
                    ├── modeling
                        └── stat
                            ├── models.py
                            ├── exponential_process.py
                            ├── mixturemodels.py
                            ├── poisson_process.py
                            ├── poles_selector.py
                            └── __init__.py
                    └── handles
                        ├── data_hand.py
                        └── __init__.py
                ├── pv_production_data
                    └── data.csv
            └── transition
                ├── __init__.py
                ├── Simulate_Station_Base.py
                ├── Simulate_Station_FIFO.py
                └── Constants.py

        ├── solvers
            ├── reinforcement_learning
                ├── mppo_customized_train_with_evaluation.py
                ├── ppo_customized_train_with_evaluation.py
                ├── utils_rl.py
                ├── display_trained_models.py
                └── CustomActorCriticPolicies.py
            ├── interactive
                ├── automatic_interaction.py
                ├── main.py
                └── scenario_all_1.test
            ├── offline
                ├── 1-Phase
                    ├── main.cpp
                    ├── CMakeLists.txt
                    └── cmake
                        └── FindCPLEX.cmake
                ├── 2-Phase
                    ├── main.cpp
                    ├── CMakeLists.txt
                    └── cmake
                        └── FindCPLEX.cmake
                └── 3-Phase
                    ├── main.cpp
                    ├── CMakeLists.txt
                    └── cmake
                        └── FindCPLEX.cmake
            ├── rule_based
                ├── baseline.py
                ├── ps2c.py
                └── rb_main.py
            └── rolling
                ├── lib_online_rolling.so
                ├── rolling_main.py
                └── 1-Phase rolling
                    ├── online_cplex.cpp
                    ├── online_cplex.h
                    ├── CMakeLists.txt
                    └── cmake
                        └── FindCPLEX.cmake

        ├── dataset
            ├── ev_scenario-50
            ├── ev_scenario-60
            ├── ev_scenario-70
            ├── ev_scenario-90
            ├── ev_scenario-120
            └── ev_scenario-150

        └── results
            ├── Input Data Visualization.ipynb
            ├── Results Visualization.ipynb
            ├── rolling
                └── analysis_rolling.py
            ├── rule_based
                └── analysis_rb.py
            ├── reinforcement_learning
                └── analysis_ppo.py
            └── offline
                ├── analysis_offline.py
                ├── utils_offline.py
                └── solution_checker.py


## Customized Environment
Customizable configuration file:
```
{
    "schema_name": "Identical_chargers+Grid_limit__constant_charging__15_timestep",
    "seed": 0,
    
    "step_time": 15,
    "number_of_days": 1,
    "price_flag": "random",
    "solar_flag": 1,
    "grid_limit": 75.0,
    
    "Types_of_EV": {
        "classic": {
            "capacity": 45,
            "chargers_type_compatibilities": ["low", "medium", "high"]
        },
    },
    
    "EV_config": {
        "considered_ev": ["classic"]
    },
    
    "Types_of_chargers": {
        "low": {
            "charging_rate": 11,
            "discharging_rate": 11,
            "charging_efficiency": 0.91,
            "discharging_efficiency": 0.91
        },
    },
    
    "Chargers_config": {
        "list_chargers": ["low", "low", "low", "low", "low", "low", "low", "low", "low", "low"],
        
        "preemptive": 1,
        "comment on preemptive": "value in {0, 1}",

        "use_V2G": 0,
        "comment on use_V2G": "value in {0, 1}",

        "charging_mode": "constant",
        "comment on charging_mode": "value in [variable, constant, discrete]"
    },
    
    "data": {
        "sdg_path": "../../charging_station_env/initializer/synthetic_data_generator",
        "sdg_model_name": "ACpoisson_fit",
        "pv_data_path": "../../charging_station_env/initializer/pv_production_data/data.csv"
    }
}
```


## Citation
Please cite the attached paper if you use this environment in your work:
```

```

This research work is inspired from the previous environment called "Chargym":
```
https://github.com/georkara/Chargym-Charging-Station
```


## License
MIT License

Copyright (c) 2024 Rémi Gauchotte

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
