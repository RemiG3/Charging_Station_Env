# Charging Station Environment
Charging_Station_Env is an open-source OpenAI Gymnasium environment to simulate the Electric Vehicle Charging Scheduling problem with admission decision control and photovoltaic panels production.
This environment is design for benchmarking online algorithms, such as reinforcement learning algorithms, rolling horizon methods, and rule-based approaches.
This work also includes comparison of online methods against the (optimal) offline solutions using mixed-integer linear programming model.

Paper: **Study of electric vehicle charging scheduling with renewable energy: Offline and stochastic online optimization**

Links:
- DOI: https://doi.org/10.1016/j.ejor.2026.01.015
- Free access (valid until March 10, 2026): https://kwnsfk27.r.eu-west-1.awstrack.me/L0/https:%2F%2Fauthors.elsevier.com%2Fa%2F1mTV31LnJ6%257EjJw/1/0102019bd7fe2df9-b9d76259-16e3-4650-8dc4-d2f80b738f76-000000/AreCO0hPMDKAKQI7SeFBjMDw0ks=461


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

        ├── env.yml

        ├── Readme.md

        ├── requirements.txt

        ├── schema.json

        ├── setup.py

        ├── utils.py

        ├── charging_station_env
            ├── __init__.py
            ├── Charging_Station_Enviroment.py
            ├── action
                ├── __init__.py
                ├── Simulate_Actions_Base.py
                └── Simulate_Actions_FIFO.py
            ├── initializer
                ├── __init__.py
                ├── Energy_Initializer.py
                ├── Energy_Initializer_Base.py
                ├── Initializer_Base.py
                ├── Initializer_FIFO.py
                ├── Initializer_General.py
                ├── pv_production_data
                    └── data.csv
                └── synthetic_data_generator
                    ├── generate_sample.py
                    ├── SDG_sample_generate.py
                    ├── SDG Model (AC,poisson_fit)
                    ├── handles
                        ├── __init__.py
                        └── data_hand.py
                    └── modeling
                        └── stat
                            ├── __init__.py
                            ├── exponential_process.py
                            ├── mixturemodels.py
                            ├── models.py
                            ├── poisson_process.py
                            └── poles_selector.py
            ├── transition
                ├── __init__.py
                ├── Constants.py
                ├── Simulate_Station_Base.py
                ├── Simulate_Station_FIFO.py
                └── Simulate_Station_FIFO_without_Price_Prediction.py
            └── visualizer
                ├── __init__.py
                ├── Console_Rendering.py
                ├── Matplotlib_Rendering.py
                └── Rendering_Base.py

        ├── dataset
            ├── ev_scenario-50
            ├── ev_scenario-60
            ├── ev_scenario-70
            ├── ev_scenario-90
            ├── ev_scenario-120
            └── ev_scenario-150

        ├── results
            ├── Input Data Visualization.ipynb
            ├── Results Visualization.ipynb
            ├── offline
                ├── analysis_offline.py
                ├── solution_checker.py
                └── utils_offline.py
            ├── reinforcement_learning
                └── analysis_ppo.py
            ├── rolling
                └── analysis_rolling.py
            └── rule_based
                └── analysis_rb.py

        └── solvers
            ├── interactive
                ├── automatic_interaction.py
                ├── main.py
                └── scenario_all_1.test
            ├── offline
                ├── CMakeLists.txt
                ├── main.cpp
                └── cmake
                    └── FindGUROBI.cmake
            ├── pv_predictions
                ├── Timeseries_33.967_-6.843_SA3_75kWp_crystSi_14_30deg_8deg_2014_2014.csv
                ├── Timeseries_48.857_2.352_SA3_75kWp_crystSi_14_38deg_-6deg_2014_2014.csv
                ├── training_pv_models_15min.ipynb
                ├── training_pv_models_3min.ipynb
                ├── xgboost_pv_model_15min.json
                └── xgboost_pv_model_3min.json
            ├── reinforcement_learning
                ├── display_trained_models.py
                ├── mppo_train_with_evaluation.py
                ├── mppo_with_postprocess_train_with_evaluation.py
                ├── ppo_customized_train_with_evaluation.py
                └── utils_rl.py
            ├── rolling
                ├── lib_online_rolling.so
                ├── rolling_main.py
                └── 1-Phase rolling
                    ├── CMakeLists.txt
                    ├── online_gurobi.cpp
                    ├── online_gurobi.h
                    └── cmake
                        └── FindGUROBI.cmake
            └── rule_based
                ├── baseline.py
                ├── ps2c.py
                └── rb_main.py


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
    
    "EV_types": {
        "classic": {
            "capacity": 45,
            "chargers_type_compatibilities": ["low"]
        }
    },
    
    "EV_config": {
        "considered_ev": ["classic"]
    },
    
    "Charger_types": {
        "low": {
            "charging_rate": 11,
            "discharging_rate": 11,
            "charging_efficiency": 0.91,
            "discharging_efficiency": 0.91
        }
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
@article{gauchotte2026study,
  title={Study of Electric Vehicle Charging Scheduling with Renewable Energy: Offline and Stochastic Online Optimization},
  author={Gauchotte, R and Oulamara, A and Ghogho, M and Oudani, M},
  journal={European Journal of Operational Research},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.ejor.2026.01.015},
  url={https://doi.org/10.1016/j.ejor.2026.01.015}
}
```
Free access link (valid until March 10, 2026): https://kwnsfk27.r.eu-west-1.awstrack.me/L0/https:%2F%2Fauthors.elsevier.com%2Fa%2F1mTV31LnJ6%257EjJw/1/0102019bd7fe2df9-b9d76259-16e3-4650-8dc4-d2f80b738f76-000000/AreCO0hPMDKAKQI7SeFBjMDw0ks=461

This research work is inspired from the previous environment called "Chargym":
[https://github.com/georkara/Chargym-Charging-Station](https://github.com/georkara/Chargym-Charging-Station)


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
