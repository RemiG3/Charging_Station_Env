# Electric Vehicle Charging Scheduling Algorithms

This repository contains scripts and implementations for managing Electric Vehicle (EV) charging using various algorithmic approaches. The main approaches implemented are **Rule-Based Methods**, **Reinforcement Learning**, and **Rolling-Horizon Optimization**. Below is an explanation of the structure and purpose of each method.

---

## Rule-Based Methods

The scripts in the `rule_based` folder implement predefined, logic-based charging strategies. These methods aim to manage charging station operations using a priority-based or baseline approach.

### Usage:
```bash
# Priority Soc-Centric Charging Strategy
for num in 50 60 70 90 120 150; do
  python rb_main.py \
    --schema ../../schema.json \
    --module_algorithm ps2c \
    --algorithm PrioritySocCentricCharging \
    --initializer Initializer \
    --simulation Simulate_Station_FIFO_without_Price_Prediction \
    --action Simulate_Actions_FIFO \
    --current_folder ../../dataset/ev_scenario-"$num"/ \
    --results_folder ../../results/rule_based/ev_scenario-"$num"/ \
    --reset 1 \
    --simulation_args nb_predict_timestep=3,pv_model_path='../../charging_station_env/transition/xgboost_pv_model.json'
done

# Baseline Charging Strategy
for num in 50 60 70 90 120 150; do
  python rb_main.py \
    --schema ../../schema.json \
    --module_algorithm baseline \
    --algorithm Baseline \
    --initializer Initializer \
    --simulation Simulate_Station_FIFO_without_Price_Prediction \
    --action Simulate_Actions_FIFO \
    --current_folder ../../dataset/ev_scenario-"$num"/ \
    --results_folder ../../results/rule_based/ev_scenario-"$num"/ \
    --reset 1 \
    --simulation_args nb_predict_timestep=3,pv_model_path='../../charging_station_env/transition/xgboost_pv_model.json'
done
```

### Purpose:
- **Priority Soc-Centric Charging**: Allocates charging slots based on state-of-charge (SOC) and prioritizes EVs accordingly.
- **Baseline Strategy**: Implements a FIFO (First-In-First-Out) policy as a benchmark for comparison with other strategies.

---

## Reinforcement Learning Algorithm

The `reinforcement_learning` folder contains implementations of RL-based strategies to optimize EV charging policies. These methods train agents using PPO (Proximal Policy Optimization) variants to learn charging actions that maximize rewards over time.

### Usage:
```bash
# MMPO with Post-Processing
python mppo_with_postprocess_train_with_evaluation.py \
  --episodes 10000000 \
  --eval_episodes 500 \
  --eval_freq 20000 \
  --schema ../../schema.json \
  --initializer Initializer \
  --simulation Simulate_Station_FIFO_without_Price_Prediction \
  --action Simulate_Actions_FIFO \
  --energy Energy_Initializer \
  --initializer_args nb_ev_min_range=50,nb_ev_max_range=150 \
  --simulation_args nb_predict_timestep=3,pv_model_path='../../charging_station_env/transition/xgboost_pv_model.json' \
  --learning_rate 0.006 \
  --batch_size 512 \
  --gamma 0.99 \
  --n_steps 4096 \
  --n_epochs 10 \
  --clip_range 0.2 \
  --gae_lambda 0.98 \
  --max_grad_norm 0.97 \
  --ent_coef 0.01 \
  --vf_coef 0.2 \
  --normalize_advantage True \
  --policy_activation tanh \
  --policy_net_pi 512,512 \
  --policy_net_vf 512,512

# MMPO
python mppo_train_with_evaluation.py <same parameters as above>

# PPO
python ppo_customized_train_with_evaluation.py <same parameters as above>
```

### Purpose:
- **MMPO with Post-Processing**: An enhanced version of PPO with post-processing mechanisms for action masking and improved policy optimization.
- **MMPO and PPO**: Standard and baseline PPO implementations for policy learning in EV charging.

---

## Rolling-Horizon Algorithm

The `rolling` folder contains a rolling-horizon optimization implementation for managing EV charging. This method uses a sliding time window approach to optimize actions iteratively.

### Usage:
```bash
for num in 50 60 70 90 120 150; do
  python rolling_main.py \
    --simulation Simulate_Station_FIFO_without_Price_Prediction \
    --action Simulate_Actions_FIFO \
    --simulation_args nb_predict_timestep=0 \
    --current_folder ../../dataset/ev_scenario-"$num"/ \
    --results_folder ../../results/rolling/ev_scenario-"$num"/ \
    --reset 1 \
    --libmodel_name ./lib_online_rolling.so
done
```

### Purpose:
- **Rolling-Horizon Optimization**: Performs step-by-step optimization over a finite horizon to dynamically adjust charging actions in response to current and predicted system states.

---

## Dataset and Results Structure

- **Dataset**: Located at `../../dataset/ev_scenario-*`, containing EV scenario data for different configurations.
- **Results**: Stored in the corresponding folders under `../../results/`.

---

## Key Arguments

- **`--schema`**: Defines the schema for simulation configurations.
- **`--simulation_args`**: Simulation-specific parameters, including prediction timestep and PV model path.
- **`--reset`**: Resets the simulation environment between runs.
- **`--current_folder`** and **`--results_folder`**: Paths to input data and output results, respectively.



