# Analysis Scripts for EV Charging Scenarios

This document provides instructions on how to run analysis scripts for evaluating the results of different EV charging management approaches. Each command should be executed in its respective directory as specified below.

---

## Rule-Based Analysis

### Command:
```bash
for num in 50 60 70 90 120 150; do 
  python analysis_rb.py --num "$num" --eval_dir ./ev_scenario-"$num" --current_folder ../../dataset/ev_scenario-"$num"; 
done
```

### Directory:
- Navigate to the `rule_based` folder before executing the above command.

### Purpose:
This script evaluates the performance of rule-based methods on different EV scenarios. It processes results from the corresponding dataset folder and stores evaluation metrics in the specified evaluation directory.

---

## Rolling-Horizon Analysis

### Command:
```bash
for num in 50 60 70 90 120 150; do 
  python analysis_rolling.py --num "$num" --eval_dir ./ev_scenario-"$num" --current_folder ../../dataset/ev_scenario-"$num"; 
done
```

### Directory:
- Navigate to the `rolling` folder before executing the above command.

### Purpose:
This script analyzes results from the rolling-horizon optimization algorithm. It evaluates scenario data from the dataset folder and saves the evaluation outputs in the specified directory.

---

## Reinforcement Learning Analysis

### Command:
```bash
for num in 50 60 70 90 120 150; do 
  python analysis_ppo.py --num "$num" \
  --schema ../../schema.json \
  --current_folder ../../dataset/ev_scenario-"$num"/ \
  --eval_dir ./ev_scenario-"$num" \
  --analysis_dir ./ \
  --module_algorithm mppo_with_postprocess_train_with_evaluation \
  --algorithm CustomMPPO \
  --logs_dir ../../solvers/reinforcement_learning/logs/eval/ \
  --model_folder ../../solvers/reinforcement_learning/models/eval/ \
  --simulation_args nb_predict_timestep=3,pv_model_path='../../solvers/pv_predictions/xgboost_pv_model_3min.json',input_window_pv=5,price_model_path=None,input_window_price=0 \
  --simulation Simulate_Station_FIFO_without_Price_Prediction; 
done
```

### Directory:
- Navigate to the `reinforcement_learning` folder before executing the above command.

### Purpose:
This script evaluates the performance of reinforcement learning algorithms (e.g., MMPO). It processes logs and models from the training phase and generates evaluation metrics for each EV scenario.

---

## Offline Analysis

### Command:
```bash
for num in 50 60 70 90 120 150; do 
  python analysis_offline.py --current_folder ../../dataset/ev_scenario-"$num" \
  --results_folder ./ev_scenario-"$num" \
  --save metrics_ev_scenario-"$num".pickle \
  --num "$num"; 
done
```

### Directory:
- Navigate to the `offline` folder before executing the above command.

### Purpose:
This script evaluates the offline analysis results for predefined EV scenarios. The metrics are saved as `.pickle` files for further inspection and analysis.

---

## Dataset and Results Structure

- **Dataset Location**: `../../dataset/ev_scenario-*`
- **Results/Evaluation Output**: Saved in respective folders under the current working directory for each method.


