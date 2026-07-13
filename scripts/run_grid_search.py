import os
import sys
import torch
import datetime
import itertools
import argparse
from pathlib import Path
import random

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env import is_colab
from src.utils.seed import deep_set_seed
from src.training.trainer import run_all_deep_models_day_by_day_stable_earlystop

def run_grid():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help="Run in nuked testing mode to verify the grid search end-to-end.")
    args = parser.parse_args()
    
    IN_COLAB = is_colab()
    
    # Define the Permutations & Combinations (Grid)
    grid_params = {
        "loss_mode": ["ce", "cb_focal"],
        "normalization_method": ["robust", "zscore"],
        "enable_smote": [False, True],
        "smote_method": ["smote", "adasyn"],
        "enable_stationarity": [False, True],
        "lr": [3e-4, 1e-4],
        "seq_len": [50, 100],
        "batch_size": [128, 256],
        "randomize_split": [True],
        "run_architectures": [
            ["mlp_baseline", "tcn_baseline", "dilated_transformer", "hybrid_cnn_inception_lstm"]
        ]
    }
    
    # Number of random seeds to test per configuration
    num_trials = 3
    base_seeds = [42, 100, 999]
    
    TEST_MODE = args.test
    
    keys = list(grid_params.keys())
    combinations = list(itertools.product(*[grid_params[k] for k in keys]))
    
    # Filter combinations (if smote is false, don't test different smote methods)
    filtered_combinations = []
    seen = set()
    for combo in combinations:
        c_dict = dict(zip(keys, combo))
        if not c_dict["enable_smote"]:
            c_dict["smote_method"] = "none" # Unify them to avoid duplicate runs
        
        # Hashable representation
        h_rep = tuple(sorted([(k, v) for k,v in c_dict.items() if k != "run_architectures"]))
        if h_rep not in seen:
            seen.add(h_rep)
            filtered_combinations.append(c_dict)
            
    if TEST_MODE:
        filtered_combinations = filtered_combinations[:2] # Only run 2 configurations in test mode
        num_trials = 1 # Only 1 trial in test mode
        base_seeds = [42]
    
    print(f"Total unique configurations to test: {len(filtered_combinations)}")
    print(f"Trials per configuration: {num_trials}")
    print(f"Total pipeline runs: {len(filtered_combinations) * num_trials}")
    
    base_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    
    for idx, current_cfg_overrides in enumerate(filtered_combinations):
        
        # Build a descriptive run name
        desc_parts = []
        for k, v in sorted(current_cfg_overrides.items()):
            if k in ["run_architectures", "randomize_split"]: continue
            val_str = str(v)
            if isinstance(v, bool): val_str = "T" if v else "F"
            # Format float without scientific notation if possible or just stringify
            if isinstance(v, float): val_str = f"{v:.1e}"
            desc_parts.append(f"{k[:4]}-{val_str}")
            
        desc = "_".join(desc_parts)
        
        print(f"\n{'='*80}")
        print(f"Running Grid Configuration {idx+1}/{len(filtered_combinations)}")
        print(f"Params: {current_cfg_overrides}")
        
        for trial in range(num_trials):
            current_seed = base_seeds[trial]
            run_id = f"grid_{base_timestamp}_trial-{trial+1}_{desc}"
            
            print(f"\n  --- Trial {trial+1}/{num_trials} (Seed: {current_seed}) ---")
            print(f"  Run ID: {run_id}")
            
            # Build directories
            if IN_COLAB:
                drive_dir = Path('/content/drive/MyDrive/multi-horizon-ofi')
                DATA_DIR    = str(drive_dir / 'data' / 'processed')
                WEIGHTS_DIR = str(drive_dir / 'model_weights' / run_id)
                RESULTS_DIR = str(drive_dir / 'results' / run_id)
            else:
                DATA_DIR    = str(PROJECT_ROOT / 'data' / 'processed')
                WEIGHTS_DIR = str(PROJECT_ROOT / 'model_weights' / run_id)
                RESULTS_DIR = str(PROJECT_ROOT / 'results' / run_id)
            
            Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
            Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
            
            # Base config
            DEEP_CONFIG = {
                "data_dir":    DATA_DIR,
                "weights_dir": WEIGHTS_DIR,
                "results_dir": RESULTS_DIR,
                "tickers": None,
                "horizons": [10, 20, 50, 100],
                "alpha": 0.00005,
                "seq_len": 100,
                "train_file_fraction": 0.8,
                "base_subsample":             4,
                "high_pressure_subsample":    8,
                "critical_pressure_subsample": 16,
                "max_rows_per_day_train":  8000,
                "max_rows_per_day_eval":  10000,
                "max_files_per_ticker":       0,
                "batch_size":   256,
                "num_workers":  0,
                "lr":            3e-4,
                "weight_decay":  1e-4,
                "grad_clip":     1.0,
                "amp":           True,
                "loss_mode":       "cb_focal",
                "cb_beta":          0.999,
                "cb_min_w":         0.5,
                "cb_max_w":         3.0,
                "cb_eps":           1.0,
                "focal_gamma":      2.0,
                "label_smoothing":  0.02,
                "seed": current_seed,
                "run_architectures": ["mlp_baseline"],
                "result_suffix": "",
                "normalization_method": "robust",
                "enable_smote":   False,
                "smote_method":   "smote",
                "smote_k":        5,
                "smote_min_per_class": 10,
                "enable_stationarity": False,
                "stationarity_method": "auto",
                "stationarity_d_fixed": 0.4,
                "randomize_split": True,
            }
            
            # Apply overrides
            DEEP_CONFIG.update(current_cfg_overrides)
            # Ensure the trial seed is used
            DEEP_CONFIG["seed"] = current_seed
            
            if TEST_MODE:
                DEEP_CONFIG["max_rows_per_day_train"] = 200
                DEEP_CONFIG["max_rows_per_day_eval"] = 200
                DEEP_CONFIG["max_files_per_ticker"] = 2
                DEEP_CONFIG["train_file_fraction"] = 0.5
                DEEP_CONFIG["batch_size"] = 32
                max_epochs = 1
            else:
                max_epochs = 10
                
            deep_set_seed(DEEP_CONFIG["seed"])
            
            # Execute
            try:
                run_all_deep_models_day_by_day_stable_earlystop(
                    DEEP_CONFIG,
                    max_epochs=max_epochs,
                    patience=3,
                    min_delta=1e-4
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"FAILED on combo {run_id}: {e}")
                continue
                
    print("\nGrid Search Complete!")

if __name__ == "__main__":
    run_grid()
