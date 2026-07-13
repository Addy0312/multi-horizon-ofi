import os
import sys
import torch
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env import is_colab
from src.utils.seed import deep_set_seed
from src.training.trainer import run_all_deep_models_day_by_day_stable_earlystop

def main():
    parser = argparse.ArgumentParser(description="Run deep learning training pipeline for Multi-Horizon OFI.")
    parser.add_argument('--test', action='store_true', help="Run in nuked testing mode to verify the pipeline ends-to-end.")
    args = parser.parse_args()

    IN_COLAB = is_colab()
    
    if IN_COLAB:
        # Note: In Colab, the drive must be mounted in a notebook cell BEFORE running this script!
        # drive.mount('/content/drive') will crash if run inside a !python subprocess.
        # Store data, weights, and results on Google Drive
        drive_dir = Path('/content/drive/MyDrive/multi-horizon-ofi')
        DATA_DIR    = str(drive_dir / 'data' / 'processed')
        WEIGHTS_DIR = str(drive_dir / 'model_weights')
        RESULTS_DIR = str(drive_dir / 'results')
    else:
        DATA_DIR    = str(PROJECT_ROOT / 'data' / 'processed')
        WEIGHTS_DIR = str(PROJECT_ROOT / 'model_weights')
        RESULTS_DIR = str(PROJECT_ROOT / 'results')
    
    Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    if not os.path.isdir(DATA_DIR):
        print(f"WARNING: DATA_DIR not found: {DATA_DIR}. Please check data path.")
        
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
        "seed": 42,
        "run_architectures": ["dilated_transformer", "hybrid_cnn_inception_lstm", "seq2seq_attn"],
        "result_suffix": "_stable_es",
        "normalization_method": "robust",
        "enable_smote":   False,
        "smote_method":   "smote",
        "smote_k":        5,
        "smote_min_per_class": 10,
        "enable_stationarity": False,
        "stationarity_method": "auto",
        "stationarity_d_fixed": 0.4,
    }

    if args.test:
        print("\n" + "!" * 50)
        print("  WARNING: RUNNING IN NUKED TEST MODE  ")
        print("!" * 50 + "\n")
        DEEP_CONFIG["max_rows_per_day_train"] = 200
        DEEP_CONFIG["max_rows_per_day_eval"] = 200
        DEEP_CONFIG["max_files_per_ticker"] = 1
        DEEP_CONFIG["batch_size"] = 32
        DEEP_CONFIG["result_suffix"] = "_TEST_RUN"
        max_epochs = 1
    else:
        max_epochs = 10

    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Architectures: {DEEP_CONFIG['run_architectures']}")
    
    deep_set_seed(DEEP_CONFIG["seed"])
    
    # Run pipeline
    run_all_deep_models_day_by_day_stable_earlystop(
        DEEP_CONFIG,
        max_epochs=max_epochs,
        patience=3,
        min_delta=1e-4
    )

if __name__ == "__main__":
    main()
