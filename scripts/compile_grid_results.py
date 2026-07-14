import os
import json
import pandas as pd
import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env import is_colab
import argparse

def compile_grid_results():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', type=str, default=None, help="Optional parent grid run ID directory. If not provided, scans for all grid runs.")
    args = parser.parse_args()
    
    IN_COLAB = is_colab()
    if IN_COLAB:
        base_dir = Path('/content/drive/MyDrive/multi-horizon-ofi')
    else:
        base_dir = PROJECT_ROOT
        
    base_results_dir = base_dir / 'results'
    base_weights_dir = base_dir / 'model_weights'
    
    if not base_results_dir.exists():
        print(f"No results directory found at {base_results_dir}")
        return
        
    grid_runs = []
    
    if args.run_id:
        target_dir = base_results_dir / args.run_id
        if target_dir.exists():
            grid_runs.extend([d for d in target_dir.iterdir() if d.is_dir() and (d.name.startswith("trial-") or d.name.startswith("grid_"))])
    else:
        # Scan for stray top-level grid runs and move them to legacy
        stray_runs = [d for d in base_results_dir.iterdir() if d.is_dir() and d.name.startswith("grid_")]
        if stray_runs:
            import shutil
            legacy_res_dir = base_results_dir / "legacy_grid_runs"
            legacy_res_dir.mkdir(exist_ok=True)
            legacy_weights_dir = base_weights_dir / "legacy_grid_runs"
            legacy_weights_dir.mkdir(exist_ok=True, parents=True)
            
            print(f"Found {len(stray_runs)} stray grid runs. Moving them to {legacy_res_dir.name}/")
            for run_dir in stray_runs:
                # Move result dir
                dest_res = legacy_res_dir / run_dir.name
                shutil.move(str(run_dir), str(dest_res))
                
                # Try to move corresponding weights dir
                weight_dir = base_weights_dir / run_dir.name
                if weight_dir.exists():
                    shutil.move(str(weight_dir), str(legacy_weights_dir / run_dir.name))
            
            grid_runs.extend([d for d in legacy_res_dir.iterdir() if d.is_dir() and d.name.startswith("grid_")])

        # Scan for new-style nested trial runs inside parent grid folders
        for parent_dir in base_results_dir.iterdir():
            if parent_dir.is_dir() and "grid" in parent_dir.name.lower() and parent_dir.name != "legacy_grid_runs":
                grid_runs.extend([d for d in parent_dir.iterdir() if d.is_dir() and d.name.startswith("trial-")])
                
    if not grid_runs:
        print("No grid/trial runs found.")
        return
        
    all_results = []
    
    for run_dir in grid_runs:
        for file in run_dir.iterdir():
            if file.name.endswith("_run_metadata.json"):
                try:
                    with open(file, 'r') as f:
                        meta = json.load(f)
                    
                    arch = meta.get("architecture", "Unknown")
                    best_f1 = meta.get("early_stopping", {}).get("best_mean_macro_f1", 0.0)
                    
                    # Compute R2 and Acc from metrics
                    metrics_file = run_dir / f"{arch}_metrics.json"
                    avg_r2 = 0.0
                    avg_acc = 0.0
                    if metrics_file.exists():
                        with open(metrics_file, 'r') as mf:
                            metrics = json.load(mf)
                            r2_keys = [k for k in metrics.keys() if 'r2' in k.lower() and k != 'mean_r2']
                            acc_keys = [k for k in metrics.keys() if 'accuracy' in k.lower()]
                            if r2_keys:
                                avg_r2 = metrics.get('mean_r2', sum(metrics[k] for k in r2_keys) / len(r2_keys))
                            if acc_keys:
                                avg_acc = sum(metrics[k] for k in acc_keys) / len(acc_keys)
                    
                    # Extract the configuration identifier
                    if "trial-" in run_dir.name:
                        parts = run_dir.name.split("trial-")[1].split("_", 1)
                        if len(parts) > 1:
                            config_desc = parts[1]
                        else:
                            config_desc = run_dir.name
                    else:
                        config_desc = run_dir.name
                        
                    run_data = {
                        "Run_ID": run_dir.name,
                        "Config_Desc": config_desc,
                        "Architecture": arch,
                        "Mean_Macro_F1": best_f1,
                        "Avg_Accuracy": avg_acc,
                        "Avg_R2": avg_r2,
                        "Epochs_Ran": meta.get("early_stopping", {}).get("epochs_ran", 0),
                        "Loss_Mode": meta.get("train_config", {}).get("loss_mode", ""),
                        "Normalization": meta.get("train_config", {}).get("normalization_method", ""),
                        "SMOTE": meta.get("train_config", {}).get("enable_smote", ""),
                        "SMOTE_Method": meta.get("train_config", {}).get("smote_method", ""),
                        "Stationarity": meta.get("train_config", {}).get("enable_stationarity", ""),
                        "Seq_Len": meta.get("seq_len", ""),
                        "LR": meta.get("train_config", {}).get("lr", ""),
                        "Batch_Size": meta.get("train_config", {}).get("batch_size", ""),
                    }
                    all_results.append(run_data)
                except Exception as e:
                    print(f"Failed to parse {file}: {e}")
                    
    if not all_results:
        print("No valid metadata files found.")
        return
        
    df = pd.DataFrame(all_results)
    
    # Aggregate by Config_Desc and Architecture
    agg_funcs = {
        "Mean_Macro_F1": ["mean", "std"],
        "Avg_Accuracy": ["mean", "std"],
        "Avg_R2": ["mean", "std"],
        "Epochs_Ran": "mean",
        "Loss_Mode": "first",
        "Normalization": "first",
        "SMOTE": "first",
        "SMOTE_Method": "first",
        "Stationarity": "first",
        "Seq_Len": "first",
        "LR": "first",
        "Batch_Size": "first",
        "Run_ID": "count" # Represents number of trials
    }
    
    grouped = df.groupby(["Config_Desc", "Architecture"]).agg(agg_funcs).reset_index()
    
    # Flatten multi-index columns
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    # Rename columns for clarity
    grouped = grouped.rename(columns={
        "Config_Desc_first": "Config_Desc",
        "Architecture_first": "Architecture",
        "Loss_Mode_first": "Loss_Mode",
        "Normalization_first": "Normalization",
        "SMOTE_first": "SMOTE",
        "SMOTE_Method_first": "SMOTE_Method",
        "Stationarity_first": "Stationarity",
        "Seq_Len_first": "Seq_Len",
        "LR_first": "LR",
        "Batch_Size_first": "Batch_Size",
        "Run_ID_count": "Trials"
    })
    
    grouped = grouped.sort_values(by="Mean_Macro_F1_mean", ascending=False)
    
    if args.run_id:
        out_csv = base_results_dir / args.run_id / "grid_leaderboard_aggregated.csv"
    else:
        out_csv = base_results_dir / "all_grids_leaderboard_aggregated.csv"
        
    grouped.to_csv(out_csv, index=False)
    
    # Create a clean display dataframe
    display_df = grouped.copy()
    display_df["F1 (mean±std)"] = display_df["Mean_Macro_F1_mean"].round(4).astype(str) + " ± " + display_df["Mean_Macro_F1_std"].fillna(0).round(4).astype(str)
    display_df["Acc (mean±std)"] = display_df["Avg_Accuracy_mean"].round(4).astype(str) + " ± " + display_df["Avg_Accuracy_std"].fillna(0).round(4).astype(str)
    display_df["R2 (mean±std)"] = display_df["Avg_R2_mean"].round(4).astype(str) + " ± " + display_df["Avg_R2_std"].fillna(0).round(4).astype(str)
    
    display_cols = ["Architecture", "F1 (mean±std)", "Acc (mean±std)", "R2 (mean±std)", "Loss_Mode", "Normalization", "SMOTE", "Stationarity", "Trials"]
    
    print("\n" + "="*120)
    print("GRID SEARCH AGGREGATED LEADERBOARD (Top 15)")
    print("="*120)
    print(display_df[display_cols].head(15).to_string(index=False))
    print("="*120)
    print(f"\nFull aggregated leaderboard saved to: {out_csv}")

if __name__ == "__main__":
    compile_grid_results()
