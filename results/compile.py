import os
import json
import csv
import glob

grid_dir = "/home/illionar/Projects/multi-horizon-ofi/results/2026-07-15-17-31-55428ac_grid"

compiled_data = []
all_keys = set()

for trial_dir in os.listdir(grid_dir):
    trial_path = os.path.join(grid_dir, trial_dir)
    if os.path.isdir(trial_path):
        for metrics_file in glob.glob(os.path.join(trial_path, "*_metrics.json")):
            model_name = os.path.basename(metrics_file).replace("_metrics.json", "")
            with open(metrics_file, "r") as f:
                data = json.load(f)
            
            row = {
                "trial_dir": trial_dir,
                "model": model_name
            }
            row.update(data)
            all_keys.update(data.keys())
            compiled_data.append(row)

output_file = os.path.join(grid_dir, "compiled_results.csv")
fieldnames = ["trial_dir", "model"] + sorted(list(all_keys))

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(compiled_data)

print(f"Successfully compiled {len(compiled_data)} results into {output_file}")
