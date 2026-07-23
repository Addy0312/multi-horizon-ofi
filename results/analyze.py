import csv
from collections import defaultdict
import math

# Calculate mean and std
def calc_mean_std(values):
    if not values: return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(var)

filepath = '/home/illionar/Projects/multi-horizon-ofi/results/2026-07-15-17-31-55428ac_grid/compiled_results.csv'

data = defaultdict(lambda: defaultdict(list))
horizons = ['h10', 'h20', 'h50', 'h100']
metrics = ['accuracy', 'f1_macro', 'f1_weighted']

with open(filepath, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        model = row['model']
        for h in horizons:
            for m in metrics:
                col = f"{h}_{m}"
                if col in row and row[col]:
                    data[model][col].append(float(row[col]))

print("# Model Performance Summary (Averaged over 3 trials)\n")

for h in horizons:
    print(f"## Horizon: {h.replace('h', '')}")
    print("| Model | Accuracy | F1 Macro | F1 Weighted |")
    print("|---|---|---|---|")
    for model in sorted(data.keys()):
        acc_vals = data[model].get(f"{h}_accuracy", [])
        f1_mac_vals = data[model].get(f"{h}_f1_macro", [])
        f1_w_vals = data[model].get(f"{h}_f1_weighted", [])
        
        acc_mean, acc_std = calc_mean_std(acc_vals)
        f1m_mean, f1m_std = calc_mean_std(f1_mac_vals)
        f1w_mean, f1w_std = calc_mean_std(f1_w_vals)
        
        print(f"| {model} | {acc_mean:.4f} ± {acc_std:.4f} | {f1m_mean:.4f} ± {f1m_std:.4f} | {f1w_mean:.4f} ± {f1w_std:.4f} |")
    print("\n")
