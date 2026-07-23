import os, json, glob
import numpy as np

legacy_dirs = glob.glob('/home/illionar/Projects/multi-horizon-ofi/results/legacy_run_*')
data = {}

for d in legacy_dirs:
    for f in glob.glob(os.path.join(d, '*_results*.json')):
        with open(f, 'r') as fh:
            try:
                res = json.load(fh)
                arch = res.get('architecture')
                metrics = res.get('test_metrics', {})
                if not arch or not metrics: continue
                # check if it's all zeros (TEST_RUN)
                if all(v == 0.0 for v in metrics.values()): continue
                
                if arch not in data:
                    data[arch] = []
                data[arch].append(metrics)
            except:
                pass

for arch, runs in data.items():
    print(f"\nArchitecture: {arch} ({len(runs)} runs)")
    for h in [10, 20, 50, 100]:
        accs = [r.get(f'h{h}_accuracy', 0) for r in runs if f'h{h}_accuracy' in r]
        f1ms = [r.get(f'h{h}_f1_macro', 0) for r in runs if f'h{h}_f1_macro' in r]
        f1ws = [r.get(f'h{h}_f1_weighted', 0) for r in runs if f'h{h}_f1_weighted' in r]
        if accs:
            print(f"  H{h} - Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}, F1 Macro: {np.mean(f1ms):.4f} ± {np.std(f1ms):.4f}, F1 Weighted: {np.mean(f1ws):.4f} ± {np.std(f1ws):.4f}")
