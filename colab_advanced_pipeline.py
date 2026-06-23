import os
import gc
import json
import time
import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, f1_score, classification_report
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from statsmodels.tsa.stattools import adfuller

# ---------------------------------------------------------
# 1. SETUP & QoL FEATURES
# ---------------------------------------------------------
def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

if is_colab():
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    PROJECT_ROOT = Path('/content/drive/MyDrive/multi-horizon-ofi')
else:
    PROJECT_ROOT = Path(os.getcwd())

DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
TODAY_STR = datetime.now().strftime('%Y-%m-%d')
RESULTS_DIR = PROJECT_ROOT / 'results' / TODAY_STR
WEIGHTS_DIR = PROJECT_ROOT / 'model_weights' / TODAY_STR

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
print(f"Results will be saved to: {RESULTS_DIR}")

# ---------------------------------------------------------
# 2. STATIONARITY CHECK
# ---------------------------------------------------------
def check_stationarity(series: np.ndarray, sample_size: int = 10000) -> dict:
    """Check stationarity using Augmented Dickey-Fuller test on a subset."""
    series = series[~np.isnan(series)]
    if len(series) > sample_size:
        series = np.random.choice(series, sample_size, replace=False)
    
    if len(series) < 10:
        return {"is_stationary": False, "p_value": 1.0}
        
    result = adfuller(series)
    return {
        "adf_statistic": float(result[0]),
        "p_value": float(result[1]),
        "is_stationary": bool(result[1] < 0.05)
    }

# ---------------------------------------------------------
# 3. PREPROCESSING & SCALING & SMOTE
# ---------------------------------------------------------
def preprocess_and_scale(X: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """Flattens, scales, and reshapes back to 3D."""
    n_samples, seq_len, n_features = X.shape
    X_2d = X.reshape(-1, n_features)
    
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'zscore':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
        
    X_scaled_2d = scaler.fit_transform(X_2d)
    return X_scaled_2d.reshape(n_samples, seq_len, n_features)

def advanced_smote_preprocess(X: np.ndarray, y: np.ndarray, target_idx: int = -1):
    """
    Applies SMOTE to balance the classes. 
    In real continuous data, SMOTE performs better when data is properly scaled 
    and anomalies are smoothed out.
    """
    n_samples, seq_len, n_features = X.shape
    
    # Target label extraction
    y_target = y[:, target_idx] if y.ndim > 1 else y
    
    # We only SMOTE up to a safe number of samples to avoid RAM spikes
    max_samples = 30000
    if n_samples > max_samples:
        indices = np.random.choice(n_samples, max_samples, replace=False)
        X_sub = X[indices]
        y_sub = y_target[indices]
    else:
        X_sub = X
        y_sub = y_target
        
    X_2d = X_sub.reshape(X_sub.shape[0], -1)
    
    smote = SMOTE(random_state=42)
    X_res_2d, y_res = smote.fit_resample(X_2d, y_sub)
    
    X_res_3d = X_res_2d.reshape(X_res_2d.shape[0], seq_len, n_features)
    return X_res_3d, y_res

# ---------------------------------------------------------
# 4. DATASET DEFINITION & R2 TARGET GENERATION
# ---------------------------------------------------------
class AdvancedDataset(Dataset):
    def __init__(self, X: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32) if y_reg is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y_reg is not None:
            return self.X[idx], self.y_cls[idx], self.y_reg[idx]
        return self.X[idx], self.y_cls[idx]

# Dummy Data Generator for completely self-contained execution
def generate_dummy_data():
    X = np.random.randn(5000, 10, 10).astype(np.float32) 
    y_reg = np.random.randn(5000, 4).astype(np.float32)
    # Threshold for classification
    y_cls = np.zeros_like(y_reg, dtype=np.int64)
    y_cls[y_reg > 0.5] = 2
    y_cls[y_reg < -0.5] = 0
    y_cls[(y_reg >= -0.5) & (y_reg <= 0.5)] = 1
    return X, y_cls, y_reg

# ---------------------------------------------------------
# 5. ROBUST TRAINER WITH CHECKPOINTING
# ---------------------------------------------------------
def calculate_r2(y_true, y_pred):
    try:
        return r2_score(y_true, y_pred)
    except:
        return float('nan')

def run_advanced_training(arch_name, model, dataloader, epochs=5, is_regression=False):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.02)
    criterion_reg = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    checkpoint_path = WEIGHTS_DIR / f"{arch_name}_checkpoint.pt"
    
    start_epoch = 1
    best_loss = float('inf')
    epoch_results = []
    
    # Robust load check
    if checkpoint_path.exists():
        print(f"Resuming {arch_name} from checkpoint...")
        chk = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(chk['model'])
        optimizer.load_state_dict(chk['optimizer'])
        
        # Guard: explicitly move loaded optimizer states to correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)
                    
        start_epoch = chk['epoch'] + 1
        best_loss = chk['best_loss']
        epoch_results = chk.get('history', [])
        
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0
        all_preds = []
        all_trues = []
        
        for batch in dataloader:
            # Unpack batch properly
            if is_regression:
                X_b, _, y_b = batch
            else:
                # For classification: may have 2 or 3 items
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    X_b, y_b, _ = batch
                else:
                    X_b, y_b = batch

            X_b = X_b.to(DEVICE)
            y_b = y_b.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                preds = model(X_b)
                if is_regression:
                    # R2 target: the dummy y_reg has shape (batch, 4) horizons! 
                    # We configured our SimpleLSTM to output shape (batch, 1). 
                    # So we must compare prediction against just the first horizon y_b[:, 0].
                    y_b_reg = y_b[:, 0] if y_b.dim() > 1 else y_b
                    loss = criterion_reg(preds.squeeze(), y_b_reg)
                else:
                    # Classification: y_b should be 1D (class indices)
                    loss = criterion_cls(preds, y_b)
            
            if not torch.isfinite(loss):
                continue
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            all_preds.append(preds.detach().cpu().numpy())
            all_trues.append(y_b.detach().cpu().numpy())
            
        avg_loss = total_loss / max(1, len(dataloader))
        print(f"[{arch_name}] Epoch {epoch} Loss: {avg_loss:.4f}")
        
        # Calculate R2 or F1
        if all_preds:
            preds_concat = np.concatenate(all_preds, axis=0)
            
            if is_regression:
                # Truncate the ground truths to just horizon 0
                trues_concat = np.concatenate([t[:, 0] if t.ndim > 1 else t for t in all_trues], axis=0)
            else:
                trues_concat = np.concatenate(all_trues, axis=0)
            
            metrics = {'avg_loss': avg_loss}
            if is_regression:
                # For R2: flatten both to 1D for proper scoring
                preds_flat = preds_concat.flatten()
                trues_flat = trues_concat.flatten()
                r2 = calculate_r2(trues_flat, preds_flat)
                metrics['r2_score'] = float(r2) if not np.isnan(r2) else 0.0
                print(f"[{arch_name}] Epoch {epoch} R2 Score: {metrics['r2_score']:.4f}")
            else:
                # For classification: argmax on logits
                preds_cls = np.argmax(preds_concat, axis=-1)
                f1 = f1_score(trues_concat, preds_cls, average='macro', zero_division=0)
                metrics['f1_macro'] = float(f1)
                print(f"[{arch_name}] Epoch {epoch} F1 Macro: {metrics['f1_macro']:.4f}")
                
            epoch_results.append(metrics)
        
        # Checkpointing
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss,
                'history': epoch_results
            }, checkpoint_path)
            
    # Save final results individually
    res_path = RESULTS_DIR / f"{arch_name}_results.json"
    with open(res_path, 'w') as f:
        json.dump({
            "architecture": arch_name,
            "best_loss": best_loss,
            "history": epoch_results
        }, f, indent=2)
        
    return epoch_results

# ---------------------------------------------------------
# 6. SIMPLE MODEL FOR TESTING
# ---------------------------------------------------------
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, is_regression=False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, output_dim)
        
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


# ---------------------------------------------------------
# 7. MAIN PIPELINE
# ---------------------------------------------------------
def main():
    try:
        print("Generating Dummy Continuous Data...")
        X, y_cls, y_reg = generate_dummy_data()
        
        # 1. Stationarity
        print("\n--- Checking Stationarity ---")
        stat_res = check_stationarity(X[:, :, 0].flatten())
        print(json.dumps(stat_res, indent=2))
        
        # Save stationarity results
        stat_path = RESULTS_DIR / "stationarity_check.json"
        with open(stat_path, 'w') as f:
            json.dump(stat_res, f, indent=2)
        
        # 2. Scaling Method Comparisons
        print("\n--- Applying advanced Scalers ---")
        X_minmax = preprocess_and_scale(X, method='minmax')
        X_robust = preprocess_and_scale(X, method='robust')
        print(f"MinMax scaled shape: {X_minmax.shape}")
        print(f"Robust scaled shape: {X_robust.shape}")
        
        # 3. SMOTE implementation
        print("\n--- Applying robust SMOTE ---")
        X_smote, y_smote = advanced_smote_preprocess(X_minmax, y_cls, target_idx=0)
        print(f"SMOTE Output shapes - X: {X_smote.shape}, Y: {y_smote.shape}")
        
        # Save SMOTE results
        smote_path = RESULTS_DIR / "smote_preprocessing.json"
        with open(smote_path, 'w') as f:
            json.dump({
                "original_shape": f"{X.shape}",
                "smote_output_shape": f"{X_smote.shape}",
                "class_distribution": {str(k): int(v) for k, v in zip(*np.unique(y_smote, return_counts=True))}
            }, f, indent=2)
        
        # 4. Training Regression (to get R2)
        print("\n--- Training Regression Model (R2 Target) ---")
        reg_dataset = AdvancedDataset(X_minmax, y_cls, y_reg)
        reg_loader = DataLoader(reg_dataset, batch_size=128, shuffle=True)
        reg_model = SimpleLSTM(input_dim=10, output_dim=1, is_regression=True)
        run_advanced_training("lstm_regressor_minmax", reg_model, reg_loader, epochs=3, is_regression=True)
        
        # 5. Training Classification (on SMOTE data)
        print("\n--- Training Classification Model (SMOTE + CE) ---")
        cls_dataset = AdvancedDataset(X_smote, y_smote)
        cls_loader = DataLoader(cls_dataset, batch_size=128, shuffle=True)
        cls_model = SimpleLSTM(input_dim=10, output_dim=3, is_regression=False)
        run_advanced_training("lstm_classifier_smote", cls_model, cls_loader, epochs=3, is_regression=False)
        
        print(f"\n✓ All pipeline tasks executed successfully!")
        print(f"✓ Results saved to: {RESULTS_DIR}")
        
    except Exception as e:
        print(f"ERROR in main pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
