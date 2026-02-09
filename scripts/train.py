"""
Unified training script for all models.

Usage:
    python scripts/train.py --model ols
    python scripts/train.py --model ridge
    python scripts/train.py --model lasso
    python scripts/train.py --model mlp
    python scripts/train.py --model lstm
    python scripts/train.py --model cnn
    python scripts/train.py --model deeplob
    python scripts/train.py --model seq2seq
    python scripts/train.py --model transformer
    python scripts/train.py --model temporal_attention
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Ensure project root is on path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import (
    load_all_days,
    temporal_split,
    LOBDataset,
    FlatDataset,
    create_dataloaders,
)
from src.data.preprocessor import normalize_splits
from src.features.labels import DEFAULT_HORIZONS, get_class_weights

# --- Linear baselines ---
from src.models.baselines.linear import (
    OLSBaseline,
    RegularizedLinearModel,
    evaluate_linear_model,
)

# --- Deep learning models ---
from src.models.baselines.mlp import MLPClassifier
from src.models.sequential.lstm import LSTMClassifier
from src.models.sequential.cnn import CNNClassifier
from src.models.sequential.cnn_lstm import DeepLOB
from src.models.sequential.seq2seq import Seq2SeqAttention
from src.models.transformer.transformer import TransformerClassifier
from src.models.sequential.seq2seq_temporal_attention import (
    Seq2SeqTemporalAttention,
)

from src.training.trainer import Trainer
from src.metrics.classification import (
    compute_all_horizon_metrics,
    print_classification_report,
)
from src.metrics.regression import compute_all_horizon_regression_metrics


# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────

CONFIGS = {
    "ols": {
        "type": "linear",
        "description": "OLS Regression (Cont et al. 2014)",
    },
    "ridge": {
        "type": "linear",
        "method": "ridge",
        "alpha": 1.0,
        "description": "Ridge Regression (Xu et al. 2019)",
    },
    "lasso": {
        "type": "linear",
        "method": "lasso",
        "alpha": 0.001,
        "description": "Lasso Regression (Xu et al. 2019)",
    },
    "mlp": {
        "type": "deep_flat",
        "hidden_dims": [256, 128, 64],
        "dropout": 0.3,
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 512,
        "description": "MLP (Kolm et al. 2023)",
    },
    "lstm": {
        "type": "deep_seq",
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.3,
        "bidirectional": False,
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 256,
        "seq_len": 100,
        "description": "LSTM (Kolm et al. 2023)",
    },
    "cnn": {
        "type": "deep_seq",
        "channels": [32, 64, 64],
        "kernel_sizes": [5, 5, 3],
        "dropout": 0.3,
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 256,
        "seq_len": 100,
        "description": "1D-CNN (Kolm et al. 2023)",
    },
    "deeplob": {
        "type": "deep_seq",
        "conv_channels": 32,
        "lstm_hidden": 64,
        "dropout": 0.2,
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 256,
        "seq_len": 100,
        "description": "DeepLOB CNN-LSTM (Zhang et al. 2019)",
    },
    "seq2seq": {
        "type": "deep_seq",
        "encoder_hidden": 64,
        "decoder_hidden": 64,
        "encoder_layers": 2,
        "attn_dim": 64,
        "dropout": 0.3,
        "lr": 5e-4,
        "epochs": 50,
        "batch_size": 256,
        "seq_len": 100,
        "description": "Seq2Seq + Attention (Zhang & Zohren 2021)",
    },
    "transformer": {
        "type": "deep_seq",
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "lr": 5e-4,
        "epochs": 50,
        "batch_size": 256,
        "seq_len": 100,
        "description": "Transformer (Wallbridge 2020)",
    },
    "temporal_attention": {
        "type": "deep_seq",
        "encoder_hidden": 64,
        "decoder_hidden": 64,
        "encoder_layers": 2,
        "attn_dim": 64,
        "dropout": 0.3,
        "use_multi_scale": True,
        "lr": 5e-4,
        "epochs": 50,
        "batch_size": 256,
        "seq_len": 100,
        "description": "Seq2Seq Temporal Attention (Ours)",
    },
}


# ──────────────────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────────────────

def build_model(model_name: str, n_features: int, n_horizons: int, cfg: dict):
    """Instantiate a model by name."""
    if model_name == "mlp":
        return MLPClassifier(
            n_features=n_features,
            n_horizons=n_horizons,
            hidden_dims=cfg.get("hidden_dims", [256, 128, 64]),
            dropout=cfg.get("dropout", 0.3),
        )
    elif model_name == "lstm":
        return LSTMClassifier(
            n_features=n_features,
            n_horizons=n_horizons,
            hidden_size=cfg.get("hidden_size", 64),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.3),
            bidirectional=cfg.get("bidirectional", False),
        )
    elif model_name == "cnn":
        return CNNClassifier(
            n_features=n_features,
            n_horizons=n_horizons,
            channels=cfg.get("channels", [32, 64, 64]),
            kernel_sizes=cfg.get("kernel_sizes", [5, 5, 3]),
            dropout=cfg.get("dropout", 0.3),
        )
    elif model_name == "deeplob":
        return DeepLOB(
            n_features=n_features,
            n_horizons=n_horizons,
            conv_channels=cfg.get("conv_channels", 32),
            lstm_hidden=cfg.get("lstm_hidden", 64),
            dropout=cfg.get("dropout", 0.2),
        )
    elif model_name == "seq2seq":
        return Seq2SeqAttention(
            n_features=n_features,
            n_horizons=n_horizons,
            encoder_hidden=cfg.get("encoder_hidden", 64),
            decoder_hidden=cfg.get("decoder_hidden", 64),
            encoder_layers=cfg.get("encoder_layers", 2),
            attn_dim=cfg.get("attn_dim", 64),
            dropout=cfg.get("dropout", 0.3),
        )
    elif model_name == "transformer":
        return TransformerClassifier(
            n_features=n_features,
            n_horizons=n_horizons,
            d_model=cfg.get("d_model", 64),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 2),
            dim_feedforward=cfg.get("dim_feedforward", 256),
            dropout=cfg.get("dropout", 0.1),
        )
    elif model_name == "temporal_attention":
        return Seq2SeqTemporalAttention(
            n_features=n_features,
            n_horizons=n_horizons,
            encoder_hidden=cfg.get("encoder_hidden", 64),
            decoder_hidden=cfg.get("decoder_hidden", 64),
            encoder_layers=cfg.get("encoder_layers", 2),
            attn_dim=cfg.get("attn_dim", 64),
            dropout=cfg.get("dropout", 0.3),
            use_multi_scale=cfg.get("use_multi_scale", True),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ──────────────────────────────────────────────────────────────────────────
# Training functions
# ──────────────────────────────────────────────────────────────────────────

def train_linear(
    model_name: str,
    cfg: dict,
    processed_dir: str,
    ticker: str,
    horizons: list,
    results_dir: str,
):
    """Train and evaluate a linear model (OLS / Ridge / Lasso)."""
    print(f"\n{'='*60}")
    print(f"Training: {cfg['description']}")
    print(f"{'='*60}\n")

    X, y_reg, y_cls, fnames = load_all_days(
        processed_dir, ticker, horizons=horizons
    )
    (X_tr, yr_tr, yc_tr), (X_va, yr_va, yc_va), (X_te, yr_te, yc_te) = \
        temporal_split(X, y_reg, y_cls)

    t0 = time.time()

    if model_name == "ols":
        model = OLSBaseline(horizons=horizons)
        model.fit(X_tr, yr_tr)
        r2 = model.get_r_squared()
        print("R² per horizon:", {k: f"{v:.4f}" for k, v in r2.items()})
    else:
        model = RegularizedLinearModel(
            method=cfg["method"],
            alpha=cfg["alpha"],
            horizons=horizons,
        )
        model.fit(X_tr, yr_tr)
        if model_name in ("ridge", "lasso"):
            importance = model.get_feature_importance(fnames)
            for h in horizons[:1]:
                top5 = list(importance[h].items())[:5]
                print(f"  Top-5 features (h={h}): {top5}")

    elapsed = time.time() - t0
    print(f"Training time: {elapsed:.1f}s")

    # Evaluate
    metrics = evaluate_linear_model(
        model, X_te, yr_te, yc_te, horizons=horizons
    )

    print("\nTest metrics:")
    for h in horizons:
        print(f"  h={h}: R²={metrics.get(f'h{h}_r2', 0):.4f}, "
              f"Accuracy={metrics.get(f'h{h}_accuracy', 0):.4f}, "
              f"F1={metrics.get(f'h{h}_f1_macro', 0):.4f}")

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f"{model_name}_results.json")
    with open(result_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nResults saved to {result_path}")

    return metrics


def train_deep(
    model_name: str,
    cfg: dict,
    processed_dir: str,
    ticker: str,
    horizons: list,
    results_dir: str,
    device: str = "cpu",
):
    """Train and evaluate a deep learning model."""
    print(f"\n{'='*60}")
    print(f"Training: {cfg['description']}")
    print(f"{'='*60}\n")

    is_flat = cfg["type"] == "deep_flat"
    seq_len = cfg.get("seq_len", 100)
    batch_size = cfg.get("batch_size", 256)
    n_epochs = cfg.get("epochs", 50)
    lr = cfg.get("lr", 1e-3)

    # Load data
    X, y_reg, y_cls, fnames = load_all_days(
        processed_dir, ticker, horizons=horizons
    )

    # Normalise
    (X_tr, yr_tr, yc_tr), (X_va, yr_va, yc_va), (X_te, yr_te, yc_te) = \
        temporal_split(X, y_reg, y_cls)
    X_tr, X_va, X_te, scaler = normalize_splits(X_tr, X_va, X_te, method="zscore")

    n_features = X_tr.shape[1]
    n_horizons = len(horizons)

    # Create datasets
    if is_flat:
        from torch.utils.data import DataLoader
        ds_tr = FlatDataset(X_tr, yr_tr, yc_tr)
        ds_va = FlatDataset(X_va, yr_va, yc_va)
        ds_te = FlatDataset(X_te, yr_te, yc_te)
    else:
        from torch.utils.data import DataLoader
        ds_tr = LOBDataset(X_tr, yr_tr, yc_tr, seq_len=seq_len)
        ds_va = LOBDataset(X_va, yr_va, yc_va, seq_len=seq_len)
        ds_te = LOBDataset(X_te, yr_te, yc_te, seq_len=seq_len)

    train_loader = DataLoader(
        ds_tr, batch_size=batch_size, shuffle=False, drop_last=True
    )
    val_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(ds_tr)} | Val: {len(ds_va)} | Test: {len(ds_te)}")
    print(f"Features: {n_features} | Horizons: {horizons}")

    # Build model
    model = build_model(model_name, n_features, n_horizons, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Class weights for imbalanced labels
    weights = get_class_weights(yc_tr[:, 0])
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    # Train
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        horizons=horizons,
        early_stopping_patience=10,
        checkpoint_dir=os.path.join(results_dir, "checkpoints"),
        model_name=model_name,
    )

    t0 = time.time()
    history = trainer.train(train_loader, val_loader, n_epochs=n_epochs)
    elapsed = time.time() - t0
    print(f"\nTraining time: {elapsed:.1f}s")

    # Evaluate
    test_metrics, preds, labels = trainer.evaluate(test_loader)

    print("\nTest metrics:")
    for h in horizons:
        print(f"  h={h}: Accuracy={test_metrics.get(f'h{h}_accuracy', 0):.4f}, "
              f"F1_macro={test_metrics.get(f'h{h}_f1_macro', 0):.4f}, "
              f"F1_weighted={test_metrics.get(f'h{h}_f1_weighted', 0):.4f}")

    # Print per-horizon classification reports
    for i, h in enumerate(horizons):
        report = print_classification_report(labels[:, i], preds[:, i], h)
        print(report)

    # Save
    trainer.save_checkpoint()
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f"{model_name}_results.json")
    all_results = {
        "config": {k: v for k, v in cfg.items() if k != "description"},
        "description": cfg["description"],
        "n_params": n_params,
        "training_time_s": elapsed,
        "test_metrics": test_metrics,
        "history": {
            "train_loss": [float(v) for v in history["train_loss"]],
            "val_loss": [float(v) for v in history["val_loss"]],
        },
    }
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {result_path}")

    return test_metrics


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train LOB prediction models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(CONFIGS.keys()),
        help="Model to train",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AMZN",
        help="Ticker symbol (default: AMZN)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: cpu or cuda",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    args = parser.parse_args()

    cfg = CONFIGS[args.model].copy()
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    horizons = DEFAULT_HORIZONS

    if cfg["type"] == "linear":
        train_linear(
            args.model, cfg, args.data_dir, args.ticker,
            horizons, args.results_dir,
        )
    else:
        train_deep(
            args.model, cfg, args.data_dir, args.ticker,
            horizons, args.results_dir, args.device,
        )


if __name__ == "__main__":
    main()
