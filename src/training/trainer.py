"""
Generic training loop for PyTorch models.

Handles:
    - Training with classification + optional regression loss
    - Validation with early stopping
    - Metric logging per epoch
    - Model checkpointing
"""

import os
import time
import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from src.metrics.classification import compute_all_horizon_metrics
from src.metrics.regression import compute_all_horizon_regression_metrics
from src.features.labels import DEFAULT_HORIZONS


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Generic trainer for multi-horizon LOB classification models.

    Works with any model that takes (batch_x) and returns
    logits of shape (batch, n_horizons, n_classes).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module | None = None,
        scheduler: _LRScheduler | None = None,
        device: str = "cpu",
        horizons: List[int] | None = None,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = "checkpoints",
        model_name: str = "model",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.scheduler = scheduler
        self.device = device
        self.horizons = horizons or DEFAULT_HORIZONS
        self.early_stop = EarlyStopping(patience=early_stopping_patience)
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }
        self.best_model_state = None

    def _run_epoch(
        self, loader: DataLoader, train: bool = True
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Run one epoch (train or eval).

        Returns
        -------
        avg_loss, all_preds (N, H), all_labels (N, H)
        """
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_labels = []

        ctx = torch.no_grad() if not train else _nullcontext()
        with ctx:
            for batch in loader:
                x = batch["x"].to(self.device)
                y_cls = batch["y_cls"].to(self.device)  # (B, H)

                logits = self.model(x)  # (B, H, C)

                # Compute loss across all horizons
                loss = 0.0
                n_h = y_cls.shape[1]
                for h_idx in range(n_h):
                    loss = loss + self.criterion(
                        logits[:, h_idx, :], y_cls[:, h_idx]
                    )
                loss = loss / n_h

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

                preds = logits.argmax(dim=-1).cpu().numpy()  # (B, H)
                all_preds.append(preds)
                all_labels.append(y_cls.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return avg_loss, all_preds, all_labels

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 50,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Full training loop with validation and early stopping.
        """
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            train_loss, _, _ = self._run_epoch(train_loader, train=True)
            val_loss, val_preds, val_labels = self._run_epoch(
                val_loader, train=False
            )

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Track best model
            if (
                self.early_stop.best_loss is None
                or val_loss <= self.early_stop.best_loss
            ):
                self.best_model_state = copy.deepcopy(self.model.state_dict())

            elapsed = time.time() - t0

            if verbose and epoch % max(1, n_epochs // 20) == 0:
                metrics = compute_all_horizon_metrics(
                    val_labels, val_preds, self.horizons
                )
                acc_str = " | ".join(
                    f"h{h}={metrics[f'h{h}_accuracy']:.3f}"
                    for h in self.horizons
                )
                print(
                    f"Epoch {epoch:3d}/{n_epochs} | "
                    f"train_loss={train_loss:.5f} | "
                    f"val_loss={val_loss:.5f} | "
                    f"val_acc: {acc_str} | "
                    f"{elapsed:.1f}s"
                )

            if self.early_stop.step(val_loss):
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.history

    def evaluate(
        self, test_loader: DataLoader
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluate on test set.

        Returns
        -------
        metrics : dict
        preds : np.ndarray (N, H)
        labels : np.ndarray (N, H)
        """
        test_loss, preds, labels = self._run_epoch(test_loader, train=False)
        metrics = compute_all_horizon_metrics(labels, preds, self.horizons)
        metrics["test_loss"] = test_loss
        return metrics, preds, labels

    def save_checkpoint(self, path: str | None = None):
        """Save model weights."""
        if path is None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            path = os.path.join(
                self.checkpoint_dir, f"{self.model_name}_best.pt"
            )
        torch.save(self.model.state_dict(), path)
        return path

    def load_checkpoint(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )


class _nullcontext:
    """Minimal no-op context manager for Python < 3.10 compat."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
