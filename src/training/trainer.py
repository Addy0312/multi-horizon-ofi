import torch
import torch.nn as nn
import time
import numpy as np
import os
import json
import pandas as pd
from typing import List, Tuple, Dict
from .checkpoint import *
from .loss import _deep_class_weights_cb, _deep_focal_loss, _deep_multihorizon_loss_advanced
from ..metrics.classification import *
from ..data.dataset import _deep_build_day_dataset, _deep_make_loader, _deep_resolve_tickers, _deep_collect_files_by_ticker, _deep_split_train_eval_files
from ..data.helpers import _deep_update_confusion, _deep_metrics_from_confusion, _mean_macro_f1
from ..utils.system import _avail_ram_gb, _deep_cleanup_cuda
from ..utils.env import DEEP_DEVICE
from ..models.builder import build_deep_model
from ..models.autoencoder.lstm_ae import LSTMAutoencoder
from ..etl.preprocess import DEEP_RAW_LOB_10_COLS

def _train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, scaler: 'torch.cuda.amp.GradScaler', cfg: dict, train_files: List[Tuple[str, str]], horizons: List[int], epoch_idx: int, total_epochs: int, amp_enabled: bool, arch: str, start_file_idx: int=1, mid_state: dict=None) -> dict:
    model.train()
    grad_clip = float(cfg.get('grad_clip', 1.0))
    label_smoothing = float(cfg.get('label_smoothing', 0.0))
    if mid_state:
        day_losses = mid_state.get('day_losses', [])
        bad_batches = mid_state.get('bad_batches', 0)
        fitted_days = mid_state.get('fitted_days', 0)
        train_rows = mid_state.get('train_rows', {h: 0 for h in horizons})
        train_counts = mid_state.get('train_counts', {h: np.zeros(3, dtype=np.int64) for h in horizons})
    else:
        day_losses: List[float] = []
        bad_batches = 0
        fitted_days = 0
        train_rows = {h: 0 for h in horizons}
        train_counts = {h: np.zeros(3, dtype=np.int64) for h in horizons}
    for file_idx, (ticker, path) in enumerate(train_files, start=1):
        if file_idx < start_file_idx:
            continue
        ds, stats = _deep_build_day_dataset(path, cfg, is_train=True)
        if ds is None:
            continue
        weight_tensors = None
        if cfg.get('loss_mode', 'ce') == 'cb_focal':
            cb_ws = _deep_class_weights_cb(ds.labels, cfg)
            weight_tensors = [torch.tensor(w, dtype=torch.float32, device=DEEP_DEVICE) for w in cb_ws]
        loader = _deep_make_loader(ds, cfg, is_train=True)
        file_losses: List[float] = []
        for xb, yb in loader:
            xb = xb.to(DEEP_DEVICE, non_blocking=True)
            xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0).clamp(-10.0, 10.0)
            yb = yb.to(DEEP_DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(xb)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
                loss = _deep_multihorizon_loss_advanced(
                    logits=logits,
                    targets=yb,
                    weight_tensors=weight_tensors,
                    label_smoothing=float(cfg.get('label_smoothing', 0.0)),
                    loss_mode=str(cfg.get('loss_mode', 'ce')),
                    focal_gamma=float(cfg.get('focal_gamma', 2.0))
                )
            if not torch.isfinite(loss):
                bad_batches += 1
                del xb, yb, logits, loss
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            file_losses.append(loss.item())
            batch_n = int(yb.shape[0])
            for i, h in enumerate(horizons):
                train_rows[h] += batch_n
                train_counts[h] += np.bincount(yb[:, i].detach().cpu().numpy(), minlength=3)
            del xb, yb, logits, loss
        if file_losses:
            day_losses.append(float(np.mean(file_losses)))
            fitted_days += 1
        print(f'[{arch}][ep {epoch_idx}/{total_epochs}][train {file_idx}/{len(train_files)}] {ticker}/{stats.file_name}  rows={stats.rows_kept}  sub={stats.step}  avg_loss={day_losses[-1]:.4f}  ram={_avail_ram_gb():.1f}GB')
        save_mid_epoch_checkpoint(cfg, arch, epoch_idx, file_idx, model, optimizer, scaler, day_losses, train_rows, train_counts, bad_batches, fitted_days, cfg.get('result_suffix', '_stable_es'))
        del loader, ds
        _deep_cleanup_cuda()
    return {'avg_train_loss': float(np.mean(day_losses)) if day_losses else float('nan'), 'fitted_days': int(fitted_days), 'bad_batches': int(bad_batches), 'train_rows_seen': train_rows, 'train_class_counts': train_counts}

def _eval_epoch(model: nn.Module, cfg: dict, eval_files: List[Tuple[str, str]], horizons: List[int], arch: str) -> dict:
    confusion = {h: np.zeros((3, 3), dtype=np.int64) for h in horizons}
    eval_rows = {h: 0 for h in horizons}
    eval_counts = {h: np.zeros(3, dtype=np.int64) for h in horizons}
    model.eval()
    with torch.no_grad():
        for file_idx, (ticker, path) in enumerate(eval_files, start=1):
            ds, stats = _deep_build_day_dataset(path, cfg, is_train=False)
            if ds is None:
                continue
            loader = _deep_make_loader(ds, cfg, is_train=False)
            for xb, yb in loader:
                xb = torch.nan_to_num(xb.to(DEEP_DEVICE, non_blocking=True), nan=0.0, posinf=0.0, neginf=0.0).clamp(-10.0, 10.0)
                logits = torch.nan_to_num(model(xb), nan=0.0, posinf=0.0, neginf=0.0)
                preds = logits.argmax(dim=2).detach().cpu().numpy().astype(np.int64)
                y_true = yb.numpy().astype(np.int64)
                for i, h in enumerate(horizons):
                    _deep_update_confusion(confusion[h], y_true[:, i], preds[:, i])
                    eval_rows[h] += int(y_true.shape[0])
                    eval_counts[h] += np.bincount(y_true[:, i], minlength=3)
                del xb, logits, preds, y_true
            print(f'[{arch}][eval {file_idx}/{len(eval_files)}] {ticker}/{stats.file_name}  rows={stats.rows_kept}')
            del loader, ds
            _deep_cleanup_cuda()
    metrics: Dict[str, float] = {}
    for h in horizons:
        metrics.update(_deep_metrics_from_confusion(confusion[h], h))
    return {'metrics': metrics, 'confusion': confusion, 'eval_rows_seen': eval_rows, 'eval_class_counts': eval_counts}

def train_one_architecture(arch: str, cfg: dict, max_epochs: int=10, patience: int=3, min_delta: float=0.0001, force_restart: bool=False) -> dict:
    """
    Train one architecture with per-epoch checkpoint recovery.

    On restart:
    - If the architecture is already marked COMPLETED, return the saved result.
    - If partially trained, load the last saved weights + state and resume
      from the next epoch.
    - If force_restart=True, ignore any existing checkpoint and start fresh.
    """
    suffix = str(cfg.get('result_suffix', '_stable_es'))
    horizons = list(cfg['horizons'])
    amp_enabled = bool(cfg.get('amp', False)) and DEEP_DEVICE.type == 'cuda'
    if not force_restart and is_arch_completed(cfg, arch, suffix):
        ckpt = load_checkpoint(cfg, arch, suffix)
        print(f"[{arch}] Already COMPLETED (epoch={ckpt['epoch']}  best_f1={ckpt['best_score']:.4f}). Skipping.")
        results_path = os.path.join(cfg['results_dir'], f'{arch}_results_day_streaming{suffix}.json')
        if os.path.exists(results_path):
            with open(results_path) as f:
                return json.load(f)
        return {'architecture': arch, 'completed': True, 'skipped': True}
    tickers_list = _deep_resolve_tickers(cfg)
    if not tickers_list:
        raise FileNotFoundError(f"No tickers found in {cfg['data_dir']}")
    files_by_ticker = _deep_collect_files_by_ticker(cfg['data_dir'], tickers_list, int(cfg.get('max_files_per_ticker', 0)))
    if not files_by_ticker:
        raise FileNotFoundError('No parquet files found for selected tickers.')
    train_files, eval_files = _deep_split_train_eval_files(files_by_ticker, float(cfg.get('train_file_fraction', 0.8)))
    print(f"\n{'═' * 80}")
    print(f'  Training: {arch}')
    print(f'  Device={DEEP_DEVICE}  train_files={len(train_files)}  eval_files={len(eval_files)}')
    print(f'  max_epochs={max_epochs}  patience={patience}  amp={amp_enabled}')
    model = build_deep_model(arch=arch, input_dim=len(DEEP_RAW_LOB_10_COLS), horizon_count=len(horizons), num_classes=3).to(DEEP_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get('lr', 0.0003)), weight_decay=float(cfg.get('weight_decay', 0.0001)))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 1
    start_file_idx = 1
    mid_state = None
    best_score = -1.0
    best_epoch = 0
    best_metrics: dict = {}
    no_improve = 0
    epoch_history: list = []
    best_eval_rows: dict = {h: 0 for h in horizons}
    best_eval_counts: dict = {h: np.zeros(3, dtype=np.int64) for h in horizons}
    agg_train_rows: dict = {h: 0 for h in horizons}
    agg_train_counts: dict = {h: np.zeros(3, dtype=np.int64) for h in horizons}
    total_bad_batches = 0
    total_train_days = 0
    if not force_restart:
        ckpt = load_checkpoint(cfg, arch, suffix)
        if ckpt is not None:
            mid = load_mid_epoch_checkpoint(cfg, arch, model, optimizer, scaler, suffix)
            if mid is not None:
                start_epoch = mid['epoch']
                start_file_idx = mid['file_idx'] + 1
                best_score = float(ckpt['best_score'])
                best_epoch = int(ckpt['best_epoch'])
                no_improve = int(ckpt['no_improve'])
                epoch_history = list(ckpt.get('epoch_history', []))
                best_metrics = dict(ckpt.get('best_metrics', {}))
                print(f"  ✓ Resumed from MID-EPOCH {start_epoch}, file index {start_file_idx}")
                mid_state = mid
                if start_file_idx > len(train_files):
                    start_epoch += 1
                    start_file_idx = 1
                    mid_state = None
            else:
                loaded = restore_weights_to_model(cfg, arch, model, label='current', suffix=suffix)
                if loaded:
                    model.to(DEEP_DEVICE)
                    start_epoch = int(ckpt['epoch']) + 1
                    best_score = float(ckpt['best_score'])
                    best_epoch = int(ckpt['best_epoch'])
                    no_improve = int(ckpt['no_improve'])
                    epoch_history = list(ckpt.get('epoch_history', []))
                    best_metrics = dict(ckpt.get('best_metrics', {}))
                    print(f'  ✓ Resumed from epoch {start_epoch - 1}  best_f1={best_score:.4f}  no_improve={no_improve}')
                    if no_improve >= patience:
                        print(f'  Early stopping already triggered before restart. Returning saved best.')
                        save_checkpoint(cfg, arch, int(ckpt['epoch']), model, optimizer, best_score, best_epoch, no_improve, epoch_history, best_metrics, completed=True, suffix=suffix)
                        return _finalize_arch(arch, cfg, model, best_metrics, best_score, best_epoch, epoch_history, train_files, eval_files, horizons, agg_train_rows, agg_train_counts, best_eval_rows, best_eval_counts, total_bad_batches, total_train_days, 0.0, max_epochs, patience, min_delta, suffix)
                else:
                    print(f'  WARNING: Checkpoint found but weights missing for {arch}. Starting fresh.')
    if start_epoch > max_epochs:
        print(f'[{arch}] All {max_epochs} epochs already done. Marking complete.')
        save_checkpoint(cfg, arch, max_epochs, model, optimizer, best_score, best_epoch, no_improve, epoch_history, best_metrics, completed=True, suffix=suffix)
        return _finalize_arch(arch, cfg, model, best_metrics, best_score, best_epoch, epoch_history, train_files, eval_files, horizons, agg_train_rows, agg_train_counts, best_eval_rows, best_eval_counts, total_bad_batches, total_train_days, 0.0, max_epochs, patience, min_delta, suffix)
    start_t = time.time()
    for epoch in range(start_epoch, max_epochs + 1):
        print(f'\n[{arch}] ── Epoch {epoch}/{max_epochs} ──────────────────')
        train_info = _train_epoch(model=model, optimizer=optimizer, scaler=scaler, cfg=cfg, train_files=train_files, horizons=horizons, epoch_idx=epoch, total_epochs=max_epochs, amp_enabled=amp_enabled, arch=arch, start_file_idx=start_file_idx, mid_state=mid_state)
        
        start_file_idx = 1
        mid_state = None
        clear_mid_epoch_checkpoint(cfg, arch, suffix)
        eval_info = _eval_epoch(model=model, cfg=cfg, eval_files=eval_files, horizons=horizons, arch=arch)
        metrics = eval_info['metrics']
        score = _mean_macro_f1(metrics, horizons)
        improved = score > best_score + min_delta
        total_bad_batches += int(train_info['bad_batches'])
        total_train_days += int(train_info['fitted_days'])
        for h in horizons:
            agg_train_rows[h] += int(train_info['train_rows_seen'][h])
            agg_train_counts[h] += train_info['train_class_counts'][h]
        epoch_history.append({'epoch': int(epoch), 'mean_macro_f1': float(score), 'avg_train_loss': float(train_info['avg_train_loss']), 'bad_batches': int(train_info['bad_batches']), 'improved': bool(improved)})
        print(f'[{arch}] epoch={epoch}  mean_macro_f1={score:.6f}  best={best_score:.6f}  improved={improved}')
        if improved:
            best_score = float(score)
            best_epoch = int(epoch)
            best_metrics = dict(metrics)
            best_eval_rows = {h: int(eval_info['eval_rows_seen'][h]) for h in horizons}
            best_eval_counts = {h: eval_info['eval_class_counts'][h].copy() for h in horizons}
            save_best_weights(cfg, arch, model, suffix)
            no_improve = 0
        else:
            no_improve += 1
        save_checkpoint(cfg=cfg, arch=arch, epoch=epoch, model=model, optimizer=optimizer, best_score=best_score, best_epoch=best_epoch, no_improve=no_improve, epoch_history=epoch_history, best_metrics=best_metrics, completed=False, suffix=suffix)
        print(f'[{arch}] ✓ Checkpoint saved (epoch={epoch})')
        if no_improve >= patience:
            print(f'[{arch}] Early stopping at epoch {epoch} (no improvement for {patience} epochs)')
            break
        _deep_cleanup_cuda()
    restored = restore_weights_to_model(cfg, arch, model, label='best', suffix=suffix)
    if not restored:
        print(f'[{arch}] WARNING: best weights not found, using current weights.')
    elapsed = time.time() - start_t
    save_checkpoint(cfg=cfg, arch=arch, epoch=best_epoch, model=model, optimizer=optimizer, best_score=best_score, best_epoch=best_epoch, no_improve=no_improve, epoch_history=epoch_history, best_metrics=best_metrics, completed=True, suffix=suffix)
    return _finalize_arch(arch=arch, cfg=cfg, model=model, best_metrics=best_metrics, best_score=best_score, best_epoch=best_epoch, epoch_history=epoch_history, train_files=train_files, eval_files=eval_files, horizons=horizons, agg_train_rows=agg_train_rows, agg_train_counts=agg_train_counts, best_eval_rows=best_eval_rows, best_eval_counts=best_eval_counts, total_bad_batches=total_bad_batches, total_train_days=total_train_days, elapsed=elapsed, max_epochs=max_epochs, patience=patience, min_delta=min_delta, suffix=suffix)

def _finalize_arch(arch, cfg, model, best_metrics, best_score, best_epoch, epoch_history, train_files, eval_files, horizons, agg_train_rows, agg_train_counts, best_eval_rows, best_eval_counts, total_bad_batches, total_train_days, elapsed, max_epochs, patience, min_delta, suffix) -> dict:
    """Save final weights + JSON result, return run_meta dict."""
    os.makedirs(cfg['weights_dir'], exist_ok=True)
    os.makedirs(cfg['results_dir'], exist_ok=True)
    weights_path = os.path.join(cfg['weights_dir'], f'{arch}{suffix}_weights.pt')
    torch.save({'architecture': arch, 'state_dict': model.state_dict(), 'input_dim': len(DEEP_RAW_LOB_10_COLS), 'horizons': horizons, 'seq_len': int(cfg['seq_len']), 'alpha': float(cfg['alpha']), 'device_trained': str(DEEP_DEVICE), 'best_epoch': int(best_epoch), 'best_macro_f1': float(best_score)}, weights_path)
    run_meta = {'timestamp': pd.Timestamp.now().isoformat(), 'mode': 'day_by_day_streaming_deep_stable_earlystop', 'architecture': arch, 'horizons': horizons, 'seq_len': int(cfg['seq_len']), 'alpha': float(cfg['alpha']), 'train_config': {'max_epochs': int(max_epochs), 'patience': int(patience), 'min_delta': float(min_delta), 'batch_size': int(cfg.get('batch_size', 256)), 'lr': float(cfg.get('lr', 0.0003)), 'weight_decay': float(cfg.get('weight_decay', 0.0001)), 'grad_clip': float(cfg.get('grad_clip', 1.0)), 'amp': bool(cfg.get('amp', False)), 'label_smoothing': float(cfg.get('label_smoothing', 0.0)), 'device': str(DEEP_DEVICE)}, 'early_stopping': {'best_epoch': int(best_epoch), 'best_mean_macro_f1': float(best_score), 'epochs_ran': int(len(epoch_history))}, 'epoch_history': epoch_history, 'files': {'train_files': len(train_files), 'eval_files': len(eval_files), 'days_fitted': int(total_train_days)}, 'train_rows_seen': {f'h{h}': int(agg_train_rows[h]) for h in horizons}, 'eval_rows_seen': {f'h{h}': int(best_eval_rows[h]) for h in horizons}, 'train_class_counts': {f'h{h}': [int(x) for x in agg_train_counts[h].tolist()] for h in horizons}, 'eval_class_counts': {f'h{h}': [int(x) for x in best_eval_counts[h].tolist()] for h in horizons}, 'bad_batches_total': int(total_bad_batches), 'test_metrics': best_metrics, 'model_path': weights_path, 'runtime_seconds': round(elapsed, 2)}
    results_path = os.path.join(cfg['results_dir'], f'{arch}_results_day_streaming{suffix}.json')
    with open(results_path, 'w') as f:
        json.dump(run_meta, f, indent=2)
    print(f'[{arch}] ✓ Weights  → {weights_path}')
    print(f'[{arch}] ✓ Results  → {results_path}')
    del model
    _deep_cleanup_cuda()
    return run_meta

def _load_model_for_eval(arch: str, cfg: dict, suffix: str) -> 'nn.Module | None':
    weights_path = os.path.join(cfg['weights_dir'], f'{arch}{suffix}_weights.pt')
    if not os.path.exists(weights_path):
        best_path = os.path.join(cfg['weights_dir'], f'{arch}{suffix}_best_weights.pt')
        if not os.path.exists(best_path):
            print(f'  [{arch}] No weights found at {weights_path}')
            return None
        weights_path = best_path
    try:
        ckpt = torch.load(weights_path, map_location='cpu')
        model = build_deep_model(arch=arch, input_dim=len(DEEP_RAW_LOB_10_COLS), horizon_count=len(horizons), num_classes=3)
        sd = ckpt.get('state_dict', ckpt)
        model.load_state_dict(sd)
        model.eval()
        return model
    except Exception as e:
        print(f'  [{arch}] Failed to load weights: {e}')
        return None

def _collect_probs_and_labels(model: 'nn.Module', cfg: dict, eval_files: List[Tuple[str, str]], horizons: List[int], max_batches: int=50) -> Tuple['np.ndarray | None', 'np.ndarray | None']:
    """Run model on eval files, collect softmax probs and true labels."""
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    n_batches = 0
    model.to(DEEP_DEVICE)
    with torch.no_grad():
        for ticker, path in eval_files:
            ds, _ = _deep_build_day_dataset(path, cfg, is_train=False)
            if ds is None:
                continue
            loader = _deep_make_loader(ds, cfg, is_train=False)
            for xb, yb in loader:
                xb = torch.nan_to_num(xb.to(DEEP_DEVICE, non_blocking=True), nan=0.0, posinf=0.0, neginf=0.0).clamp(-10.0, 10.0)
                logits = model(xb)
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs.detach().cpu().numpy())
                all_labels.append(yb.numpy())
                n_batches += 1
                del xb, logits, probs
                if n_batches >= max_batches:
                    break
            del loader, ds
            _deep_cleanup_cuda()
            if n_batches >= max_batches:
                break
    if not all_probs:
        return (None, None)
    return (np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0))

def _is_final_weight(filename: str) -> bool:
    name = os.path.basename(filename)
    if any((name.endswith(p.replace('*', '')) for p in _WEIGHTS_EXCLUDE_PATTERNS if p.startswith('*'))):
        return False
    return True

def train_lstm_autoencoder_day_by_day(cfg: dict, max_epochs: int=5):
    """
    Robust day-by-day training loop for the LSTM Autoencoder with per-file, per-epoch checkpointing.
    Train ONLY on Class 1 (Stationary) sequences.
    """
    os.makedirs(cfg.get('weights_dir', 'model_weights'), exist_ok=True)
    os.makedirs(cfg.get('results_dir', 'results'), exist_ok=True)
    arch = 'lstm_autoencoder'
    checkpoint_path = os.path.join(cfg.get('weights_dir', 'model_weights'), f'{arch}_checkpoint.pt')
    weights_path = os.path.join(cfg.get('weights_dir', 'model_weights'), f'{arch}_weights.pt')
    results_path = os.path.join(cfg.get('results_dir', 'results'), f'{arch}_results.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = len(DEEP_RAW_LOB_10_COLS)
    seq_len = int(cfg.get('seq_len', 10))
    model = LSTMAutoencoder(seq_len=seq_len, n_features=input_dim, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    tickers = _deep_resolve_tickers(cfg)
    files_by_ticker = _deep_collect_files_by_ticker(cfg.get('data_dir', 'data/processed'), tickers, cfg.get('max_files_per_ticker', 0))
    train_files, eval_files = _deep_split_train_eval_files(files_by_ticker, float(cfg.get('train_file_fraction', 0.8)))
    start_epoch = 1
    start_file_idx = 1
    best_loss = float('inf')
    total_train_days = 0
    epoch_history = []
    if os.path.exists(checkpoint_path):
        print(f'[{arch}] Found checkpoint at {checkpoint_path}, resuming...')
        chk = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(chk['model'])
        optimizer.load_state_dict(chk['optimizer'])
        start_epoch = chk['epoch']
        start_file_idx = chk['file_idx'] + 1
        best_loss = chk.get('best_loss', float('inf'))
        epoch_history = chk.get('epoch_history', [])
        total_train_days = chk.get('total_train_days', 0)
        if start_file_idx > len(train_files):
            start_epoch += 1
            start_file_idx = 1

    def save_checkpoint(ep, f_idx):
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': ep, 'file_idx': f_idx, 'best_loss': best_loss, 'epoch_history': epoch_history, 'total_train_days': total_train_days}, checkpoint_path)
    target_h_idx = 0
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        print(f'\n[{arch}] Epoch {epoch}/{max_epochs}')
        loss_sum = 0.0
        good_batches = 0
        for file_idx, (ticker, path) in enumerate(train_files, start=1):
            if file_idx < start_file_idx:
                continue
            ds, stats = _deep_build_day_dataset(path, cfg, is_train=True)
            if ds is None:
                continue
            mask_c1 = ds.labels[:, target_h_idx] == 1
            raw_c1 = ds.raw[ds.starts[mask_c1]]
            starts_c1 = ds.starts[mask_c1]
            labels_c1 = ds.labels[mask_c1]
            if starts_c1.size == 0:
                continue
            ds.starts = starts_c1
            ds.labels = labels_c1
            loader = _deep_make_loader(ds, cfg, is_train=True)
            for xb, _ in loader:
                xb = xb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(xb)
                loss = criterion(outputs, xb)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += loss.item()
                good_batches += 1
            total_train_days += 1
            save_checkpoint(epoch, file_idx)
            del ds, loader
            _deep_cleanup_cuda()
        start_file_idx = 1
        avg_loss = loss_sum / max(1, good_batches)
        print(f'[{arch}] Epoch {epoch} complete, Avg Loss: {avg_loss:.6f}')
        epoch_history.append({'epoch': epoch, 'avg_train_loss': avg_loss})
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), weights_path)
            print(f'[{arch}] Saved new best weights to {weights_path}')
    print(f'[{arch}] Training complete. Best Loss: {best_loss:.6f}')
    res = {'architecture': arch, 'epoch_history': epoch_history, 'best_train_loss': best_loss, 'total_train_days': total_train_days, 'checkpoint_used': checkpoint_path, 'weights_path': weights_path}
    with open(results_path, 'w') as f:
        json.dump(res, f, indent=2)


def run_all_deep_models_day_by_day_stable_earlystop(config: dict, max_epochs: int=10, patience: int=3, min_delta: float=1e-4) -> dict:
    from ..utils.seed import deep_set_seed
    import time
    import pandas as pd
    import json
    
    cfg = dict(config)
    cfg.setdefault('result_suffix', '_stable_es')

    deep_set_seed(int(cfg['seed']))
    os.makedirs(cfg['weights_dir'], exist_ok=True)
    os.makedirs(cfg['results_dir'], exist_ok=True)

    all_runs = {}
    all_results_paths = {}
    t0 = time.time()
    
    for arch in cfg['run_architectures']:
        run_meta = train_one_architecture(
            arch=arch,
            cfg=cfg,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
        )
        all_runs[arch] = run_meta
        suffix = str(cfg.get('result_suffix', '_stable_es'))
        if suffix and not suffix.startswith('_'):
            suffix = '_' + suffix
        all_results_paths[arch] = os.path.join(cfg['results_dir'], f'{arch}_results_day_streaming{suffix}.json')
        
        from ..data.helpers import _deep_cleanup_cuda
        _deep_cleanup_cuda()

    suffix = str(cfg.get('result_suffix', '_stable_es'))
    if suffix and not suffix.startswith('_'):
        suffix = '_' + suffix

    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'mode': 'day_by_day_streaming_deep_multi_arch_stable_earlystop',
        'architectures': list(cfg['run_architectures']),
        'horizons': list(cfg['horizons']),
        'seq_len': int(cfg['seq_len']),
        'alpha': float(cfg['alpha']),
        'early_stopping': {
            'max_epochs': int(max_epochs),
            'patience': int(patience),
            'min_delta': float(min_delta),
        },
        'results_paths': all_results_paths,
        'runtime_seconds': round(time.time() - t0, 2),
    }

    summary_path = os.path.join(cfg['results_dir'], f'deep_models_day_streaming_summary{suffix}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n' + '=' * 88)
    print('Stable deep early-stop run complete')
    print(f"Summary saved -> {summary_path}")

    return {
        'summary_path': summary_path,
        'summary': summary,
        'runs': all_runs,
    }
