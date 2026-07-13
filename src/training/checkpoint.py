import os
import json
import torch
import torch.nn as nn
import pandas as pd

def _ckpt_path(cfg: dict, arch: str, suffix: str='_stable_es') -> str:
    """Path for the JSON checkpoint file."""
    return os.path.join(cfg.get('weights_dir', 'model_weights'), f'{arch}{suffix}_checkpoint.json')

def _ckpt_weights_path(cfg: dict, arch: str, label: str='current', suffix: str='_stable_es') -> str:
    """Path for a .pt weights file (current or best)."""
    return os.path.join(cfg.get('weights_dir', 'model_weights'), f'{arch}{suffix}_{label}_weights.pt')

def save_checkpoint(cfg: dict, arch: str, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, best_score: float, best_epoch: int, no_improve: int, epoch_history: list, best_metrics: dict, completed: bool=False, suffix: str='_stable_es') -> None:
    """
    Persist model + training state after each epoch.
    Called inside the training loop — safe to interrupt during next epoch.
    """
    cur_w_path = _ckpt_weights_path(cfg, arch, 'current', suffix)
    torch.save(model.state_dict(), cur_w_path)
    meta = {'arch': arch, 'epoch': int(epoch), 'best_score': float(best_score), 'best_epoch': int(best_epoch), 'no_improve': int(no_improve), 'completed': bool(completed), 'epoch_history': epoch_history, 'best_metrics': best_metrics, 'timestamp': pd.Timestamp.now().isoformat()}
    ckpt_p = _ckpt_path(cfg, arch, suffix)
    tmp_p = ckpt_p + '.tmp'
    with open(tmp_p, 'w') as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_p, ckpt_p)

def save_best_weights(cfg: dict, arch: str, model: nn.Module, suffix: str='_stable_es') -> str:
    """Save the best model weights separately."""
    best_w_path = _ckpt_weights_path(cfg, arch, 'best', suffix)
    torch.save(model.state_dict(), best_w_path)
    return best_w_path

def load_checkpoint(cfg: dict, arch: str, suffix: str='_stable_es') -> dict | None:
    """
    Load checkpoint metadata for `arch`. Returns None if no checkpoint exists.
    """
    ckpt_p = _ckpt_path(cfg, arch, suffix)
    if not os.path.exists(ckpt_p):
        return None
    try:
        with open(ckpt_p) as f:
            return json.load(f)
    except Exception as e:
        print(f'[ckpt] WARNING: could not read checkpoint for {arch}: {e}')
        return None

def is_arch_completed(cfg: dict, arch: str, suffix: str='_stable_es') -> bool:
    """Return True if this architecture finished training."""
    ckpt = load_checkpoint(cfg, arch, suffix)
    return ckpt is not None and bool(ckpt.get('completed', False))

def restore_weights_to_model(cfg: dict, arch: str, model: nn.Module, label: str='current', suffix: str='_stable_es') -> bool:
    """
    Load saved weights into model (in-place). Returns True on success.
    label: "current" or "best"
    """
    path = _ckpt_weights_path(cfg, arch, label, suffix)
    if not os.path.exists(path):
        return False
    try:
        sd = torch.load(path, map_location='cpu')
        model.load_state_dict(sd)
        return True
    except Exception as e:
        print(f'[ckpt] WARNING: could not load {label} weights for {arch}: {e}')
        return False

def print_checkpoint_status(cfg: dict, suffix: str='_stable_es') -> None:
    """Print recovery status for all architectures in cfg."""
    archs = cfg.get('run_architectures', [])
    print('\n' + '═' * 60)
    print('  Checkpoint Status')
    print('═' * 60)
    for arch in archs:
        ckpt = load_checkpoint(cfg, arch, suffix)
        if ckpt is None:
            status = 'NOT STARTED'
        elif ckpt.get('completed'):
            ep = ckpt.get('epoch', '?')
            sc = ckpt.get('best_score', 0.0)
            status = f'COMPLETED  epoch={ep}  best_f1={sc:.4f}'
        else:
            ep = ckpt.get('epoch', 0)
            sc = ckpt.get('best_score', 0.0)
            ni = ckpt.get('no_improve', 0)
            status = f'IN PROGRESS  last_epoch={ep}  best_f1={sc:.4f}  no_improve={ni}'
        print(f'  {arch:<30} {status}')
    print('═' * 60 + '\n')


def save_mid_epoch_checkpoint(cfg: dict, arch: str, epoch: int, file_idx: int, model: nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, day_losses: list, train_rows: dict, train_counts: dict, bad_batches: int, fitted_days: int, suffix: str='_stable_es') -> None:
    path = os.path.join(cfg.get('weights_dir', 'model_weights'), f'{arch}{suffix}_midepoch.pt')
    state = {
        'epoch': epoch,
        'file_idx': file_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'day_losses': day_losses,
        'train_rows': train_rows,
        'train_counts': train_counts,
        'bad_batches': bad_batches,
        'fitted_days': fitted_days
    }
    torch.save(state, path)

def load_mid_epoch_checkpoint(cfg: dict, arch: str, model: nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, suffix: str='_stable_es') -> dict | None:
    path = os.path.join(cfg.get('weights_dir', 'model_weights'), f'{arch}{suffix}_midepoch.pt')
    if not os.path.exists(path):
        return None
    try:
        state = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scaler.load_state_dict(state['scaler_state_dict'])
        return state
    except Exception as e:
        print(f"[ckpt] WARNING: Failed to load mid-epoch checkpoint: {e}")
        return None

def clear_mid_epoch_checkpoint(cfg: dict, arch: str, suffix: str='_stable_es') -> None:
    path = os.path.join(cfg.get('weights_dir', 'model_weights'), f'{arch}{suffix}_midepoch.pt')
    if os.path.exists(path):
        os.remove(path)
