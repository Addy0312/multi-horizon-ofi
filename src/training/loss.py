import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def _deep_focal_loss(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None=None, gamma: float=2.0, label_smoothing: float=0.0) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, weight=weight, reduction='none', label_smoothing=float(label_smoothing))
    p_t = torch.exp(-ce)
    loss = (1.0 - p_t) ** float(gamma) * ce
    return loss.mean()

class WeightedCrossEntropyLoss(nn.Module):
    """
    Calculates class weights dynamically from the training labels (inverse frequency).
    """

    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, inputs, targets):
        if self.weight is not None:
            self.weight = self.weight.to(inputs.device)
        return F.cross_entropy(inputs, targets, weight=self.weight)

class FocalLoss(nn.Module):
    """
    Implements Focal Loss to heavily penalize missing the minority classes 
    while ignoring the easy majority class.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if alpha is None:
            self.alpha = torch.tensor([1.0, 1.0, 1.0])
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        self.alpha = self.alpha.to(inputs.device)
        alpha_t = self.alpha.gather(0, targets.view(-1)).view(targets.shape)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def _deep_multihorizon_loss_advanced(logits: torch.Tensor, targets: torch.Tensor, weight_tensors: list[torch.Tensor] | None=None, label_smoothing: float=0.0, loss_mode: str | None=None, focal_gamma: float | None=None) -> torch.Tensor:
    if loss_mode is None:
        loss_mode = str(globals().get('DEEP_CONFIG', {}).get('loss_mode', 'ce'))
    losses = []
    for i in range(logits.shape[1]):
        w = None if weight_tensors is None else weight_tensors[i]
        if loss_mode == 'focal_advanced':
            if 'advanced_focal_criterion' not in globals():
                globals()['advanced_focal_criterion'] = FocalLoss(gamma=2.0, reduction='mean')
            loss_i = globals()['advanced_focal_criterion'](logits[:, i, :], targets[:, i])
        elif loss_mode == 'weighted_ce_advanced':
            criterion = WeightedCrossEntropyLoss(weight=w)
            loss_i = criterion(logits[:, i, :], targets[:, i])
        elif loss_mode == 'cb_focal':
            loss_i = _deep_focal_loss(logits[:, i, :], targets[:, i], weight=w, gamma=float(focal_gamma if focal_gamma is not None else 2.0), label_smoothing=float(label_smoothing))
        else:
            loss_i = F.cross_entropy(logits[:, i, :], targets[:, i], weight=w, label_smoothing=float(label_smoothing))
        losses.append(loss_i)
    return torch.stack(losses).mean()

