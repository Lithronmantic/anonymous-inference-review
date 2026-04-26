import math
import torch
import torch.nn.functional as F

def kl_divergence(student_logits: torch.Tensor, teacher_prob: torch.Tensor, T: float=1.0, reduction: str='mean') -> torch.Tensor:
    eps = 1e-08
    t = teacher_prob.clamp_min(eps)
    t = t / t.sum(dim=1, keepdim=True).clamp_min(eps)
    log_ps = F.log_softmax(student_logits / T, dim=1)
    log_pt = t.log()
    kl = (t * (log_pt - log_ps)).sum(dim=1)
    if reduction == 'mean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    return kl

def soft_ce(student_logits: torch.Tensor, teacher_prob: torch.Tensor, T: float=1.0, reduction: str='mean') -> torch.Tensor:
    log_ps = F.log_softmax(student_logits / T, dim=1)
    t = teacher_prob / teacher_prob.sum(dim=1, keepdim=True).clamp_min(1e-08)
    ce = -(t * log_ps).sum(dim=1)
    return ce.mean() if reduction == 'mean' else ce

def hard_ce(student_logits: torch.Tensor, hard_targets: torch.Tensor, reduction: str='mean') -> torch.Tensor:
    return F.cross_entropy(student_logits, hard_targets, reduction=reduction)

def ramp_up(curr_epoch: int, ramp_epochs: int) -> float:
    if ramp_epochs <= 0:
        return 1.0
    t = max(0.0, min(1.0, curr_epoch / float(ramp_epochs)))
    return 3 * t ** 2 - 2 * t ** 3
