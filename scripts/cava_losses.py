from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
EPS = 1e-08

def _flatten_bt(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        batch_size, time_steps, dim = x.shape
        return x.reshape(batch_size * time_steps, dim)
    if x.ndim == 2:
        return x
    raise ValueError(f'expect [B,T,D] or [N,D], got {tuple(x.shape)}')

def _mask_to_weights(mask: Optional[torch.Tensor], n: int, device, dtype=torch.float32) -> torch.Tensor:
    if mask is None:
        return torch.ones(n, device=device, dtype=dtype)
    weight = mask
    if weight.ndim == 3 and weight.size(-1) == 1:
        weight = weight.squeeze(-1)
    if weight.ndim == 2:
        weight = weight.reshape(-1)
    if weight.ndim != 1:
        raise ValueError(f'mask must be [B,T], [B,T,1] or [N], got {tuple(mask.shape)}')
    return weight.to(device=device, dtype=dtype).clamp(0.0, 1.0)

def _apply_temporal_exclusion(logits: torch.Tensor, radius: int) -> torch.Tensor:
    if int(radius) <= 0:
        return logits
    time_steps = logits.size(0)
    idx = torch.arange(time_steps, device=logits.device)
    dist = torch.abs(idx.view(-1, 1) - idx.view(1, -1))
    exclude = (dist <= int(radius)) & ~torch.eye(time_steps, device=logits.device, dtype=torch.bool)
    return logits.masked_fill(exclude, -10000.0)

def info_nce_align(audio_feat: torch.Tensor, video_feat: torch.Tensor, mask: Optional[torch.Tensor]=None, tau: float=0.07, normalize: bool=True, reduction: str='mean') -> torch.Tensor:
    audio_flat = _flatten_bt(audio_feat).float()
    video_flat = _flatten_bt(video_feat).float()
    if normalize:
        audio_flat = F.normalize(audio_flat, dim=-1, eps=EPS)
        video_flat = F.normalize(video_flat, dim=-1, eps=EPS)
    if audio_flat.shape[0] != video_flat.shape[0]:
        raise ValueError(f'mismatched flattened length: {audio_flat.shape[0]} vs {video_flat.shape[0]}')
    logits = audio_flat @ video_flat.t() / max(float(tau), EPS)
    logits = logits.clamp(-60.0, 60.0)
    target = torch.arange(audio_flat.shape[0], device=logits.device)
    loss_per_item = F.cross_entropy(logits, target, reduction='none')
    weight = _mask_to_weights(mask, audio_flat.shape[0], logits.device, logits.dtype)
    if reduction == 'none':
        return loss_per_item * weight
    if reduction == 'sum':
        return (loss_per_item * weight).sum()
    return (loss_per_item * weight).sum() / weight.sum().clamp_min(EPS)

def corr_diag_align(audio_feat: torch.Tensor, video_feat: torch.Tensor, mask: Optional[torch.Tensor]=None, reduction: str='mean') -> torch.Tensor:
    audio_flat = _flatten_bt(audio_feat).float()
    video_flat = _flatten_bt(video_feat).float()
    audio_flat = audio_flat - audio_flat.mean(dim=0, keepdim=True)
    video_flat = video_flat - video_flat.mean(dim=0, keepdim=True)
    audio_flat = F.normalize(audio_flat, dim=-1, eps=EPS)
    video_flat = F.normalize(video_flat, dim=-1, eps=EPS)
    corr = audio_flat @ video_flat.t()
    loss_per_item = 1.0 - torch.diag(corr)
    weight = _mask_to_weights(mask, loss_per_item.numel(), loss_per_item.device, loss_per_item.dtype)
    if reduction == 'none':
        return loss_per_item * weight
    if reduction == 'sum':
        return (loss_per_item * weight).sum()
    return (loss_per_item * weight).sum() / weight.sum().clamp_min(EPS)

def prior_l2(delta: torch.Tensor, mu: Optional[float], sigma: Optional[float]) -> torch.Tensor:
    if mu is None and sigma is None:
        return delta.new_zeros(())
    if mu is None or sigma is None:
        raise ValueError('prior_l2: both `prior_mu` and `prior_sigma` must be set when the prior regulariser is enabled (got mu=%r, sigma=%r).' % (mu, sigma))
    if float(sigma) <= 0:
        raise ValueError(f'prior_l2: `prior_sigma` must be > 0, got {sigma!r}.')
    z = (delta.float() - float(mu)) / float(sigma)
    return (z * z).mean()

def edge_hinge(delta: torch.Tensor, low: float, high: float, margin_ratio: float=0.25) -> torch.Tensor:
    if float(high) < float(low):
        raise ValueError('edge_hinge: high must >= low')
    low_v = float(low)
    high_v = float(high)
    margin = float(margin_ratio) * (high_v - low_v)
    delta = delta.float()
    left = F.relu(low_v + margin - delta) / (margin + EPS)
    right = F.relu(delta - (high_v - margin)) / (margin + EPS)
    return (left + right).mean()

def gate_temporal_smoothness(gate: torch.Tensor) -> torch.Tensor:
    gate = gate.float()
    if gate.ndim == 2:
        gate = gate.unsqueeze(-1)
    if gate.ndim != 3:
        raise ValueError(f'Expected gate to be [B,T] or [B,T,1], got {tuple(gate.shape)}')
    if gate.size(1) <= 1:
        return gate.new_zeros(())
    diff = gate[:, 1:, :] - gate[:, :-1, :]
    return (diff * diff).mean()

class CAVALoss(nn.Module):

    def __init__(self, cfg: Optional[Dict]=None):
        super().__init__()
        self.update_cfg(cfg or {})

    def update_cfg(self, cfg: Dict):
        c = dict(cfg or {})
        self.beta_align = float(c.get('lambda_cava', c.get('lambda_align', c.get('beta_align', 0.0))))
        self.beta_edge = float(c.get('lambda_edge', c.get('beta_edge', 0.0)))
        self.beta_prior = float(c.get('lambda_prior', c.get('beta_prior', 0.0)))
        self.beta_gate = float(c.get('lambda_gate', c.get('beta_gate', 0.0)))
        self.tau = float(c.get('tau_nce', c.get('tau', 0.2)))
        self.edge_margin_ratio = float(c.get('edge_margin_ratio', 0.25))
        self.prior_mu = c.get('prior_mu', c.get('delta_prior', None))
        self.prior_sigma = c.get('prior_sigma', 1.0 if self.prior_mu is not None else None)
        self.negative_mode = str(c.get('negative_mode', 'batch_global')).lower()
        self.temporal_exclusion_radius = int(c.get('temporal_exclusion_radius', 0))
        self._cfg = c

    def _resolve_delta_range(self, outputs: Dict) -> tuple[float, float]:
        low = outputs.get('delta_low', self._cfg.get('delta_low_frames', self._cfg.get('delta_low', -1.0)))
        high = outputs.get('delta_high', self._cfg.get('delta_high_frames', self._cfg.get('delta_high', 1.0)))
        return (float(low), float(high))

    def _align_loss(self, audio_feat: torch.Tensor, video_feat: torch.Tensor, gate: Optional[torch.Tensor]) -> torch.Tensor:
        mode = self.negative_mode
        if mode == 'batch_global':
            return info_nce_align(audio_feat, video_feat, mask=gate, tau=self.tau)
        if audio_feat.ndim != 3 or video_feat.ndim != 3:
            raise ValueError(f'{mode} requires [B,T,D] tensors')
        if audio_feat.shape != video_feat.shape:
            raise ValueError('audio_feat and video_feat must have the same shape for intra-sequence modes')
        batch_size, time_steps, _ = audio_feat.shape
        audio_norm = F.normalize(audio_feat.float(), dim=-1, eps=EPS)
        video_norm = F.normalize(video_feat.float(), dim=-1, eps=EPS)
        target = torch.arange(time_steps, device=audio_feat.device)
        if gate is None:
            weight = audio_feat.new_ones((batch_size, time_steps))
        else:
            weight = gate.squeeze(-1) if gate.ndim == 3 else gate
            weight = weight.float().clamp(0.0, 1.0)
        loss_sum = audio_feat.new_zeros(())
        weight_sum = audio_feat.new_zeros(())
        for batch_idx in range(batch_size):
            logits = audio_norm[batch_idx] @ video_norm[batch_idx].transpose(0, 1) / max(self.tau, EPS)
            logits = logits.clamp(-60.0, 60.0)
            if mode == 'intra_sequence_exclude_neighbors':
                logits = _apply_temporal_exclusion(logits, self.temporal_exclusion_radius)
            elif mode != 'intra_sequence_all':
                raise ValueError(f'unsupported negative_mode: {mode}')
            ce = F.cross_entropy(logits, target, reduction='none')
            loss_sum = loss_sum + (ce * weight[batch_idx]).sum()
            weight_sum = weight_sum + weight[batch_idx].sum()
        return loss_sum / weight_sum.clamp_min(EPS)

    def forward(self, outputs: Dict) -> Dict[str, torch.Tensor]:
        device = outputs['clip_logits'].device if 'clip_logits' in outputs and isinstance(outputs['clip_logits'], torch.Tensor) else None
        zero = torch.tensor(0.0, device=device) if device is not None else torch.tensor(0.0)
        audio_aligned = outputs.get('audio_aligned', outputs.get('audio_masked', outputs.get('audio_seq')))
        video_shifted = outputs.get('video_shifted', outputs.get('video_proj', outputs.get('video_seq')))
        gate = outputs.get('causal_gate', None)
        delta = outputs.get('delay_frames_cont', outputs.get('delay_frames', None))
        r_align = zero if audio_aligned is None or video_shifted is None else self._align_loss(audio_aligned, video_shifted, gate)
        loss_align = self.beta_align * r_align
        r_edge = zero
        if delta is not None:
            low, high = self._resolve_delta_range(outputs)
            r_edge = edge_hinge(delta, low, high, margin_ratio=self.edge_margin_ratio)
        loss_edge = self.beta_edge * r_edge
        r_prior = zero if delta is None or self.beta_prior <= 0.0 else prior_l2(delta, self.prior_mu, self.prior_sigma)
        loss_prior = self.beta_prior * r_prior
        r_gate = zero
        if gate is not None and self.beta_gate > 0.0:
            r_gate = gate_temporal_smoothness(gate)
        loss_gate = self.beta_gate * r_gate
        loss_total = loss_align + loss_edge + loss_prior + loss_gate
        return {'loss_total': loss_total, 'loss_align': loss_align, 'loss_edge': loss_edge, 'loss_prior': loss_prior, 'loss_gate': loss_gate, 'r_align': r_align, 'r_edge': r_edge, 'r_prior': r_prior, 'r_gate': r_gate}

def compute_cava_losses(outputs: Dict, cfg: Dict) -> Dict[str, torch.Tensor]:
    mod = CAVALoss(cfg)
    out = mod(outputs)
    return {'align': out['loss_align'], 'edge': out['loss_edge'], 'prior': out['loss_prior'], 'gate': out['loss_gate'], 'total': out['loss_total']}

def causal_supervised_loss(audio_proj: torch.Tensor, video_proj: torch.Tensor, class_labels: torch.Tensor, cava_module, weight: float=1.0) -> torch.Tensor:
    if cava_module is None or getattr(cava_module, 'class_delay', None) is None:
        return audio_proj.new_zeros(())
    scores = cava_module._corr_scores(audio_proj, video_proj)
    prob = F.softmax(scores, dim=1)
    max_delay = (prob.size(1) - 1) // 2
    offsets = torch.arange(-max_delay, max_delay + 1, device=prob.device, dtype=prob.dtype)
    exp_dt = (prob * offsets.unsqueeze(0)).sum(1)
    class_delay = cava_module.class_delay[class_labels.to(cava_module.class_delay.device)]
    return float(weight) * F.mse_loss(exp_dt, class_delay)

def causal_self_supervised_loss(audio_proj: torch.Tensor, video_proj: torch.Tensor, temperature: float=0.07) -> torch.Tensor:
    return info_nce_align(audio_proj, video_proj, mask=None, tau=float(temperature))
