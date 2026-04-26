import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple
import torch
import torch.nn as nn
DEFAULT_7D_SOURCES = ['max_prob', 'entropy', 'margin', 'student_feat_norm', 'loss_mean', 'loss_std', 'g_bar']
EXTENDED_7D_SOURCES = ['max_prob', 'entropy', 'margin', 'g_bar', 'delta_prior_deviation', 'loss_trend', 'student_feat_norm']

def _safe_cfg_get(cfg: Any, *keys: str, default: Any=None) -> Any:
    cur = cfg
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k, None)
        else:
            cur = getattr(cur, k, None)
    return default if cur is None else cur

def _as_1d_tensor(x: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype, batch_size: int) -> torch.Tensor:
    if x is None:
        return torch.zeros(batch_size, device=device, dtype=dtype)
    if not torch.is_tensor(x):
        return torch.full((batch_size,), float(x), device=device, dtype=dtype)
    if x.ndim == 0:
        return x.to(device=device, dtype=dtype).expand(batch_size)
    if x.ndim == 1:
        return x.to(device=device, dtype=dtype)
    return x.to(device=device, dtype=dtype).reshape(x.shape[0], -1).mean(dim=1)

class MetaNet(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int=64, architecture: str='mlp2', num_hidden_layers: int=1, dropout: float=0.0) -> None:
        super().__init__()
        self.architecture = architecture
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        if architecture == 'mlp2':
            self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        else:
            layers = [nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
            for _ in range(max(0, num_hidden_layers - 1)):
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)])
            layers.extend([nn.Linear(hidden_dim, 1), nn.Sigmoid()])
            self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)

class MetaReweighter(nn.Module):

    def __init__(self, cfg: Optional[Any]=None, **kwargs: Any) -> None:
        super().__init__()
        self.cfg = cfg
        profile = kwargs.get('profile', _safe_cfg_get(cfg, 'profile', default='default'))
        explicit_mode = kwargs.get('feature_mode', _safe_cfg_get(cfg, 'mlpr', 'feature_mode', default=None))
        if explicit_mode is None:
            self.feature_mode = '7d' if profile == 'default' else 'research_extended_7d'
        else:
            self.feature_mode = explicit_mode
        self.history_window = int(kwargs.get('history_window', _safe_cfg_get(cfg, 'mlpr', 'history_window', default=5)))
        self.history_window = max(1, self.history_window)
        self._loss_history: deque[float] = deque(maxlen=self.history_window)
        clip = kwargs.get('weight_clip', _safe_cfg_get(cfg, 'mlpr', 'weight_clip', default=(0.05, 0.95)))
        if isinstance(clip, (list, tuple)) and len(clip) == 2:
            self.weight_low = float(clip[0])
            self.weight_high = float(clip[1])
        else:
            self.weight_low, self.weight_high = (0.05, 0.95)
        hidden_dim = int(kwargs.get('hidden_dim', _safe_cfg_get(cfg, 'mlpr', 'hidden_dim', default=64)))
        num_hidden_layers = int(kwargs.get('num_hidden_layers', _safe_cfg_get(cfg, 'mlpr', 'num_hidden_layers', default=1)))
        dropout = float(kwargs.get('dropout', _safe_cfg_get(cfg, 'mlpr', 'dropout', default=0.0)))
        arch = kwargs.get('architecture', _safe_cfg_get(cfg, 'mlpr', 'architecture', default=None))
        if arch is None:
            arch = 'mlp2' if profile == 'default' else 'extended_mlp'
        self.feature_sources = self._resolve_feature_sources(self.feature_mode)
        self.feature_dim = len(self.feature_sources)
        self.meta_net = MetaNet(in_dim=self.feature_dim, hidden_dim=hidden_dim, architecture=arch, num_hidden_layers=num_hidden_layers, dropout=dropout)
        print(f'[MLPR] feature_mode={self.feature_mode}, feature_dim={self.feature_dim}')
        print(f'[MLPR] feature_sources={self.feature_sources}')

    def _resolve_feature_sources(self, mode: str) -> Sequence[str]:
        if mode == '7d':
            return DEFAULT_7D_SOURCES
        if mode in ('research_extended_7d', 'extended_7d'):
            return EXTENDED_7D_SOURCES
        return DEFAULT_7D_SOURCES

    def update_loss_history(self, step_loss: Any) -> None:
        if torch.is_tensor(step_loss):
            val = float(step_loss.detach().mean().item())
        else:
            val = float(step_loss)
        if math.isfinite(val):
            self._loss_history.append(val)

    def _history_mean_std(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self._loss_history) == 0:
            mean = torch.zeros(batch_size, device=device, dtype=dtype)
            std = torch.zeros(batch_size, device=device, dtype=dtype)
            return (mean, std)
        hist = torch.tensor(list(self._loss_history), device=device, dtype=dtype)
        mean = hist.mean().expand(batch_size)
        std = hist.std(unbiased=False).expand(batch_size)
        return (mean, std)

    def build_features(self, *, logits: Optional[torch.Tensor]=None, probs: Optional[torch.Tensor]=None, student_feat: Optional[torch.Tensor]=None, gate: Optional[torch.Tensor]=None, per_sample_loss: Optional[torch.Tensor]=None, delta: Optional[torch.Tensor]=None, delta_prior: Optional[torch.Tensor]=None) -> torch.Tensor:
        if probs is None:
            if logits is None:
                raise ValueError('Either `probs` or `logits` must be provided.')
            probs = torch.softmax(logits, dim=-1)
        device = probs.device
        dtype = probs.dtype
        bsz = probs.shape[0]
        max_prob = probs.max(dim=-1).values
        top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
        if top2.shape[-1] == 1:
            margin = top2[..., 0]
        else:
            margin = top2[..., 0] - top2[..., 1]
        entropy = -(probs * probs.clamp_min(1e-08).log()).sum(dim=-1)
        if student_feat is None:
            student_feat_norm = torch.zeros(bsz, device=device, dtype=dtype)
        else:
            student_feat_norm = student_feat.reshape(student_feat.shape[0], -1).norm(p=2, dim=1)
        g_bar = _as_1d_tensor(gate, device=device, dtype=dtype, batch_size=bsz)
        if self.feature_mode == '7d':
            loss_mean, loss_std = self._history_mean_std(bsz, device, dtype)
            feat = torch.stack([max_prob, entropy, margin, student_feat_norm, loss_mean, loss_std, g_bar], dim=-1)
            if per_sample_loss is not None:
                self.update_loss_history(_as_1d_tensor(per_sample_loss, device, dtype, bsz).mean())
            return feat
        delta_t = _as_1d_tensor(delta, device=device, dtype=dtype, batch_size=bsz)
        if delta_prior is None:
            delta_prior_t = torch.zeros_like(delta_t)
        else:
            delta_prior_t = _as_1d_tensor(delta_prior, device=device, dtype=dtype, batch_size=bsz)
        delta_dev = (delta_t - delta_prior_t).abs()
        loss_t = _as_1d_tensor(per_sample_loss, device=device, dtype=dtype, batch_size=bsz)
        if len(self._loss_history) > 0:
            hist_mean = float(sum(self._loss_history) / len(self._loss_history))
            loss_trend = loss_t - hist_mean
        else:
            loss_trend = torch.zeros_like(loss_t)
        feat = torch.stack([max_prob, entropy, margin, g_bar, delta_dev, loss_trend, student_feat_norm], dim=-1)
        return feat

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        w01 = self.meta_net(features)
        return self.weight_low + (self.weight_high - self.weight_low) * w01

    def build_meta_features(self, **kwargs: Any) -> torch.Tensor:
        return self.build_features(**kwargs)

    def compute_weights(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward(features)

    def reweight(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward(features)

    @property
    def meta_architecture(self) -> str:
        return self.meta_net.architecture

def build_meta_reweighter(cfg: Any) -> MetaReweighter:
    return MetaReweighter(cfg=cfg)

def build_mlpr_features(*, teacher_prob: torch.Tensor, student_feat: Optional[torch.Tensor]=None, history_mean: Optional[torch.Tensor]=None, history_std: Optional[torch.Tensor]=None, cava_gate_mean: Optional[torch.Tensor]=None, use_prob_vector: bool=False, feature_mode: str='7d', delay_frames: Optional[torch.Tensor]=None, delta_prior: Optional[float]=0.0, loss_trend: Optional[torch.Tensor]=None) -> torch.Tensor:
    probs = teacher_prob
    device = probs.device
    dtype = probs.dtype
    bsz = probs.shape[0]
    max_prob = probs.max(dim=-1).values
    top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
    if top2.shape[-1] >= 2:
        margin = top2[..., 0] - top2[..., 1]
    else:
        margin = top2[..., 0]
    entropy = -(probs * probs.clamp_min(1e-08).log()).sum(dim=-1)
    if student_feat is None:
        student_feat_norm = torch.zeros(bsz, device=device, dtype=dtype)
    else:
        student_feat_norm = student_feat.reshape(student_feat.shape[0], -1).norm(p=2, dim=1)
    if cava_gate_mean is None:
        g_bar = torch.zeros(bsz, device=device, dtype=dtype)
    else:
        g_bar = cava_gate_mean.reshape(-1)[:bsz].to(dtype)
    if feature_mode in ('7d', 'legacy'):
        if history_mean is not None:
            loss_mean = history_mean.reshape(bsz).to(dtype)
        else:
            loss_mean = torch.zeros(bsz, device=device, dtype=dtype)
        if history_std is not None:
            loss_std = history_std.reshape(bsz).to(dtype)
        else:
            loss_std = torch.zeros(bsz, device=device, dtype=dtype)
        feat = torch.stack([max_prob, entropy, margin, student_feat_norm, loss_mean, loss_std, g_bar], dim=-1)
    else:
        if delay_frames is not None:
            delta_t = delay_frames.reshape(bsz).to(dtype)
        else:
            delta_t = torch.zeros(bsz, device=device, dtype=dtype)
        if delta_prior is None:
            delta_dev = torch.zeros_like(delta_t)
        else:
            delta_dev = (delta_t - float(delta_prior)).abs()
        if loss_trend is not None:
            lt = loss_trend.reshape(bsz).to(dtype)
        else:
            lt = torch.zeros(bsz, device=device, dtype=dtype)
        feat = torch.stack([max_prob, entropy, margin, g_bar, delta_dev, lt, student_feat_norm], dim=-1)
    return feat
