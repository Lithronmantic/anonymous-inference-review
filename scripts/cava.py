from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
EPS = 1e-08

def _normalize_seq(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1, eps=EPS)

def _mask_log_bias(mask: Optional[torch.Tensor], dtype: torch.dtype) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.ndim != 3:
        raise ValueError(f'temporal mask must be [B,T,T], got {tuple(mask.shape)}')
    return torch.log(mask.float().clamp_min(EPS)).to(dtype=dtype)

class LearnableDelay(nn.Module):

    def __init__(self, delta_low: float, delta_high: float, init_mid: bool=True):
        super().__init__()
        if float(delta_high) < float(delta_low):
            raise ValueError('delta_high must be >= delta_low')
        self.register_buffer('delta_low', torch.tensor(float(delta_low)))
        self.register_buffer('delta_high', torch.tensor(float(delta_high)))
        init = 0.0 if init_mid else -2.0
        self.theta = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def forward(self, batch_size: int) -> torch.Tensor:
        theta = torch.clamp(self.theta, -12.0, 12.0)
        delta = self.delta_low + (self.delta_high - self.delta_low) * torch.sigmoid(theta)
        return delta.expand(int(batch_size))

class SampleLevelDelayPredictor(nn.Module):

    def __init__(self, d_model: int, delta_low: float, delta_high: float):
        super().__init__()
        if float(delta_high) < float(delta_low):
            raise ValueError('delta_high must be >= delta_low')
        self.register_buffer('delta_low', torch.tensor(float(delta_low)))
        self.register_buffer('delta_high', torch.tensor(float(delta_high)))
        self.mlp = nn.Sequential(nn.Linear(3 * int(d_model), int(d_model)), nn.GELU(), nn.Linear(int(d_model), 1))

    def forward(self, video_seq: torch.Tensor, audio_seq: torch.Tensor) -> torch.Tensor:
        video_pool = video_seq.mean(dim=1)
        audio_pool = audio_seq.mean(dim=1)
        feat = torch.cat([video_pool, audio_pool, video_pool * audio_pool], dim=-1)
        logits = self.mlp(feat).squeeze(-1).clamp(-12.0, 12.0)
        return self.delta_low + (self.delta_high - self.delta_low) * torch.sigmoid(logits)

class SoftTemporalShift(nn.Module):

    def forward(self, seq: torch.Tensor, delta_frames: torch.Tensor) -> torch.Tensor:
        if seq.ndim != 3:
            raise ValueError(f'seq must be [B,T,D], got {tuple(seq.shape)}')
        batch_size, time_steps, feat_dim = seq.shape
        if time_steps <= 1:
            return seq
        if delta_frames.ndim == 0:
            delta_frames = delta_frames.view(1).expand(batch_size)
        delta = delta_frames.view(batch_size, 1, 1).clamp(min=-float(time_steps - 1), max=float(time_steps - 1))
        tidx = torch.arange(time_steps, device=seq.device, dtype=seq.dtype).view(1, time_steps, 1)
        position = torch.clamp(tidx + delta.to(seq.dtype), 0.0, float(time_steps - 1))
        idx0_float = torch.floor(position)
        alpha = position - idx0_float
        idx0 = idx0_float.long()
        idx1 = torch.clamp(idx0 + 1, 0, time_steps - 1)
        x0 = torch.gather(seq, 1, idx0.expand(batch_size, time_steps, feat_dim))
        x1 = torch.gather(seq, 1, idx1.expand(batch_size, time_steps, feat_dim))
        return (1.0 - alpha) * x0 + alpha * x1

def soft_shift_right(audio_seq: torch.Tensor, delta_frames: torch.Tensor) -> torch.Tensor:
    return SoftTemporalShift()(audio_seq, delta_frames)

class DisplacementAwareCausalMask(nn.Module):

    def __init__(self, window_size: int=5, mask_type: str='hard', multi_scale: bool=False):
        super().__init__()
        if int(window_size) < 1:
            raise ValueError('window_size must be >= 1')
        mask_type = str(mask_type).lower()
        if mask_type not in {'hard', 'gaussian'}:
            raise ValueError('mask_type must be one of: hard, gaussian')
        self.window_size = int(window_size)
        self.mask_type = mask_type
        self.multi_scale = bool(multi_scale)

    def _single_scale(self, delta: torch.Tensor, tlen: int, window: int, device: torch.device) -> torch.Tensor:
        batch_size = delta.shape[0]
        t = torch.arange(tlen, device=device, dtype=torch.float32).view(1, tlen, 1)
        tau = torch.arange(tlen, device=device, dtype=torch.float32).view(1, 1, tlen)
        center = torch.clamp(t + delta.view(batch_size, 1, 1).float(), 0.0, float(tlen - 1))
        if self.mask_type == 'hard':
            raw = (torch.abs(tau - center) <= float(window)).float()
        else:
            sigma = max(float(window), 1.0)
            raw = torch.exp(-0.5 * ((tau - center) / sigma) ** 2)
        return raw / raw.sum(dim=-1, keepdim=True).clamp_min(EPS)

    def forward(self, delta_frames: torch.Tensor, tlen: int) -> torch.Tensor:
        if int(tlen) <= 0:
            raise ValueError('tlen must be positive')
        if delta_frames.ndim == 0:
            delta_frames = delta_frames.view(1)
        if not self.multi_scale:
            return self._single_scale(delta_frames, int(tlen), self.window_size, delta_frames.device)
        windows = [self.window_size, self.window_size * 2, self.window_size * 4]
        masks = [self._single_scale(delta_frames, int(tlen), w, delta_frames.device) for w in windows]
        out = torch.stack(masks, dim=0).mean(dim=0)
        return out / out.sum(dim=-1, keepdim=True).clamp_min(EPS)

class AlignmentGate(nn.Module):

    def __init__(self, d_model: int, gate_min: float=0.05, gate_max: float=0.95):
        super().__init__()
        if float(gate_max) <= float(gate_min):
            raise ValueError('gate_max must be > gate_min')
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        hidden = max(64, min(4 * int(d_model), 2048))
        self.net = nn.Sequential(nn.Linear(3 * int(d_model), hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, aligned_seq: torch.Tensor, shifted_seq: torch.Tensor) -> torch.Tensor:
        aligned_norm = _normalize_seq(aligned_seq)
        shifted_norm = _normalize_seq(shifted_seq)
        feat = torch.cat([aligned_norm, shifted_norm, aligned_norm * shifted_norm], dim=-1)
        batch_size, time_steps, _ = feat.shape
        logits = self.net(feat.reshape(batch_size * time_steps, -1)).reshape(batch_size, time_steps, 1)
        gate = torch.sigmoid(torch.clamp(logits, -12.0, 12.0))
        return gate.clamp(min=self.gate_min, max=self.gate_max)

@dataclass
class CAVAConfigResolved:
    delta_low: float
    delta_high: float
    d_model: int
    window_size: int
    mask_type: str
    multi_scale: bool
    gate_min: float
    gate_max: float
    dist_max_delay: int

def _resolve_gate_range(cava_cfg: Dict) -> tuple[float, float]:
    if 'gate_range' in cava_cfg and isinstance(cava_cfg['gate_range'], (list, tuple)) and (len(cava_cfg['gate_range']) == 2):
        return (float(cava_cfg['gate_range'][0]), float(cava_cfg['gate_range'][1]))
    mode = str(cava_cfg.get('gate_range_mode', 'strict')).lower()
    if mode == 'legacy':
        return (0.01, 0.99)
    return (float(cava_cfg.get('gate_min', 0.05)), float(cava_cfg.get('gate_max', 0.95)))

class CAVAModule(nn.Module):

    def __init__(self, video_dim: int, audio_dim: int, d_model: int=256, delta_low_frames: float=2.0, delta_high_frames: float=6.0, delta_prior: float=0.0, gate_clip_min: Optional[float]=None, gate_clip_max: Optional[float]=None, num_classes: Optional[int]=None, dist_max_delay: int=6, window_size: int=5, mask_type: str='hard', multi_scale: bool=False, gate_range_mode: str='strict', gate_range: Optional[list]=None, use_learnable_delay: bool=True, use_mask: bool=True, use_gate: bool=True):
        super().__init__()
        cava_cfg = {'gate_range_mode': gate_range_mode, 'gate_range': gate_range, 'gate_min': gate_clip_min if gate_clip_min is not None else 0.05, 'gate_max': gate_clip_max if gate_clip_max is not None else 0.95}
        gate_min, gate_max = _resolve_gate_range(cava_cfg)
        self.cfg = CAVAConfigResolved(delta_low=float(delta_low_frames), delta_high=float(delta_high_frames), d_model=int(d_model), window_size=int(window_size), mask_type=str(mask_type).lower(), multi_scale=bool(multi_scale), gate_min=float(gate_min), gate_max=float(gate_max), dist_max_delay=int(dist_max_delay))
        self.d_model = self.cfg.d_model
        self.use_learnable_delay = bool(use_learnable_delay)
        self.use_mask = bool(use_mask)
        self.use_gate = bool(use_gate)
        self.v_proj = nn.Linear(video_dim, self.cfg.d_model) if int(video_dim) != self.cfg.d_model else nn.Identity()
        self.a_proj = nn.Linear(audio_dim, self.cfg.d_model) if int(audio_dim) != self.cfg.d_model else nn.Identity()
        if self.use_learnable_delay:
            self.delay = SampleLevelDelayPredictor(d_model=self.cfg.d_model, delta_low=self.cfg.delta_low, delta_high=self.cfg.delta_high)
        else:
            self.delay = None
            self.register_buffer('delta_fixed', torch.tensor(float(delta_prior)))
        self.shift = SoftTemporalShift()
        self.mask = DisplacementAwareCausalMask(window_size=self.cfg.window_size, mask_type=self.cfg.mask_type, multi_scale=self.cfg.multi_scale) if self.use_mask else None
        self.gate = AlignmentGate(d_model=self.cfg.d_model, gate_min=self.cfg.gate_min, gate_max=self.cfg.gate_max) if self.use_gate else None
        self.class_delay = None
        self.dist_max_delay = self.cfg.dist_max_delay
        self.register_buffer('delta_low', torch.tensor(self.cfg.delta_low))
        self.register_buffer('delta_high', torch.tensor(self.cfg.delta_high))

    def _directional_cross_attention(self, audio_proj: torch.Tensor, video_shifted: torch.Tensor, temporal_mask: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if audio_proj.shape != video_shifted.shape:
            raise ValueError(f'audio_proj and video_shifted must have identical [B,T,D] shapes, got {tuple(audio_proj.shape)} vs {tuple(video_shifted.shape)}')
        d_model = float(audio_proj.size(-1))
        attn_logits = torch.matmul(audio_proj.float(), video_shifted.float().transpose(1, 2)) / max(d_model ** 0.5, 1.0)
        mask_bias = _mask_log_bias(temporal_mask, attn_logits.dtype)
        if mask_bias is not None:
            attn_logits = attn_logits + mask_bias
        attn_weights = F.softmax(attn_logits, dim=-1)
        audio_context = torch.bmm(attn_weights, video_shifted.float())
        audio_aligned = F.layer_norm(audio_proj.float() + audio_context, [self.cfg.d_model])
        return (audio_aligned, attn_weights)

    def _corr_scores(self, audio_seq: torch.Tensor, video_seq: torch.Tensor) -> torch.Tensor:
        if audio_seq.ndim == 2:
            audio_seq = audio_seq.unsqueeze(1)
        if video_seq.ndim == 2:
            video_seq = video_seq.unsqueeze(1)
        batch_size = audio_seq.shape[0]
        time_steps = min(audio_seq.shape[1], video_seq.shape[1])
        audio_norm = _normalize_seq(audio_seq[:, :time_steps, :])
        video_norm = _normalize_seq(video_seq[:, :time_steps, :])
        max_delay = min(int(self.dist_max_delay), time_steps - 1)
        scores = []
        for delay in range(-max_delay, max_delay + 1):
            if delay == 0:
                score = (audio_norm * video_norm).sum(dim=-1).mean(dim=1)
            elif delay > 0:
                score = (audio_norm[:, :-delay, :] * video_norm[:, delay:, :]).sum(dim=-1).mean(dim=1) if delay < time_steps else audio_norm.new_zeros((batch_size,))
            else:
                dd = -delay
                score = (audio_norm[:, dd:, :] * video_norm[:, :-dd, :]).sum(dim=-1).mean(dim=1) if dd < time_steps else audio_norm.new_zeros((batch_size,))
            scores.append(score)
        return torch.stack(scores, dim=1)

    def get_predicted_delay(self, audio_seq: torch.Tensor, video_seq: torch.Tensor) -> torch.Tensor:
        scores = self._corr_scores(audio_seq, video_seq)
        prob = F.softmax(scores, dim=1)
        time_steps = min(audio_seq.shape[1] if audio_seq.ndim == 3 else 1, video_seq.shape[1] if video_seq.ndim == 3 else 1)
        max_delay = min(int(self.dist_max_delay), time_steps - 1)
        return prob.argmax(dim=1) - max_delay

    def forward(self, video_seq: torch.Tensor, audio_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        if video_seq.ndim != 3 or audio_seq.ndim != 3:
            raise ValueError('CAVAModule expects [B,T,D] inputs')
        batch_size, time_steps, _ = video_seq.shape
        if time_steps != audio_seq.shape[1]:
            time_steps = min(time_steps, audio_seq.shape[1])
            video_seq = video_seq[:, :time_steps, :]
            audio_seq = audio_seq[:, :time_steps, :]
        video_proj = F.layer_norm(self.v_proj(video_seq.float()), [self.cfg.d_model])
        audio_proj = F.layer_norm(self.a_proj(audio_seq.float()), [self.cfg.d_model])
        if self.use_learnable_delay:
            delay = self.delay(video_proj, audio_proj)
        else:
            delay = self.delta_fixed.expand(batch_size)
        video_shifted = self.shift(video_proj, delay)
        temporal_mask = self.mask(delay, time_steps) if self.mask is not None else None
        audio_aligned, cross_attn_weights = self._directional_cross_attention(audio_proj=audio_proj, video_shifted=video_shifted, temporal_mask=temporal_mask)
        gate = self.gate(audio_aligned, video_shifted) if self.gate is not None else None
        delay_distribution = F.softmax(self._corr_scores(audio_proj, video_proj), dim=1)
        max_delay = min(int(self.dist_max_delay), time_steps - 1)
        pred_delay = delay_distribution.argmax(dim=1) - max_delay
        return {'video_for_fusion': video_shifted, 'audio_for_fusion': audio_aligned, 'audio_aligned': audio_aligned, 'audio_masked': audio_aligned, 'audio_context': audio_aligned - audio_proj, 'audio_proj': audio_proj, 'video_proj': video_proj, 'video_shifted': video_shifted, 'audio_seq': audio_proj, 'causal_gate': gate, 'causal_mask': temporal_mask, 'cross_attn_weights': cross_attn_weights, 'delay_frames': delay, 'delay_frames_cont': delay, 'delta_low': float(self.delta_low.item()), 'delta_high': float(self.delta_high.item()), 'causal_prob': gate.squeeze(-1) if gate is not None else None, 'causal_prob_dist': delay_distribution, 'pred_delay': pred_delay}
