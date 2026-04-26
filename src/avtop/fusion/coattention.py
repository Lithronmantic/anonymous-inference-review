from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from src.avtop.utils.logging import get_logger
except Exception:
    import logging

    def get_logger(name):
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
        return logging.getLogger(name)
log = get_logger(__name__)

def _ensure_btd(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x if x.shape[1] <= x.shape[2] else x.transpose(1, 2).contiguous()
    if x.dim() == 4:
        B, C, T, Dp = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, T, C * Dp)
    if x.dim() == 5:
        B, C, T, H, W = x.shape
        return x.permute(0, 2, 1, 3, 4).contiguous().view(B, T, C * H * W)
    raise ValueError(f'[coatt._ensure_btd] Unsupported shape: {tuple(x.shape)}')

def _match_time(v: torch.Tensor, a: torch.Tensor, to: str='video') -> Tuple[torch.Tensor, torch.Tensor]:
    Tv, Ta = (v.shape[1], a.shape[1])
    if Tv == Ta:
        return (v, a)
    if to == 'video':
        a2 = F.interpolate(a.transpose(1, 2), size=Tv, mode='linear', align_corners=False).transpose(1, 2)
        return (v, a2)
    else:
        v2 = F.interpolate(v.transpose(1, 2), size=Ta, mode='linear', align_corners=False).transpose(1, 2)
        return (v2, a)

class _FeedForward(nn.Module):

    def __init__(self, d_model: int, dropout: float=0.1, mult: int=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, mult * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(mult * d_model, d_model), nn.Dropout(dropout))
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.ln(x + self.net(x))

class _BiCoAttnLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float, bottleneck_tokens: int):
        super().__init__()
        self.d_model = d_model
        self.bottleneck_tokens = int(bottleneck_tokens)
        if self.bottleneck_tokens > 0:
            self.q_a2v = nn.Parameter(torch.randn(self.bottleneck_tokens, d_model))
            self.q_v2a = nn.Parameter(torch.randn(self.bottleneck_tokens, d_model))
            nn.init.xavier_uniform_(self.q_a2v)
            nn.init.xavier_uniform_(self.q_v2a)
        self.attn_v_from_a = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_a_from_v = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        if self.bottleneck_tokens > 0:
            self.attn_a_reduce = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            self.attn_v_reduce = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln_v1 = nn.LayerNorm(d_model)
        self.ln_v2 = nn.LayerNorm(d_model)
        self.ln_a1 = nn.LayerNorm(d_model)
        self.ln_a2 = nn.LayerNorm(d_model)
        self.ffn_v = _FeedForward(d_model, dropout)
        self.ffn_a = _FeedForward(d_model, dropout)
        self.drop = nn.Dropout(dropout)

    def _reduce(self, reducer: nn.MultiheadAttention, src: torch.Tensor, q_param: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = src.size(0)
        q = q_param.unsqueeze(0).expand(B, -1, -1)
        out, w = reducer(query=q, key=src, value=src, need_weights=True)
        return (out, w)

    def forward(self, v: torch.Tensor, a: torch.Tensor, need_weights: bool=False):
        attn_logs: Dict[str, List[torch.Tensor]] = {} if need_weights else None
        if self.bottleneck_tokens > 0:
            a_sum, w_a = self._reduce(self.attn_a_reduce, a, self.q_a2v)
            v_sum, w_v = self._reduce(self.attn_v_reduce, v, self.q_v2a)
            if need_weights:
                attn_logs.setdefault('a_reduce', []).append(w_a)
                attn_logs.setdefault('v_reduce', []).append(w_v)
            k_a = v = v
            k_v = a = a
            kv_a = a_sum
            kv_v = v_sum
        else:
            kv_a = a
            kv_v = v
        v2, w_va = self.attn_v_from_a(query=v, key=kv_a, value=kv_a, need_weights=need_weights)
        v = self.ln_v1(v + self.drop(v2))
        v = self.ln_v2(v + self.drop(self.ffn_v(v)))
        a2, w_av = self.attn_a_from_v(query=a, key=kv_v, value=kv_v, need_weights=need_weights)
        a = self.ln_a1(a + self.drop(a2))
        a = self.ln_a2(a + self.drop(self.ffn_a(a)))
        if need_weights:
            attn_logs.setdefault('v_from_a', []).append(w_va)
            attn_logs.setdefault('a_from_v', []).append(w_av)
        return (v, a, attn_logs)

class EnhancedCoAttention(nn.Module):

    def __init__(self, video_dim: int, audio_dim: int, d_model: int=256, bottleneck_dim: int=128, num_layers: int=2, num_heads: int=8, dropout: float=0.1, match_time: str='video'):
        super().__init__()
        assert match_time in ('video', 'audio')
        self.d_model = int(d_model)
        self.match_time = match_time
        self.v_in = nn.Linear(video_dim, d_model) if video_dim != d_model else nn.Identity()
        self.a_in = nn.Linear(audio_dim, d_model) if audio_dim != d_model else nn.Identity()
        M = int(max(0, bottleneck_dim))
        self.layers = nn.ModuleList([_BiCoAttnLayer(d_model=d_model, num_heads=num_heads, dropout=dropout, bottleneck_tokens=M) for _ in range(num_layers)])
        self.fuse_proj = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        log.info(f'[CoAttn] init: d_model={d_model}, layers={num_layers}, heads={num_heads}, bottleneck_tokens={M}, match_time={match_time}')

    def forward(self, v: torch.Tensor, a: torch.Tensor, return_attn: bool=False):
        v = _ensure_btd(v)
        a = _ensure_btd(a)
        v, a = _match_time(v, a, to=self.match_time)
        v = self.v_in(v)
        a = self.a_in(a)
        attn_info = {'a_reduce': [], 'v_reduce': [], 'v_from_a': [], 'a_from_v': []} if return_attn else None
        for i, blk in enumerate(self.layers):
            need_w = return_attn
            v, a, logs = blk(v, a, need_weights=need_w)
            if need_w and logs is not None:
                for k in logs.keys():
                    attn_info[k].append(logs[k])
        fused = self.fuse_proj(torch.cat([v, a], dim=-1))
        z_v = v.mean(dim=1)
        z_a = a.mean(dim=1)
        if log.level <= 10:
            log.debug(f'[CoAttn] v:{tuple(v.shape)} a:{tuple(a.shape)} fused:{tuple(fused.shape)} z_v:{tuple(z_v.shape)} z_a:{tuple(z_a.shape)}')
        if return_attn:
            summary = {}
            for k, lst in attn_info.items():
                if not lst:
                    summary[k] = []
                    continue
                comp = []
                for w in lst:
                    dims = w.dim()
                    if dims == 4:
                        comp.append(w.mean(dim=(0, 1)))
                    elif dims == 3:
                        comp.append(w.mean(dim=0))
                    else:
                        comp.append(w)
                summary[k] = comp
            return (fused, z_v, z_a, summary)
        return (fused, z_v, z_a)
import torch
import torch.nn as nn

class CoAttentionFusion(nn.Module):

    def __init__(self, video_dim: int, audio_dim: int, d_model: int=256, num_layers: int=2, num_heads: int=8, dropout: float=0.1, bottleneck_dim: int=0, match_time: str='video'):
        super().__init__()
        self.core = EnhancedCoAttention(video_dim=video_dim, audio_dim=audio_dim, d_model=d_model, bottleneck_dim=bottleneck_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout, match_time=match_time)

    def forward(self, video_feat: torch.Tensor, audio_feat: torch.Tensor):
        out = self.core(video_feat, audio_feat, return_attn=False)
        fused, z_v, z_a = out[:3]
        aux = {'video_emb': z_v, 'audio_emb': z_a}
        return (fused, aux)
