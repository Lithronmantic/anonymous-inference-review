#!/usr/bin/env python3
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from typing import Dict, Optional, Tuple
try:
    from src.avtop.models.enhanced_audio_backbones import LightVGGishAudioBackbone, ModerateVGGishAudioBackbone, ImprovedAudioBackbone
    _HAS_EXT_AUDIO = True
except Exception:
    _HAS_EXT_AUDIO = False
try:
    from src.avtop.fusion.cfa_fusion import CFAFusion
    HAS_CFA = True
except Exception:
    HAS_CFA = False
    print(' CFAFusion not found, using default fusion')
try:
    from src.avtop.fusion.ib_fusion import InformationBottleneckFusion
    HAS_IB = True
except Exception:
    HAS_IB = False
try:
    from src.avtop.fusion.coattention import CoAttentionFusion
    HAS_COATTN = True
except Exception:
    HAS_COATTN = False
try:
    from src.avtop.fusion.mult_style_transformer import MulTStyleFusion
    HAS_MULT_STYLE = True
except Exception:
    HAS_MULT_STYLE = False
try:
    from src.avtop.models.backbones import VideoBackbone as ImportedVideoBackbone, AudioBackbone as ImportedAudioBackbone
    HAS_BACKBONES = True
except Exception:
    HAS_BACKBONES = False
    print(' Backbones not found, using local/dummy backbones')
try:
    from src.avtop.models.temporal_encoder import SimpleTemporalEncoder
    HAS_TEMPORAL = True
except Exception:
    HAS_TEMPORAL = False
try:
    from cava import CAVAModule
    HAS_CAVA = True
except Exception:
    try:
        from scripts.cava import CAVAModule
        HAS_CAVA = True
    except Exception:
        HAS_CAVA = False
        print('[WARN] CAVA module not found, proceeding without causal alignment')
try:
    from config_system import resolve_runtime_config
except Exception:
    from scripts.config_system import resolve_runtime_config
import logging
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)
try:
    from enhanced_mil import EnhancedMIL
except Exception:
    from scripts.enhanced_mil import EnhancedMIL

def _extract(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

class SafeCoAttention(nn.Module):

    def __init__(self, core, video_dim: int, audio_dim: int, fusion_dim: Optional[int]=None):
        super().__init__()
        self.core = core
        self.video_dim = int(video_dim)
        self.audio_dim = int(audio_dim)
        self.fusion_dim = int(fusion_dim) if fusion_dim is not None else None
        vin, ain = self._guess_expected_dims()
        if (vin == self.audio_dim and ain == self.video_dim) and (not (vin == self.video_dim and ain == self.audio_dim)):
            self.call_order = 'av'
            self.want_video_dim = int(ain)
            self.want_audio_dim = int(vin)
        else:
            self.call_order = 'va'
            self.want_video_dim = int(vin)
            self.want_audio_dim = int(ain)
        self.v_proj: Optional[nn.Module] = None
        if self.video_dim != self.want_video_dim:
            self.v_proj = nn.Linear(self.video_dim, self.want_video_dim, bias=True).float()
        self.a_proj: Optional[nn.Module] = None
        if self.audio_dim != self.want_audio_dim:
            self.a_proj = nn.Linear(self.audio_dim, self.want_audio_dim, bias=True).float()
        core_obj = getattr(self.core, 'core', self.core)
        core_out_dim = int(getattr(core_obj, 'd_model', self.want_video_dim))
        self.out_proj: Optional[nn.Module] = None
        if self.fusion_dim is not None and core_out_dim != self.fusion_dim:
            self.out_proj = nn.Linear(core_out_dim, self.fusion_dim, bias=True).float()

    def _first_linear_in_features(self, mod: nn.Module) -> Optional[int]:
        if isinstance(mod, nn.Linear):
            return int(mod.in_features)
        if isinstance(mod, nn.Sequential):
            for m in mod.modules():
                if isinstance(m, nn.Linear):
                    return int(m.in_features)
        return None

    def _guess_expected_dims(self) -> Tuple[Optional[int], Optional[int]]:
        obj = getattr(self.core, 'core', self.core)
        vin = self._first_linear_in_features(getattr(obj, 'v_in', None)) if hasattr(obj, 'v_in') else None
        ain = self._first_linear_in_features(getattr(obj, 'a_in', None)) if hasattr(obj, 'a_in') else None
        if vin is None and ain is None:
            dm = None
            for k in ('d_model', 'embed_dim', 'model_dim', 'hidden_dim'):
                dm = getattr(obj, k, None)
                if dm is not None:
                    try:
                        dm = int(dm)
                        break
                    except Exception:
                        dm = None
            if dm is not None:
                vin = vin or dm
                ain = ain or dm
        vin = vin or self.video_dim
        ain = ain or self.audio_dim
        return (int(vin), int(ain))

    def _adapt_lastdim(self, x: torch.Tensor, want: int, proj: Optional[nn.Module], name: str):
        B, T, Din = x.shape
        if Din == want:
            return x
        if proj is None:
            raise ValueError(f'{name} dim mismatch: got {Din}, expected {want}')
        x2 = x.reshape(B * T, Din).to(proj.weight.dtype)
        x2 = proj(x2)
        return x2.reshape(B, T, want)

    def _ensure_btd(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, int]:
        if x.dim() == 3:
            return (x, x.size(0), x.size(1), x.size(2))
        if x.dim() == 2:
            return (x.unsqueeze(1), x.size(0), 1, x.size(1))
        raise ValueError(f'SafeCoAttention expects a 2D or 3D tensor, got shape {tuple(x.shape)}')

    def forward(self, v: torch.Tensor, a: torch.Tensor, **kw):
        v, Bv, Tv, Dv = self._ensure_btd(v)
        a, Ba, Ta, Da = self._ensure_btd(a)
        assert Bv == Ba, 'CoAttn batch sizes must match'
        vin, ain = self._guess_expected_dims()
        call = 'va'
        if (vin == Da and ain == Dv) and (not (vin == Dv and ain == Da)):
            call = 'av'
            want_v, want_a = (ain, vin)
        else:
            want_v, want_a = (vin, ain)
        v = self._adapt_lastdim(v, self.want_video_dim, self.v_proj, 'video')
        a = self._adapt_lastdim(a, self.want_audio_dim, self.a_proj, 'audio')
        if Tv != Ta:
            T = max(Tv, Ta)
            if Tv < T:
                v = F.pad(v, (0, 0, 0, T - Tv))
            if Ta < T:
                a = F.pad(a, (0, 0, 0, T - Ta))
        else:
            T = Tv
        out = self.core(v, a, **kw) if self.call_order == 'va' else self.core(a, v, **kw)
        if isinstance(out, tuple):
            fused, aux = out
        else:
            fused, aux = (out, {})
        if fused.dim() == 2:
            fused = fused.reshape(Bv, T, -1)
        if self.fusion_dim is not None and fused.size(-1) != self.fusion_dim:
            if self.out_proj is None:
                raise ValueError(f'Fusion dim mismatch: got {fused.size(-1)}, expected {self.fusion_dim}')
            fused = self.out_proj(fused.reshape(Bv * T, -1).to(self.out_proj.weight.dtype)).reshape(Bv, T, self.fusion_dim)
        aux.setdefault('video_seq', v)
        aux.setdefault('audio_seq', a)
        aux.setdefault('video_emb', v.mean(dim=1))
        aux.setdefault('audio_emb', a.mean(dim=1))
        aux.setdefault('call_order', self.call_order)
        aux.setdefault('want_dims', (self.want_video_dim, self.want_audio_dim))
        return (fused, aux)

class VideoBackbone(nn.Module):

    def __init__(self, backbone_type='resnet18', output_dim=512, pretrained=True):
        super().__init__()
        self.backbone_type = backbone_type
        self.output_dim = output_dim
        try:
            import torchvision.models as models
            from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
            if backbone_type == 'resnet18':
                weights = ResNet18_Weights.DEFAULT if pretrained else None
                try:
                    resnet = models.resnet18(weights=weights)
                except Exception as e:
                    print(f'[WARN] Failed to load pretrained ResNet18 weights. Falling back to random init: {e}')
                    resnet = models.resnet18(weights=None)
                self.base_dim = 512
            elif backbone_type == 'resnet34':
                weights = ResNet34_Weights.DEFAULT if pretrained else None
                try:
                    resnet = models.resnet34(weights=weights)
                except Exception as e:
                    print(f'[WARN] Failed to load pretrained ResNet34 weights. Falling back to random init: {e}')
                    resnet = models.resnet34(weights=None)
                self.base_dim = 512
            elif backbone_type == 'resnet50':
                weights = ResNet50_Weights.DEFAULT if pretrained else None
                try:
                    resnet = models.resnet50(weights=weights)
                except Exception as e:
                    print(f'[WARN] Failed to load pretrained ResNet50 weights. Falling back to random init: {e}')
                    resnet = models.resnet50(weights=None)
                self.base_dim = 2048
            self.features = nn.Sequential(*list(resnet.children())[:-1])
            self.proj = nn.Linear(self.base_dim, output_dim) if self.base_dim != output_dim else nn.Identity()
        except Exception:
            print('[WARN] torchvision is unavailable. Using the simplified backbone instead.')
            self.features = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, 1), nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
            self.base_dim = 128
            self.proj = nn.Linear(128, output_dim)

    def forward(self, x):
        if x.ndim == 3 and x.size(-1) == self.output_dim:
            return x
        if x.ndim == 2:
            return x
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            feat = self.features(x).squeeze(-1).squeeze(-1)
            feat = self.proj(feat)
            return feat.reshape(B, T, -1)
        else:
            feat = self.features(x).squeeze(-1).squeeze(-1)
            return self.proj(feat)

class AudioBackbone(nn.Module):

    def __init__(self, n_mels=128, hidden_dim=512):
        super().__init__()
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.conv = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(256, hidden_dim)

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)
        B, T = x.shape[:2]
        mel = x.reshape(B * T, *x.shape[2:])
        mel = mel.unsqueeze(1)
        feat = self.conv(mel).squeeze(-1).squeeze(-1)
        feat = self.fc(feat)
        feat = feat.view(B, T, -1)
        return feat

class DefaultFusion(nn.Module):

    def __init__(self, video_dim, audio_dim, fusion_dim):
        super().__init__()
        self.proj = nn.Linear(video_dim + audio_dim, fusion_dim)

    def forward(self, video_feat, audio_feat):
        combined = torch.cat([video_feat, audio_feat], dim=-1)
        fused = self.proj(combined)
        return fused
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedMILHead(nn.Module):

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int=256, dropout: float=0.3, topk_ratio: float=0.0, attn_temp: float=1.0, pooling_mode: str='attention'):
        super().__init__()
        assert 0.0 <= topk_ratio <= 1.0, 'topk_ratio should be within [0, 1]'
        self.topk_ratio = float(topk_ratio)
        self.attn_temp = float(attn_temp)
        self.pooling_mode = str(pooling_mode).lower()
        if self.pooling_mode not in {'attention', 'mean'}:
            raise ValueError(f"Unsupported MIL pooling_mode={pooling_mode!r}; expected 'attention' or 'mean'.")
        self.frame_classifier = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        self.attention = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor):
        if x.dim() != 3:
            raise AssertionError(f'EnhancedMILHead expects [B,T,D], got {tuple(x.shape)}')
        batch_size, seq_len, _ = x.shape
        seg_logits = self.frame_classifier(x)
        if self.pooling_mode == 'mean':
            weights = seg_logits.new_full((batch_size, seq_len), 1.0 / float(max(seq_len, 1)))
        else:
            attn_scores = self.attention(x).squeeze(-1)
            temp = max(self.attn_temp, 1e-06)
            base_weights = F.softmax(attn_scores / temp, dim=1)
            if self.topk_ratio > 0.0:
                k = max(1, min(seq_len, int(round(seq_len * self.topk_ratio))))
                if k < seq_len:
                    _, idx = torch.topk(attn_scores, k, dim=1)
                    mask = torch.zeros_like(base_weights)
                    mask.scatter_(1, idx, 1.0)
                    weights = base_weights * mask
                else:
                    weights = base_weights
            else:
                weights = base_weights
        weights = torch.clamp(weights, min=0.0)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-08)
        clip_logits = torch.einsum('btc,bt->bc', seg_logits, weights)
        return {'clip_logits': clip_logits, 'seg_logits': seg_logits, 'weights': weights}

class EnhancedAVTopDetector(nn.Module):

    def _freeze_stages(self, backbone: nn.Module, frozen_stages: int):
        if backbone is None or frozen_stages <= 0:
            return
        for m in backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                m.eval()
        layers = []
        for name in ('conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4'):
            if hasattr(backbone, name):
                layers.append(getattr(backbone, name))
        if not layers:
            children = list(backbone.children())
            if children:
                layers = children[:min(frozen_stages, len(children))]
        cnt = 0
        for block in layers:
            if cnt >= frozen_stages:
                break
            for p in block.parameters():
                p.requires_grad = False
            cnt += 1

    def __init__(self, cfg: Dict):
        super().__init__()
        cfg = resolve_runtime_config(cfg if isinstance(cfg, dict) else {})
        model_cfg = cfg.get('model', {})
        self.video_dim = model_cfg.get('video_dim', 512)
        self.audio_dim = model_cfg.get('audio_dim', 512)
        self.fusion_dim = model_cfg.get('fusion_dim', 256)
        self.num_classes = int(model_cfg.get('num_classes', cfg.get('data', {}).get('num_classes', 2)))
        self.d_model = cfg.get('d_model', self.audio_dim)
        self.cfg = cfg
        self.use_aux_heads = bool(model_cfg.get('use_aux_heads', cfg.get('use_aux_heads', True)))
        self.emit_aux_logits = bool(model_cfg.get('emit_aux_logits', cfg.get('emit_aux_logits', self.use_aux_heads)))
        self.video_backbone_net = self._build_video_backbone(cfg)
        self.audio_backbone_net = self._build_audio_backbone(cfg)
        self.video_dim = int(getattr(self, 'vbb_out_dim', self.video_dim))
        self.audio_dim = int(getattr(self, 'abb_out_dim', self.audio_dim))
        self.video_backbone = self.video_backbone_net
        self.audio_backbone = self.audio_backbone_net
        if cfg.get('use_temporal_encoder', False) and HAS_TEMPORAL:
            self.video_temporal = SimpleTemporalEncoder(input_dim=self.video_dim, hidden_dim=model_cfg.get('hidden_dim', 256))
            self.audio_temporal = SimpleTemporalEncoder(input_dim=self.audio_dim, hidden_dim=model_cfg.get('hidden_dim', 256))
        else:
            self.video_temporal = None
            self.audio_temporal = None
        fusion_cfg = cfg.get('fusion', model_cfg.get('fusion', {'type': 'default'}))
        fusion_type = str(fusion_cfg.get('type', fusion_cfg.get('name', model_cfg.get('fusion_type', cfg.get('fusion_type', 'default'))))).lower()
        if fusion_type == 'coattn' and HAS_COATTN:
            core = CoAttentionFusion(video_dim=self.video_dim, audio_dim=self.audio_dim, d_model=fusion_cfg.get('d_model', self.fusion_dim), num_layers=fusion_cfg.get('num_layers', 2), num_heads=fusion_cfg.get('num_heads', 8), dropout=fusion_cfg.get('dropout', 0.1))
            self.fusion = SafeCoAttention(core, video_dim=self.video_dim, audio_dim=self.audio_dim, fusion_dim=self.fusion_dim)
            self.fusion_type = 'coattn'
        elif fusion_type == 'ib' and HAS_IB:
            self.fusion = InformationBottleneckFusion(video_dim=self.video_dim, audio_dim=self.audio_dim, fusion_dim=self.fusion_dim, beta=fusion_cfg.get('beta', 0.1))
            self.fusion_type = 'ib'
        elif fusion_type == 'cfa' and HAS_CFA:
            self.fusion = CFAFusion(video_dim=self.video_dim, audio_dim=self.audio_dim, fusion_dim=self.fusion_dim)
            self.fusion_type = 'cfa'
        elif fusion_type in ('mult_style', 'cross_modal_transformer') and HAS_MULT_STYLE:
            self.fusion = MulTStyleFusion(video_dim=self.video_dim, audio_dim=self.audio_dim, d_model=int(fusion_cfg.get('d_model', self.fusion_dim)), num_layers=int(fusion_cfg.get('num_layers', 2)), num_heads=int(fusion_cfg.get('num_heads', 8)), dropout=float(fusion_cfg.get('dropout', 0.1)), ffn_mult=int(fusion_cfg.get('ffn_mult', 4)), output_timeline=str(fusion_cfg.get('output_timeline', 'video')))
            self.fusion_type = 'mult_style'
        else:
            self.fusion = DefaultFusion(video_dim=self.video_dim, audio_dim=self.audio_dim, fusion_dim=self.fusion_dim)
            self.fusion_type = 'default'
        print(f'[EnhancedDetector] using fusion strategy: {self.fusion_type}')
        cava_cfg = cfg.get('cava', {}) if isinstance(cfg, dict) else {}
        self.use_cava = bool(cava_cfg.get('enabled', False) and HAS_CAVA)
        if self.use_cava:
            self.cava = CAVAModule(video_dim=self.video_dim, audio_dim=self.audio_dim, d_model=int(cava_cfg.get('d_model', self.fusion_dim)), delta_low_frames=float(cava_cfg.get('delta_low_frames', 2.0)), delta_high_frames=float(cava_cfg.get('delta_high_frames', 6.0)), delta_prior=float(cava_cfg.get('delta_prior', 0.0)), gate_clip_min=float(cava_cfg.get('gate_clip_min', cava_cfg.get('gate_min', 0.05))), gate_clip_max=float(cava_cfg.get('gate_clip_max', cava_cfg.get('gate_max', 0.95))), num_classes=int(cfg.get('data', {}).get('num_classes', cfg.get('model', {}).get('num_classes', self.num_classes))), dist_max_delay=int(cava_cfg.get('dist_max_delay', int(cava_cfg.get('delta_high_frames', 6.0)))), window_size=int(cava_cfg.get('window_size', 5)), mask_type=str(cava_cfg.get('mask_type', 'hard')), multi_scale=bool(cava_cfg.get('multi_scale', False)), gate_range_mode=str(cava_cfg.get('gate_range_mode', 'strict')), gate_range=cava_cfg.get('gate_range', None), use_learnable_delay=bool(cava_cfg.get('use_learnable_delay', True)), use_mask=bool(cava_cfg.get('use_mask', True)), use_gate=bool(cava_cfg.get('use_gate', True)))
        else:
            self.cava = None
        self.cava_to_audio = None
        self.cava_to_video = None
        try:
            cava_d = int(self.cava.d_model) if self.cava is not None else None
        except Exception:
            cava_d = None
        if self.cava is not None and cava_d is not None and (cava_d != self.audio_dim):
            self.cava_to_audio = nn.Linear(cava_d, self.audio_dim)
        if self.cava is not None and cava_d is not None and (cava_d != self.video_dim):
            self.cava_to_video = nn.Linear(cava_d, self.video_dim)
        mil_cfg = model_cfg.get('mil', cfg.get('mil', {}))
        self.mil_head = EnhancedMILHead(input_dim=self.fusion_dim, num_classes=self.num_classes, dropout=float(mil_cfg.get('dropout')), topk_ratio=float(mil_cfg.get('topk_ratio')), attn_temp=float(mil_cfg.get('attn_temp')), pooling_mode=str(mil_cfg.get('pooling_mode', 'attention')))
        self.classifier = self.mil_head.frame_classifier
        if self.use_aux_heads:
            self.video_head = nn.Sequential(nn.Linear(self.video_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, self.num_classes))
            self.audio_head = nn.Sequential(nn.Linear(self.audio_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, self.num_classes))
        else:
            self.video_head = None
            self.audio_head = None

    def _build_video_backbone(self, cfg):
        mb = cfg.get('model', {}) or {}
        vb_cfg = mb.get('video_backbone', 'resnet18')
        if isinstance(vb_cfg, dict):
            name = str(vb_cfg.get('name', 'resnet18')).lower()
            weights = str(vb_cfg.get('weights', 'imagenet')).lower()
            pretrained = weights != 'none'
            out_dim = int(vb_cfg.get('out_dim', self.video_dim))
            frozen = int(vb_cfg.get('frozen_stages', 0))
        else:
            name = str(vb_cfg).lower()
            pretrained = bool(mb.get('pretrained', False))
            out_dim = int(mb.get('video_dim', self.video_dim))
            frozen = 0
        bb = VideoBackbone(backbone_type=name, output_dim=out_dim, pretrained=pretrained)
        self._freeze_stages(bb, frozen)
        self.vbb_out_dim = out_dim
        return bb

    def _build_audio_backbone(self, cfg):
        mb = cfg.get('model', {}) or {}
        ab_cfg = mb.get('audio_backbone', 'cnn')
        n_mels = int(cfg.get('audio', {}).get('n_mels', cfg.get('n_mels', 128)))
        if isinstance(ab_cfg, dict):
            name = str(ab_cfg.get('name', 'cnn')).lower()
            weights = str(ab_cfg.get('weights', 'audioset')).lower()
            pretrained = weights != 'none'
            out_dim = int(ab_cfg.get('out_dim', self.audio_dim))
            frozen = int(ab_cfg.get('frozen_stages', 0))
        else:
            name = str(ab_cfg).lower()
            pretrained = bool(mb.get('pretrained_audio', False))
            out_dim = int(mb.get('audio_dim', self.audio_dim))
            frozen = 0
        if name in ('vggish', 'light_vggish'):
            backbone = LightVGGishAudioBackbone(n_mels=n_mels, hidden_dim=out_dim)
        elif name == 'moderate_vggish':
            backbone = ModerateVGGishAudioBackbone(n_mels=n_mels, hidden_dim=out_dim)
        elif name in ('cnn', 'improved'):
            backbone = ImprovedAudioBackbone(n_mels=n_mels, hidden_dim=out_dim) if name == 'improved' else AudioBackbone(n_mels=n_mels, hidden_dim=out_dim)
        else:
            raise ValueError(f'Unknown audio backbone: {name}')
        self._freeze_stages(backbone, frozen)
        self.abb_out_dim = out_dim
        return backbone

    def forward(self, video, audio, return_aux: bool=True):
        for name in ('video_backbone_net', 'audio_backbone_net'):
            mod = getattr(self, name, None)
            if not isinstance(mod, nn.Module) or not hasattr(mod, 'forward'):
                raise TypeError(f'{name} must be an nn.Module with a forward method, got {type(mod)}')
        video_feat = self.video_backbone_net(video)
        audio_feat = self.audio_backbone_net(audio)
        if video_feat.ndim == 2:
            video_feat = video_feat.unsqueeze(1)
        if audio_feat.ndim == 2:
            audio_feat = audio_feat.unsqueeze(1)
        if getattr(self, 'video_temporal', None) is not None:
            video_feat = self.video_temporal(video_feat)
        if getattr(self, 'audio_temporal', None) is not None:
            audio_feat = self.audio_temporal(audio_feat)
        audio_seq_raw = audio_feat
        cava_aux = {}
        use_cava_flag = bool(getattr(self, 'use_cava', False)) and getattr(self, 'cava', None) is not None
        if use_cava_flag:
            C = self.cava(video_feat, audio_feat)
            video_aligned = C.get('video_for_fusion', C.get('video_shifted', video_feat))
            audio_aligned = C.get('audio_for_fusion', C.get('audio_aligned', audio_feat))
            need_v_proj = hasattr(self, 'video_dim') and video_aligned.size(-1) != self.video_dim
            need_proj = hasattr(self, 'audio_dim') and audio_aligned.size(-1) != self.audio_dim
            if need_v_proj:
                if not hasattr(self, 'cava_to_video') or self.cava_to_video is None:
                    raise ValueError(f'CAVA video output dim {video_aligned.size(-1)} does not match video_dim={self.video_dim}, but cava_to_video was not initialized.')
                Bv_, Tv_, Dv_ = video_aligned.shape
                video_feat = self.cava_to_video(video_aligned.reshape(Bv_ * Tv_, Dv_)).reshape(Bv_, Tv_, self.video_dim)
            else:
                video_feat = video_aligned
            if need_proj:
                if not hasattr(self, 'cava_to_audio') or self.cava_to_audio is None:
                    raise ValueError(f'CAVA output dim {audio_aligned.size(-1)} does not match audio_dim={self.audio_dim}, but cava_to_audio was not initialized.')
                B_, T_, D_ = audio_aligned.shape
                audio_feat = self.cava_to_audio(audio_aligned.reshape(B_ * T_, D_)).reshape(B_, T_, self.audio_dim)
            else:
                audio_feat = audio_aligned
            causal_gate = C.get('causal_gate', None)
            if causal_gate is not None and causal_gate.ndim == 2:
                causal_gate = causal_gate.unsqueeze(-1)
            delta_low_v = getattr(self.cava, 'delta_low', None) if hasattr(self.cava, 'delta_low') else None
            delta_high_v = getattr(self.cava, 'delta_high', None) if hasattr(self.cava, 'delta_high') else None
            if delta_low_v is not None and (not torch.is_tensor(delta_low_v)):
                delta_low_v = torch.tensor(float(delta_low_v), device=video_feat.device)
            if delta_high_v is not None and (not torch.is_tensor(delta_high_v)):
                delta_high_v = torch.tensor(float(delta_high_v), device=video_feat.device)
            if torch.is_tensor(delta_low_v):
                delta_low_v = delta_low_v.detach()
            if torch.is_tensor(delta_high_v):
                delta_high_v = delta_high_v.detach()
            cava_aux = {'delay_frames': C.get('delay_frames', None), 'causal_gate': causal_gate, 'audio_aligned': C.get('audio_aligned', None), 'audio_masked': C.get('audio_masked', None), 'audio_context': C.get('audio_context', None), 'video_proj': C.get('video_proj', None), 'video_shifted': C.get('video_shifted', None), 'video_for_fusion': C.get('video_for_fusion', None), 'audio_proj': C.get('audio_proj', None), 'causal_mask': C.get('causal_mask', None), 'cross_attn_weights': C.get('cross_attn_weights', None), 'delta_low': delta_low_v, 'delta_high': delta_high_v, 'causal_prob': C.get('causal_prob', C.get('causal_gate', None).squeeze(-1) if C.get('causal_gate', None) is not None else None), 'causal_prob_dist': C.get('causal_prob_dist', None), 'pred_delay': C.get('pred_delay', None)}
        fusion_aux = {}
        fusion_out = self.fusion(video_feat, audio_feat)
        if isinstance(fusion_out, tuple):
            fused = fusion_out[0]
            if len(fusion_out) > 1 and isinstance(fusion_out[1], dict):
                fusion_aux = fusion_out[1]
        else:
            fused = fusion_out
        if fused.ndim == 2:
            fused = fused.unsqueeze(1)
        video_emb = fusion_aux.get('video_emb', video_feat.mean(dim=1))
        audio_emb = fusion_aux.get('audio_emb', audio_feat.mean(dim=1))
        video_seq = fusion_aux.get('video_seq', video_feat)
        audio_seq = fusion_aux.get('audio_seq', audio_feat)
        mil_outputs = self.mil_head(fused)
        outputs = {'clip_logits': mil_outputs['clip_logits'], 'seg_logits': mil_outputs['seg_logits'], 'weights': mil_outputs['weights']}
        if return_aux:
            if self.emit_aux_logits and getattr(self, 'video_head', None) is not None and (getattr(self, 'audio_head', None) is not None):
                video_pooled = video_seq.mean(dim=1)
                audio_pooled = audio_seq.mean(dim=1)
                outputs['video_logits'] = self.video_head(video_pooled)
                outputs['audio_logits'] = self.audio_head(audio_pooled)
            outputs['video_emb'] = video_emb
            outputs['audio_emb'] = audio_emb
            outputs['video_seq'] = video_seq
            outputs['audio_seq'] = audio_seq
            outputs['audio_seq_raw'] = audio_seq_raw
            try:
                if fusion_aux.get('fusion_token', None) is not None:
                    outputs['fusion_token'] = fusion_aux['fusion_token']
                else:
                    v_tok = cava_aux.get('video_shifted', cava_aux.get('video_proj', video_emb)) if cava_aux else video_emb
                    a_tok = cava_aux.get('audio_aligned', audio_emb) if cava_aux else audio_emb
                    if v_tok is not None and v_tok.ndim == 3:
                        v_tok = v_tok.mean(dim=1)
                    if a_tok is not None and a_tok.ndim == 3:
                        a_tok = a_tok.mean(dim=1)
                    if v_tok is None:
                        v_tok = video_emb
                    if a_tok is None:
                        a_tok = audio_emb
                    outputs['fusion_token'] = torch.cat([v_tok, a_tok], dim=-1)
            except Exception:
                outputs['fusion_token'] = torch.cat([video_emb, audio_emb], dim=-1)
            for key in ('fusion_type', 'loss_dava', 'dava_loss', 'g_bar', 'causal_gate', 'delay_frames', 'delay', 'alignment_gate_mean'):
                if key in fusion_aux and outputs.get(key) is None:
                    outputs[key] = fusion_aux[key]
            if cava_aux:
                outputs.update(cava_aux)
        return outputs

class EnhancedAVDetector(nn.Module):

    def __init__(self, cfg: Dict):
        super().__init__()
        self.num_classes = cfg.get('num_classes', 2)
        video_dim = cfg.get('video_dim', 512)
        audio_dim = cfg.get('audio_dim', 256)
        fusion_dim = cfg.get('fusion_dim', 256)
        self.video_enc = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, video_dim))
        self.audio_enc = nn.Sequential(nn.Conv1d(1, 64, 7, 2, 3), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, audio_dim))
        self.fusion = nn.Sequential(nn.Linear(video_dim + audio_dim, fusion_dim), nn.ReLU(), nn.Dropout(0.3))
        self.classifier = nn.Linear(fusion_dim, self.num_classes)

    def forward(self, video, audio, return_aux=False):
        B = video.size(0)
        if video.ndim == 5:
            T = video.size(1)
            video = video.reshape(B * T, *video.shape[2:])
            v_feat = self.video_enc(video)
            v_feat = v_feat.reshape(B, T, -1).mean(dim=1)
        else:
            v_feat = self.video_enc(video)
        if audio.ndim == 3 and audio.size(1) > 10:
            T = audio.size(1)
            audio = audio.reshape(B * T, 1, -1)
            a_feat = self.audio_enc(audio)
            a_feat = a_feat.reshape(B, T, -1).mean(dim=1)
        else:
            if audio.ndim == 2:
                audio = audio.unsqueeze(1)
            a_feat = self.audio_enc(audio)
        combined = torch.cat([v_feat, a_feat], dim=1)
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        outputs = {'clip_logits': logits}
        if return_aux:
            outputs.update({'video_emb': v_feat, 'audio_emb': a_feat, 'video_logits': logits, 'audio_logits': logits})
            try:
                v_tok = outputs.get('video_proj', outputs.get('video_emb', None))
                a_tok = outputs.get('audio_aligned', outputs.get('audio_emb', None))
                if v_tok is not None and a_tok is not None:
                    if v_tok.dim() == 3:
                        v_tok = v_tok.mean(dim=1)
                    if a_tok.dim() == 3:
                        a_tok = a_tok.mean(dim=1)
                    fusion_token = torch.cat([v_tok, a_tok], dim=-1)
                    outputs['fusion_token'] = fusion_token
            except Exception:
                pass
        return outputs if return_aux else outputs['clip_logits']
if __name__ == '__main__':
    print('=' * 70)
    print('Enhanced Detector smoke test')
    print('=' * 70)
    cfg = {'model': {'video_dim': 512, 'audio_dim': 512, 'fusion_dim': 256, 'num_classes': 12, 'video_backbone': 'resnet18', 'audio_backbone': 'cnn', 'pretrained': False}, 'fusion': {'type': 'default', 'd_model': 256, 'num_layers': 2, 'num_heads': 8}, 'use_temporal_encoder': False, 'use_aux_heads': True, 'n_mels': 128}
    print('\nTesting compact EnhancedAVDetector:')
    simple_model = EnhancedAVDetector({'num_classes': 12, 'video_dim': 512, 'audio_dim': 256, 'fusion_dim': 256})
    B = 4
    video_simple = torch.randn(B, 3, 224, 224)
    audio_simple = torch.randn(B, 1, 16000)
    out_simple = simple_model(video_simple, audio_simple, return_aux=True)
    print(f"  clip_logits: {out_simple['clip_logits'].shape}")
    print(f"  video_emb: {out_simple['video_emb'].shape}")
    assert out_simple['clip_logits'].shape == (B, 12), 'Output shape mismatch.'
    print('  Simple detector smoke test passed.')
    print('\nTesting full EnhancedAVTopDetector:')
    model = EnhancedAVTopDetector(cfg)
    B = 4
    T = 8
    video = torch.randn(B, T, 3, 224, 224)
    audio = torch.randn(B, T, 128, 32)
    outputs = model(video, audio, return_aux=True)
    print(f"  video_seq: {outputs['video_seq'].shape}")
    print(f"  audio_seq: {outputs['audio_seq'].shape}")
    print(f'\nOutputs:')
    print(f"  clip_logits: {outputs['clip_logits'].shape}")
    print(f"  seg_logits: {outputs['seg_logits'].shape}")
    print(f"  weights: {outputs['weights'].shape}")
    if 'video_logits' in outputs:
        print(f'\nAuxiliary outputs (single modality):')
        print(f"  video_logits: {outputs['video_logits'].shape}")
        print(f"  audio_logits: {outputs['audio_logits'].shape}")
    if 'video_emb' in outputs:
        print(f'\nGlobal embeddings (for contrastive use):')
        print(f"  video_emb: {outputs['video_emb'].shape}")
        print(f"  audio_emb: {outputs['audio_emb'].shape}")
    assert outputs['clip_logits'].shape == (B, 12), f"Clip-level output shape mismatch. Expected {(B, 12)}, got {outputs['clip_logits'].shape}"
    assert outputs['seg_logits'].shape == (B, T, 12), f"Segment-level output shape mismatch. Expected {(B, T, 12)}, got {outputs['seg_logits'].shape}"
    assert torch.allclose(outputs['weights'].sum(dim=1), torch.ones(B), atol=1e-05), 'Attention weights should sum to 1.'
    print(f'\nAll smoke tests passed. The implementation is ready for regular use.')
try:
    SimpleAVDetector = EnhancedAVDetector
    EnhancedAVDetector = EnhancedAVTopDetector
except Exception:
    pass
