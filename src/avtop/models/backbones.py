from typing import Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision.models as tvm
except Exception:
    tvm = None
try:
    import torchaudio
    import torchaudio.transforms as TAT
except Exception:
    torchaudio = None
    TAT = None
from torch.nn.utils.rnn import pad_sequence

def _flatten_bt(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    assert x.dim() == 5, f'Expected (B, T, C, H, W), got {tuple(x.shape)}'
    B, T, C, H, W = x.shape
    x = x.contiguous().view(B * T, C, H, W)
    return (x, B, T)

def _require(module, name: str):
    if module is None:
        raise ImportError(f"Required dependency for '{name}' is not available. Please install it or choose another backbone.")

class _ResNet18PerFrame(nn.Module):

    def __init__(self, pretrained: bool=True, freeze: bool=False):
        super().__init__()
        _require(tvm, 'torchvision')
        try:
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = tvm.resnet18(weights=weights)
        except Exception:
            base = tvm.resnet18(pretrained=pretrained)
        self.feature = nn.Sequential(*list(base.children())[:-1])
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, B, T = _flatten_bt(x)
        feat = self.feature(x).flatten(1)
        return feat.view(B, T, 512)

class _TimeSformerWrapper(nn.Module):

    def __init__(self, pretrained: bool=True, freeze: bool=False):
        super().__init__()
        try:
            from transformers import TimesformerModel, TimesformerConfig
        except Exception as e:
            raise ImportError("TimeSformer requires 'transformers' (pip install transformers>=4.31).") from e
        model_name = 'facebook/timesformer-base-finetuned-k400' if pretrained else None
        if model_name is not None:
            self.model = TimesformerModel.from_pretrained(model_name)
        else:
            self.model = TimesformerModel(TimesformerConfig())
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        self.out_dim = self.model.config.hidden_size
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        out = self.model(pixel_values=x)
        clip_emb = getattr(out, 'pooler_output', None)
        if clip_emb is None:
            clip_emb = out.last_hidden_state[:, 0, :]
        B, T = (x.shape[0], x.shape[1])
        return clip_emb.unsqueeze(1).expand(B, T, -1)

class VideoBackbone(nn.Module):

    def __init__(self, backbone_type: str='resnet18_2d', pretrained: bool=True, freeze: bool=False):
        super().__init__()
        b = backbone_type.lower()
        if b in ['resnet18', 'resnet18_2d', 'resnet-18']:
            self.impl = _ResNet18PerFrame(pretrained=pretrained, freeze=freeze)
            self.out_dim = 512
        elif b in ['timesformer', 'time-sformer', 'timesformer_base']:
            self.impl = _TimeSformerWrapper(pretrained=pretrained, freeze=freeze)
            self.out_dim = self.impl.out_dim
        else:
            raise ValueError(f'Unsupported video backbone_type: {backbone_type}')

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.impl(video)

class MelSpectrogramCNN(nn.Module):

    def __init__(self, mel_bins: int=64, d_out: int=128, dropout: float=0.1, sample_rate: int=16000, n_fft: int=400, hop_length: int=160):
        super().__init__()
        self.mel_bins = mel_bins
        self.out_dim = d_out
        self.sample_rate = sample_rate
        _require(torchaudio, 'torchaudio')
        _require(TAT, 'torchaudio.transforms')
        self.mel_extractor = TAT.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=mel_bins)
        self.amp_to_db = TAT.AmplitudeToDB()
        self.temporal_conv = nn.Sequential(nn.Conv1d(mel_bins, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Conv1d(128, 256, kernel_size=5, padding=2), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Conv1d(256, d_out, kernel_size=3, padding=1), nn.BatchNorm1d(d_out), nn.ReLU(inplace=True))
        self.use_rnn = True
        if self.use_rnn:
            self.rnn = nn.GRU(d_out, d_out // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)

    def _detect_input_type(self, x: torch.Tensor) -> str:
        if x.dim() == 2:
            B, T = x.shape
            if T > 1000:
                return 'waveform'
            else:
                return 'unknown'
        elif x.dim() == 3:
            B, T, M = x.shape
            if M in [64, 80, 128, 256] and T < 1000:
                return 'melspec'
            elif T > 1000 and M < 10:
                return 'waveform_3d'
            else:
                return 'unknown'
        else:
            return 'unknown'

    def _waveform_to_mel(self, wave: torch.Tensor, target_frames: int=None) -> torch.Tensor:
        if wave.dim() == 2:
            pass
        elif wave.dim() == 3 and wave.shape[1] == 1:
            wave = wave.squeeze(1)
        else:
            raise ValueError(f'Cannot convert waveform shape {wave.shape}')
        mel = self.mel_extractor(wave)
        mel = self.amp_to_db(mel)
        mel = mel.transpose(1, 2)
        if target_frames is not None and mel.shape[1] != target_frames:
            mel = mel.transpose(1, 2)
            mel = F.interpolate(mel, size=target_frames, mode='linear', align_corners=False)
            mel = mel.transpose(1, 2)
        return mel

    def forward(self, audio_input: torch.Tensor, target_frames: int=None) -> torch.Tensor:
        input_type = self._detect_input_type(audio_input)
        if input_type == 'waveform':
            mel_spec = self._waveform_to_mel(audio_input, target_frames)
        elif input_type == 'waveform_3d':
            wave = audio_input.squeeze(1) if audio_input.shape[1] == 1 else audio_input.reshape(audio_input.shape[0], -1)
            mel_spec = self._waveform_to_mel(wave, target_frames)
        elif input_type == 'melspec':
            mel_spec = audio_input
            B, T, M = mel_spec.shape
            if M != self.mel_bins:
                raise ValueError(f'Expected mel_bins={self.mel_bins}, got {M}')
        else:
            raise ValueError(f'Cannot determine input type. Shape: {audio_input.shape}\nExpected: (B, T_samples) for waveform or (B, T, {self.mel_bins}) for mel spectrogram')
        B, T, M = mel_spec.shape
        x = mel_spec.transpose(1, 2)
        x = self.temporal_conv(x)
        x = x.transpose(1, 2)
        if self.use_rnn:
            x, _ = self.rnn(x)
        return x

class VGGishBackbone(nn.Module):

    def __init__(self, pretrained: bool=True, sample_rate: int=16000, freeze: bool=False):
        super().__init__()
        self.sample_rate = sample_rate
        self.uses_ta_vggish = False
        force_tvg = os.environ.get('AVTOP_USE_TORCHVGGISH', '0') == '1'
        if not force_tvg:
            try:
                _require(torchaudio, 'torchaudio')
                from torchaudio.prototype.pipelines import VGGISH as TA_VGGISH
                self._ta_sr = TA_VGGISH.sample_rate
                self._ta_processor = TA_VGGISH.get_input_processor()
                self._ta_model = TA_VGGISH.get_model()
                self.uses_ta_vggish = True
                if freeze:
                    for p in self._ta_model.parameters():
                        p.requires_grad = False
            except Exception:
                self.uses_ta_vggish = False
        if not self.uses_ta_vggish:
            try:
                from torchvggish import vggish as tvg_vggish, vggish_input as tvg_input
            except Exception as e:
                raise ImportError("VGGish requires either torchaudio.prototype.pipelines.VGGISH or 'torchvggish' (pip install torchvggish>=0.2.1).") from e
            self._tvg_model = tvg_vggish()
            self._tvg_model.train(not freeze)
            self._tvg_input = tvg_input
        self.out_dim = 128
        self.debug = os.environ.get('AVTOP_DEBUG_VGGISH', '0') == '1'

    @staticmethod
    def _to_BT_mono(wave: torch.Tensor) -> torch.Tensor:
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        elif wave.dim() == 2:
            dim0, dim1 = wave.shape
            if dim0 <= 2 and dim1 > 8000:
                wave = wave.mean(dim=0, keepdim=True)
            elif dim0 > 2 and dim1 < dim0:
                wave = wave.transpose(0, 1)
        elif wave.dim() == 3:
            if wave.shape[1] <= 2 and wave.shape[2] > 8000:
                wave = wave.mean(dim=1)
            elif wave.shape[2] <= 2 and wave.shape[1] > 8000:
                wave = wave.mean(dim=2)
            else:
                wave = wave.reshape(wave.shape[0], -1)
        else:
            B = wave.shape[0]
            wave = wave.view(B, -1)
        return wave.to(dtype=torch.float32, copy=False)

    def _pad_min_len(self, one: torch.Tensor, target_len: int) -> torch.Tensor:
        if one.numel() < target_len:
            pad = target_len - one.numel()
            one = F.pad(one, (0, pad))
        return one

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        wave = self._to_BT_mono(wave)
        if self.debug:
            print(f'[VGGish] after _to_BT_mono: shape={tuple(wave.shape)}')
        if self.uses_ta_vggish:
            if self.sample_rate != getattr(self, '_ta_sr', 16000):
                wave = torchaudio.functional.resample(wave, self.sample_rate, self._ta_sr)
            feats = []
            device = wave.device
            sr = getattr(self, '_ta_sr', 16000)
            win_len = max(1, int(round(0.025 * sr)))
            min_len_2nd = int(round(0.96 * sr))
            for b in range(wave.size(0)):
                one = wave[b].detach().cpu()
                if one.dim() != 1:
                    one = one.reshape(-1)
                one = self._pad_min_len(one, win_len)
                proc = self._ta_processor(one)
                if proc.ndim == 4 and proc.shape[0] == 0:
                    one = self._pad_min_len(one, min_len_2nd)
                    proc = self._ta_processor(one)
                if proc.ndim == 4 and proc.shape[0] == 0:
                    proc = torch.zeros(1, 1, 96, 64, dtype=torch.float32)
                emb = self._ta_model(proc)
                feats.append(emb)
            out = pad_sequence(feats, batch_first=True).to(device)
            if self.debug:
                print(f'[VGGish] ta_out: {tuple(out.shape)}')
            return out
        else:
            outs = []
            device = wave.device
            min_len = int(round(0.96 * self.sample_rate))
            for b in range(wave.size(0)):
                one = wave[b].detach().cpu()
                if one.dim() != 1:
                    one = one.reshape(-1)
                one = self._pad_min_len(one, min_len)
                ex = self._tvg_input.waveform_to_examples(one.numpy(), self.sample_rate)
                ex = torch.from_numpy(ex).unsqueeze(1).float()
                with torch.set_grad_enabled(self._tvg_model.training):
                    emb = self._tvg_model(ex)
                outs.append(emb.to(device))
            out = pad_sequence(outs, batch_first=True)
            if self.debug:
                print(f'[VGGish] tvg_out: {tuple(out.shape)}')
            return out

class _MelCNN(nn.Module):

    def __init__(self, sample_rate: int=16000, n_mels: int=64, freeze_frontend: bool=False):
        super().__init__()
        _require(torchaudio, 'torchaudio')
        _require(TAT, 'torchaudio.transforms')
        self.sample_rate = sample_rate
        self.melspec = TAT.MelSpectrogram(sample_rate=sample_rate, n_fft=400, win_length=400, hop_length=160, n_mels=n_mels)
        self.amptoDB = TAT.AmplitudeToDB()
        if freeze_frontend:
            for p in self.melspec.parameters():
                p.requires_grad = False
        self.cnn = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.proj = nn.Linear(n_mels, 128)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        spec = self.melspec(wave)
        spec = self.amptoDB(spec).unsqueeze(1)
        feat = self.cnn(spec)
        feat = feat.mean(dim=-1)
        return feat.transpose(1, 2).contiguous()

class AudioBackbone(nn.Module):

    def __init__(self, backbone_type: str='mel_spectrogram_cnn', pretrained: bool=True, sample_rate: int=16000, mel_bins: int=64, freeze: bool=False):
        super().__init__()
        b = backbone_type.lower()
        if b in ['mel_spectrogram_cnn', 'melspec_cnn', 'mel_cnn_direct', 'mel_direct']:
            self.impl = MelSpectrogramCNN(mel_bins=mel_bins, d_out=128, dropout=0.1)
            self.out_dim = 128
            self.expects_waveform = False
        elif b in ['vggish', 'vggish_ta', 'vggish_torchvggish']:
            self.impl = VGGishBackbone(pretrained=pretrained, sample_rate=sample_rate, freeze=freeze)
            self.out_dim = 128
            self.expects_waveform = True
        elif b in ['mel', 'mel_cnn', 'melcnn']:
            self.impl = _MelCNN(sample_rate=sample_rate, n_mels=mel_bins, freeze_frontend=freeze)
            self.out_dim = 128
            self.expects_waveform = True
        else:
            raise ValueError(f'Unsupported audio backbone_type: {backbone_type}\nAvailable options:\n  - For mel spectrogram input: mel_spectrogram_cnn, melspec_cnn, mel_cnn_direct\n  - For waveform input: vggish, mel_cnn')

    def forward(self, audio_input: torch.Tensor) -> torch.Tensor:
        return self.impl(audio_input)
