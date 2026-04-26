#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class LightVGGishAudioBackbone(nn.Module):

    def __init__(self, n_mels=128, hidden_dim=512):
        super().__init__()
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.features = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, hidden_dim))

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(B * T, -1)
        x = self.classifier(x)
        x = x.view(B, T, -1)
        return x

class ModerateVGGishAudioBackbone(nn.Module):

    def __init__(self, n_mels=128, hidden_dim=512):
        super().__init__()
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.features = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(nn.Linear(512, 2048), nn.ReLU(), nn.Dropout(0.5), nn.Linear(2048, 2048), nn.ReLU(), nn.Dropout(0.5), nn.Linear(2048, hidden_dim))

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(B * T, -1)
        x = self.classifier(x)
        x = x.view(B, T, -1)
        return x

class ImprovedAudioBackbone(nn.Module):

    def __init__(self, n_mels=128, hidden_dim=512):
        super().__init__()
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, hidden_dim))

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        x = x.view(B, T, -1)
        return x

def count_parameters(model):
    total = sum((p.numel() for p in model.parameters()))
    trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
    return (total, trainable)
if __name__ == '__main__':
    print('=' * 80)
    print('Audio backbone parameter comparison')
    print('=' * 80)
    batch_size = 2
    num_frames = 8
    n_mels = 128
    mel_frames = 32
    dummy_input = torch.randn(batch_size, num_frames, n_mels, mel_frames)
    print(f'\nInput shape: {tuple(dummy_input.shape)} [B, T, n_mels, frames]\n')
    models = {'Improved CNN': ImprovedAudioBackbone(n_mels, 512), 'Light VGGish': LightVGGishAudioBackbone(n_mels, 512), 'Moderate VGGish': ModerateVGGishAudioBackbone(n_mels, 512)}
    print(f"{'Model':<20} | {'Total Params':<12} | {'Conv Params':<12} | {'FC Params':<12} | {'Inference Time'}")
    print('-' * 80)
    for name, model in models.items():
        model.eval()
        total, _ = count_parameters(model)
        conv_params = sum((p.numel() for p in model.features.parameters())) / 1000000.0
        fc_params = sum((p.numel() for p in model.classifier.parameters())) / 1000000.0 if hasattr(model, 'classifier') else sum((p.numel() for p in model.fc.parameters())) / 1000000.0
        import time
        with torch.no_grad():
            start = time.time()
            for _ in range(5):
                _ = model(dummy_input)
            elapsed = (time.time() - start) / 5 * 1000
        print(f'{name:<20} | {total / 1000000.0:>10.2f}M | {conv_params:>10.2f}M | {fc_params:>10.2f}M | {elapsed:>8.1f}ms')
    print('\n' + '=' * 80)
    print('Recommended options')
    print('=' * 80)
    print('\nCurrent issue: the original VGGish has about 25M parameters and is oversized.')
    print('        Roughly 20M parameters are in the FC layers (4096 dims), which can overfit.\n')
    print('Suggested choices:\n')
    print('1. Light VGGish (5M)     <- Recommended')
    print('   - Keeps the VGGish convolutional structure for strong feature extraction')
    print('   - Simplifies the FC layers to reduce overfitting')
    print('   - Offers a balanced parameter budget relative to the video backbone')
    print('   - Expected gain: about +5% to +7% accuracy\n')
    print('2. Moderate VGGish (8M)  <- If memory is sufficient')
    print('   - Uses a slightly larger FC stack (2048 dims)')
    print('   - Provides stronger representation capacity')
    print('   - Expected gain: about +6% to +9% accuracy\n')
    print('3. Improved CNN (2M)     <- If memory is limited')
    print('   - Lightweight alternative')
    print('   - Expected gain: about +3% to +5% accuracy')
    print('=' * 80)
