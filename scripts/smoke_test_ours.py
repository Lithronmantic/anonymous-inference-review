#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from config_system import resolve_runtime_config
from enhanced_detector import EnhancedAVTopDetector


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise TypeError("Config must be a dict.")
    return resolve_runtime_config(cfg)


def build_dummy_inputs(cfg: dict, batch_size: int = 2):
    data_cfg = cfg.get("data", {})
    video_cfg = cfg.get("video", {})
    audio_cfg = cfg.get("audio", {})
    time_steps = int(video_cfg.get("num_frames", video_cfg.get("frames", 16)))
    side = int(video_cfg.get("size", 224))
    n_mels = int(audio_cfg.get("n_mels", audio_cfg.get("mel_bins", 128)))
    seg_frames = int(audio_cfg.get("segment_frames", audio_cfg.get("target_len", 64)))
    video = torch.randn(batch_size, time_steps, 3, side, side)
    audio = torch.randn(batch_size, time_steps, n_mels, seg_frames)
    return video, audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = dict(cfg.get("model", {}))
    model_cfg["num_classes"] = int(cfg.get("data", {}).get("num_classes", model_cfg.get("num_classes", 12)))
    full_cfg = {
        "model": model_cfg,
        "fusion": model_cfg.get("fusion", cfg.get("fusion", {})),
        "cava": cfg.get("cava", {}),
        "data": cfg.get("data", {}),
        "video": cfg.get("video", {}),
        "audio": cfg.get("audio", {}),
        "mlpr": cfg.get("mlpr", {}),
        "training": cfg.get("training", {}),
    }

    device = torch.device(args.device)
    model = EnhancedAVTopDetector(full_cfg).to(device)
    model.eval()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print({"checkpoint_loaded": str(Path(args.checkpoint).resolve()), "missing": len(missing), "unexpected": len(unexpected)})

    video, audio = build_dummy_inputs(cfg)
    video = video.to(device)
    audio = audio.to(device)
    with torch.no_grad():
        out = model(video, audio, return_aux=True)

    required = ["clip_logits", "seg_logits", "weights", "fusion_token"]
    missing_keys = [key for key in required if key not in out]
    if missing_keys:
        raise RuntimeError(f"Missing required output keys: {missing_keys}")

    print(
        {
            "clip_logits": tuple(out["clip_logits"].shape),
            "seg_logits": tuple(out["seg_logits"].shape),
            "weights": tuple(out["weights"].shape),
            "fusion_token": tuple(out["fusion_token"].shape),
        }
    )


if __name__ == "__main__":
    main()
