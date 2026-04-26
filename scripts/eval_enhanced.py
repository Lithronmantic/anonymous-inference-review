#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
import yaml
from dataset import AVFromCSV, safe_collate_fn
try:
    from config_system import resolve_runtime_config
except Exception:
    from scripts.config_system import resolve_runtime_config
try:
    from enhanced_detector import EnhancedAVTopDetector
except Exception:
    from scripts.enhanced_detector import EnhancedAVTopDetector

def _unpack(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) == 4:
            return (batch[0], batch[1], batch[2], batch[3])
        return (batch[0], batch[1], batch[2], None)
    if isinstance(batch, dict):
        keys = {k.lower(): k for k in batch.keys()}
        v = batch[keys.get('video', 'video')]
        a = batch[keys.get('audio', 'audio')]
        y = batch[keys.get('label', 'label')]
        i = batch.get(keys.get('ids', 'ids'))
        return (v, a, y, i)
    raise ValueError(f'Unsupported batch type: {type(batch)}')

@torch.no_grad()
def evaluate(cfg: dict, checkpoint: str, out_dir: str, device_mode: str='auto', max_batches: int=0):
    dm = str(device_mode).lower()
    if dm == 'cpu':
        device = torch.device('cpu')
    elif dm == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c = resolve_runtime_config(cfg)
    data_cfg = c['data']
    model_cfg = dict(c.get('model', {}))
    model_cfg['num_classes'] = int(data_cfg['num_classes'])
    model = EnhancedAVTopDetector({'model': model_cfg, 'fusion': model_cfg.get('fusion', c.get('fusion', {})), 'cava': c.get('cava', {}), 'data': data_cfg, 'video': c.get('video', {}), 'audio': c.get('audio', {})}).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    sd = ckpt.get('state_dict', ckpt)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    test_csv = data_cfg.get('test_csv') or data_cfg.get('val_csv')
    ds = AVFromCSV(test_csv, data_cfg.get('data_root', ''), int(data_cfg['num_classes']), list(data_cfg['class_names']), c.get('video', {}), c.get('audio', {}), is_unlabeled=False)
    loader = DataLoader(ds, batch_size=int(c.get('training', {}).get('batch_size', 16)), shuffle=False, num_workers=int(data_cfg.get('num_workers_val', 2)), pin_memory=device.type == 'cuda', collate_fn=safe_collate_fn)
    y_true, y_pred = ([], [])
    for bi, b in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        v, a, y, _ = _unpack(b)
        if hasattr(y, 'ndim') and y.ndim == 2:
            y = y.argmax(dim=1)
        v, a, y = (v.to(device), a.to(device), y.to(device))
        out = model(v, a, return_aux=True)
        logits = out['clip_logits'] if isinstance(out, dict) else out
        pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
        y_true.append(y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
    y_true = np.concatenate(y_true) if y_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred) if y_pred else np.array([], dtype=np.int64)
    acc = float(accuracy_score(y_true, y_pred)) if y_true.size > 0 else 0.0
    f1m = float(f1_score(y_true, y_pred, average='macro')) if y_true.size > 0 else 0.0
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metrics = {'accuracy': acc, 'macro_f1': f1m, 'num_samples': int(y_true.size)}
    (out_path / 'eval_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(metrics)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', default='outputs/eval_fixed')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--max_batches', type=int, default=0, help='0 means full eval')
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    evaluate(cfg, args.checkpoint, args.output, device_mode=args.device, max_batches=args.max_batches)
if __name__ == '__main__':
    main()
