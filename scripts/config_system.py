from __future__ import annotations
import copy
import json
from pathlib import Path
from typing import Any, Dict, Tuple
import yaml
RUNTIME_DEFAULTS: Dict[str, Any] = {'profile': 'safe', 'seed': 42, 'data': {'nominal_fps': 30, 'split_ratio': '5:3:2'}, 'model': {'mil': {'topk_ratio': 0.15, 'attn_temp': 0.9, 'dropout': 0.3, 'pooling_mode': 'attention'}}, 'mlpr': {'inner_lr': 0.01}, 'training': {'batch_size': 256, 'learning_rate': 0.0001, 'weight_decay': 0.001, 'num_epochs': 30, 'ssl': {'ema_decay_init': 0.999, 'ema_decay_base': 0.999, 'warmup_epochs': 3}}, 'cava': {'enabled': True, 'tau_nce': 0.07, 'delta_low_frames': -10.0, 'delta_high_frames': 150.0, 'delta_prior': 37.0, 'window_size': 5, 'mask_type': 'hard', 'multi_scale': False, 'negative_mode': 'intra_sequence_all', 'eq7_negative_definition': 'strict', 'temporal_exclusion_radius': 0, 'gate_range_mode': 'strict', 'gate_range': [0.05, 0.95], 'lambda_cava': 0.1, 'lambda_align': 0.1, 'lambda_edge': 0.05, 'lambda_prior': 0.05, 'lambda_gate': 0.05, 'edge_margin_ratio': 0.25}}
PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {'default': {'profile': 'default', 'training': {'batch_size': 256, 'learning_rate': 0.0001, 'weight_decay': 0.001, 'num_epochs': 30, 'ssl': {'ema_decay_init': 0.999, 'ema_decay_base': 0.999}}, 'mlpr': {'inner_lr': 0.01}, 'cava': {'tau_nce': 0.07, 'delta_low_frames': -10.0, 'delta_high_frames': 150.0, 'delta_prior': 37.0, 'mask_type': 'hard', 'window_size': 5, 'multi_scale': False, 'negative_mode': 'intra_sequence_all', 'eq7_negative_definition': 'strict', 'temporal_exclusion_radius': 0, 'gate_range_mode': 'strict', 'gate_range': [0.05, 0.95], 'lambda_cava': 0.1, 'lambda_align': 0.1, 'lambda_edge': 0.05, 'lambda_prior': 0.05, 'lambda_gate': 0.05}, 'model': {'mil': {'topk_ratio': 0.15}}, 'data': {'nominal_fps': 30, 'split_ratio': '5:3:2'}, 'video': {'fps': 30}}, 'safe': {'profile': 'safe', 'cava': {'mask_type': 'hard', 'window_size': 5, 'multi_scale': False, 'negative_mode': 'batch_global', 'temporal_exclusion_radius': 0, 'gate_range_mode': 'strict', 'gate_range': [0.05, 0.95]}, 'training': {'ssl': {'ema_decay_init': 0.99, 'ema_decay_base': 0.999}}}, 'research_extended': {'profile': 'research_extended', 'cava': {'mask_type': 'gaussian', 'window_size': 3, 'multi_scale': True, 'negative_mode': 'intra_sequence_exclude_neighbors', 'eq7_negative_definition': 'text_strict', 'temporal_exclusion_radius': 1, 'gate_range_mode': 'legacy', 'gate_range': [0.01, 0.99]}}}

def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f'Config must parse to dict, got: {type(cfg)}')
    return cfg

def resolve_runtime_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg_in = copy.deepcopy(cfg)
    profile = str(cfg_in.get('profile', RUNTIME_DEFAULTS['profile']))
    resolved = copy.deepcopy(RUNTIME_DEFAULTS)
    if profile in PROFILE_PRESETS:
        _deep_update(resolved, PROFILE_PRESETS[profile])
    _deep_update(resolved, cfg_in)
    cava = resolved.setdefault('cava', {})
    eq7_mode = str(cava.get('eq7_negative_definition', 'strict'))
    cava['eq7_negative_definition'] = eq7_mode
    if 'negative_mode' not in cava or cfg_in.get('cava', {}).get('negative_mode') is None:
        cava['negative_mode'] = 'intra_sequence_all' if eq7_mode == 'strict' else 'batch_global'
    tau_nce = float(cava.get('tau_nce', cava.get('tau', RUNTIME_DEFAULTS['cava']['tau_nce'])))
    cava['tau_nce'] = tau_nce
    cava['tau'] = tau_nce
    if 'lambda_cava' not in cava:
        cava['lambda_cava'] = float(cava.get('lambda_align', RUNTIME_DEFAULTS['cava']['lambda_cava']))
    cava['lambda_align'] = float(cava.get('lambda_align', cava['lambda_cava']))
    for key, beta_key, default in [('lambda_edge', 'beta_edge', RUNTIME_DEFAULTS['cava']['lambda_edge']), ('lambda_prior', 'beta_prior', RUNTIME_DEFAULTS['cava']['lambda_prior']), ('lambda_gate', 'beta_gate', RUNTIME_DEFAULTS['cava']['lambda_gate'])]:
        cava[key] = float(cava.get(key, cava.get(beta_key, default)))
    cava['delta_prior'] = float(cava.get('delta_prior', 37.0))
    if not isinstance(cava.get('gate_range'), (list, tuple)) or len(cava['gate_range']) != 2:
        mode = str(cava.get('gate_range_mode', 'strict')).lower()
        cava['gate_range'] = [0.01, 0.99] if mode == 'legacy' else [0.05, 0.95]
    ssl = resolved.setdefault('training', {}).setdefault('ssl', {})
    if 'ema_momentum' in ssl:
        ssl['ema_decay_base'] = float(ssl['ema_momentum'])
    ssl['ema_decay_base'] = float(ssl.get('ema_decay_base', ssl.get('ema_decay', RUNTIME_DEFAULTS['training']['ssl']['ema_decay_base'])))
    ssl['ema_decay_init'] = float(ssl.get('ema_decay_init', ssl['ema_decay_base']))
    ssl['ema_decay'] = ssl['ema_decay_base']
    ssl['ema_momentum'] = ssl['ema_decay_base']
    mlpr = resolved.setdefault('mlpr', {})
    if 'inner_lr_alpha' in mlpr:
        mlpr['inner_lr'] = float(mlpr['inner_lr_alpha'])
    mlpr['inner_lr'] = float(mlpr.get('inner_lr', RUNTIME_DEFAULTS['mlpr']['inner_lr']))
    mlpr['inner_lr_alpha'] = mlpr['inner_lr']
    mil = resolved.setdefault('model', {}).setdefault('mil', {})
    if 'mil_topk_ratio' in mil:
        mil['topk_ratio'] = float(mil['mil_topk_ratio'])
    mil['topk_ratio'] = float(mil.get('topk_ratio', RUNTIME_DEFAULTS['model']['mil']['topk_ratio']))
    mil['attn_temp'] = float(mil.get('attn_temp', RUNTIME_DEFAULTS['model']['mil']['attn_temp']))
    mil['dropout'] = float(mil.get('dropout', RUNTIME_DEFAULTS['model']['mil']['dropout']))
    mil['pooling_mode'] = str(mil.get('pooling_mode', RUNTIME_DEFAULTS['model']['mil']['pooling_mode'])).lower()
    return resolved

def extract_key_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    c = resolve_runtime_config(cfg)
    cava = c['cava']
    ssl = c['training']['ssl']
    mil = c['model']['mil']
    mlpr = c.get('mlpr', {})
    feature_mode = str(mlpr.get('feature_mode', 'legacy')).lower()
    use_hist = bool(mlpr.get('use_history_stats', True))
    use_cava = bool(mlpr.get('use_cava_signal', True))
    use_prob = bool(mlpr.get('use_prob_vector', False))
    meta_feat_dim = 7 if feature_mode == '7d' else 3 + 1 + (2 if use_hist else 0) + (1 if use_cava else 0) + (12 if use_prob else 0)
    return {'profile': c.get('profile'), 'tau_nce': float(cava['tau_nce']), 'delta_low_frames': float(cava['delta_low_frames']), 'delta_high_frames': float(cava['delta_high_frames']), 'window_size': int(cava['window_size']), 'mask_type': str(cava['mask_type']), 'multi_scale': bool(cava['multi_scale']), 'negative_mode': str(cava['negative_mode']), 'temporal_exclusion_radius': int(cava['temporal_exclusion_radius']), 'gate_range': [float(cava['gate_range'][0]), float(cava['gate_range'][1])], 'lambda_align': float(cava['lambda_align']), 'lambda_cava': float(cava['lambda_cava']), 'lambda_edge': float(cava['lambda_edge']), 'lambda_prior': float(cava['lambda_prior']), 'lambda_gate': float(cava['lambda_gate']), 'delta_prior': float(cava['delta_prior']), 'inner_lr': float(c.get('mlpr', {}).get('inner_lr', RUNTIME_DEFAULTS['mlpr']['inner_lr'])), 'inner_lr_alpha': float(c.get('mlpr', {}).get('inner_lr_alpha', c.get('mlpr', {}).get('inner_lr', RUNTIME_DEFAULTS['mlpr']['inner_lr']))), 'mlpr_weight_clip': [float(mlpr.get('weight_clip', [0.05, 0.95])[0]), float(mlpr.get('weight_clip', [0.05, 0.95])[1])], 'mlpr_meta_feature_dim': int(meta_feat_dim), 'mlpr_feature_mode': feature_mode, 'ema_decay_init': float(ssl['ema_decay_init']), 'ema_decay_base': float(ssl['ema_decay_base']), 'ema_momentum': float(ssl['ema_momentum']), 'mil_topk_ratio': float(mil['topk_ratio']), 'mil_attn_temp': float(mil['attn_temp']), 'mil_dropout': float(mil['dropout']), 'mil_pooling_mode': str(mil['pooling_mode']), 'batch_size': int(c.get('training', {}).get('batch_size', RUNTIME_DEFAULTS['training']['batch_size'])), 'learning_rate': float(c.get('training', {}).get('learning_rate', RUNTIME_DEFAULTS['training']['learning_rate'])), 'weight_decay': float(c.get('training', {}).get('weight_decay', RUNTIME_DEFAULTS['training']['weight_decay'])), 'epochs': int(c.get('training', {}).get('num_epochs', RUNTIME_DEFAULTS['training']['num_epochs'])), 'eq7_negative_definition': str(cava.get('eq7_negative_definition', 'strict')), 'dataset_nominal_fps': int(c.get('data', {}).get('nominal_fps', 30)), 'working_fps': int(c.get('video', {}).get('fps', 30)), 'split_ratio': str(c.get('data', {}).get('split_ratio', '5:3:2'))}

def _flat(d: Dict[str, Any], prefix: str='') -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        name = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            out.update(_flat(v, name))
        else:
            out[name] = v
    return out

def audit_against_default(cfg: Dict[str, Any], default_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cur = extract_key_config(cfg)
    ref = extract_key_config(default_cfg)
    diff = []
    for k in sorted(set(cur.keys()) | set(ref.keys())):
        if cur.get(k) != ref.get(k):
            diff.append({'key': k, 'current': cur.get(k), 'reference': ref.get(k)})
    return {'is_default': len(diff) == 0, 'num_diffs': len(diff), 'diffs': diff, 'current': cur, 'reference': ref}

def load_default_config(repo_root: str | Path) -> Dict[str, Any]:
    return load_yaml(Path(repo_root) / 'configs' / 'default.yaml')

def save_audit_summary(path: str | Path, summary: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
