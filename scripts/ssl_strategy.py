from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn.functional as F

def compute_weighted_ssl_loss(student_logits: torch.Tensor, pseudo_labels: torch.Tensor, weights: torch.Tensor, teacher_prob: Optional[torch.Tensor]=None, alpha_ce: float=1.0, alpha_kl: float=0.0, temperature: float=1.0) -> torch.Tensor:
    if weights.dim() > 1:
        weights = weights.squeeze(1)
    weights = weights.float()
    if weights.sum() < 1e-06:
        return student_logits.sum() * 0.0
    loss_elem = float(alpha_ce) * F.cross_entropy(student_logits, pseudo_labels, reduction='none')
    if teacher_prob is not None and float(alpha_kl) > 0.0:
        temp = max(float(temperature), 1e-08)
        teacher = teacher_prob.detach().float().clamp_min(1e-08)
        teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-08)
        log_student = F.log_softmax(student_logits / temp, dim=1)
        kl_elem = F.kl_div(log_student, teacher, reduction='none').sum(dim=1) * temp ** 2
        loss_elem = loss_elem + float(alpha_kl) * kl_elem
    return (loss_elem * weights).sum() / weights.sum().clamp_min(1e-06)

class SSLStrategy(ABC):
    use_dist_align: bool = True

    @abstractmethod
    def build_pseudo_targets(self, teacher_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def compute_sample_weights(self, teacher_prob: torch.Tensor, mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        pass

    def compute_unsup_loss(self, student_logits: torch.Tensor, pseudo_labels: torch.Tensor, weights: torch.Tensor, teacher_prob: Optional[torch.Tensor]=None, alpha_ce: float=1.0, alpha_kl: float=0.0, temperature: float=1.0) -> torch.Tensor:
        return compute_weighted_ssl_loss(student_logits=student_logits, pseudo_labels=pseudo_labels, weights=weights, teacher_prob=teacher_prob, alpha_ce=alpha_ce, alpha_kl=alpha_kl, temperature=temperature)

    @abstractmethod
    def update_method_state(self, teacher_prob: torch.Tensor, pseudo_labels: torch.Tensor, mask: torch.Tensor, **kwargs: Any) -> None:
        pass

    def after_optimizer_step(self, trainer: Any, global_step: int) -> None:
        pass

    def get_log_dict(self) -> Dict[str, float]:
        return {}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(dist_align={self.use_dist_align})'

class EMAFixedStrategy(SSLStrategy):
    use_dist_align = True

    def __init__(self, threshold: float=0.9) -> None:
        self.threshold = threshold

    def build_pseudo_targets(self, teacher_prob):
        t_max, t_idx = teacher_prob.max(dim=1)
        mask = (t_max >= self.threshold).float()
        return (t_idx, mask)

    def compute_sample_weights(self, teacher_prob, mask, **kwargs):
        return mask

    def update_method_state(self, teacher_prob, pseudo_labels, mask, **kwargs):
        pass

class FixMatchStrategy(SSLStrategy):
    use_dist_align = False

    def __init__(self, threshold: float=0.95) -> None:
        self.threshold = threshold

    def build_pseudo_targets(self, teacher_prob):
        t_max, t_idx = teacher_prob.max(dim=1)
        mask = (t_max >= self.threshold).float()
        return (t_idx, mask)

    def compute_sample_weights(self, teacher_prob, mask, **kwargs):
        return mask

    def update_method_state(self, teacher_prob, pseudo_labels, mask, **kwargs):
        pass

class FlexMatchStrategy(SSLStrategy):
    use_dist_align = False

    def __init__(self, num_classes: int, base_threshold: float=0.95, min_threshold: float=0.5, ema: float=0.9) -> None:
        self.C = num_classes
        self.base_thresh = base_threshold
        self.min_thresh = min_threshold
        self.ema = ema
        self._class_counts = torch.zeros(num_classes)

    def _class_thresholds(self, device: torch.device) -> torch.Tensor:
        counts = self._class_counts.to(device)
        max_c = counts.max().clamp(min=1.0)
        beta = (counts / max_c).clamp(min=1e-06)
        thresh = (self.base_thresh * beta).clamp(min=self.min_thresh, max=self.base_thresh)
        return thresh

    def build_pseudo_targets(self, teacher_prob):
        device = teacher_prob.device
        t_max, t_idx = teacher_prob.max(dim=1)
        thresh_per_class = self._class_thresholds(device)
        thresh_per_sample = thresh_per_class[t_idx]
        mask = (t_max >= thresh_per_sample).float()
        return (t_idx, mask)

    def compute_sample_weights(self, teacher_prob, mask, **kwargs):
        return mask

    def update_method_state(self, teacher_prob, pseudo_labels, mask, **kwargs):
        with torch.no_grad():
            for c in range(self.C):
                selected = ((pseudo_labels == c).float() * mask).sum().cpu().item()
                self._class_counts[c] = self.ema * self._class_counts[c] + (1.0 - self.ema) * selected

    def get_log_dict(self) -> Dict[str, float]:
        counts = self._class_counts
        return {'flexmatch/min_class_count': counts.min().item(), 'flexmatch/max_class_count': counts.max().item(), 'flexmatch/mean_thresh': self._class_thresholds(torch.device('cpu')).mean().item()}

class FreeMatchStrategy(SSLStrategy):
    use_dist_align = False

    def __init__(self, num_classes: int, ema: float=0.9, init_threshold: float=0.5) -> None:
        self.C = num_classes
        self.ema = ema
        self._global_thresh = init_threshold
        self._class_thresh = torch.full((num_classes,), init_threshold)

    def build_pseudo_targets(self, teacher_prob):
        t_max, t_idx = teacher_prob.max(dim=1)
        mask = (t_max >= self._global_thresh).float()
        return (t_idx, mask)

    def compute_sample_weights(self, teacher_prob, mask, **kwargs):
        return mask

    def update_method_state(self, teacher_prob, pseudo_labels, mask, **kwargs):
        with torch.no_grad():
            t_max = teacher_prob.max(dim=1).values
            batch_mean = t_max.mean().cpu().item()
            self._global_thresh = self.ema * self._global_thresh + (1.0 - self.ema) * batch_mean
            for c in range(self.C):
                cls_sel = (pseudo_labels == c) & (mask > 0.5)
                if cls_sel.any():
                    cls_mean = t_max[cls_sel].mean().cpu().item()
                    self._class_thresh[c] = self.ema * self._class_thresh[c] + (1.0 - self.ema) * cls_mean

    def get_log_dict(self) -> Dict[str, float]:
        return {'freematch/global_thresh': self._global_thresh}

class SoftMatchStrategy(SSLStrategy):
    use_dist_align = False

    def __init__(self, num_classes: int, base_threshold: float=0.5, ema: float=0.9, init_mu: float=0.9, init_var: float=0.01) -> None:
        self.C = num_classes
        self.base_thresh = base_threshold
        self.ema = ema
        self._class_mu = torch.full((num_classes,), init_mu)
        self._class_var = torch.full((num_classes,), init_var)

    def build_pseudo_targets(self, teacher_prob):
        t_max, t_idx = teacher_prob.max(dim=1)
        mask = (t_max >= self.base_thresh).float()
        return (t_idx, mask)

    def compute_sample_weights(self, teacher_prob, mask, **kwargs):
        device = teacher_prob.device
        t_max, t_idx = teacher_prob.max(dim=1)
        mu = self._class_mu.to(device)
        std = self._class_var.to(device).clamp(min=0.0001).sqrt()
        mu_i = mu[t_idx]
        std_i = std[t_idx]
        log_w = -0.5 * ((t_max - mu_i) / std_i) ** 2
        w = log_w.exp()
        w = w * mask
        n_eff = mask.sum().clamp(min=1.0)
        B = float(teacher_prob.shape[0])
        w = (w * B / n_eff).clamp(min=0.0, max=2.0)
        return w

    def compute_unsup_loss(self, student_logits, pseudo_labels, weights, teacher_prob: Optional[torch.Tensor]=None, alpha_ce: float=1.0, alpha_kl: float=0.0, temperature: float=1.0):
        return compute_weighted_ssl_loss(student_logits=student_logits, pseudo_labels=pseudo_labels, weights=weights, teacher_prob=teacher_prob, alpha_ce=alpha_ce, alpha_kl=alpha_kl, temperature=temperature)

    def update_method_state(self, teacher_prob, pseudo_labels, mask, **kwargs):
        with torch.no_grad():
            t_max = teacher_prob.max(dim=1).values
            for c in range(self.C):
                cls_sel = (pseudo_labels == c) & (mask > 0.5)
                if cls_sel.any():
                    vals = t_max[cls_sel].cpu()
                    b_mu = vals.mean().item()
                    b_var = vals.var(unbiased=False).item() if len(vals) > 1 else 0.0
                    self._class_mu[c] = self.ema * self._class_mu[c] + (1.0 - self.ema) * b_mu
                    self._class_var[c] = (self.ema * self._class_var[c] + (1.0 - self.ema) * b_var).clamp(min=0.0001)

    def get_log_dict(self) -> Dict[str, float]:
        return {'softmatch/mean_mu': self._class_mu.mean().item(), 'softmatch/mean_std': self._class_var.sqrt().mean().item()}

class OursMLPRStrategy(SSLStrategy):
    use_dist_align = True

    def __init__(self, trainer_ref: Any) -> None:
        self._t = trainer_ref
        self._meta_fail_count = 0

    def build_pseudo_targets(self, teacher_prob):
        t_max, t_idx = teacher_prob.max(dim=1)
        thresh = getattr(self._t, 'ssl_final_thresh', 0.9)
        mask = (t_max >= thresh).float()
        return (t_idx, mask)

    def compute_sample_weights(self, teacher_prob: torch.Tensor, mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        trainer = self._t
        if trainer.meta is None:
            return mask
        student_out: Dict[str, Any] = kwargs.get('student_out', {})
        ids_u = kwargs.get('ids_u', None)
        labeled_batch = kwargs.get('labeled_batch', (None, None, None))
        try:
            try:
                from meta_reweighter import build_mlpr_features
            except ImportError:
                from scripts.meta_reweighter import build_mlpr_features
            stu_feat: Optional[torch.Tensor] = None
            if 'fusion_token' in student_out:
                ftok = student_out['fusion_token']
                stu_feat = ftok.mean(dim=tuple(range(1, ftok.dim()))) if ftok.dim() > 2 else ftok
            else:
                v_emb = student_out.get('video_emb')
                a_emb = student_out.get('audio_emb')
                if v_emb is not None and a_emb is not None:
                    if v_emb.dim() > 2:
                        v_emb = v_emb.mean(dim=1)
                    if a_emb.dim() > 2:
                        a_emb = a_emb.mean(dim=1)
                    stu_feat = torch.cat([v_emb, a_emb], dim=-1)
            hist_mu: Optional[torch.Tensor] = None
            hist_std: Optional[torch.Tensor] = None
            id_list = None
            if trainer.hist_bank is not None and ids_u is not None:
                id_list = ids_u.cpu().tolist() if torch.is_tensor(ids_u) else list(ids_u)
                h_mu, h_sd = trainer.hist_bank.query([int(x) for x in id_list])
                hist_mu = h_mu.view(-1).to(trainer.device)
                hist_std = h_sd.view(-1).to(trainer.device)
            cava_gate_mean: Optional[torch.Tensor] = None
            if student_out.get('causal_gate') is not None:
                cg = student_out['causal_gate']
                cava_gate_mean = cg.mean(dim=tuple(range(1, cg.dim()))).view(-1, 1)
            _cur_kl: Optional[torch.Tensor] = None
            _s_logits = student_out.get('clip_logits')
            if _s_logits is not None and hist_mu is not None:
                _s_log_prob = F.log_softmax(_s_logits.detach(), dim=1)
                _cur_kl = F.kl_div(_s_log_prob, teacher_prob.detach(), reduction='none').sum(dim=1)
                _loss_trend = (_cur_kl - hist_mu).detach()
            else:
                _loss_trend = None
            feats = build_mlpr_features(teacher_prob=teacher_prob.detach(), student_feat=stu_feat.detach() if stu_feat is not None else None, history_mean=hist_mu, history_std=hist_std, cava_gate_mean=cava_gate_mean.detach() if cava_gate_mean is not None else None, use_prob_vector=trainer._mlpr_flags.get('use_prob_vec', False), feature_mode=trainer._mlpr_feature_mode, delay_frames=student_out.get('delay_frames_cont', student_out.get('delay_frames')), delta_prior=float(trainer.cava_cfg.get('delta_prior', 0.0)) if trainer._mlpr_flags.get('use_delay_prior', True) else None, loss_trend=_loss_trend)
            with torch.no_grad():
                w = trainer.meta(feats)
            w_eff = (w * mask).clamp(0.0, 1.0)
            vu, au = kwargs.get('unlabeled_batch', (None, None))
            if vu is not None and au is not None:
                t_idx_cache = teacher_prob.max(dim=1).indices
                v_l, a_l, y_l = labeled_batch
                with torch.no_grad():
                    trainer._last_labeled_batch = (v_l.detach() if v_l is not None else None, a_l.detach() if a_l is not None else None, y_l.detach() if y_l is not None else None)
                    trainer._last_unlabeled_batch = (vu.detach(), au.detach(), t_idx_cache.detach())
                    trainer._last_teacher_prob = teacher_prob.detach()
                    trainer._last_w_features = feats.detach()
                    trainer._last_w_mask = mask.detach()
                    trainer._last_ssl_loss_cfg = {'alpha_ce': float(kwargs.get('alpha_ce', 1.0)), 'alpha_kl': float(kwargs.get('alpha_kl', 0.0)), 'temperature': float(kwargs.get('temperature', 1.0)), 'lambda_u': float(kwargs.get('lambda_u', 1.0))}
            if trainer.hist_bank is not None and id_list is not None:
                s_logits = student_out.get('clip_logits')
                if s_logits is not None:
                    s_log_prob = F.log_softmax(s_logits.detach(), dim=1)
                    kl = F.kl_div(s_log_prob, teacher_prob.detach(), reduction='none').sum(dim=1)
                    trainer.hist_bank.update(id_list, kl.detach())
            return w_eff
        except Exception as exc:
            import traceback
            self._meta_fail_count += 1
            if self._meta_fail_count == 1:
                traceback.print_exc()
                print(f"[OursMLPR] weight-gen FAILED on first call - MetaNet will be bypassed. Fix the root cause; runs labeled 'ours_mlpr' are degraded to plain mask.")
            elif self._meta_fail_count % 10 == 0:
                print(f'[OursMLPR] weight-gen still failing (#{self._meta_fail_count}): {exc}')
            if hasattr(trainer, '_mlpr_weight_gen_fail_count'):
                trainer._mlpr_weight_gen_fail_count = self._meta_fail_count
            trainer._last_w_features = None
            trainer._last_labeled_batch = None
            trainer._last_unlabeled_batch = None
            trainer._last_teacher_prob = None
            trainer._last_w_mask = None
            trainer._last_ssl_loss_cfg = None
            return mask

    def update_method_state(self, teacher_prob, pseudo_labels, mask, **kwargs):
        pass

    def after_optimizer_step(self, trainer: Any, global_step: int) -> None:
        if not trainer.mlpr_enabled:
            return
        interval = getattr(trainer, '_mlpr_meta_interval', 200)
        if global_step % interval == 0:
            trainer.opt.zero_grad(set_to_none=True)
            trainer._meta_update_step(global_step)
            trainer.opt.zero_grad(set_to_none=True)
            trainer.model.zero_grad(set_to_none=True)

def build_ssl_strategy(cfg: Dict[str, Any], trainer_ref: Any=None) -> SSLStrategy:
    method = cfg.get('training', {}).get('ssl_method', 'ours_mlpr').lower().strip()
    C = cfg.get('data', {}).get('num_classes', 12)
    ssl_cfg = cfg.get('training', {}).get('ssl', {})
    base_thresh = float(ssl_cfg.get('final_thresh', 0.9))
    ema = float(ssl_cfg.get('ema_decay_base', 0.999))
    state_ema = float(ssl_cfg.get('strategy_state_ema', 0.9))
    if method == 'ema_fixed':
        return EMAFixedStrategy(threshold=base_thresh)
    if method == 'fixmatch':
        return FixMatchStrategy(threshold=float(ssl_cfg.get('fixmatch_thresh', 0.95)))
    if method == 'flexmatch':
        return FlexMatchStrategy(num_classes=C, base_threshold=float(ssl_cfg.get('flexmatch_base_thresh', 0.95)), min_threshold=float(ssl_cfg.get('flexmatch_min_thresh', 0.5)), ema=state_ema)
    if method == 'freematch':
        return FreeMatchStrategy(num_classes=C, ema=state_ema, init_threshold=float(ssl_cfg.get('freematch_init_thresh', 0.5)))
    if method == 'softmatch':
        return SoftMatchStrategy(num_classes=C, base_threshold=float(ssl_cfg.get('softmatch_base_thresh', 0.5)), ema=state_ema)
    if method == 'ours_mlpr':
        if trainer_ref is None:
            raise ValueError('ours_mlpr requires trainer_ref; pass trainer instance to build_ssl_strategy().')
        return OursMLPRStrategy(trainer_ref=trainer_ref)
    raise ValueError(f"Unknown ssl_method '{method}'. Choose from: ema_fixed, fixmatch, flexmatch, freematch, softmatch, ours_mlpr")
SSL_METHOD_NAMES = ['ema_fixed', 'fixmatch', 'flexmatch', 'freematch', 'softmatch', 'ours_mlpr']
