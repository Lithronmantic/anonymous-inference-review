from typing import Dict, Iterable, Tuple
import torch
from torch import nn
import torch.nn.functional as F
try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

def _extract_clip_logits(output):
    if isinstance(output, dict):
        return output.get('clip_logits', output)
    return output

def _mixed_unsup_loss_per_example(student_logits: torch.Tensor, pseudo_labels: torch.Tensor, teacher_prob: 'torch.Tensor | None', *, alpha_ce: float, alpha_kl: float, temperature: float) -> torch.Tensor:
    loss_elem = float(alpha_ce) * F.cross_entropy(student_logits, pseudo_labels, reduction='none')
    if teacher_prob is not None and float(alpha_kl) > 0.0:
        temp = max(float(temperature), 1e-08)
        teacher = teacher_prob.detach().float().clamp_min(1e-08)
        teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-08)
        log_student = F.log_softmax(student_logits / temp, dim=1)
        kl_elem = F.kl_div(log_student, teacher, reduction='none').sum(dim=1) * temp ** 2
        loss_elem = loss_elem + float(alpha_kl) * kl_elem
    return loss_elem

def sgd_fast_weights_detached(model: nn.Module, det_params: Dict[str, torch.Tensor], loss: torch.Tensor, lr: float) -> Dict[str, torch.Tensor]:
    grads = torch.autograd.grad(loss, list(det_params.values()), create_graph=True, allow_unused=True)
    fast_state = {}
    for (n, p), g in zip(det_params.items(), grads):
        fast_state[n] = p - lr * g if g is not None else p
    return fast_state

@torch.no_grad()
def _flatten_probs(p: torch.Tensor) -> torch.Tensor:
    if p.dim() == 1:
        return p.unsqueeze(0)
    return p

def meta_step_first_order(student_model: nn.Module, meta_net: nn.Module, meta_opt: torch.optim.Optimizer, *, v_l: torch.Tensor, a_l: torch.Tensor, y_l: torch.Tensor, v_tr: torch.Tensor, a_tr: torch.Tensor, yhat_tr: torch.Tensor, w_tr: torch.Tensor, teacher_prob_tr: 'torch.Tensor | None', v_val: torch.Tensor, a_val: torch.Tensor, y_val: torch.Tensor, lr_inner: float=0.001, mask_tr: 'torch.Tensor | None'=None, alpha_ce: float=1.0, alpha_kl: float=0.0, temperature: float=1.0, lambda_u: float=1.0) -> Dict[str, float]:
    det_params = {n: p.clone().detach().requires_grad_(True) for n, p in student_model.named_parameters() if p.requires_grad}
    w_eff = w_tr.view(-1)
    if mask_tr is not None:
        w_eff = w_eff * mask_tr.view(-1).to(w_eff)
    if w_eff.sum() < 0.5:
        with torch.no_grad():
            w_flat = w_tr.view(-1).detach()
            return {'meta_val_loss': float('nan'), 'meta_train_loss': float('nan'), 'w_mean': float(w_flat.mean().cpu()), 'w_std': float(w_flat.std(unbiased=False).cpu()) if w_flat.numel() > 1 else 0.0, 'w_min': float(w_flat.min().cpu()), 'w_max': float(w_flat.max().cpu()), 'skipped': True}
    logits_sup = functional_call(student_model, det_params, (v_l, a_l), kwargs={'return_aux': False})
    logits_sup = _extract_clip_logits(logits_sup)
    L_sup = F.cross_entropy(logits_sup, y_l)
    logits_tr = functional_call(student_model, det_params, (v_tr, a_tr), kwargs={'return_aux': False})
    logits_tr = _extract_clip_logits(logits_tr)
    per_ex = _mixed_unsup_loss_per_example(logits_tr, yhat_tr, teacher_prob_tr, alpha_ce=alpha_ce, alpha_kl=alpha_kl, temperature=temperature)
    L_unsup = (w_eff * per_ex).sum() / w_eff.sum().clamp(min=1e-06)
    L_tr = L_sup + float(lambda_u) * L_unsup
    fast_state = sgd_fast_weights_detached(student_model, det_params, L_tr, lr=lr_inner)
    logits_val = functional_call(student_model, fast_state, (v_val, a_val), kwargs={'return_aux': False})
    if isinstance(logits_val, dict):
        logits_val = logits_val.get('clip_logits', logits_val)
    L_val = F.cross_entropy(logits_val, y_val)
    meta_opt.zero_grad(set_to_none=True)
    L_val.backward()
    torch.nn.utils.clip_grad_norm_(meta_net.parameters(), max_norm=1.0)
    meta_opt.step()
    with torch.no_grad():
        w_flat = w_tr.view(-1).detach()
        return {'meta_val_loss': float(L_val.detach().cpu().item()), 'meta_train_loss': float(L_tr.detach().cpu().item()), 'meta_sup_loss': float(L_sup.detach().cpu().item()), 'meta_unsup_loss': float(L_unsup.detach().cpu().item()), 'w_mean': float(w_flat.mean().cpu().item()), 'w_std': float(w_flat.std(unbiased=False).cpu().item()) if w_flat.numel() > 1 else 0.0, 'w_min': float(w_flat.min().cpu().item()), 'w_max': float(w_flat.max().cpu().item()), 'skipped': False}

def meta_step_first_order_from_features(student_model: nn.Module, meta_net: nn.Module, meta_opt: torch.optim.Optimizer, *, w_features: torch.Tensor, v_l: torch.Tensor, a_l: torch.Tensor, y_l: torch.Tensor, v_tr: torch.Tensor, a_tr: torch.Tensor, yhat_tr: torch.Tensor, teacher_prob_tr: 'torch.Tensor | None', v_val: torch.Tensor, a_val: torch.Tensor, y_val: torch.Tensor, lr_inner: float=0.001, mask_tr: 'torch.Tensor | None'=None, alpha_ce: float=1.0, alpha_kl: float=0.0, temperature: float=1.0, lambda_u: float=1.0) -> Dict[str, float]:
    w_tr = meta_net(w_features)
    return meta_step_first_order(student_model=student_model, meta_net=meta_net, meta_opt=meta_opt, v_l=v_l, a_l=a_l, y_l=y_l, v_tr=v_tr, a_tr=a_tr, yhat_tr=yhat_tr, w_tr=w_tr, teacher_prob_tr=teacher_prob_tr, v_val=v_val, a_val=a_val, y_val=y_val, lr_inner=lr_inner, mask_tr=mask_tr, alpha_ce=alpha_ce, alpha_kl=alpha_kl, temperature=temperature, lambda_u=lambda_u)

def meta_step_neumann(student_model: nn.Module, meta_net: nn.Module, meta_opt: torch.optim.Optimizer, *, v_tr: torch.Tensor, a_tr: torch.Tensor, yhat_tr: torch.Tensor, w_tr: torch.Tensor, v_val: torch.Tensor, a_val: torch.Tensor, y_val: torch.Tensor, lr_inner: float=0.001, neumann_iter: int=5, damping: float=0.5) -> float:
    logits_tr = student_model(v_tr, a_tr, return_aux=False)
    if isinstance(logits_tr, dict):
        logits_tr = logits_tr.get('clip_logits', logits_tr)
    per_ex = F.cross_entropy(logits_tr, yhat_tr, reduction='none')
    L_tr = (w_tr.view(-1) * per_ex).mean()
    params = [p for p in student_model.parameters() if p.requires_grad]
    g = torch.autograd.grad(L_tr, params, create_graph=True, retain_graph=True, allow_unused=True)
    logits_val = student_model(v_val, a_val, return_aux=False)
    if isinstance(logits_val, dict):
        logits_val = logits_val.get('clip_logits', logits_val)
    L_val = F.cross_entropy(logits_val, y_val)
    g_val = torch.autograd.grad(L_val, params, create_graph=True, retain_graph=True, allow_unused=True)
    v_vec = [gv.detach() for gv in g_val]
    for _ in range(neumann_iter):
        hv = torch.autograd.grad(g, params, grad_outputs=v_vec, retain_graph=True, allow_unused=True)
        v_vec = [gv + (1 - damping) * vv - lr_inner * (hv_i if hv_i is not None else 0.0) for gv, vv, hv_i in zip(g_val, v_vec, hv)]
    meta_opt.zero_grad(set_to_none=True)
    L_tr.backward()
    torch.nn.utils.clip_grad_norm_(meta_net.parameters(), max_norm=1.0)
    meta_opt.step()
    return float(L_val.detach().cpu().item())
