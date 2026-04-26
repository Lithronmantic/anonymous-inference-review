from typing import Union

def compute_ema_decay(epoch: int, ema_decay_base: Union[float, int], ema_decay_init: Union[float, int], warmup_epochs: int) -> float:
    base = float(ema_decay_base)
    init = float(ema_decay_init)
    if warmup_epochs <= 0:
        return base
    e = max(1, int(epoch))
    if e >= warmup_epochs:
        return base
    t = float(e - 1) / float(max(1, warmup_epochs - 1))
    return init + (base - init) * t
