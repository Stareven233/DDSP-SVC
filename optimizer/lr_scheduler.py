import math
from functools import partial

from torch.optim import Optimizer
from torch.optim import lr_scheduler


# step scheduler
def fn_linear_warmup(warmup_steps, step):
  if step < warmup_steps:  # linear warmup
    return float(step) / float(max(1, warmup_steps))
  else:
    return 1.0


def linear_warmup(optimizer: Optimizer, warmup_steps):
  # return partial(fn_linear_warmup, warmup_steps)
  scheduler = lr_scheduler.LambdaLR(optimizer, partial(fn_linear_warmup, warmup_steps))
  return scheduler


def fn_linear_warmup_cosine_decay(warmup_steps, max_steps, multipler_min, step):
  if step < warmup_steps:  # linear warmup
    return float(step) / float(max(1, warmup_steps))
  else:  # cosine learning rate schedule
    multipler = 0.5 * (math.cos((step-warmup_steps) / (max_steps-warmup_steps) * math.pi) + 1)
    return max(multipler, multipler_min)


def linear_warmup_cosine_decay(optimizer: Optimizer, warmup_steps, max_steps, multipler_min):
  # return partial(fn_linear_warmup_cosine_decay, warmup_steps, max_steps, multipler_min)
  scheduler = lr_scheduler.LambdaLR(optimizer, partial(fn_linear_warmup_cosine_decay, warmup_steps, max_steps, multipler_min))
  return scheduler


def linear_warmup_drop(optimizer: Optimizer, steps_per_epoch: int, warmup_epochs: int, drop_epoch_list: None|tuple[int], drop_rate: float=0.1):
  # if drop_epochs is not None:
  #   drop_steps = tuple()
  warmup_steps = max(1, int(steps_per_epoch * warmup_epochs))
  rate = 1.0

  def inner(step):
    nonlocal rate
    nonlocal drop_epoch_list
    if step < warmup_steps:  # linear warmup
      return step / warmup_steps
    elif not isinstance(drop_epoch_list, (list, tuple)) or len(drop_epoch_list)==0:
      return rate
    i = 0
    for e in drop_epoch_list:
      if step > e*steps_per_epoch:
        i += 1
      else:
        break
    rate = rate * (drop_rate**i)
    drop_epoch_list = drop_epoch_list[i:]
    return rate

  scheduler = lr_scheduler.LambdaLR(optimizer, inner)
  return scheduler


def linear_warmup_decay(optimizer: Optimizer, warmup_steps: int, decay_per_steps: int, decay_rate: float=0.1, last_steps=-1):
  warmup_steps = max(0, min(warmup_steps, decay_per_steps))
  rate = 1.0
  last_decay_step = warmup_steps
  if last_steps > warmup_steps:
    n = (last_steps - warmup_steps) // decay_per_steps
    rate = decay_rate ** n
    last_decay_step += decay_per_steps * n

  def inner(step):
    nonlocal rate
    nonlocal last_decay_step
    if step < warmup_steps:  # linear warmup
      return step / warmup_steps
    elif (step - last_decay_step) >= decay_per_steps:
      last_decay_step = step
      rate *= decay_rate
    return rate

  scheduler = lr_scheduler.LambdaLR(optimizer, inner, last_steps)
  return scheduler


def cosine_annealing(optimizer: Optimizer, max_lr: float, max_steps: int, warmup_ratio=0, final_lr_ratio=1e-4, **kwargs):
  div_factor = kwargs.get('div_factor', 25)
  final_div_factor = 1 / (div_factor * final_lr_ratio)
  scheduler = lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    total_steps=max_steps,
    pct_start=warmup_ratio,  # 从 initial_lr warmup 到 max_lr 所占的step比例
    div_factor=div_factor,  # initial_lr = max_lr/div_factor
    final_div_factor=final_div_factor,  # 余弦退火最终学习率相比initial_lr降低的倍数，div_factor默认25，因此分子就是相比max_lr降低的倍数
    # **kwargs,
  )
  return scheduler


def warmup_stable_decay(optimizer: Optimizer, max_steps: int, warmup_ratio=0, decay_ratio=0, **_):
  # WSD策略（Warmup-Stable-Decay） @Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations
  n_warmup = max_steps * warmup_ratio
  n_decay = max_steps * decay_ratio  # n大于20k，可小于0.2
  def inner(step):
    if step < n_warmup:
      return step / n_warmup  # 线性增长
    elif step <= max_steps - n_decay:
      return 1  # 稳定
    else:
      t = (step - (max_steps - n_decay)) / n_decay  # (0 -> 1)
      t = 1 - t**0.5  # (1 -> 0)
      return t
  scheduler = lr_scheduler.LambdaLR(optimizer, inner)
  return scheduler


def warmup_stage_decay(optimizer: Optimizer, decay_per_steps: int, max_steps: int, warmup_ratio=0.05, decay_ratio=0.2, decay_rate=0.1, last_steps=-1):
  n_warmup = max_steps * warmup_ratio
  n_warmup = max(0, min(n_warmup, decay_per_steps))
  n_decay = max_steps * decay_ratio
  decay_steps = max_steps - n_decay
  rate = 1.0
  last_decay_step = n_warmup
  if last_steps > n_warmup:
    n = (last_steps - n_warmup) // decay_per_steps
    rate = decay_rate ** n
    last_decay_step += decay_per_steps * n

  def inner(step):
    nonlocal rate
    nonlocal last_decay_step
    if step < n_warmup:  # linear warmup
      return step / n_warmup
    elif step < decay_steps and (step - last_decay_step) >= decay_per_steps:
      last_decay_step = step
      rate *= decay_rate
    elif step >= decay_steps:
      t = (step - decay_steps) / n_decay  # (0 -> 1)
      t =  1 - t**0.5  # (1 -> 0)
      return rate * t
    return rate

  scheduler = lr_scheduler.LambdaLR(optimizer, inner, last_steps)
  return scheduler
