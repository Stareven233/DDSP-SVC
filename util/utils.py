import random
import time
from pathlib import Path
from collections import abc
import shutil

import numpy as np
import torch

ROOT_DIR = Path(__file__, '../..').resolve()


def exclusive_sample(seq: list | tuple, exclude: list | tuple | None = None, k=1) -> tuple:
  '''从seq中抽取不重复，且不属于exclude的k个元素'''

  if exclude is None:
    return random.sample(seq, k)
  n = k + len(exclude)
  x = set(random.sample(seq, n))
  exclude = set(exclude)
  # remove exclude from x
  x = tuple(x - exclude)[:k]
  return x


def counter(initial_num=0):
  num = initial_num

  def inner():
    nonlocal num
    t = num
    num += 1
    return t

  return inner


def format_float(x: float | np.ndarray | torch.Tensor, ndigits=-1) -> str:
  # template = '{' + f':.{ndigits}f' + '}'
  if isinstance(x, (
      np.ndarray,
      torch.Tensor,
  )):
    assert len(x.shape) == 1 and x.shape[0] == 1
    x = x.item()

  if ndigits == -1 or ndigits is None:
    return str(x)
  return f'{x:.{ndigits}f}'


def randint(a=None, b=None):
  '''返回[a,b]中的随机整数，只填a则范围为[0,a]，都不填就随机返回整数'''
  if a is None:
    b = time.time_ns()
    a = -b
  elif b is None:
    b, a = a, 0
  return random.randint(a, b)


def rand_half_true() -> bool:
  return random.random() < 0.5


def chain_sequence(*args) -> list:
  ret = []
  for r in args:
    if not isinstance(r, str) and isinstance(r, abc.Iterable):
      ret.extend(r)
      continue
    ret.append(r)
  return ret


def chain_iterable(*args):
  for r in args:
    if isinstance(r, abc.Iterable):
      yield from r


class Timer:

  def __init__(self, start=None) -> None:
    self.last = start or time.time()
    self.epoch = 0

  def next_epoch(self, num=1):
    self.epoch += num

  def timeit(self, name='pass'):
    print(f'{self.epoch}-{name}: {time.time() - self.last:.4f}s')
    self.last = time.time()


def timeit(func):

  def run(*args, **kwargs):
    t = time.time()
    res = func(*args, **kwargs)
    print('executing \'%s\' costed %.3fs' % (func.__name__, time.time() - t))
    return res

  return run


def chr2hex(ch):
  return hex(ord(ch))[2:].upper()


def hex2chr(uni):
  return chr(int(uni, 16))


def setup_seed(seed, strict=True):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  if strict:
    torch.backends.cudnn.deterministic = True
  # https://zhuanlan.zhihu.com/p/76472385
  # 评论提到这个会让训练变得很慢


__desc_counter = counter(0)


def _desc_var(f):

  def _preview(s: str, threshold=47):
    h = threshold
    if (l := len(s)) <= h + 3:
      return s
    return f'{s[:h]}...{s[-min(h, l-h):]}'

  def _array_tensor_format(x: np.ndarray | torch.Tensor):
    minmax = ''
    if x.numel() > 0:
      minmax = f', min={x.min()}, max={x.max()}'
    return f'var: type={type(x)}{minmax}, shape={x.shape}, dtype={x.dtype}; '

  f_repr = repr(f)
  desc = ''

  if isinstance(f, (np.ndarray, torch.Tensor)):
    desc = _array_tensor_format(f)
  elif isinstance(f, (list, tuple)):
    tmp = f[0]
    cnt = 1
    while isinstance(tmp, (list, tuple)):
      tmp = tmp[0]
      cnt += 1
    desc = f'var{"[0]"*cnt}: type={type(tmp)}, len={len(f)}, repr={_preview(f_repr)}; '
  elif not isinstance(f, dict):
    desc = f'var: type={type(f)}, repr={_preview(f_repr)}; '

  if desc:
    print(desc)
    return

  for k, v in f.items():
    desc += f'{k}, {type(v)}, '
    if isinstance(v, (np.ndarray, torch.Tensor)):
      desc += _array_tensor_format(v)
    elif isinstance(v, (
        list,
        tuple,
    )):
      desc += f'len={len(v)}; '
    else:
      r = repr(v)
      desc += f'repr={_preview(r)}; '

  print(desc)


def desc_var(*args, finish=False):
  print()
  for v in args:
    print(__desc_counter(), end='# ')
    _desc_var(v)
  if finish:
    exit(0)


def desc_var_exit(*args):
  desc_var(*args, finish=True)


def show_cuda_info():
  print(f'{torch.__version__=}')
  if torch.cuda.is_available():
    print(f'{torch.version.cuda=}')
    print(f'{torch.backends.cudnn.version()=}')
    print(f'number of available gpu: {torch.cuda.device_count()}')  # 有几个可用的gpu
    print(f'index of current device: {torch.cuda.current_device()}')  # 可用gpu编号
    print(f'device capability: {".".join(map(str, torch.cuda.get_device_capability()))}')  # 可用gpu算力
    print(f'device name: {torch.cuda.get_device_name()}')  # 可用gpu的名字
  else:
    print('No CUDA GPUs are available')


def count_parameters(model: torch.nn.Module):
  # trainable, frozen
  bin1, bin2 = [], []
  for p in model.parameters():
    if p.requires_grad:
      bin1.append(p.numel())
    else:
      bin2.append(p.numel())
  return sum(bin1), sum(bin2)


def show_parameters_num(model: torch.nn.Module):
  b1, b2 = tuple(map(lambda x: x / 1e6, count_parameters(model)))
  print(f'<{model.__class__.__name__}> number of parameters: {b1+b2:.2f}M = trainable {b1:.2f}M + frozen {b2:.2f}')


def backup_codes(dst: str | Path):
  '''将项目文件夹整个备份到 dst 指定的位置'''

  if isinstance(dst, str):
    dst = Path(dst)
  if dst.is_dir() and dst.exists():
    print(f'{dst} already exists, removing')
    shutil.rmtree(dst)
  shutil.copytree(ROOT_DIR, dst, symlinks=True)
