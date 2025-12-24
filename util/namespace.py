from pathlib import Path
from collections.abc import Sequence

import yaml

from util import io


class Dummy:

  def __getattr__(self, _):
    return self

  def __setattr__(self, _, __):
    pass

  def __getitem__(self, _):
    return self

  def __setitem__(self, _, __):
    pass

  def __call__(self, *_, **__):
    return self


class Namespace:

  def __init__(self, *args: dict, **kwargs):
    data = dict()
    for a in args:
      if isinstance(a, dict):
        data |= a
      elif isinstance(a, self.__class__):
        data |= a.as_dict()

    data |= kwargs
    object.__setattr__(self, '__c', data)  # 避免触发 __setattr__

  def __get_impl(self, key):
    # 使用 self.__dict__ 以避免一般属性的循环调用
    if isinstance((c := self.__dict__['__c'][key]), dict):
      c = self.__class__(c)
      self.__dict__['__c'][key] = c
      # return self.__class__(c)
    return c

  def __getitem__(self, key):
    if key not in self.__dict__['__c']:
      raise KeyError(f'{key} not found in {self.__class__.__name__}')
    return self.__get_impl(key)

  def __getattr__(self, key):
    if key not in self.__dict__['__c']:
      raise AttributeError(f'{key} not found in {self.__class__.__name__}')
    return self.__get_impl(key)

  def __set_impl(self, key, value, /, another=None, **kwargs):
    if key is not None and value is not None:
      self.__dict__['__c'][key] = value
      return
    if another is None:
      another = dict()
    if isinstance(another, self.__class__):
      another = another.as_dict()
    elif not isinstance(another, dict):
      raise ValueError(f'only support dict or {self.__class__.__name__} type')
    kwargs |= another
    self.__dict__['__c'] |= kwargs

  def __setitem__(self, key, value):
    self.__set_impl(key, value)

  def __setattr__(self, key, value):
    self.__set_impl(key, value)

  def __len__(self):
    return len(self.__dict__['__c'])

  def __contains__(self, key):
    return key in self.__dict__['__c']

  def __eq__(self, another):
    if isinstance(another, self.__class__):
      another = another.as_dict()
    if not isinstance(another, dict):
      return False
    return self.__dict__['__c'] == another

  def __or__(self, another):
    if not isinstance(another, (dict, self.__class__)):
      raise TypeError(f'only support dict or {self.__class__.__name__} type')
    return self.__class__(self, another)

  def __ior__(self, another):
    '''self |= another'''
    self.__set_impl(None, None, another=another)
    return self

  def __delitem__(self, key: str) -> None:
    del self.__dict__['__c'][key]

  def __delattr__(self, key: str) -> None:
    try:
      del self.__dict__['__c'][key]
    except KeyError as e:
      raise AttributeError(key) from e

  def __getstate__(self):
    # 序列化时调用
    return self.__dict__

  def __setstate__(self, state):
    # 反序列化时调用
    self.__dict__.update(state)

  def __str__(self) -> str:
    return str(self.as_dict())

  def get(self, key, default=None):
    if key not in self.__dict__['__c']:
      return default
    return self.__get_impl(key)

  def pop(self, key):
    ret = self[key]
    del self.__dict__['__c'][key]
    return ret

  def keys(self):
    # 支持 **namespace 的关键
    return self.__dict__['__c'].keys()

  def update(self, another=dict(), **kwargs):
    '''若key已存在则值会被覆盖'''
    self.__set_impl(None, None, another=another, **kwargs)

  def accumulate(self, another=dict(), **kwargs):
    '''v = v + v_prev, 同样是更新值，但不覆盖而是加到原值上'''
    if isinstance(another, self.__class__):
      another = another.as_dict()
    if isinstance(another, dict):
      kwargs |= another
    for k, v in kwargs.items():
      self.__set_impl(k, self.get(k, v.__class__()) + v)

  def as_dict(self) -> dict:
    ret = self.__dict__['__c'].copy()
    for k in tuple(ret.keys()):
      v = ret[k]
      if isinstance(v, self.__class__):
        ret[k] = v.as_dict()
      elif isinstance(v, Path):
        ret[k] = v.as_posix()
    return ret

  def copy(self):
    return self.__class__(self)


class Config(Namespace):
  '''加载yaml文件/dict，支持以dict/attr形式进行读取。
  OmegaConf的简化版
  '''

  def __init__(self, args, /, **kwargs):
    # args: dict|str|Path|Sequence|None
    if args is not None and not isinstance(args, (dict, str, Path, Sequence,)):
      raise TypeError(f'cannot use {type(args)=}: "{args}" to init {self.__class__.__name__}')

    if isinstance(args, (str, Path)):
      args = (args,)
    if isinstance(args, Sequence) and len(args) > 0:
      for path in args:
        if not isinstance(path, Path):
          path = Path(path)
        kwargs |= self.load_yaml(path)
      args = None

    super().__init__(args, **kwargs)

  @staticmethod
  def load_yaml(path):
    if isinstance(path, Path):
      path = path.as_posix()
    with open(path, 'r', encoding='utf-8') as f:
      return yaml.safe_load(f)

  def __repr__(self) -> str:
    msg = '----------------- Config ---------------\n'
    for k, v in sorted(self.as_dict().items()):
      msg += f'{k:>25}: {str(v):<30}\n'
    msg += '----------------- End -------------------\n'
    return msg

  def print_configs(self, save_path=None):
    """Print and save configs

    It will print both current configs and default values(if different).
    It will save configs into a text file / [checkpoints_dir] / opt.txt
    """

    message = repr(self)
    print(message)
    if save_path is None:
      return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    io.write_json(save_path, self.as_dict())
