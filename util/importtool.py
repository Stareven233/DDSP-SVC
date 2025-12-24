import importlib
from pathlib import Path
from typing import Sequence

from omegaconf import OmegaConf, DictConfig


def get_obj_from_str(str_, reload=False):
  module, cls = str_.rsplit(".", 1)
  if reload:
    module_imp = importlib.import_module(module)
    importlib.reload(module_imp)
  return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, cls='target', args='params'):
  if not cls in config:
    raise KeyError('Expected key `target` to instantiate.')
  return get_obj_from_str(config[cls])(**config.get(args, dict()))


def instantiate_from_config_recursively(config: dict|DictConfig):
  '''模仿 Lightning CLI 做的，根据config递归构建对象
  input: {class_path: models.nanogpt.GPT, init_args:..., }
  '''

  cls = config.get('class_path', None)
  if cls is None:
    raise KeyError('Expected key `class_path` to instantiate.')
  if isinstance(config, DictConfig):
    config = OmegaConf.to_object(config)
  need_init = lambda d: isinstance(d, dict) and 'class_path' in d
  
  params = dict()
  new_v = None
  t = config.get('init_args', dict()) | config.get('dict_kwargs', dict())
  for k, v in t.items():
    if need_init(v):
      new_v = instantiate_from_config_recursively(v)
    elif isinstance(v, list):
      new_v = []
      for i in v:
        new_v.append(instantiate_from_config_recursively(i) if need_init(i) else i)
      # 太麻烦，不考虑嵌套列表
    else:
      new_v = v
    params[k] = new_v

  return get_obj_from_str(cls)(**params)


def instantiate_from_yaml_recursively(path: str|Path|Sequence, target_key:str|None=None):
  '''根据yaml文件递归构建对象'''

  if isinstance(path, Sequence):
    config = OmegaConf.merge(*tuple(OmegaConf.load(p).model for p in path))
  else:
    config = OmegaConf.load(path)
  if target_key is not None:
    if '.' in target_key:
      target_key = target_key.split('.')
    else:
      target_key = [target_key]
    for k in target_key:
      config = config.get(k)
  return instantiate_from_config_recursively(config)
