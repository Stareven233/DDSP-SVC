import json
import pickle
from pathlib import Path
import yaml
import re


def check_path_exist(func):

  def inner(*args, **kwargs):
    p = args[0]
    if not isinstance(p, Path):
      p = Path(p)
      args = (p, *args[1:])
    if not p.is_file():
      raise FileNotFoundError(p)
    res = func(*args, **kwargs)
    return res

  return inner


@check_path_exist
def load_json(path, encoding='utf-8', **kwargs) -> dict:
  with path.open('r', encoding=encoding) as f:
    return json.load(f, **kwargs)


def write_json(path, obj, encoding='utf-8', **kwargs) -> None:
  with open(path, 'w', encoding=encoding) as f:
    json.dump(obj, f, ensure_ascii=False, **kwargs)


@check_path_exist
def load_pickle(path, **kwargs):
  with path.open('rb') as f:
    return pickle.load(f, **kwargs)


def write_pickle(path, obj, **kwargs) -> None:
  with open(path, 'wb') as f:
    pickle.dump(obj, f, **kwargs)


@check_path_exist
def load_yaml(path) -> dict:
  with path.open('r') as f:
    return yaml.safe_load(f)


def write_yaml(path, obj, encoding='utf-8', **kwargs) -> None:
  with open(path, 'w', encoding=encoding) as f:
    yaml.dump(obj, f, encoding=encoding, **kwargs)


def find_nth_sub_path(path: Path, r: str, index=-1) -> Path:
  '''
    验证p是一个目录，然后对其中的每个路径名应用正则表达式r提取数字，
    过滤无法提取数字的路径，并按提取的数字排序返回有效路径列表。
  
    用于从权重目录下获取最新的训练权重

    :param path: 要扫描的目录路径
    :param r: 用于提取数字的正则表达式字符串，例如 r'model_(\d+)\.pt'
    :param index: 从排序好的（默认从小到大）路径中获取第index个
    :return: 按提取数字排序的有效子路径列表
    :raises: ValueError 如果p不是目录
    '''
  if not path.is_dir():
    raise ValueError(f'\'{path}\' is not a directory')

  compiled_regex = re.compile(r)
  valid_paths = []

  for p in path.iterdir():
    if not p.is_file():
      continue
    match = compiled_regex.search(p.name)
    if match is None:
      continue
    try:
      number = int(match.group(1))
      valid_paths.append((number, p))
    except (ValueError, IndexError):
      # 忽略无法转换为数字的匹配
      continue

  # 按数字排序并返回目标路径
  target = sorted(valid_paths, key=lambda x: x[0])
  return target[-1][index]
