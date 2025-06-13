from pathlib import Path
from tqdm import tqdm
import soundfile as sf


MIN_LENGTH = 2


def check_duration(file, min_duration=MIN_LENGTH):
  # 打开wav文件
  f = sf.SoundFile(file)
  # 获取帧数和帧率
  frames = f.frames
  rate = f.samplerate
  # 计算时长（秒）
  duration = frames / rate
  # 关闭文件
  f.close()
  # 返回时长是否大于最短时长的布尔值
  return duration > min_duration


if __name__ == '__main__':
  root = Path(r'D:\code\Projects\DDSP-SVC\data\train\audio')
  # for a in tqdm(short_audios):
  for a in tqdm(root.iterdir()):
    if check_duration(a):
      continue
    a.unlink()
