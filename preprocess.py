'''
cd D:\Code\projects\DDSP-SVC
$python="D:\Software\SVC-Fusion\project\.conda\python.exe"
& $python preprocess.py -c configs/reflow_kazuma.yaml
& $python preprocess.py -c configs/reflow_megumin.yaml
& $python preprocess.py -c configs/reflow_fritia.yaml

cd /data/cxp/toys/DDSP-SVC/
conda activate ddsp
nvidia-smi
python preprocess.py -c configs/reflow.yaml -s train_5_16
'''

import os
from pathlib import Path
import numpy as np
import random
import librosa
import torch
# import pyworld as pw
# import parselmouth
import argparse
import shutil
from tqdm import tqdm

from logger import utils
from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
from reflow.vocoder import Vocoder

# import concurrent.futures


def parse_args(args=None, namespace=None):
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument("-c", "--config", type=str, required=True, help="path to the config file")
  parser.add_argument("-d", "--device", type=str, default='cuda:0', required=False, help="cpu or cuda, auto if not set")
  parser.add_argument("-s", "--split", type=str, default=None, required=False, help="train/val_1_8: 训练或验证集数据，划分八份，预处理第一份")
  return parser.parse_args(args=args, namespace=namespace)


def preprocess(path, f0_extractor, volume_extractor, mel_extractor, units_encoder, sample_rate, hop_size, device='cuda', use_pitch_aug=False, extensions=['wav'], frange=None):
  path_srcdir = Path(path, 'audio')
  path_unitsdir = Path(path, 'units')
  path_f0dir = Path(path, 'f0')
  path_volumedir = Path(path, 'volume')
  path_augvoldir = Path(path, 'aug_vol')
  path_meldir = Path(path, 'mel')
  path_augmeldir = Path(path, 'aug_mel')
  path_skipdir = Path(path, 'skip')
  data_dirs = (path_unitsdir, path_f0dir, path_volumedir, path_meldir, path_augmeldir, path_augvoldir, path_skipdir)
  tuple(d.mkdir(exist_ok=True) for d in data_dirs)
  sub_item = next(path_srcdir.iterdir())
  if sub_item.is_dir():
    n_spk = len(tuple(path_srcdir.iterdir()))
    for i in range(1, n_spk+1):
      for d in data_dirs:
        (d / str(i)).mkdir(parents=True, exist_ok=True)

  # list files
  filelist = utils.traverse_dir(path_srcdir.as_posix(), extensions=extensions, is_pure=True, is_sort=True, is_ext=True)
  if frange is not None:
    i, n = tuple(map(int, frange))
    i = min(n, max(1, i))
    len_chunk = int((len(filelist) + n - 1) / n)
    filelist = filelist[(i-1) * len_chunk:i * len_chunk]

  # pitch augmentation dictionary
  pitch_aug_dict = {}

  # run
  def process(file):
    binfile = file + '.npy'
    path_srcfile = path_srcdir / file
    path_unitsfile = path_unitsdir / binfile
    path_f0file = path_f0dir / binfile
    path_volumefile = path_volumedir / binfile
    path_augvolfile = path_augvoldir / binfile
    path_melfile = path_meldir / binfile
    path_augmelfile = path_augmeldir / binfile
    path_skipfile = path_skipdir / file

    # load audio
    audio, _ = librosa.load(path_srcfile, sr=sample_rate)
    if len(audio.shape) > 1:
      audio = librosa.to_mono(audio)
    audio_t = torch.from_numpy(audio).float().to(device)
    audio_t = audio_t.unsqueeze(0)

    # extract volume
    volume = volume_extractor.extract(audio)

    # extract mel and volume augmentaion
    if mel_extractor is not None:
      mel_t = mel_extractor.extract(audio_t, sample_rate)
      mel = mel_t.squeeze().to('cpu').numpy()

      max_amp = float(torch.max(torch.abs(audio_t))) + 1e-5
      max_shift = min(1, np.log10(1 / max_amp))
      log10_vol_shift = random.uniform(-1, max_shift)
      if use_pitch_aug:
        keyshift = random.uniform(-5, 5)
      else:
        keyshift = 0

      aug_mel_t = mel_extractor.extract(audio_t * (10**log10_vol_shift), sample_rate, keyshift=keyshift)
      aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
      aug_vol = volume_extractor.extract(audio * (10**log10_vol_shift))

    # units encode
    units_t = units_encoder.encode(audio_t, sample_rate, hop_size)
    units = units_t.squeeze().to('cpu').numpy()
    # extract f0
    f0 = f0_extractor.extract(audio, uv_interp=False)

    uv = f0 == 0
    if len(f0[~uv]) > 0:
      # interpolate the unvoiced f0
      f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

      # save npy
      np.save(path_unitsfile, units)
      np.save(path_f0file, f0)
      np.save(path_volumefile, volume)
      if mel_extractor is not None:
        pitch_aug_dict[file] = keyshift
        np.save(path_melfile, mel)
        np.save(path_augmelfile, aug_mel)
        np.save(path_augvolfile, aug_vol)
    else:
      print(f'\n[Error] F0 extraction failed: {path_srcfile}')
      shutil.move(path_srcfile, path_skipdir)
      print(f'This file has been moved to {path_skipfile}')

  print(f'Preprocess the audio clips in : {path_srcdir}')
  # single process
  for file in tqdm(filelist):
    process(file)

  if mel_extractor is not None:
    path_pitchaugdict = os.path.join(path, 'pitch_aug_dict.npy')
    np.save(path_pitchaugdict, pitch_aug_dict)
  # multi-process (have bugs)
  '''
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        list(tqdm(executor.map(process, filelist), total=len(filelist)))
    '''


if __name__ == '__main__':
  # parse commands
  print('pasing arguments')
  cmd = parse_args()
  split = None
  frange = None
  if cmd.split is not None:
    split, *frange = cmd.split.split('_')
  print(f'{split=}, {frange=}')

  device = cmd.device
  if device is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'{device=}')

  # load config
  args = utils.load_config(cmd.config)
  sample_rate = args.data.sampling_rate
  hop_size = args.data.block_size

  extensions = args.data.extensions

  # initialize f0 extractor
  f0_extractor = F0_Extractor(args.data.f0_extractor, args.data.sampling_rate, args.data.block_size, args.data.f0_min, args.data.f0_max)
  print('F0_Extractor initialized')

  # initialize volume extractor
  volume_extractor = Volume_Extractor(args.data.block_size, args.data.volume_smooth_size)
  print('Volume_Extractor initialized')

  # initialize mel extractor
  mel_extractor = None
  use_pitch_aug = False
  mel_extractor = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
  print('Vocoder initialized')
  if mel_extractor.vocoder_sample_rate != sample_rate or mel_extractor.vocoder_hop_size != hop_size:
    mel_extractor = None
    print('Unmatch vocoder parameters, mel extraction is ignored!')
  elif args.model.use_pitch_aug:
    use_pitch_aug = True

  # initialize units encoder
  if args.data.encoder == 'cnhubertsoftfish':
    cnhubertsoft_gate = args.data.cnhubertsoft_gate
  else:
    cnhubertsoft_gate = 10
  units_encoder = Units_Encoder(args.data.encoder, args.data.encoder_ckpt, args.data.encoder_sample_rate, args.data.encoder_hop_size, cnhubertsoft_gate=cnhubertsoft_gate, device=device)
  print('Units_Encoder initialized')

  if split is None or split == 'train':
    # preprocess training set
    preprocess(args.data.train_path, f0_extractor, volume_extractor, mel_extractor, units_encoder, sample_rate, hop_size, device=device, use_pitch_aug=use_pitch_aug, extensions=extensions, frange=frange)
  if split is None or split == 'val':
    # preprocess validation set
    preprocess(args.data.valid_path, f0_extractor, volume_extractor, mel_extractor, units_encoder, sample_rate, hop_size, device=device, use_pitch_aug=False, extensions=extensions, frange=frange)
