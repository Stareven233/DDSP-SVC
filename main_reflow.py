r'''
cd D:/code/Projects/DDSP-SVC
nvidia-smi
$python = "D:/Software/SVC-Fusion/project/.conda/python.exe"

$model = "exp/kazuma/model_11600.pt"
$model = "exp/acacia/model_2000.pt"
$model = "exp/fritia/model_3500.pt"
$model = "exp/megumin/model_3200.pt"
$indir = "D:\Document\ai-sings\銀の龍の背に乗って"
$indir = "D:\Document\ai-sings"
$path = "$indir\God Knows\4K高清修复音源升级God Knows_Vocals_vocals_noreverb-new-au.flac"
$path = "$indir\黄昏\黄昏_人声2.flac"
$path = "$indir\銀の龍の背に乗って\骑在银龙的背上_vnV.flac"
$path = "$indir\TAIDADA\TAIDADA_反相不纯人声_Vocals_vocals_noreverb.flac"
$path = "$indir\虫儿飞\童声歌唱家冯晓菲奶声虫儿飞带你净化心灵_Vocals_vocals_noreverb.flac"
$path = "$indir\最后一页\顾疚疚最后一页_Vocals_vocals.flac"
$path = "$indir\Ending Note\Ending Note 門谷純_Vocals_vocals.flac"


$key=0
$vocal_key=0
$formant_key=0

& $python main_reflow.py -m $model -i "$path" -k $key -f $formant_key -v $vocal_key
$mix="{1:0.8,2:0.2}"
& $python main_reflow.py -m $model -i "$path" -k $key -f $formant_key -v $vocal_key -mix $mix
& $python main_reflow.py -m $model -i "$indir" -k $key -f $formant_key -v $vocal_key

cd F:/CODE/!projects/DDSP-SVC
uv run python main_reflow.py -m $model -i "$indir/$filename" -k $key -f $formant_key -v $vocal_key
New-Item -Path "D:/Code/projects/ddsp6.2/pretrain/contentvec/checkpoint_best_legacy_500.pt" -ItemType HardLink -Target "D:/Software/SVC-Fusion/project/pretrain/contentvec/checkpoint_best_legacy_500.pt"
New-Item -Path "D:/Code/projects/ddsp6.2/pretrain/rmvpe/model.pt" -ItemType HardLink -Target "D:/Software/SVC-Fusion/project/pretrain/rmvpe/model.pt"
'''
#AI翻唱 #RIFT  #芙提雅  #尘白禁区  #精灵世纪 #霞光 
#童年 #怀旧 #经典 #华语MV


import os
import re
import torch
import fairseq
torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])
import librosa
import argparse
import numpy as np
import soundfile as sf
# import parselmouth
import hashlib
from ast import literal_eval
from slicer import Slicer
from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
from ddsp.core import upsample
from reflow.vocoder import load_model_vocoder
from tqdm import tqdm
from pathlib import Path


def check_args(ddsp_args, diff_args):
  if ddsp_args.data.sampling_rate != diff_args.data.sampling_rate:
    print("Unmatch data.sampling_rate!")
    return False
  if ddsp_args.data.block_size != diff_args.data.block_size:
    print("Unmatch data.block_size!")
    return False
  if ddsp_args.data.encoder != diff_args.data.encoder:
    print("Unmatch data.encoder!")
    return False
  return True


def parse_args(args=None, namespace=None):
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-m",
      "--model_ckpt",
      type=str,
      required=True,
      help="path to the model checkpoint",
  )
  parser.add_argument("-d", "--device", type=str, default=None, required=False, help="cpu or cuda, auto if not set")
  parser.add_argument(
      "-i",
      "--input",
      type=str,
      nargs='+',
      required=True,
      help="path /dir to the input audio files",
  )
  parser.add_argument(
      "-o",
      "--output",
      type=str,
      nargs='+',
      required=False,
      default=(),
      help="path to the output audio files, only works when --input not includes dir",
  )
  parser.add_argument(
      "-id",
      "--spk_id",
      type=str,
      required=False,
      default=1,
      help="speaker id (for multi-speaker model) | default: 1",
  )
  parser.add_argument(
      "-mix",
      "--spk_mix_dict",
      type=str,
      required=False,
      default="None",
      help="mix-speaker dictionary (for multi-speaker model) | default: None",
  )
  parser.add_argument(
      "-k",
      "--key",
      type=int,
      required=False,
      default=0,
      help="key changed (number of semitones) | default: 0",
  )
  parser.add_argument(
      "-f",
      "--formant_shift_key",
      type=int,
      required=False,
      default=0,
      help="formant changed (number of semitones) , only for pitch-augmented model| default: 0",
  )
  parser.add_argument(
      "-v",
      "--vocal_register_shift_key",
      type=int,
      required=False,
      default=0,
      # 输入正值表示先降key进行推理再让声码器升回来
      help="vocal register changed (number of semitones) , only for pc-type vocoder| default: 0",
  )
  parser.add_argument(
      "-pe",
      "--pitch_extractor",
      type=str,
      required=False,
      default='rmvpe',
      help="pitch extrator type: parselmouth, dio, harvest, crepe, fcpe, rmvpe (default)",
  )
  parser.add_argument(
      "-fmin",
      "--f0_min",
      type=str,
      required=False,
      default=50,
      help="min f0 (Hz) | default: 50",
  )
  parser.add_argument(
      "-fmax",
      "--f0_max",
      type=str,
      required=False,
      default=1100,
      help="max f0 (Hz) | default: 1100",
  )
  parser.add_argument(
      "-th",
      "--threhold",
      type=str,
      required=False,
      default=-60,
      help="response threhold (dB) | default: -60",
  )
  parser.add_argument(
      "-step",
      "--infer_step",
      type=str,
      required=False,
      default='auto',
      help="sample steps | default: auto",
  )
  parser.add_argument(
      "-method",
      "--method",
      type=str,
      required=False,
      default='auto',
      help="euler or rk4 | default: auto",
  )
  parser.add_argument(
      "-ts",
      "--t_start",
      type=str,
      required=False,
      default=0.0,
      help="t_start | default: auto",
  )
  return parser.parse_args(args=args, namespace=namespace)


def split(audio, sample_rate, hop_size, db_thresh=-40, min_len=5000):
  slicer = Slicer(sr=sample_rate, threshold=db_thresh, min_length=min_len)
  chunks = dict(slicer.slice(audio))
  result = []
  for k, v in chunks.items():
    tag = v["split_time"].split(",")
    if tag[0] != tag[1]:
      start_frame = int(int(tag[0]) // hop_size)
      end_frame = int(int(tag[1]) // hop_size)
      if end_frame > start_frame:
        result.append((start_frame, audio[int(start_frame * hop_size):int(end_frame * hop_size)]))
  return result


def cross_fade(a: np.ndarray, b: np.ndarray, idx: int):
  result = np.zeros(idx + b.shape[0])
  fade_len = a.shape[0] - idx
  np.copyto(dst=result[:idx], src=a[:idx])
  k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
  result[idx:a.shape[0]] = (1-k) * a[idx:] + k * b[:fade_len]
  np.copyto(dst=result[a.shape[0]:], src=b[fade_len:])
  return result


step_patten = re.compile('(?<=model_)\d+')  # model_180000.pt
def gen_metadata(args, ckpt):
  # ckpt = ckpt.removeprefix('model_').removesuffix('.pt')
  ckpt = args.model_ckpt.split('/')[-1]  # exp/megumin/model_228000.pt
  m = step_patten.search(ckpt)
  s = '0'
  if m is not None:
    s = int(m.group(0)) / 1000
  s = f'ddsp@{s}ks_{args.key}k_{args.vocal_register_shift_key}vk'
  if args.formant_shift_key != 0:
    s += f'_{args.formant_shift_key}fk'
  mix_dict = args.spk_mix_dict
  if mix_dict and mix_dict != 'None':
    s += f'_{mix_dict.replace(":", "@")}m'
  return s


@torch.no_grad()
def infer_file(model, vocoder, ucoder, args, cmd, device, in_file:Path, out_file:Path|None=None):
  # load input
  print(f'processing {in_file}...')
  audio, sample_rate = librosa.load(in_file, sr=None)
  if len(audio.shape) > 1:
    audio = librosa.to_mono(audio)
  hop_size = args.data.block_size * sample_rate / args.data.sampling_rate
  win_size = args.data.volume_smooth_size * sample_rate / args.data.sampling_rate

  # get MD5 hash from wav file
  md5_hash = ""
  with in_file.open('rb') as f:
    data = f.read()
    md5_hash = hashlib.md5(data).hexdigest()
    print("MD5: " + md5_hash)

  cache_dir_path = os.path.join(os.path.dirname(__file__), "cache")
  cache_file_path = os.path.join(cache_dir_path, f"{cmd.pitch_extractor}_{hop_size}_{cmd.f0_min}_{cmd.f0_max}_{md5_hash}.npy")

  is_cache_available = os.path.exists(cache_file_path)
  if is_cache_available:
    # f0 cache load
    print('Loading pitch curves for input audio from cache directory...')
    f0 = np.load(cache_file_path, allow_pickle=False)
  else:
    # extract f0
    print('Pitch extractor type: ' + cmd.pitch_extractor)
    pitch_extractor = F0_Extractor(cmd.pitch_extractor, sample_rate, hop_size, float(cmd.f0_min), float(cmd.f0_max))
    print('Extracting the pitch curve of the input audio...')
    f0 = pitch_extractor.extract(audio, uv_interp=True, device=device)

    # f0 cache save
    os.makedirs(cache_dir_path, exist_ok=True)
    np.save(cache_file_path, f0, allow_pickle=False)

  f0 = torch.from_numpy(f0).float().to(device).unsqueeze(-1).unsqueeze(0)

  # key change
  f0 = f0 * 2**(cmd.key / 12)

  # formant change
  formant_shift_key = torch.from_numpy(np.array([[cmd.formant_shift_key]])).float().to(device)

  # vocal register change
  if vocoder.vocoder.h.pc_aug:
    vocal_register_factor = 2**(cmd.vocal_register_shift_key / 12)
  else:
    print('Vocal register shift is not supported for current vocoder!')
    vocal_register_factor = 1

  # extract volume
  print('Extracting the volume envelope of the input audio...')
  volume_extractor = Volume_Extractor(hop_size, win_size)
  volume = volume_extractor.extract(audio)
  mask = (volume > 10**(float(cmd.threhold) / 20)).astype('float')
  mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
  mask = upsample(mask, args.data.block_size).squeeze(-1)
  volume = torch.from_numpy(volume).float().to(device).unsqueeze(-1).unsqueeze(0)

  # speaker id or mix-speaker dictionary
  spk_mix_dict = literal_eval(cmd.spk_mix_dict)
  spk_id = torch.LongTensor(np.array([[int(cmd.spk_id)]])).to(device)
  if spk_mix_dict is not None:
    print('Mix-speaker mode')
  else:
    print('Speaker ID: ' + str(int(cmd.spk_id)))

  # sampling method
  if cmd.method == 'auto':
    method = args.infer.method
  else:
    method = cmd.method

  # infer step
  if cmd.infer_step == 'auto':
    infer_step = args.infer.infer_step
  else:
    infer_step = int(cmd.infer_step)

  # t_start
  if cmd.t_start == 'auto':
    if args.model.t_start is not None:
      t_start = float(args.model.t_start)
    else:
      t_start = 0.0
  else:
    t_start = float(cmd.t_start)
    if args.model.t_start is not None and t_start < args.model.t_start:
      t_start = args.model.t_start

  if infer_step > 0:
    print('Sampling method: ' + method)
    print('infer step: ' + str(infer_step))
    print('t_start: ' + str(t_start))
  elif infer_step < 0:
    print('infer step cannot be negative!')
    exit(0)

  # forward and save the output
  result = np.zeros(0)
  current_length = 0
  segments = split(audio, sample_rate, hop_size)
  print('Cut the input audio into ' + str(len(segments)) + ' slices')
  for segment in tqdm(segments):
    start_frame = segment[0]
    seg_input = torch.from_numpy(segment[1]).float().unsqueeze(0).to(device)
    seg_units = ucoder.encode(seg_input, sample_rate, hop_size)
    seg_f0 = f0[:, start_frame:start_frame + seg_units.size(1), :]
    seg_volume = volume[:, start_frame:start_frame + seg_units.size(1), :]
    seg_mel = model(seg_units, seg_f0 / vocal_register_factor, seg_volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=formant_shift_key, vocoder=vocoder, infer_step=infer_step, method=method, t_start=t_start)
    seg_output = vocoder.infer(seg_mel, seg_f0)
    seg_output *= mask[:, start_frame * args.data.block_size:(start_frame + seg_units.size(1)) * args.data.block_size]
    seg_output = seg_output.squeeze().cpu().numpy()

    silent_length = round(start_frame * args.data.block_size) - current_length
    if silent_length >= 0:
      result = np.append(result, np.zeros(silent_length))
      result = np.append(result, seg_output)
    else:
      result = cross_fade(result, seg_output, current_length + silent_length)
    current_length = current_length + silent_length + len(seg_output)
  if out_file is None:
    *_, name, ckpt = cmd.model_ckpt.split('/')  # exp/megumin/model_228000.pt
    out_file = in_file.parent / f'{in_file.stem}_{name}_{gen_metadata(cmd, ckpt)}.flac'
  sf.write(out_file, result, args.data.sampling_rate)


if __name__ == '__main__':
  # parse commands
  cmd = parse_args()
  #device = 'cpu'
  device = cmd.device
  if device is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # load reflow model
  model, vocoder, args = load_model_vocoder(cmd.model_ckpt, device=device)
  # load units encoder
  if args.data.encoder == 'cnhubertsoftfish':
    cnhubertsoft_gate = args.data.cnhubertsoft_gate
  else:
    cnhubertsoft_gate = 10
  ucoder = Units_Encoder(args.data.encoder, args.data.encoder_ckpt, args.data.encoder_sample_rate, args.data.encoder_hop_size, cnhubertsoft_gate=cnhubertsoft_gate, device=device)

  infiles = []
  outfiles = tuple(Path(i) for i in cmd.output)
  for i in cmd.input:
    i = Path(i)
    if not i.is_dir():
      infiles.append(i)
      continue
    infiles.extend(i.iterdir())
    outfiles = None

  for i, f in enumerate(infiles):
    if f.suffix[1:].upper() not in sf.available_formats():
      continue
    f_o = len(outfiles) > 0 and outfiles[i] or None
    infer_file(model, vocoder, ucoder, args, cmd, device, f, f_o)
