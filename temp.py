import numpy as np
from pathlib import Path
from tqdm import tqdm
from shutil import copyfile
from remove_short_audios import check_duration


src = Path(r'D:\Code\projects\DDSP-SVC\data\kazuma\train\audio')
dest = Path(r'D:\Code\projects\DDSP-SVC\data\kazuma\val\audio')
# val_targets = (
#   'Kazuma_2021_Story_12gatsu_3_08_0001.wav',
#   'Kazuma_Main3_9_4_9_0001.wav',
# )
# for v in tqdm(val_targets):
#   f = src / v
#   if not f.exists():
#     print(f, 'not found')
#     continue
#   f.rename(dest / v)

dest = Path(r'D:\Code\projects\DDSP-SVC\data\kazuma\train_more')
for i, f in tqdm(enumerate(src.iterdir())):
  if i % 5 != 0 or f.stem.startswith('ちいさな冒険者'):
    continue
  # f.unlink()
  f.rename(dest / f.name)
exit()

# src = Path(r'D:/documents/!audio/konofan-audio/Voice/Story')
# dest = Path(r'D:/documents/!audio/megumin')
# for d in tqdm(src.iterdir()):
#   if not d.is_dir():
#     continue
#   for f in d.glob('megumin_*'):
#     ft = dest / f.name
#     not ft.exists() and check_duration(f) and copyfile(f, ft)
# exit()


# root = Path(r'/data/cxp/toys/DDSP-SVC/data/val/')
# for d in tqdm(tuple(root.iterdir())):
#   if not d.is_dir():
#     continue
#   for f in d.glob('ちいさな冒険者 -カズマ Ver.- - 福島潤_*'):
#     print(f'removing {f}')
#     f.unlink()

# path_pitchaugdict = Path(r'D:\code\Projects\DDSP-SVC\data\val\pitch_aug_dict.npy')
# path_pitchaugdict1 = Path(r'D:\code\Projects\DDSP-SVC\data\val\pitch_aug_dict1.npy')
# pitch_aug_dict = np.load(path_pitchaugdict, allow_pickle=True).item()
# pitch_aug_dict1 = np.load(path_pitchaugdict1, allow_pickle=True).item()
# # print(pitch_aug_dict.keys())
# for k in tqdm(tuple(pitch_aug_dict.keys())):
#   if k.startswith('ちいさな冒険者 -カズマ Ver.- - 福島潤_'):
#     print(k)
#     assert k in pitch_aug_dict
#     del pitch_aug_dict[k]
#     assert k not in pitch_aug_dict

# pitch_aug_dict |= pitch_aug_dict1
# np.save(path_pitchaugdict, pitch_aug_dict)
