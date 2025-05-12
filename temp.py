import numpy as np
from pathlib import Path
from tqdm import tqdm
from shutil import copyfile
from remove_short_audios import check_duration


src = Path(r'D:\documents\!audio\kazuma')
dest = Path(r'D:\documents\!audio\t')
i = 0b0
for f in tqdm(src.iterdir()):
  i ^= 0b1
  if i == 0b1:
    continue
  f.unlink()
  # f.rename(dest / f.name)
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
