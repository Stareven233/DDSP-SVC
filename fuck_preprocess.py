'''
cd /data/cxp/toys/DDSP-SVC/
nvidia-smi
pkill -u cxp -f python
python fuck_preprocess.py

cd D:\code\Projects\DDSP-SVC
conda activate ddsp
python preprocess.py -c configs/reflow_megumin.yaml -d cuda:0

cd D:\Code\projects\DDSP-SVC
$python="D:\Software\SVC-Fusion\project\.conda\python.exe"
& $python preprocess.py -c configs/reflow_megumin.yaml -d cuda:0
'''

import os
import subprocess
from pathlib import Path


f0_dir = Path(r'data/train/f0')
# 计数f0_dir目录下的所有文件，并统计数量
i = 0
for f in f0_dir.iterdir():
  if not f.is_file():
    continue
  i += 1
i //= 64

# 定义需要运行的 Python 文件名
target_script = 'preprocess.py'

# for i in range(2, 17):
print(f'Starting process for iteration {i} / 16')
# 使用 subprocess 创建一个新进程运行目标脚本
process = subprocess.Popen(['python', target_script, '-c', 'configs/reflow.yaml', '-s', f'train_{i}_16'], cwd=os.getcwd())
# 等待当前进程执行结束
return_code = process.wait()
if return_code != 0:
  print(f'Process failed with return code {return_code}')
  # exit()
print(f'Process for iteration {i} / 16 completed.\n')
