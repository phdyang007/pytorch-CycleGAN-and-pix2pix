'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2021-04-08 15:04:21
LastEditTime: 2021-04-08 17:00:14
Contact: cgjhaha@qq.com
Description:
transfer the sdf dataset to the tsdf dataset
'''


import os
import sys
sys.path.append('/home/guojin/projects/develset_opc/levelset_net')
import torch
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path
from src.models.const import *


# default_path = '/home/guojin/data/datasets/iccad_2013/ls_params'
# default_path = '/home/guojin/data/datasets/develset/train/ls_params'

default_in = '/home/guojin/data/datasets/iccad_2013'
default_out = '/home/guojin/data/datasets/iccad_2013_tsdf'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--in_dir', type=str, default=default_in, help='the input ls dir')
parser.add_argument('--out_dir', type=str, default=default_out, help='the output ls dir')
cfg = parser.parse_args()

in_dir = Path(cfg.in_dir)
out_dir = Path(cfg.out_dir)

out_dir.mkdir(exist_ok=True)

in_tar = in_dir / 'targets'
in_ls = in_dir / 'ls_params'

out_tar = out_dir / 'targets'
out_ls = out_dir / 'ls_params'

# copy the target image first
# out_tar.mkdir()
if not out_tar.exists():
    shutil.copytree(str(in_tar), str(out_tar))

# transfer the ls
out_ls.mkdir(exist_ok=True)

in_ls_lists = list(sorted(in_ls.glob('*.pt')))

for ls in tqdm(in_ls_lists):
    # print(str(ls))
    ls_name = ls.name
    ls_t = torch.load(str(ls))
    if torch.isnan(ls_t).any():
        print(str(ls))
    ls_t[torch.where(ls_t > 1280)] = 1280
    ls_t[torch.where(ls_t < -1280)] = -1280

    out_ls_p = out_ls / ls_name
    torch.save(ls_t, str(out_ls_p))

