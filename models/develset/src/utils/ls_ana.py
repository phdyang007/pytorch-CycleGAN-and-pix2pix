'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2021-04-08 15:04:21
LastEditTime: 2021-04-10 14:56:18
Contact: cgjhaha@qq.com
Description: analyse the level set function to get a optimal TSDF
'''


import os
import sys
# sys.path.append('/home/guojin/projects/develset_opc/levelset_net')
sys.path.append('/research/d2/xfyao/guojin/develset_opc/levelset_net')
import torch
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
from src.models.const import *

# /home/guojin/data/datasets/iccad_2013/ls_params

# default_path = '/home/guojin/data/datasets/iccad_2013/ls_params'
# default_path = '/home/guojin/data/datasets/iccad_2013_tsdf/ls_params'
# default_path = '/home/guojin/data/datasets/develset/train/ls_params'
default_path = '/research/d2/xfyao/guojin/develset_opc/levelset_net/l11ganopc_outputs/ckpts_2021-04-09_20-02-35/mask_ls'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ls_dir', type=str, default=default_path, help='the input ls dir')
cfg = parser.parse_args()


ls_dir = Path(cfg.ls_dir)

ls_lists = list(sorted(ls_dir.glob('*.npy')))

max_ls = -1000
min_ls = 1000

num_up = 0
num_down = 0

nan_num = 0


nan_lists = []


for ls in tqdm(ls_lists):
    # print(str(ls))
    ls_t = torch.from_numpy(np.load(str(ls)))
    if torch.isnan(ls_t).any():
        print(str(ls))
        nan_lists.append(ls)
        nan_num += 1
    # ls_t = ls_t[:,LITHOSIM_OFFSET:MASK_TILE_END_Y,LITHOSIM_OFFSET:MASK_TILE_END_X]
    t_max = torch.max(ls_t)
    t_min = torch.min(ls_t)
    max_ls = max(max_ls, t_max)
    min_ls = min(min_ls, t_min)
    if t_max > 900:
        num_up += 1

    if t_min < -100:
        num_down +=1

print('max: {}'.format(max_ls))
print('min: {}'.format(min_ls))
print('num up: {}'.format(num_up))
print('num down: {}'.format(num_down))
print('nan num: {}'.format(nan_num))



# for nan_ls in tqdm(nan_lists):
#     par = nan_ls.parent.parent
#     name = nan_ls.stem
#     t = par / 'target' / '{}.png'.format(name)
#     tls = par / 'target_ls' / '{}.npy'.format(name)
#     m = par / 'mask' / '{}.png'.format(name)
#     t.unlink()
#     tls.unlink()
#     m.unlink()
#     nan_ls.unlink()

