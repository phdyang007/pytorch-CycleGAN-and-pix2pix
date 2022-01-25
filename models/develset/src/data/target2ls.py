'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2020-12-27 12:21:13
LastEditTime: 2021-04-08 15:15:42
Contact: cgjhaha@qq.com
Description: get the levelset distance of target image

input:
    target image
output:
    save the levelset function
'''


import os
import sys
sys.path.append('/research/d2/xfyao/guojin/develset_opc/levelset_net')
import time
import torch
import shutil
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from src.models.const import *
import torch.nn.functional as F
from src.utils.levelset_utils import *
import torchvision.transforms as transforms
# from src.utils.unit_tests.test_case1 import *

def makeifnotexist(dir_p):
    dir_p = Path(dir_p)
    if not dir_p.exists():
        dir_p.mkdir()
    return dir_p



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--in_dir', type=str, required=True, help='the input target dir')
parser.add_argument('--out_dir', type=str, required=True, help='the output mask dir')
parser.add_argument('--train_num', type=int, default=4000, help='the total training set number')
parser.add_argument('--val_num', type=int, default=0, help='the total validataion set number')
# parser.add_argument('--val_num', type=int, default=1000, help='the total validataion set number')
cfg = parser.parse_args()
in_dir = Path(cfg.in_dir)
out_dir = makeifnotexist(cfg.out_dir)
train_num = cfg.train_num
val_num = cfg.val_num

# prepare train/val list
all_list = list(sorted(in_dir.glob('*.png')))
assert train_num + val_num <= len(all_list), 'train_num: {} + val_num:{} should <= len(all data) : {}'.format(train_num, val_num, len(all_list))
train_list = all_list[:train_num]
# use all
val_list = all_list[train_num:]

all_list = {
    'train': train_list,
    'val': val_list
}

# prepare dir
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# copy and compute
for mode in ['train', 'val']:
    t_out_dir = out_dir / mode / 'targets'
    t_out_dir.mkdir(parents=True, exist_ok=True)
    ls_dir = out_dir / mode / 'ls_params'
    ls_dir.mkdir(parents=True, exist_ok=True)
    for t in tqdm(all_list[mode]):
        # first copy target
        t_name = t.name
        t_out = t_out_dir / t_name
        if not t_out.is_file():
            shutil.copy(str(t), str(t_out))
        timg = Image.open(str(t)).convert('L')
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        timg = transform(timg).unsqueeze(1)
        p_v,p_h = t2boudry_lines_cpu(timg, device)
        ls = t2lsparams(p_v, p_h, timg, device)
        ls = ls.squeeze(0)
        ls_opt_path = ls_dir / '{}.pt'.format(t.stem)
        torch.save(ls, str(ls_opt_path))