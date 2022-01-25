'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2021-04-07 18:59:11
LastEditTime: 2021-04-09 19:01:13
Contact: cgjhaha@qq.com
Description:
1. transfer the levelset \phi to grayscale image
2. replace the nan to the original \phi.
'''
import os
import sys
sys.path.append('/home/guojin/projects/develset_opc/levelset_net')
import torch
import torchvision
import numpy as np
from PIL import Image
from pathlib import Path
from src.models.const import *
from src.utils.levelset_utils import gray2ls
import torchvision.transforms as transforms

# ls_path = '/home/guojin/projects/develset_opc/levelset_net/ganopc_outputs/ckpts_2021-03-23_18-47-11/mask_ls'
# ls_im = '/home/guojin/projects/develset_opc/levelset_net/ls_images'
# ls_im = Path(ls_im)
# ls_path = Path(ls_path)
# ckpts = 'ckpts_2021-04-09_18-50-28'
ckpts = 'ckpts_2021-04-09_19-00-15'

ls_path = '/home/guojin/projects/develset_opc/levelset_net/iccad13_outputs/{}/mask_ls_p/'.format(ckpts)
ls_im = '/home/guojin/projects/develset_opc/levelset_net/iccad13_outputs/{}/mask_ls'.format(ckpts)
ls_im = Path(ls_im)
ls_path = Path(ls_path)


name = 'M1_test0'
ls_path = ls_path / '{}.pt'.format(name)
ls_im = ls_im / '{}.npz'.format(name)

ls = torch.load(str(ls_path))
# print(torch.where(torch.isnan(ls) == True))
# print(torch.isnan(ls).any())
# ls = torch.nan_to_num(ls, nan=-100)
# ls = ls[:,LITHOSIM_OFFSET:MASK_TILE_END_Y,LITHOSIM_OFFSET:MASK_TILE_END_X]



print(torch.max(ls))
print(torch.min(ls))
print(torch.sum(ls))



# gray_im = Image.open(str(ls_im)).convert('L')
im_npz = np.load(str(ls_im))['mask_ls']
print(im_npz.shape)
# trans = transforms.Compose([
#     transforms.ToTensor()
# ])
# im_npz = trans(im_npz).cuda()
im_npz = torch.from_numpy(im_npz).cuda()
print(im_npz.shape)
# ls_gray = gray2ls(gray_im)

print(torch.max(im_npz))
print(torch.min(im_npz))
print(torch.sum(im_npz))
print(im_npz.shape)


X = 800
Y = 1000

for i in range(0, 100, 10):
    x = X + i
    y = Y + i
    print(ls[0, x+LITHOSIM_OFFSET, y+LITHOSIM_OFFSET])
    print(im_npz[0, x, y])
    print("================")


# print(ls)
# ls[torch.where(ls > 0)] = -1
# ls[torch.where(ls <= 0)] = 255
# ls = ((-ls / 1280) + 1)/2
# # ls = -1*ls
# torchvision.utils.save_image(ls, str(ls_im))

# trans = transforms.Compose([
#     transforms.ToTensor()
# ])
# n_im = Image.open(str(ls_im)).convert('L')
# n_t = trans(n_im).cuda()
# # print(torch.max(n_t))
# # print(torch.min(n_t))

# # n_t = -1 * n_t / 255 * 1280
# n_t = 1280 * (1 - 2*n_t)
# print(torch.max(n_t))
# print(torch.min(n_t))
# print(torch.sum(n_t))

# print(ls - n_t)

