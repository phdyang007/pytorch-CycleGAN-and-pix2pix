'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2021-04-20 12:40:36
LastEditTime: 2021-04-25 15:02:50
Contact: cgjhaha@qq.com
Description: the litho simulation for single image.
'''


import os
import sys
# sys.path.append('/home/guojin/projects/develset_opc/levelset_net')
import torch
import torch.nn
import hydra
import numpy as np
import torchvision
from PIL import Image
from pathlib import Path
from src.models.const import *
from src.models.kernels import Kernel
from src.metrics.metrics import Metrics
import torchvision.transforms as transforms
from src.models.litho_layer import CUDA_LITHO

def lithosim(target, mask):
    kernel = Kernel()
    cl = CUDA_LITHO(kernel)
    metric = []
    nominal_aerial, nominal = cl.simulateImageOpt(mask, LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
    outer_aerial, outer = cl.simulateImageOpt(mask, LITHO_KERNEL_FOCUS, MAX_DOSE)
    inner_aerial, inner = cl.simulateImageOpt(mask, LITHO_KERNEL_DEFOCUS, MIN_DOSE)
    aerials = [nominal_aerial, inner_aerial, outer_aerial]
    m = Metrics(target, aerials)
    l2, pvb = m.get_all()
    print(l2, pvb)
    #nom_save_path = f'./test_nom_pvb_{pvb}_l2_{l2}.png'
    #torchvision.utils.save_image(m.nominal[0], nom_save_path)
    #torchvision.utils.save_image(m.target[0], './target.png')

def litho_trans(img_path):
    im = Image.open(str(img_path)).convert('L')
    #print(max(im))
    trans = transforms.Compose([transforms.ToTensor()])
    im = trans(im)
    im = im.unsqueeze(0)
    #im = torch.nn.ZeroPad2d(24)(im)
    im = im.cuda()
    print(im.shape, torch.max(im))
    return im

#@hydra.main(config_path='config', config_name='lithosim')
#def main(cfg):
#    target_p = Path(cfg.target_path)
#    mask_p = Path(cfg.mask_path)
#    target = litho_trans(target_p)
#    mask = litho_trans(mask_p)
#    lithosim(target, mask, cfg)


#if __name__ == "__main__":
#    main()