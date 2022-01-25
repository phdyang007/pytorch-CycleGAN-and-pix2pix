'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2021-04-25 10:49:08
LastEditTime: 2021-04-25 10:54:56
Contact: cgjhaha@qq.com
Description: read metrics.
'''

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

tp = '/home/guojin/projects/cu-ilt-litho/build/targetImg.png'
target = Image.open(tp).convert('L')
# T1 = transform(T1).unsqueeze(1)      #(1,2048,2048) to (1,1,2048,2048)
np = '/home/guojin/projects/cu-ilt-litho/build/img_nom_e0_pvb_67161_epe_84_l2_120959.png'
nominal = Image.open(np).convert('L')

target = transform(target)
nominal = transform(nominal)


l2 = torch.sum(torch.abs(target - nominal))
print(l2)
# pvb = torch.zeros_like(nominal)
# pvb[torch.where()]