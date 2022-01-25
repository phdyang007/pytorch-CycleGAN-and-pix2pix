'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2020-12-16 09:56:28
LastEditTime: 2021-04-14 20:34:40
Contact: cgjhaha@qq.com
Description: functions for image process
'''
import torch
import numpy as np
from src.models.const import *

def get_pvb_aerial(inner_aerial, outer_aerial):
    inner = intensity_thres(inner_aerial)
    outer = intensity_thres(outer_aerial)
    pvb = torch.zeros_like(inner).cuda()
    pvb[torch.where(outer == 1)] = 1
    pvb[torch.where(inner == 1)] = 0
    return pvb

def get_tb_images(*imgs):
    tb_imgs = []
    for img in imgs:
        img = intensity_thres(img)
        img = print_thres(img)
        img = img.detach().cpu().numpy()[0][0]
        img = img.reshape(1, img.shape[0], img.shape[1])
        tb_imgs.append(img)
    return np.array(tb_imgs)


def intensity_thres(img):
    out_img = torch.zeros_like(img).cuda()
    out_img[torch.where(img >= TARGET_INTENSITY)] = 1
    return out_img


def print_thres(img):
    out_img = torch.zeros_like(img).cuda()
    out_img[torch.where(img >= MASK_PRINTABLE_THRESHOLD)] = 1
    return out_img