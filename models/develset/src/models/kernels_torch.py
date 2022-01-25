'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2020-12-07 15:08:44
LastEditTime: 2021-03-05 20:10:46
Contact: cgjhaha@qq.com
Description: kernel objects
'''
# import cupy as np
import torch
import logging

class Kernel():
    def __init__(self):
        #self.cfg = config
        self.optKernels = self.getOptKernels()
        self.comboKernels = self.getComboKernels()

    def getOptKernels(self):
        # 4, 24, 35, 35
        kernel_head = torch.load("./models/develset/kernels/np/optKernel.npy")
        kernel_head = kernel_head[:, :24].cuda()
        # 4, 24
        kernel_scale = torch.load("./models/develset/kernels/np/optKernel_scale.npy")
        kernel_scale = kernel_scale[:, :24]
        a, b = kernel_scale.shape
        kernel_scale = kernel_scale.reshape(a, b, 1, 1).cuda()
        print('kernel used: ', kernel_head.shape)
        print('scale used: ', kernel_scale.shape)
        return {
            'kernel_head': kernel_head,
            'kernel_scale': kernel_scale
        }
    def getComboKernels(self):
        # 4, 24, 35, 35
        kernel_head = torch.load("./models/develset/kernels/np/comboOptKernel.npy")
        nku = 24
        # 4, 1, 35, 35
        kernel_head = kernel_head[:, nku - 1:nku].cuda()
        # 4, 1
        kernel_scale = torch.Tensor([[1],[1], [1], [1]]).cuda()
        return {
            'kernel_head': kernel_head,
            'kernel_scale': kernel_scale
        }