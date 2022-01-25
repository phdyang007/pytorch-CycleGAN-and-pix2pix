'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2020-12-07 15:08:44
LastEditTime: 2021-04-14 21:55:42
Contact: cgjhaha@qq.com
Description: kernel objects
'''
import logging
import cupy as np

class Kernel():
    def __init__(self):
        #self.cfg = config
        self.optKernels = self.getOptKernels()
        self.comboKernels = self.getComboKernels()

    def getOptKernels(self):
        # 4, 24, 35, 35
        kernel_head = np.load("./models/develset/kernels/np/optKernel.npy")
        # kernel_head = kernel_head[:, :self.cfg.num_kernel_used]
        nku = 24
        kernel_head = kernel_head[:, :nku]
        # 4, 24
        kernel_scale = np.load("./models/develset/kernels/np/optKernel_scale.npy")
        kernel_scale = kernel_scale[:, :nku]
        a, b = kernel_scale.shape
        kernel_scale = kernel_scale.reshape(a, b, 1, 1)
        pl_logger = logging.getLogger("lightning")
        pl_logger.info('opt kernel used: {} and scale used {}'.format(kernel_head.shape, kernel_scale.shape))
        return {
            'kernel_head': kernel_head,
            'kernel_scale': kernel_scale
        }
    def getComboKernels(self):
        # 4, 24, 35, 35
        kernel_head = np.load("./models/develset/kernels/np/comboOptKernel.npy")
        nku = 9
        # 4, 1, 35, 35
        kernel_head = kernel_head[:, nku - 1:nku]
        # 4, 1
        kernel_scale = np.array([[1],[1], [1], [1]])
        return {
            'kernel_head': kernel_head,
            'kernel_scale': kernel_scale
        }