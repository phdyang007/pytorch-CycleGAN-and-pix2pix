'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2021-01-25 14:38:20
LastEditTime: 2021-04-20 13:38:36
Contact: cgjhaha@qq.com
Description: the testing metric
'''
import torch
from src.utils.img_utils import intensity_thres, print_thres

'''
TODO:
1. save the metrics images.
2. save a table for all the images.
'''
class Metrics:
    def __init__(self, target, aerials):
    # def __init__(self, target, nominal, inner, outer):
        nominal_aerial, inner_aerial, outer_aerial = aerials[0], aerials[1], aerials[2]
        self.target = target
        # self.nominal = nominal_aerial
        # self.inner = inner_aerial
        # self.outer = outer_aerial
        # self.target = print_thres(target)
        # self.nominal = print_thres(nominal_aerial)
        # self.inner = print_thres(inner_aerial)
        # self.outer = print_thres(outer_aerial)

# before
        # self.nominal = nominal_aerial
        self.nominal = intensity_thres(nominal_aerial)
        self.inner = intensity_thres(inner_aerial)
        self.outer = intensity_thres(outer_aerial)

        self.nominal = print_thres(self.nominal)
        self.inner = print_thres(self.inner)
        self.outer = print_thres(self.outer)

    def get_L2(self):
        return torch.sum(torch.abs(self.target - self.nominal))
        # return torch.sum((self.target - self.nominal).pow(2))

    def get_PVB(self):
        # this may be error
        pvb = torch.zeros(self.nominal.shape).cuda()
        pvb[torch.where(self.outer == 1)] = 1
        pvb[torch.where(self.inner == 1)] = 0
        return torch.sum(pvb)

    # ToDo:
    # def getShot(self):

    # ToDo:
    # def getEPE(self):

    def get_all(self):
        l2 = self.get_L2()
        pvb = self.get_PVB()
        return l2, pvb


# epe