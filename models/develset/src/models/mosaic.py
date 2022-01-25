'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2020-11-30 13:24:10
LastEditTime: 2021-01-03 21:19:02
Contact: cgjhaha@qq.com
Description: Deep level-set methods models
'''


import os
import torch
import torch.nn as nn
from models.const import *
import pytorch_lightning as pl
import torch.nn.functional as F
from models.kernels import Kernel
from utils.img_utils import get_tb_images
# from pl_bolts.models.vision import UNet
from models.litho_layer import CUDA_LITHO, ILT_LAYER, PVB_LAYER

class Develset(pl.LightningModule):
    def __init__(self, cfg):
        super(Develset, self).__init__()
        self.cfg = cfg
        self.kernel = Kernel(cfg.kernel)
        self.cuda_litho = CUDA_LITHO(self.kernel)
        self.mask_params = nn.Parameter(torch.zeros(MASK_SHAPE))

    def forward(self, mask, target):
        self.__updateMask()
        ilt_loss, nominal = ILT_LAYER.apply(mask, target, self.cuda_litho)
        pvb_loss, inner, outer = PVB_LAYER.apply(mask, target, self.cuda_litho)
        return ilt_loss, pvb_loss, nominal, inner, outer
        # return pvb_loss, inner, outer

    def __initParams(self, mask):
        self.mask_params.data[torch.where(mask > 0.5)] = 2
        self.mask_params.data.sub_(1)


    def __updateMask(self):
        self.mask = 1/(1+torch.exp(-MASKRELAX_SIGMOID_STEEPNESS*self.mask_params))
    '''
    may be many bugs, the mask sometimes is the binary
    sometimes is the sigmod mask
    should use different array to differentiate
    '''

    def training_step(self, batch, batch_idx):
        if self.current_epoch == 0:
            target = batch['target']
            target_path = batch['target_path']
            self.target = target
            self.__initParams(self.target)
            self.__updateMask()
        ilt_loss, pvb_loss, nominal, inner, outer = self.forward(self.mask, self.target)
        # pvb_loss, inner, outer = self.forward(self.mask, self.target)
        ilt_weight = self.cfg.model.loss.ilt_weight
        pvb_weight = self.cfg.model.loss.pvb_weight
        loss = ilt_weight * ilt_loss + pvb_weight * pvb_loss
        self.log('train_loss', loss)
        # this part should be some hooks part.
        # visualization:
        if self.cfg.view.save_img:
            if self.current_epoch >= self.trainer.max_epochs - self.cfg.view.save_img_last_epochs:
                # tb_imgs = get_tb_images(self.target, self.mask, nominal, inner, outer)
                tb_imgs = get_tb_images(self.target, self.mask, nominal, inner, outer)
                self.logger.experiment.add_images(
                    'images in {} epoch'.format(self.current_epoch), tb_imgs
                )
        if self.cfg.view.save_loss:
            self.logger.experiment.add_scalar('Loss/train', loss, self.current_epoch)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        return optimizer


