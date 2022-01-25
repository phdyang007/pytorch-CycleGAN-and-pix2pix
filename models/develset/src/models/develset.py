'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2020-12-25 17:35:25
LastEditTime: 2021-05-03 18:20:22
Contact: cgjhaha@qq.com
Description: Deep level-set methods models
'''


import os
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pytorch_lightning as pl
from src.models.const import *
from src.utils.img_utils import get_tb_images, get_pvb_aerial
from src.metrics.metrics import Metrics
from src.utils.levelset_utils import gradient_geo, curvature_term, truncated_ls, ls2npy

class Develset(pl.LightningModule):
    def __init__(self, cfg, t_name):
        super(Develset, self).__init__()
        self.cfg = cfg
        self.target_name = t_name
        # self.automatic_optimization = False
        if self.cfg.model.torch_fft:
            from src.models.litho_layer_torch import CUDA_LITHO,  LEVELSET_LAYER
            from src.models.kernels_torch import Kernel
        else:
            from src.models.litho_layer import CUDA_LITHO, ILT_LAYER, LEVELSET_LAYER
            from src.models.kernels import Kernel
        self.LEVELSET_LAYER = LEVELSET_LAYER
        self.kernel = Kernel(cfg.kernel)
        self.cuda_litho = CUDA_LITHO(self.kernel)
        self.mask = nn.Parameter(torch.zeros(MASK_SHAPE))
        self.ls_params = nn.Parameter(torch.zeros(MASK_SHAPE))
        # self.ls_params = torch.zeros(MASK_SHAPE, requires_grad=False).cuda()
        self.metric = []

    def forward(self):
        self.__updateMask()
        levelset_loss, aerials, wafers = self.LEVELSET_LAYER.apply(self.mask, self.target, self.cuda_litho, self.cfg)
        return levelset_loss, aerials, wafers

    def __initLSParams(self, ls):
        self.ls_params.data.add_(ls)


    def __updateMask(self):
        self.mask.data[torch.where(self.ls_params <= 0)] = 1
        self.mask.data[torch.where(self.ls_params > 0)] = 0

    def __gradient_geo(self):
        '''
        input: (b,c,x,y) leveset tensor
        '''
        levelset = self.ls_params
        ls_geo = gradient_geo(levelset)
        return ls_geo

    def __geo_curv_term(self):
        levelset = self.ls_params
        curv = curvature_term(levelset)
        return curv

    def training_step(self, batch, batch_idx):
        if self.current_epoch == 0:
            target = batch['target']
            target_path = batch['target_path']
            # ls shape [1, 1, 2048, 2048]
            ls = batch['ls']
            self.target = target
            self.__initLSParams(ls)
            # self.__updateMask()
        # levelset_loss, aerials, wafers = self.forward(self.mask, self.target)
        levelset_loss, aerials, wafers = self.forward()
        nominal_aerial, inner_aerial, outer_aerial = aerials[0], aerials[1], aerials[2]
        nominal, inner, outer = wafers[0], wafers[1], wafers[2]
        self.nominal = nominal

        # we should use the aerial to calculate the epe, and maybe th same.
        save_loss = self.cfg.view.save_loss
        if self.cfg.logger.log_metrics:
            m = Metrics(self.target, aerials)
            l2, pvb = m.get_all()
            self.log('L2', l2, on_epoch=True, prog_bar=True, logger=save_loss)
            self.log('PVB', pvb, on_epoch=True, prog_bar=True, logger=save_loss)
            m_log = {
                'epoch': self.current_epoch,
                'L2': float(l2),
                'PVB': float(pvb)
            }
            self.metric.append(m_log)

        self.log('train_loss', levelset_loss, on_epoch=True, prog_bar=True, logger=save_loss)
        self.trainer.train_loop.running_loss.append(levelset_loss)

        opter = self.optimizers()
        self.manual_backward(levelset_loss, opter)

        gradient_geo = self.__gradient_geo()
        if self.mask.grad is not None:
            """
            save gradient and visualize gradient 
            """
            # np.save('/home/hongduo/school/test_' + str(self.current_epoch) + '.npy', (self.mask.grad * gradient_geo).detach().cpu().numpy())

            time_step = 2.5/torch.max(self.mask.grad)
            if self.cfg.model.add_curv:
                curv_term = self.__geo_curv_term()
                curv_weight = self.cfg.model.curv_weight
                self.ls_params.data.add_(time_step * (self.mask.grad * gradient_geo + curv_weight * curv_term))
            else:
                self.ls_params.data.add_(time_step * self.mask.grad * gradient_geo)
            # truncated levelset
            self.ls_params.data[torch.where(self.ls_params.data > UP_TRUNCATED_D)] = UP_TRUNCATED_D
            self.ls_params.data[torch.where(self.ls_params.data < DOWN_TRUNCATED_D)] = DOWN_TRUNCATED_D

        # visualization:
        # this saved the sigmod, we should save the aerial.
        '''
        TODO:
        1. should see the score in cu-ilt, how to calculate.
        2. should stop based on gradient.
        '''

        if self.cfg.view.save_img:
            if self.current_epoch >=  self.trainer.max_epochs - self.cfg.view.save_img_last_epochs:
                pvb_aerial = get_pvb_aerial(inner_aerial, outer_aerial)
                tb_imgs = get_tb_images(self.target, self.mask, nominal_aerial, pvb_aerial, inner_aerial, outer_aerial)
                self.logger.experiment.add_images(
                    'images in {} epoch'.format(self.current_epoch), tb_imgs
                )

    def test_step(self, batch, batch_idx):
        best_model = torch.load(self.best_model_path)['state_dict']

        mask = best_model['mask'][0]
        mask_ls_params = best_model['ls_params'][0]
        mask_ls_params = truncated_ls(mask_ls_params)
        mask_ls_npy = ls2npy(mask_ls_params)

        target = batch['target'][0]
        target_ls_params = batch['ls'][0]
        target_ls_params = truncated_ls(target_ls_params)
        target_ls_npy = ls2npy(target_ls_params)

        target_path = batch['target_path'][0]
        target_name = Path(target_path).stem

        ckpt_path = Path(self.cfg.model.ckpt.save_path)

        target_folder = ckpt_path / "target"
        target_folder.mkdir(exist_ok=True)
        mask_folder = ckpt_path / "mask"
        mask_folder.mkdir(exist_ok=True)

        target_ls_folder = ckpt_path / "target_ls"
        target_ls_folder.mkdir(exist_ok=True)
        mask_ls_folder = ckpt_path / "mask_ls"
        mask_ls_folder.mkdir(exist_ok=True)

        target_save_path =  target_folder / "{}.png".format(target_name)
        mask_save_path = mask_folder / "{}.png".format(target_name)
        target_ls_save_path = target_ls_folder / "{}.npy".format(target_name)
        mask_ls_save_path = mask_ls_folder / "{}.npy".format(target_name)
        torchvision.utils.save_image(target, target_save_path)
        torchvision.utils.save_image(mask, mask_save_path)
        np.save(str(target_ls_save_path), target_ls_npy)
        np.save(str(mask_ls_save_path), mask_ls_npy)

        # if self.cfg.test.save_params:
        #     target_ls_p_folder = ckpt_path / "target_ls_p"
        #     target_ls_p_folder.mkdir(exist_ok=True)
        #     mask_ls_p_folder = ckpt_path / "mask_ls_p"
        #     mask_ls_p_folder.mkdir(exist_ok=True)
        #     target_ls_p_save_path = target_ls_p_folder / "{}.pt".format(target_name)
        #     mask_ls_p_save_path = mask_ls_p_folder / "{}.pt".format(target_name)
        #     torch.save(target_ls_params, target_ls_p_save_path)
        #     torch.save(mask_ls_params, mask_ls_p_save_path)


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


