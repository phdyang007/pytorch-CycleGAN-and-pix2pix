'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2021-03-05 19:52:15
LastEditTime: 2021-03-05 20:02:35
Contact: cgjhaha@qq.com
Description: The teardown call back
'''

import torch
import pytorch_lightning as pl
# from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import Callback
from src.utils.img_utils import get_tb_images

class TearDownSaveImgCallback(Callback):
    def __init__(self, cfg, **kwargs):
        super(TearDownSaveImgCallback, self).__init__(**kwargs)
        self.cfg = cfg
    '''
    if teardown, we only save the last epoch
    '''
    def teardown(self, trainer, pl_module, stage):
        cur_epoch = pl_module.current_epoch
        if self.cfg.view.save_img:
            if cur_epoch <= (self.cfg.trainer.max_epochs - self.cfg.view.save_img_last_epochs):
                print('early tear down in {} epoch, save images.'.format(cur_epoch))
                tb_imgs = get_tb_images(pl_module.target, pl_module.mask, pl_module.nominal)
                trainer.logger.experiment.add_images(
                    'images in {} epoch'.format(cur_epoch), tb_imgs
                )
