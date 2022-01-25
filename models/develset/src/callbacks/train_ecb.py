'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2021-01-25 10:35:26
LastEditTime: 2021-03-05 19:54:51
Contact: cgjhaha@qq.com
Description: the train early callback
'''
# import torch
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import EarlyStopping
# from src.utils.img_utils import get_tb_images

# class TrainEarlyStopping(EarlyStopping):
#     def __init__(self, cfg, **kwargs):
#         super(TrainEarlyStopping, self).__init__(**kwargs)
#         self.cfg = cfg

#     # def on_validation_end(self, trainer, pl_module):
#     #     # override this to disable early stopping at the end of val loop
#     #     pass

#     # def on_train_end(self, trainer, pl_module):
#     #     # instead, do it at the end of training loop
#     #     self._run_early_stopping_check(trainer, pl_module)

#     '''
#     if teardown, we only save the last epoch
#     '''
#     def teardown(self, trainer, pl_module, stage):
#         cur_epoch = pl_module.current_epoch
#         if self.cfg.view.save_img:
#             if cur_epoch <= (self.cfg.trainer.max_epochs - self.cfg.view.save_img_last_epochs):
#                 print('early tear down in {} epoch, save images.'.format(cur_epoch))
#                 tb_imgs = get_tb_images(pl_module.target, pl_module.mask, pl_module.nominal)
#                 trainer.logger.experiment.add_images(
#                     'images in {} epoch'.format(cur_epoch), tb_imgs
#                 )
