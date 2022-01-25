'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2021-02-03 16:24:29
LastEditTime: 2021-04-15 14:57:16
Contact: cgjhaha@qq.com
Description: the model checkpoint call back
'''
import time
import torch
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.callbacks import Callback
from typing import Any, Dict, Optional, Union


class BestScoreCallback(Callback):
    def __init__(self, cfg, **kwargs):
        super(BestScoreCallback, self).__init__(**kwargs)
        self.cfg = cfg
        self.score_dicts = [
            {
                'name': 'ilt',
                'cfg': self.cfg.model.loss_weights.ilt_weight
            },
            {
                'name': 'pvb',
                'cfg': self.cfg.model.loss_weights.pvb_weight
            },
            {
                'name': 'curv',
                'cfg': self.cfg.model.add_curv,
            },
            {
                'name': 'cw',
                'cfg': self.cfg.model.curv_weight
            },
            {
                'name': 'exp',
                'cfg': self.cfg.model.ilt_exp
            }
        ]
    '''
    if teardown, we only save the last epoch
    '''

    def on_train_start(self, trainer, pl_module):
        self.start = time.time()

    def on_train_end(self, trainer, pl_module):
        self.end = time.time()
        self.elapsed = self.end - self.start

        least_L2 = float("inf")
        epoch = -1
        for record in pl_module.metric:
            if record['L2'] < least_L2:
                least_L2 = record['L2']
                epoch = record['epoch']
        line = str(pl_module.metric[epoch])
        basename = pl_module.target_name
        bsf = Path('best_results')
        if not bsf.exists():
            bsf.mkdir()

        save_file = ''
        for i, d in enumerate(self.score_dicts):
            if i == len(self.score_dicts) - 1:
                save_file += '{}_{}.txt'.format(d['name'], d['cfg'])
            else:
                save_file += '{}_{}_'.format(d['name'], d['cfg'])

        save_file = Path('best_results') / Path(save_file)
        with open(save_file, "a+") as file_object:
            file_object.write(basename)
            file_object.write(' ')
            file_object.write(str(pl_module.metric[epoch]['epoch']))
            file_object.write(' ')
            file_object.write(str(pl_module.metric[epoch]['L2']))
            file_object.write(' ')
            file_object.write(str(pl_module.metric[epoch]['PVB']))
            file_object.write(' ')
            file_object.write('{:.2f}'.format(self.elapsed))
            file_object.write('\n')
