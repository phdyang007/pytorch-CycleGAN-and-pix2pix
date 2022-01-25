'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2021-02-03 16:24:29
LastEditTime: 2021-02-03 21:24:16
Contact: cgjhaha@qq.com
Description: the model checkpoint call back
'''
import torch
import pytorch_lightning as pl
from typing import Any, Dict, Optional, Union

class ModelCKPT(pl.callbacks.ModelCheckpoint):
    def __init__(self, cfg, **kwargs):
        super(ModelCKPT, self).__init__(**kwargs)
        self.cfg = cfg

    def on_save_checkpoint(self, trainer, pl_module) -> Dict[str, Any]:
        ret_dict = {
            "monitor": self.monitor,
            "best_model_score": self.best_model_score,
            "best_model_path": self.best_model_path,
            "current_score": self.current_score,
            "dirpath": self.dirpath
        }
        pl_module.best_model_path = self.best_model_path
        pl_module.best_model_score = self.best_model_score
        return ret_dict

