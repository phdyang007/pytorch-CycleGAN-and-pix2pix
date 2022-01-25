'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2020-11-28 16:15:47
LastEditTime: 2021-05-06 20:47:10
Contact: cgjhaha@qq.com
Description: train deep levelset
'''

import os
import time
import hydra
import shutil
import torch
import logging
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.models.develset import Develset
from src.data.lsdata import LevelsetDataset
from src.callbacks.ckpt_cb import ModelCKPT
from src.utils.init_logger import init_logger
from pytorch_lightning.callbacks import EarlyStopping
from src.callbacks.score_cb import BestScoreCallback
from pytorch_lightning.loggers import TensorBoardLogger
from src.utils.split_data import find_trained_list
torch.cuda.set_device(4)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

pl_logger = logging.getLogger("lightning")
pl_logger.propagate = False
pl_logger.setLevel(logging.ERROR)

# @hydra.main(config_path='config', config_name='lhd')
# @hydra.main(config_path='config', config_name='unet_iccad13')
@hydra.main(config_path='config', config_name='iccad13')
def main(cfg):
    targets_path = Path(cfg.dataset.targets_path)
    logger_path = cfg.logger.save_path

    trained_list = find_trained_list(Path(cfg.model.ckpt.save_path)/'mask')


    for target in sorted(targets_path.glob('*.png')):
        target_name = target.stem
        if target_name in trained_list:
            print('{} is trained, pass'.format(target_name))
            continue
        target_logger_path = Path(logger_path) / target_name
        target_logger_path = Path.cwd() / target_logger_path
        tb_logger = init_logger(target_logger_path, cfg)
        levelset_dataset = LevelsetDataset(cfg, target)
        train_dataloader = DataLoader(
            dataset=levelset_dataset,
            batch_size=cfg.dataset.batch_size
        )
        develset_model = Develset(cfg, target_name)
        early_stop_callback = EarlyStopping(
            monitor='train_loss',
            patience=1,
            mode='min'
        )
        ckpt_callback = ModelCKPT(
            cfg = cfg,
            monitor='train_loss',
            # verbose = True,
            dirpath = Path.cwd() / cfg.model.ckpt.save_path / 'ckpt',
            filename = target_name +"-{epoch}-{L2:.2f}"
        )
        best_score_callback = BestScoreCallback(
            cfg = cfg
        )
        trainer = pl.Trainer(
            logger = tb_logger,
            automatic_optimization = False,
            callbacks = [
                early_stop_callback,
                # best_score_callback,
                # ckpt_callback
            ],
            **cfg.trainer
        )
        start = time.time()
        trainer.fit(develset_model, train_dataloader)
        end = time.time()
        elapsed = end - start
        print("{} used {} seconds \n\n".format(target_name, elapsed))
        # trainer.test(test_dataloaders=train_dataloader)
        trainer.logger.close()
        # break

if __name__ == "__main__":
    main()



