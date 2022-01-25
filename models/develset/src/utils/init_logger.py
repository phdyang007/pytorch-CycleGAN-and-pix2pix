'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2021-01-25 10:38:40
LastEditTime: 2021-03-10 14:16:07
Contact: cgjhaha@qq.com
Description: init logger
'''
from pathlib import Path
import shutil
from pytorch_lightning.loggers import TensorBoardLogger

def init_logger(target_logger_path, cfg):
    logger_path = target_logger_path
    print('logger_path: ', logger_path)
    if cfg.logger.refresh:
        if logger_path.exists():
            shutil.rmtree(str(logger_path))
    tb_logger = TensorBoardLogger(str(logger_path))
    return tb_logger
