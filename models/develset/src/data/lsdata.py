'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2020-12-02 16:40:16
LastEditTime: 2021-04-20 13:05:22
Contact: cgjhaha@qq.com
Description: customed levelset dataloader
'''
import os
import torch
import os.path
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
# from src.utils.debug_tools import debug_tensor

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

CKPT_EXTENSIONS = [
    '.pt'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_ckpt(filename):
    return any(filename.endswith(extension) for extension in CKPT_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]



class LevelsetDataset(Dataset):
    def __init__(self, cfg, target):
        super(LevelsetDataset, self).__init__()
        # self.dataset_path = hparams.dataset_path.rstrip('/')
        # self.dataset_name = os.path.basename(self.dataset_path)
        self.target_path = target
        self.dataset = [self.target_path]
        self.target_name = target.stem
        self.ls_path = Path(cfg.dataset.lsparams_path) / '{}.pt'.format(self.target_name)

    def __getitem__(self, index):
        # target_path = self.dataset[index]
        # ls_path = target_path.parent / '{}.pt'.format(target_name)
        target = Image.open(str(self.target_path)).convert('L')
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        target = transform(target)
        ls = torch.load(str(self.ls_path))
        # print(str(ls_path))
        return {'target': target, 'target_path': str(self.target_path), 'ls': ls}

    def __len__(self):
        return len(self.dataset)
