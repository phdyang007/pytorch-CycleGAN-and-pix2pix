import os
from data.base_dataset import BaseDataset, get_params, get_transform, get_resize_transform
from data.image_folder import make_dataset
from PIL import Image
import heapq

    
class LithoAugDataset(BaseDataset):
    """A dataset class for 3 paired image dataset.
    A: design
    B: mask
    C: resist
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B,C} 
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.high_res = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        Image.MAX_IMAGE_PIXELS = 301326592
        # read a image given a random integer index
        high_res_path = self.high_res[index]
        high_res = Image.open(high_res_path).convert('L')
        high_res_transform = get_resize_transform(self.opt, grayscale=True, convert=True, resize=False)
        real_high_res = high_res_transform(high_res)
        mask = real_high_res[:,:,:2048]
        resist = real_high_res[:,:,2048:]
        return {'mask': mask, 'resist': resist, 'image_paths':high_res_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.high_res)

    
class LithoRankDataset(BaseDataset):
    """Only store a list of 2k image paths. Need to downsample for stylegan.
    """
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.rank_size = opt.rank_buffer_size 
        self.buffer = []
        
    def update(self, model, newDataset):
        removed = []
        added = []
        for data in newDataset:
            ious, paths = model.get_iou(data)
            for iou, path in zip(ious, paths):
                if len(self.buffer) < self.rank_size:
                    heapq.heappush(self.buffer, (1-iou, path))
                elif self.buffer[0][0] < 1-ious:
                    heapq.heappop(self.buffer)
                    heapq.heappush(self.buffer, (1-iou, path))
                    
    def __getitem__(self, index):
        Image.MAX_IMAGE_PIXELS = 301326592
        _, high_res_path = self.buffer[index]
        high_res = Image.open(high_res_path).convert('L')
        high_res_transform = get_resize_transform(self.opt, grayscale=True, convert=True, resize=False)
        real_high_res = high_res_transform(high_res)
        mask = real_high_res[:,:,:2048]
        resist = real_high_res[:,:,2048:]
        return {'mask': mask, 'resist': resist, 'image_paths':high_res_path}
    
    def __len__(self):
        return len(self.buffer)