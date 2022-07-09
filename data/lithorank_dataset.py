import os
from data.base_dataset import BaseDataset, get_params, get_transform, get_resize_transform
from data.image_folder import make_dataset
from PIL import Image
import heapq

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
                elif self.buffer[0][0] < 1-iou:
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
