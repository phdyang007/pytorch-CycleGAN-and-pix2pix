import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class LithotwinDataset(BaseDataset):
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
        if opt.lt==True:
            self.dir_AB = os.path.join(opt.dataroot, 'test_lt')  # get the image directory
        else:
            self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        if opt.phase == 'test':
            if opt.update_mask:
                if opt.update_train_round==0:
                    self.dir_AB = os.path.join(opt.dataroot, 'train')
                else:
                    self.dir_AB = os.path.join(opt.dataroot, 'train_next_%g'%(opt.update_train_round))
        if opt.phase == 'train':
            if opt.use_update_mask:
                self.dir_AB = os.path.join(opt.dataroot, 'train_next_%g'%(opt.update_train_round))
        #print(self.dir_AB)
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        #self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        #self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

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
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('L')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 3)
        A = AB.crop((0, 0, w2, h)) #left top right bottom
        B = AB.crop((w2, 0, w2*2, h))
        C = AB.crop((w2*2, 0, w, h))

        # apply the same transform to both A and B
        #transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, grayscale=True, convert=True)
        B_transform = get_transform(self.opt, grayscale=True, convert=True)
        C_transform = get_transform(self.opt, grayscale=True, convert=True)
        A = A_transform(A)
        B = B_transform(B)
        C = B_transform(C)
        #v_epe_points = []
        #h_epe_points = []
        #import torch
        #print(torch.max(A), torch.min(A))
        #quit()
        return {'A': A, 'B': B, 'C': C, 'A_paths': AB_path, 'B_paths': AB_path, 'C_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
