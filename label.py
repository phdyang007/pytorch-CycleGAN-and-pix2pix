import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from tqdm import tqdm
import PIL
import torch

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    for data in tqdm(dataset):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        res = model.simulate(model.real_high_res)
        
        for i in range(res.shape[0]):
            paths = data['image_paths'][i].split('/')
            paths[2] = 'mask_2048_label'
            paths = '/'.join(paths)
            mask_golden = (res[i,0,:,:] * 255).to(torch.uint8)
            input_img = (model.real_high_res[i,0,:,:] * 255).to(torch.uint8)
            tot_image = torch.cat((input_img,mask_golden), 1)
    
            PIL.Image.fromarray(tot_image.detach().cpu().numpy(), 'L').save(paths)
            
        
