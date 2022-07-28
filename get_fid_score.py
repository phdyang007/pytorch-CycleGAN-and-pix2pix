import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import torch
import numpy as np
from scipy.linalg import sqrtm



def calculate_fid_score(model, dataset1, dataset2, size=1000):
    act1, act2 = None, None
    
    for data1, data2 in zip(dataset1, dataset2):
        with torch.no_grad():
            model.set_input(data1)
            model.forward()
            embedding1 = model.embedding.detach().cpu().numpy()
            model.set_input(data2)
            model.forward()
            embedding2 = model.embedding.detach().cpu().numpy()
        if act1 is None:
            act1, act2 = embedding1, embedding2
        else:
            act1 = np.concatenate((act1, embedding1), axis=0)
            act2 = np.concatenate((act2, embedding2), axis=0)
            if act1.shape[0] >= size:
                break
        
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return fid

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 4    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.eval = True
    
    # get dataset
    dataset2 = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    opt.dataroot= './../mask_all'
    dataset1 = create_dataset(opt)
    
    
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()
    
    fid_score = calculate_fid_score(model, dataset1, dataset2, 100)
    
    print(fid_score)
    
