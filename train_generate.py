"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import torch
from options.aug_options import AugOptions
from data import create_dataset
from models import create_model
from util.util import mkdir, get_args_from_opt

if __name__ == '__main__':
    opt = AugOptions().parse()   # get training options
    dataset_mode = opt.dataset_mode
    args = get_args_from_opt(opt)
    
    # initialize DOINN mode
    model = create_model(opt)
    model.setup(opt)
    model.load_networks(opt.load_iter)
    
    # initialize styleGAN model
    # TODO initialize args
    opt.model = 'stylegan'
    stylegan = create_model(opt)
    stylegan.setup(opt)
    stylegan.init(args.G_kwargs, args.D_kwargs)

    with open("./results.txt", 'a') as f:
        f.write("Running new experiments with {}\n".format(opt.augmode))
    
    res = []
    for iter in range(opt.aug_iter):
        
        # Generate new image with styleGAN
        print("Generating new images.")
        newDir = "./{}_{}_{}/iter_{}".format(opt.augmode, opt.rank_buffer_size, opt.aug_iter, iter)
        mkdir(newDir)
        cur = stylegan.generate_data(newDir, model, opt.augmode)
        res += cur
        with open("./results.txt", 'a') as f:
            f.write("Generated data of size {:02d} with iou_fg of {:.8f}\n".format(len(res),  sum(res)/len(res)))
        opt.dataroot = newDir
        newDataset = create_dataset(opt)
        