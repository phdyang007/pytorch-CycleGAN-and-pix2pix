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
    
    # initialize "worst" buffer
    totalDataset = create_dataset(opt)
    opt.dataroot = opt.testroot
    batch, opt.batch_size = opt.batch_size, 1 # use 1 for testing
    testDataset = create_dataset(opt)
    opt.batch_size = batch
    
    # initialize DOINN mode
    model = create_model(opt)
    model.setup(opt)
    model.load_networks(opt.load_iter)
    
    # initial testing
    model.eval()
    test = []
    for data in testDataset:
        cur, _ = model.get_iou(data, True)
        test += cur
    model.train()

    with open("./results.txt", 'a') as f:
        f.write("Running new experiments with {}\n".format(opt.augmode))
        f.write("Initial model tested with iou_fg of {:.8f}\n".format(sum(test) / len(test)))
    
    res = []
    for iter in range(opt.aug_iter):
        
        # Generate new image with styleGAN
        print("Obtaining new images.")
        newDir = "./{}_{}_{}/iter_{}".format(opt.augmode, opt.rank_buffer_size, opt.aug_iter, iter)
        opt.dataroot = newDir
        newDataset = create_dataset(opt)
        
        # update buffer dataset
        totalDataset.dataset.add(newDataset.dataset)
    
        # Train DOINN 
        torch.cuda.empty_cache()
        print("Start model retraining on all data {}.".format(len(totalDataset)))
        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
            for i, data in enumerate(totalDataset):
                model.set_input(data)
                model.optimize_parameters()
            model.update_learning_rate()
        model.save_networks("iteration_{}_{}_{}".format(opt.augmode, opt.rank_buffer_size, iter))
        
        # Test DOINN
        test = []
        model.eval()
        for data in testDataset:
            cur, _ = model.get_iou(data, True)
            test += cur
        model.train()
        with open("./results.txt", 'a') as f:
            f.write("Tested with iou_fg of {:.8f}\n".format(sum(test) / len(test)))
            
        # Undo training, reload previous weights
        model.setup(opt)
        model.load_networks(opt.load_iter)

