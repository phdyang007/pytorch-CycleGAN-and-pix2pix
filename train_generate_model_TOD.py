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
from util.util import mkdir, get_args_from_opt, set_seed

if __name__ == '__main__':
    opt = AugOptions().parse()   # get training options
    set_seed(opt.random_seed)
    dataset_mode = opt.dataset_mode
    args = get_args_from_opt(opt)

    assert opt.adv_loss_type == 'TOD'
    
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
    model_past = create_model(opt)
    model_past.setup(opt)
    model_past.load_networks(opt.load_iter)

    # initialize styleGAN model
    opt.model = 'stylegan'
    stylegan = create_model(opt)
    stylegan.setup(opt)
    stylegan.init(args.G_kwargs, args.D_kwargs)
    
    # initial testing
    model.eval()
    test = []
    for data in testDataset:
        cur, _ = model.get_iou(data, True)
        test += cur
    model.train()

    with open("./results.txt", 'a') as f:
        f.write("Running new experiments with {} iterative active learning TOD\n".format(opt.augmode))
        f.write("Initial model tested with iou_fg of {:.8f}\n".format(sum(test) / len(test)))
    
    all_train_iterations = [0,1,2,3,7,11,15]
    for iter in range(opt.aug_iter):
        # Generate new image with styleGAN
        print("Obtaining new images on iterations {:02d}".format(iter))
        newDir = "./{}_{}_{}_{}/iter_{}".format(opt.augmode, opt.adv_loss_type, opt.rank_buffer_size, opt.aug_iter, iter)
        # skip generation for 0 iter since it should already exist from previous
        mkdir(newDir)
        if iter == 0:
            res = stylegan.generate_data(newDir, model, 'random')
        else:
            res = stylegan.generate_data(newDir, model, opt.augmode, model_past)
        with open("./results.txt", 'a') as f:
            f.write("Generated data of size {:02d} with iou_fg of {:.8f}\n".format(len(res),  sum(res)/len(res)))
        opt.dataroot = newDir
        newDataset = create_dataset(opt)
        
        # add to total dataset
        totalDataset.dataset.add(newDataset.dataset)
        if iter not in all_train_iterations:
            continue
        # Train DOINN 
        if iter != 0:
            idx = all_train_iterations.index(iter)
            prev_iter = all_train_iterations[idx-1]
            model_past.load_networks("iteration_{}_{}_{}_{}".format(opt.augmode, opt.adv_loss_type, opt.rank_buffer_size, prev_iter))
        print("Start model retraining on all data {}.".format(len(totalDataset)))
        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
            for i, data in enumerate(totalDataset):
                model.set_input(data)
                model.optimize_parameters(detach=True)
            model.update_learning_rate()
        model.save_networks("iteration_{}_{}_{}_{}".format(opt.augmode, opt.adv_loss_type, opt.rank_buffer_size, iter))
        
        # Test DOINN
        test = []
        model.eval()
        for data in testDataset:
            cur, _ = model.get_iou(data, True)
            test += cur
        model.train()
        with open("./results.txt", 'a') as f:
            f.write("Tested with iou_fg of {:.8f} on iteration {:02d}\n".format(sum(test) / len(test), iter))
            
        # Undo training, reload previous weights
        model.setup(opt, load=False)
        # no need to reload previous weights.
        #model.load_networks(opt.load_iter) 

