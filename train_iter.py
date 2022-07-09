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
    opt.dataset_mode = 'lithorank'
    bufferDataset = create_dataset(opt)
    opt.dataset_mode = dataset_mode
    
    # initialize DOINN mode
    model = create_model(opt)
    model.setup(opt)
    
    # initialize styleGAN model
    # TODO initialize args
    opt.model = 'stylegan'
    stylegan = create_model(opt)
    stylegan.setup(opt)
    stylegan.init(args.G_kwargs, args.D_kwargs)
    
    for i in range(opt.aug_iter):
        
        # Generate new image with styleGAN
        newDir = "{}_{}".format(opt.outdirPrefix, i)
        mkdir(newDir)
        stylegan.generate_random(newDir, model)
        opt.dataroot = newDir
        newDataset = create_dataset(opt)
        
        # update buffer dataset
        bufferDataset.dataset.update(model, newDataset)
        totalDataset.dataset.add(newDataset.dataset)
        
        # Train styleGAN
        args.training_set_kwargs.files = bufferDataset.dataset.buffer
        stylegan.finetune(args)

    print(len(totalDataset))
    # Train DOINN 
    for epoch in range(opt.n_epochs):
        for i, data in enumerate(totalDataset):
            model.set_input(data)
            model.optimize_parameters()
    model.save_networks("iteration_{}".format(iteration))