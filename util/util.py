"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import dnnlib

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def tensor2im_layout(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()[0]  # convert it into a numpy array
        #if image_numpy.shape[0] == 1:  # grayscale to RGB
        #    image_numpy = np.tile(image_numpy, (3, 1, 1))
        #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

def save_image_layout(image_numpy, image_path):

    plt.imsave(image_path, image_numpy)

def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def get_args_from_opt(opt):
    args = dnnlib.EasyDict()
    args.num_gpus = 1 # default needed.
    args.random_seed = 0
    args.training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderResizeDataset', files=[], use_labels=False, max_size=None, xflip=True)
    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=8, prefetch_factor=2)
    
    spec = dnnlib.EasyDict(dict(ref_gpus=args.num_gpus, kimg=1, mb=opt.gan_batch_size, mbstd=opt.gan_batch_size*args.num_gpus, fmaps=-1, lrate=-1, gamma=-1, ema=-1, ramp=0.05, map=2))
    res = 256
    spec.fmaps = 1 if res >= 512 else 0.5
    spec.lrate = 0.0002 if res >= 1024 else 0.00025
    spec.gamma = 0.0002 * (res ** 2) / spec.mb * 50 # heuristic formula
    spec.ema = spec.mb * 10 / 32
    # including modified arguments
    args.G_reg_interval = None
    args.D_reg_interval = None
    args.G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=128, w_dim=128, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    args.D_kwargs = dnnlib.EasyDict(class_name='training.networks.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(spec.fmaps * 32768)
    args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 128
    args.G_kwargs.mapping_kwargs.num_layers = spec.map
    args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4 # enable mixed-precision training
    args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow
    args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd

    args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss', r1_gamma=spec.gamma)

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    
    return args