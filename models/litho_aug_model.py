import torch
from .base_model import BaseModel
from . import networks

import sys 
sys.path.append('./models/develset')
from models.develset.src.models.const import *
from models.develset.src.models.kernels import Kernel
from models.develset.src.metrics.metrics import Metrics
from models.develset.src.models.litho_layer import CUDA_LITHO
from models.develset.lithosim import *

class LithoAugModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.add_argument('--netF', type=str, default='oinnopc_seg', help='specify litho architecture [oinnopc]')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['F']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_resist', 'real_mask_img']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['F']
        # define networks (both generator and discriminator)
        self.netF = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netF, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        # define loss functions
        self.criterionLitho = torch.nn.CrossEntropyLoss()
        if self.isTrain:
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_F)
        else:
            self.kernel = Kernel()
            self.cl = CUDA_LITHO(self.kernel)
            self.MSELoss = torch.nn.MSELoss()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.mask = input['mask'].to(self.device)
        self.real_resist = input['resist'].to(self.device)
        self.image_paths = input['image_paths']
        
    def legalize_mask(self, mask, threshold=0.5):
        """Legalize the mask generated from GAN."""
        mask[mask >= threshold] = 1
        mask[mask < threshold] = 0
        return mask
    
    def legalize_resist(self, resist):
        """Legalize the resist from litho simulator."""
        resist[resist >= 0.225] = 1  # prior was 1 vs -1
        resist[resist < 0.225] = 0
        return resist
    
    def simulate(self, mask):
        """Call litho simulator with input legal masks
        """
        for id in range(mask.shape[0]):
            if id == 0:
                _, resist = self.cl.simulateImageOpt(torch.unsqueeze(mask[id],0), LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
            else:
                _, tmp = self.cl.simulateImageOpt(torch.unsqueeze(mask[id],0), LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
                resist = torch.cat((resist, tmp), dim=0)
        return self.legalize_resist(resist)
    
    def to_one_hot(self, x):
        x = x.squeeze(1).to(torch.int64)
        return torch.nn.functional.one_hot(x,num_classes=2).permute((0,3,1,2)).float()
        
    def forward(self):
        self.real_mask = self.netF(self.mask)
        self.real_mask_img = torch.argmax(self.real_mask, dim=1, keepdim=True)
        
    def forward_uncertainty(self):
        self.forward()
        return self.MSELoss(self.real_mask[...,0] - self.real_mask[...,1])
        
    def backward(self):
        self.loss_F = self.criterionLitho(self.real_mask, self.to_one_hot(self.real_resist))  
        self.loss_F.backward()
        
    def get_F_criterion(self, real=None):
        if real != None:
            self.real_resist = real
        else:
            self.real_resist = self.simulate(self.mask)
            self.real_resist = self.legalize_mask(self.real_resist, 0.5)
        self.forward()
        loss = self.criterionLitho(self.real_mask, self.to_one_hot(self.real_resist)) 
        self.real_mask = self.legalize_mask(self.real_mask_img, 0.5).int()
        self.real_resist = self.real_resist.int()
        intersection_fg = (self.real_mask & self.real_resist).float()
        union_fg = (self.real_mask | self.real_resist).float()
        self.iou_fg = intersection_fg.sum(dim=(1,2,3))/union_fg.sum(dim=(1,2,3))
        #self.iou_bg = (1-union_fg).sum()/(1-intersection_fg).sum()
        #self.iou = (self.iou_bg + self.iou_fg)/2.0
        return loss, self.iou_fg.mean()
    
    def optimize_parameters(self):
        self.set_requires_grad(self.netF, True)
        self.forward()
        self.optimizer_F.zero_grad()
        self.backward()
        self.optimizer_F.step()
