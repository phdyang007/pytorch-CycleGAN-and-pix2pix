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

class LithoGANModel(BaseModel):
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
        parser.set_defaults(netG='dcgan', netD='dcgan')
        parser.add_argument('--netF', type=str, default='oinnopc', help='specify litho architecture [oinnopc]')
        parser.add_argument('--trainGAN', type=bool, default=True, help='whether to include GAN model for train/test')
        parser.add_argument('--trainF', type=bool, default=False, help='whether to include F model')
        parser.add_argument('--input_zdim', type=int, default=128, help='input Z dimension to GAN')
        if is_train:
            parser.set_defaults(gan_mode='wgangp')
            parser.add_argument('--lambda_attack', type=float, default=10.0, help='weight for F attack loss')
            parser.add_argument('--grad_norm', type=bool, default=True, help='include gradient penalty for GAN training')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.trainGAN, self.trainF = opt.trainGAN, opt.trainF
        assert self.trainGAN or self.trainF
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = []
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = []
        if self.trainGAN:
            if self.isTrain:
                self.model_names = ['G', 'D']
            else:
                self.model_names = ['G']
            self.loss_names = ['G_GAN', 'D_real', 'D_fake']
            if self.opt.grad_norm:
                self.loss_names.append('G_L1')
            self.visual_names = ['fake', 'real_low_res']
        if self.trainF:
            self.model_names.append('F')
            self.loss_names.append('F_real')
            self.visual_names.append('real_mask')
            self.visual_names.append('real_resist')
            if self.trainGAN:
                self.loss_names.append('F_fake')
                self.loss_names.append('F_attack')
                self.visual_names.append('fake_mask')
                self.visual_names.append('fake_resist')
        # define networks (both generator and discriminator)
        if self.trainGAN:
            self.netG = networks.define_G(opt.input_zdim, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.trainF:
            self.netF = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netF, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain and self.trainGAN:  # define a discriminator; 
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionLitho = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if self.trainGAN:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_G)
                self.optimizers.append(self.optimizer_D)
            if self.trainF:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_F)
                
                self.kernel = Kernel()
                self.cl = CUDA_LITHO(self.kernel)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_low_res = input['real_low_res'].to(self.device)
        self.real_high_res = input['real_high_res'].to(self.device)
        
    def legalize_mask(self, mask):
        """Legalize the mask generated from GAN."""
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        return mask
    
    def legalize_resist(self, resist):
        """Legalize the resist from litho simulator."""
        resist[resist >= 0.225] = 1.0
        resist[resist < 0.225] = 0.0
        return resist
    
    def simulate(self, mask):
        """Call litho simulator with input legal masks
        """
        for id in range(mask[0]):
            if id == 0:
                _, resist = self.cl.simulateImageOpt(torch.unsqueeze(mask[id],0), LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
            else:
                _, tmp = self.cl.simulateImageOpt(torch.unsqueeze(mask[id],0), LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
                resist = torch.cat((resist, tmp), dim=0)
        return self.legalize_resist(resist)
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.trainGAN:
            self.forward_G()
        if self.trainF:
            self.foward_F()

    def forward_G(self):
        noise = torch.randn(self.opt.batch_size, self.opt.input_zdim, 1, 1, device=self.device)
        self.fake = self.netG(noise)
        
    def forward_F(self):
        if self.trainGAN:
            self.legal_fake = self.legalize_mask(self.fake)
            self.legal_fake_high_res = self.netG.upsample(self.legal_fak)
            self.fake_mask = self.netF(self.legal_fake_high_res)    
        self.real_mask = self.netF(self.real_high_res)
        
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        pred_fake = self.netD(self.fake.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        pred_real = self.netD(self.real_low_res)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        if self.opt.grad_norm:
            self.loss_G_L1, _ = networks.cal_gradient_penalty(self.netD, self.real_low_res, self.fake.detach(), self.device)
            self.loss_D += self.loss_G_L1
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        pred_fake = self.netD(self.fake)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G = self.loss_G_GAN
        if self.trainF:
            self.fake_resist = self.simulate(self.legal_fake)
            self.loss_F_attack = self.criterionLitho(self.fake_resist, self.fake_mask)
            self.loss_G -= self.loss_F_attack * self.opt.lambda_attack
        self.loss_G.backward()
        
    def backward_F(self):
        self.real_resist = self.simulate(self.real_high_res)
        self.loss_F_real = self.criterionLitho(self.real_resist, self.real_mask)  
        self.loss_F = self.loss_F_real 
        if self.trainGAN:
            self.fake_resist = self.simulate(self.legal_fake)
            self.loss_F_fake = self.criterionLitho(self.fake_resist, self.fake_mask)
            self.loss_F += self.loss_F_fake
        self.loss_F.backward()
            
    def optimize_parameters(self):
        # Iterative train GAN and F
        if self.trainGAN:
            self.forward()
            if self.trainF:
                self.set_requires_grad(self.netF, False) # Fix F when attacking
            # update D
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
            # update G
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G()                   # calculate graidents for G
            self.optimizer_G.step()             # udpate G's weights
            
        if self.trainF:
            self.forward()
            self.set_requires_grad(self.netF, True)
            self.optimizer_F.zero_grad()
            self.backward_F()
            self.optimizer_F.step()