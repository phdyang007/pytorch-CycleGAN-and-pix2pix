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

def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    if len(input) % 2 != 0:
        return 0
    assert len(input) % 2 == 0
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss

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
        self.loss_names = ['F', 'all', 'loss_predict']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_resist', 'real_mask_img'] #, 'union', 'inter']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['F']
        # define networks (both generator and discriminator)
        self.netF = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netF, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        # define loss functions
        self.criterionLitho = torch.nn.CrossEntropyLoss(reduction='none')
        if self.isTrain:
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_F)
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
        
    def forward(self, detach=False):
        self.real_mask, self.embedding, self.loss_predict = self.netF(self.mask, detach)
        self.real_mask_img = torch.argmax(self.real_mask, dim=1, keepdim=True)

    def forward_attack(self, original=None):
        self.forward()
        if original is None:
            self.real_resist = self.simulate(self.mask)
        else:
            self.real_resist = original
        self.loss_F = self.criterionLitho(self.real_mask, self.to_one_hot(self.real_resist))  
        return self.loss_F.mean()
        
    def forward_uncertainty(self, loss_type):
        self.forward()
        if loss_type == 'houdini':
            x = torch.mul( self.real_mask[...,0], self.real_mask[...,1])
            return -x.mean() 
        elif loss_type == 'logprob':
            x, _ =  torch.max( self.logSoftmax(self.real_mask), dim=1)
            return -x.mean()
        elif loss_type == 'mse':
            return self.MSELoss(self.real_mask[...,0], self.real_mask[...,1])
        elif loss_type == 'predict':
            return -self.loss_predict.mean()
        else:
            assert False, "{} not supported".format(loss_type)
        
    def backward(self, loss='all'):
        #print(self.real_mask.shape, self.to_one_hot(self.real_resist).shape)
        self.loss_F = self.criterionLitho(self.real_mask, self.to_one_hot(self.real_resist)).mean(dim=(1,2))
        self.loss_predict = self.loss_predict.reshape(self.loss_F.shape)
        self.loss_loss_predict = LossPredLoss(self.loss_predict, self.loss_F) * 0.1
        self.loss_F = self.loss_F.mean()
        self.loss_all = self.loss_F + self.loss_loss_predict
        if loss == 'all':
            self.loss_all.backward()
        else:
            if type(self.loss_loss_predict) == float:
                return
            self.loss_loss_predict.backward()
        
    def get_iou(self, data, relabel=True):
        self.set_input(data)
        if relabel:
            self.real_resist = self.legalize_mask(self.simulate(self.mask), 0.5)
        self.forward()
        self.real_mask = self.legalize_mask(self.real_mask_img, 0.5).int()
        self.real_resist = self.real_resist.int()
        intersection_fg = (self.real_mask & self.real_resist).float()
        union_fg = (self.real_mask | self.real_resist).float()
        iou_fg = (intersection_fg.sum(dim=(1,2,3)) + 1e-3)/ (union_fg.sum(dim=(1,2,3)) + 1e-3)
        return iou_fg.reshape(-1).cpu().numpy().tolist(), self.image_paths
        
    def get_F_criterion(self, real=None):
        if real != None:
            self.real_resist = real
        else:
            self.real_resist = self.simulate(self.mask)
            self.real_resist = self.legalize_mask(self.real_resist, 0.5)
        self.forward()
        loss = self.criterionLitho(self.real_mask, self.to_one_hot(self.real_resist)).mean(dim=(1,2)) 
        self.real_mask = self.legalize_mask(self.real_mask_img, 0.5).int()
        self.real_resist = self.real_resist.int()
        intersection_fg = (self.real_mask & self.real_resist).float()
        union_fg = (self.real_mask | self.real_resist).float()
        self.inter = (self.real_mask ^ self.real_resist).float()
        self.union = union_fg
        #self.iou_fg = intersection_fg.sum()/union_fg.sum()
        self.iou_fg = (intersection_fg.sum(dim=(1,2,3)) + 1e-3) / (union_fg.sum(dim=(1,2,3)) + 1e-3)
        #print(self.iou_fg) # here prints
        #self.iou_bg = (1-union_fg).sum()/(1-intersection_fg).sum()
        #self.iou = (self.iou_bg + self.iou_fg)/2.0
        return loss, self.iou_fg.mean()
    
    def optimize_parameters(self, detach=True, loss='all'):
        self.set_requires_grad(self.netF, True)
        self.forward(detach)
        self.optimizer_F.zero_grad()
        self.backward(loss)
        self.optimizer_F.step()
