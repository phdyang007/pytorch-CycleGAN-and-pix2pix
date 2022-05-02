import torch
import itertools
from util.image_pool import ImagePool
from util.util import mkdir
from .base_model import BaseModel
from . import networks
import os 
import sys 
#sys.path.append('./models/src')
sys.path.append('./models/develset')
import hydra
import numpy as np
import torchvision
from PIL import Image
from pathlib import Path
from models.develset.src.models.const import *
from models.develset.src.models.kernels import Kernel
from models.develset.src.metrics.metrics import Metrics
import torchvision.transforms as transforms
from models.develset.src.models.litho_layer import CUDA_LITHO
from models.develset.lithosim import *



class OPCModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> C.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. C.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - C|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper) (X)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper) (X)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            #parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            #parser.add_argument('--lambda_identity', type=float, default=0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.save_dir_pre = os.path.join(opt.checkpoints_dir, opt.name, "pretrain")
        #self.save_dir_post= os.path.join(opt.checkpoints_dir, opt.name, "postrain")
        mkdir(self.save_dir_pre)
        #mkdir(self.save_dir_post)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_A', 'G_A_GAN', 'D_A']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B']
        visual_names_B = ['real_B', 'real_C']
        #if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
        #    visual_names_A.append('idt_B')
        #    visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        if not self.isTrain:
            if self.opt.lt == False:
                self.visual_names = ['real_A', 'fake_B', 'nominal_C', 'inner_C', 'outer_C', 'real_B', 'real_C']#, 'post_B', 'post_nominal_C', 'post_inner_C', 'post_outer_C']
                if self.opt.dump_imask:
                    self.visual_names = self.visual_names + ['fake_B_1','fake_B_2','fake_B_3','fake_B_4']
            else:
                self.visual_names = ['real_A', 'fake_B', 'nominal_C', 'real_B', 'a2_nominal_C']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
  
        self.model_names = ['G_A']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        if self.opt.lt == True:
            if not (opt.netG_A == 'oinnopcg' or opt.netG_A == 'oinnopcgv2'):
                opt.netG_A = opt.netG_A + '_large'
        self.netG_A = networks.define_G(opt.A_nc, opt.B_nc, opt.ngf, opt.netG_A, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, self.isTrain)

        if self.isTrain:
            #if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
            #    assert(opt.input_nc == opt.output_nc)

            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionLitho = torch.nn.MSELoss()
            self.criterionMask = networks.LpLoss(size_average=False)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            #self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_A = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G_A)

        self.kernel = Kernel()
        self.cl = CUDA_LITHO(self.kernel)
        self.batch_size = opt.batch_size

        self.epe_th = 16
        #print(self.cl)

    def set_input(self, input, postrain=False):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A'].to(self.device)
        #if not self.opt.lt:
        self.real_B = input['B'].to(self.device)
        self.real_C = input['C'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


        self.loss_G_A = 0
        self.loss_G_A_GAN = 0
        self.loss_D_A = 0 
        self.model_win_count = 0
        self.total_data_count = 0
    def forward(self, pretrain=True):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            self.fake_B = self.netG_A(self.real_A)  # G_A(A)

            #self.G_A_logit = self.netD_A(self.fake_B)
                
        else:

            if self.opt.lt==False:
                fake_B = self.netG_A(self.real_A)  # G_A(A)
                turn_off = torch.zeros_like(self.real_A).type('torch.cuda.FloatTensor')
                turn_off[:,:,512:1536,512:1536]=1.0
                self.fake_B = fake_B
                print(self.fake_B.shape)
                
                self.fake_B[self.fake_B>0.5]=1.0
                self.fake_B[self.fake_B<0.5]=0.0
                
                golden_nominal_C, self.real_C = self.cl.simulateImageOpt(self.real_B, LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
                golden_outer_C, _ = self.cl.simulateImageOpt(self.real_B, LITHO_KERNEL_FOCUS, MAX_DOSE)
                golden_inner_C, _ = self.cl.simulateImageOpt(self.real_B, LITHO_KERNEL_DEFOCUS, MIN_DOSE)
                golden_aerials = [golden_nominal_C*turn_off, golden_inner_C*turn_off, golden_outer_C*turn_off]
                golden_m = Metrics(self.real_A*turn_off, golden_aerials)
                self.g_l2, self.g_pvb = golden_m.get_all()
                self.real_C[self.real_C>=0.225]=1.0
                self.real_C[self.real_C<0.225]=0.0
                #metric = []
                nominal_aerial, self.nominal_C = self.cl.simulateImageOpt(self.fake_B, LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
                outer_aerial, self.outer_C = self.cl.simulateImageOpt(self.fake_B, LITHO_KERNEL_FOCUS, MAX_DOSE)
                inner_aerial, self.inner_C = self.cl.simulateImageOpt(self.fake_B, LITHO_KERNEL_DEFOCUS, MIN_DOSE)
                self.nominal_C[self.nominal_C>=0.225]=1.0
                self.nominal_C[self.nominal_C<0.225]=0.0
                self.outer_C[self.outer_C>=0.225]=1.0
                self.outer_C[self.outer_C<0.225]=0.0
                self.inner_C[self.inner_C>=0.225]=1.0
                self.inner_C[self.inner_C<0.225]=0.0
                aerials = [nominal_aerial*turn_off, inner_aerial*turn_off, outer_aerial*turn_off]
                m = Metrics(self.real_A*turn_off, aerials)
                self.l2, self.pvb = m.get_all()
                if self.opt.dump_imask:
                    self.fake_B_1 = fake_B[0]
                    self.fake_B_2 = fake_B[1]
                    self.fake_B_3 = fake_B[2]
                    self.fake_B_4 = fake_B[3] 

                    #self.fake_B_1[self.fake_B_1>0.5]=1.0
                    #self.fake_B_1[self.fake_B_1<0.5]=0.0
                    #self.fake_B_2[self.fake_B_2>0.5]=1.0
                    #self.fake_B_2[self.fake_B_2<0.5]=0.0
                    #self.fake_B_3[self.fake_B_3>0.5]=1.0
                    #self.fake_B_3[self.fake_B_3<0.5]=0.0
                    #self.fake_B_4[self.fake_B_4>0.5]=1.0
                    #self.fake_B_4[self.fake_B_4<0.5]=0.0

                    nominal_aerial, _   = self.cl.simulateImageOpt(self.fake_B_1, LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
                    outer_aerial, _     = self.cl.simulateImageOpt(self.fake_B_1, LITHO_KERNEL_FOCUS, MAX_DOSE)
                    inner_aerial, _     = self.cl.simulateImageOpt(self.fake_B_1, LITHO_KERNEL_DEFOCUS, MIN_DOSE)                    
                    aerials = [nominal_aerial, inner_aerial, outer_aerial]
                    m = Metrics(self.real_A, aerials)
                    self.l2_1, self.pvb_1 = m.get_all()
                    nominal_aerial, _   = self.cl.simulateImageOpt(self.fake_B_2, LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
                    outer_aerial, _     = self.cl.simulateImageOpt(self.fake_B_2, LITHO_KERNEL_FOCUS, MAX_DOSE)
                    inner_aerial, _     = self.cl.simulateImageOpt(self.fake_B_2, LITHO_KERNEL_DEFOCUS, MIN_DOSE)   
                    aerials = [nominal_aerial, inner_aerial, outer_aerial]
                    m = Metrics(self.real_A, aerials)
                    self.l2_2, self.pvb_2 = m.get_all()
                    nominal_aerial, _   = self.cl.simulateImageOpt(self.fake_B_3, LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
                    outer_aerial, _     = self.cl.simulateImageOpt(self.fake_B_3, LITHO_KERNEL_FOCUS, MAX_DOSE)
                    inner_aerial, _     = self.cl.simulateImageOpt(self.fake_B_3, LITHO_KERNEL_DEFOCUS, MIN_DOSE)   
                    aerials = [nominal_aerial, inner_aerial, outer_aerial]
                    m = Metrics(self.real_A, aerials)
                    self.l2_3, self.pvb_3 = m.get_all()
                    nominal_aerial, _   = self.cl.simulateImageOpt(self.fake_B_4, LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
                    outer_aerial, _     = self.cl.simulateImageOpt(self.fake_B_4, LITHO_KERNEL_FOCUS, MAX_DOSE)
                    inner_aerial, _     = self.cl.simulateImageOpt(self.fake_B_4, LITHO_KERNEL_DEFOCUS, MIN_DOSE)   
                    aerials = [nominal_aerial, inner_aerial, outer_aerial]
                    m = Metrics(self.real_A, aerials)
                    self.l2_4, self.pvb_4 = m.get_all()
                if self.opt.update_mask: #check if the model gets better results for the training set. 
                    self.total_data_count += 1
                    if self.l2 < self.g_l2:
                        self.update_mask = 1
                        self.model_win_count += 1
                        print("mask updated....")
                    else:
                        self.update_mask = 0


            else:
                import timeit 
                self.real_A = self.real_A[:,:,1024:7168,1024:7168]
                turn_off = torch.zeros_like(self.real_A).type('torch.cuda.FloatTensor')
                turn_off[:,:,512:5632,512:5632]=1.0
                #self.real_A = self.real_A
                start = timeit.default_timer()
        # run inference

                self.fake_B = self.netG_A(self.real_A)
                end = timeit.default_timer()
                print("Pure Inference Time elasped: %f"%(end-start))
                #for mth in range(30,51):
                mth=50
                self.fake_B[self.fake_B>=mth/100.0]=1.0
                self.fake_B[self.fake_B<mth/100.0]=0.0
                resist_nominal = torch.zeros_like(self.fake_B).type('torch.cuda.FloatTensor')
                resist_inner = torch.zeros_like(self.fake_B).type('torch.cuda.FloatTensor')
                resist_outer = torch.zeros_like(self.fake_B).type('torch.cuda.FloatTensor')
                target_2k = torch.zeros((25,1,2048,2048)).type('torch.cuda.FloatTensor')

                for tile_x in range(5):
                    for tile_y in range(5):
                        tmp_mask = self.fake_B[:,:,1024*tile_x:1024*tile_x+2048,1024*tile_y:1024*tile_y+2048]
                        tmp_nominal, _ = self.cl.simulateImageOpt(tmp_mask, LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
                        tmp_outer, _ = self.cl.simulateImageOpt(tmp_mask, LITHO_KERNEL_FOCUS, MAX_DOSE)
                        tmp_inner, _ = self.cl.simulateImageOpt(tmp_mask, LITHO_KERNEL_DEFOCUS, MIN_DOSE)
                        resist_nominal[:,:,1024*tile_x:1024*tile_x+2048,1024*tile_y:1024*tile_y+2048][:,:,512:1536,512:1536] = tmp_nominal[:,:,512:1536,512:1536]
                        resist_inner[:,:,1024*tile_x:1024*tile_x+2048,1024*tile_y:1024*tile_y+2048][:,:,512:1536,512:1536]   = tmp_inner[:,:,512:1536,512:1536]
                        resist_outer[:,:,1024*tile_x:1024*tile_x+2048,1024*tile_y:1024*tile_y+2048][:,:,512:1536,512:1536]   = tmp_outer[:,:,512:1536,512:1536]
                        #aerials = [resist_nominal, resist_inner, resist_outer]

                aerials = [resist_nominal*turn_off, resist_inner*turn_off, resist_outer*turn_off]
                m = Metrics(self.real_A*turn_off, aerials)
                self.l2, self.pvb = m.get_all()
                print(self.l2, self.pvb)
                self.nominal_C = resist_nominal
                self.nominal_C[self.nominal_C>=0.225]=1
                self.nominal_C[self.nominal_C<0.225]=0

                a2_resist_nominal = torch.zeros_like(self.real_A).type('torch.cuda.FloatTensor')
                a2_resist_inner = torch.zeros_like(self.real_A).type('torch.cuda.FloatTensor')
                a2_resist_outer = torch.zeros_like(self.real_A).type('torch.cuda.FloatTensor')
                self.real_B = self.real_B[:,:,1024:7168,1024:7168] 
                for tile_x in range(5):
                    for tile_y in range(5):
                        tmp_mask = self.real_B[:,:,1024*tile_x:1024*tile_x+2048,1024*tile_y:1024*tile_y+2048]
                        tmp_nominal, _ = self.cl.simulateImageOpt(tmp_mask, LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
                        tmp_outer, _ = self.cl.simulateImageOpt(tmp_mask, LITHO_KERNEL_FOCUS, MAX_DOSE)
                        tmp_inner, _ = self.cl.simulateImageOpt(tmp_mask, LITHO_KERNEL_DEFOCUS, MIN_DOSE)
                        a2_resist_nominal[:,:,1024*tile_x:1024*tile_x+2048,1024*tile_y:1024*tile_y+2048][:,:,512:1536,512:1536] = tmp_nominal[:,:,512:1536,512:1536]
                        a2_resist_inner[:,:,1024*tile_x:1024*tile_x+2048,1024*tile_y:1024*tile_y+2048][:,:,512:1536,512:1536]   = tmp_inner[:,:,512:1536,512:1536]
                        a2_resist_outer[:,:,1024*tile_x:1024*tile_x+2048,1024*tile_y:1024*tile_y+2048][:,:,512:1536,512:1536]   = tmp_outer[:,:,512:1536,512:1536]
                        #a2_aerials = [a2_resist_nominal, a2_resist_inner, a2_resist_outer]
                        #m = Metrics(x_design[i], aerials)
                        #l2, pvb = m.get_all()
                        #target_2k[5*tile_x + tile_y,0,:,:]=x_design[i][0,1024*tile_x:1024*tile_x+2048,1024*tile_y:1024*tile_y+2048]*turn_off_small
                a2_aerials = [a2_resist_nominal*turn_off, a2_resist_inner*turn_off, a2_resist_outer*turn_off]
                m = Metrics(self.real_A*turn_off, a2_aerials)
                self.a2_l2, self.a2_pvb = m.get_all()
                print(self.a2_l2, self.a2_pvb)
                #if self.a2_l2 > self.l2:
                #    print("#############################################th: %g#######################################"%mth)
                self.a2_nominal_C = a2_resist_nominal
                self.a2_nominal_C[self.a2_nominal_C>=0.225]=1
                self.a2_nominal_C[self.a2_nominal_C<0.225]=0

                ##check epe 
                img_base_path = self.get_image_paths()[0][:-4]
                #print(img_base_path)
                v_epe_points_path = img_base_path+'.gds.v.csv'
                h_epe_points_path = img_base_path+'.gds.h.csv'

                v_epe_points = np.genfromtxt(v_epe_points_path, delimiter=',').astype(int)
                h_epe_points = np.genfromtxt(h_epe_points_path, delimiter=',').astype(int)

                #print(v_epe_points, h_epe_points)
                target = torch.zeros(8192,8192).type('torch.cuda.FloatTensor')
                resist_ml = torch.zeros(8192,8192).type('torch.cuda.FloatTensor')
                resist_ilt = torch.zeros(8192,8192).type('torch.cuda.FloatTensor')
                target[1024:7168,1024:7168] = (self.real_A*turn_off)[0,0,:,:]
                resist_ml[1024:7168,1024:7168]= (self.nominal_C*turn_off)[0,0,:,:]
                resist_ilt[1024:7168,1024:7168]= (self.a2_nominal_C*turn_off)[0,0,:,:]
                target = target.int()
                resist_ml=resist_ml.int()
                resist_ilt=resist_ilt.int()
                self.h_epes_ml, self.v_epes_ml, target = self.get_epe_via(target, resist_ml, h_epe_points, v_epe_points)
                self.h_epes_ilt, self.v_epes_ilt, _ = self.get_epe_via(target, resist_ilt, h_epe_points, v_epe_points)

                self.real_A[0,0,:,:] = target[1024:7168,1024:7168].type('torch.cuda.FloatTensor')

                ml_epes = np.absolute(np.array(self.h_epes_ml+self.v_epes_ml))
                ilt_epes = np.absolute(np.array(self.h_epes_ilt+self.v_epes_ilt))
                self.ml_epe_count = len(np.where(ml_epes==self.epe_th-1)[0])
                self.ilt_epe_count = len(np.where(ilt_epes==self.epe_th-1)[0])
                print("ml epe violation count %g; ilt epe violation count %g."%(self.ml_epe_count,self.ilt_epe_count))


                

                

    def get_epe_via(self,target,resist,h_epe_points,v_epe_points):
        overlap = target & resist 
        union = target | resist 
        diff = union-overlap 
        outdiff = diff & (1-target)
        indiff  = diff & target 
        h_epe_count = len(h_epe_points)
        v_epe_count = len(v_epe_points)
        h_epes = []
        v_epes = []
        #checking horizontal epe
        for point in h_epe_points:
            
            if point[0]<= 1536 or point[1]<= 1536 or point[0]>= 6656 or point[1]>= 6656:
                continue
            target[point[1]-3:point[1]+3,point[0]-3:point[0]+3]=1
            if point[-1]==-1:
                if resist[point[1],point[0]]==1: #positive epe
                    for disp in range(1,self.epe_th):
                        if resist[point[1],point[0]-disp]==0:
                            break 
                    h_epes.append(disp)
                if resist[point[1],point[0]]==0: #negative epe
                    for disp in range(1,self.epe_th):
                        if resist[point[1],point[0]+disp]==1:
                            break 
                    h_epes.append(-disp)                
            if point[-1]==1:
                if resist[point[1],point[0]]==1: #positive epe
                    for disp in range(1,self.epe_th):
                        if resist[point[1],point[0]+disp]==0:
                            break 
                    h_epes.append(disp)
                if resist[point[1],point[0]]==0: #negative epe
                    for disp in range(1,self.epe_th):
                        if resist[point[1],point[0]-disp]==1:
                            break 
                    h_epes.append(-disp)  
        
        #checking vertical epe
        for point in v_epe_points:
            if point[0]<= 1536 or point[1]<= 1536 or point[0]>= 6656 or point[1]>= 6656:
                continue
            target[point[1]-3:point[1]+3,point[0]-3:point[0]+3]=1   # debug epe control points 
            if point[-1]==-1:
                
                if resist[point[1],point[0]]==1: #positive epe
                    for disp in range(1,self.epe_th):
                        if resist[point[1]-disp,point[0]]==0:
                            break 
                    v_epes.append(disp)
                if resist[point[1],point[0]]==0: #negative epe
                    for disp in range(1,self.epe_th):
                        if resist[point[1]+disp,point[0]]==1:
                            break 
                    v_epes.append(-disp)                
            if point[-1]==1:
                if resist[point[1],point[0]]==1: #positive epe
                    for disp in range(1,self.epe_th):
                        if resist[point[1]+disp,point[0]]==0:
                            break 
                    v_epes.append(disp)
                if resist[point[1],point[0]]==0: #negative epe
                    for disp in range(1,self.epe_th):
                        if resist[point[1]-disp,point[0]]==1:
                            break 
                    v_epes.append(-disp) 


        return h_epes, v_epes, target 


    def get_epe(self,target,resist,h_epe_points,v_epe_points):
        overlap = target & resist 
        union = target | resist 
        diff = union-overlap 
        outdiff = diff & (1-target)
        indiff  = diff & target 
        h_epe_count = len(h_epe_points)
        v_epe_count = len(v_epe_points)
        h_epes = []
        v_epes = []
        #checking horizontal epe
        for point in h_epe_points:
            
            if point[0]<= 1536 or point[1]<= 1536 or point[0]>= 6656 or point[1]>= 6656:
                continue
            target[point[1]-3:point[1]+3,point[0]-3:point[0]+3]=1
            if point[-1]==-1:
                if resist[point[1],point[0]]==1: #positive epe
                    for disp in range(1,self.epe_th):
                        if resist[point[1],point[0]-disp]==0:
                            break 
                    h_epes.append(disp)
                if resist[point[1],point[0]]==0: #negative epe
                    for disp in range(1,self.epe_th):
                        if resist[point[1],point[0]+disp]==1:
                            break 
                    h_epes.append(-disp)                
            if point[-1]==1:
                if resist[point[1],point[0]]==1: #positive epe
                    for disp in range(1,self.epe_th):
                        if resist[point[1],point[0]+disp]==0:
                            break 
                    h_epes.append(disp)
                if resist[point[1],point[0]]==0: #negative epe
                    for disp in range(1,self.epe_th):
                        if resist[point[1],point[0]-disp]==1:
                            break 
                    h_epes.append(-disp)  
        
        #checking vertical epe
        for point in v_epe_points:
            if point[0]<= 1536 or point[1]<= 1536 or point[0]>= 6656 or point[1]>= 6656:
                continue
            target[point[1]-3:point[1]+3,point[0]-3:point[0]+3]=1   # debug epe control points 
            if point[-1]==-1:
                
                if resist[point[1],point[0]]==1: #positive epe
                    for disp in range(1,self.epe_th):
                        if resist[point[1]+disp,point[0]]==0:
                            break 
                    v_epes.append(disp)
                if resist[point[1],point[0]]==0: #negative epe
                    for disp in range(1,self.epe_th):
                        if resist[point[1]-disp,point[0]]==1:
                            break 
                    v_epes.append(-disp)                
            if point[-1]==1:
                if resist[point[1],point[0]]==1: #positive epe
                    for disp in range(1,self.epe_th):
                        if resist[point[1]-disp,point[0]]==0:
                            break 
                    v_epes.append(disp)
                if resist[point[1],point[0]]==0: #negative epe
                    for disp in range(1,self.epe_th):
                        if resist[point[1]+disp,point[0]]==1:
                            break 
                    v_epes.append(-disp) 


        return h_epes, v_epes, target 
        

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
    
    def backward_G_A_pretrain(self):
        
        self.loss_G_A = self.criterionMask(self.fake_B, self.real_B) #mask v.s. ground mask  us L1 loss to avoid blur for the sake of litho accuracy.


        #self.loss_G   = self.loss_G_A
 

        self.loss_G_A.backward()

    def optimize_parameters(self):


        self.forward()
        # G_A and G_B
        #self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G_A.zero_grad()  # set G_A gradients to zero
        #self.set_requires_grad([self.netG_A], True)  #update masknet
        #self.set_requires_grad([self.netG_B], False) #update masknet and lithonet seperately.
        self.backward_G_A_pretrain() #get gradients for G_A
        #self.optimizer_G_A.step()  #update G_A
        #self.optimizer_G_B.zero_grad()  # set G_B's gradients to zero
        #self.set_requires_grad([self.netG_B], True)  #update lithonet
        #self.set_requires_grad([self.netG_A], False) #update masknet and lithonet seperately.
        #self.backward_G_B_pretrain() #get gradients for G_B
        self.optimizer_G_A.step()



    

    def reset_optimizer(self):
            #self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_G_A = torch.optim.Adam(self.netG_A.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            #self.optimizer_G_B = torch.optim.Adam(self.netG_B.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            #self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            #self.optimizers.append(self.optimizer_G)
            #self.optimizers.append(self.optimizer_D)
            #self.optimizers.append(self.optimizer_G_A)
            #self.optimizers.append(self.optimizer_G_B)
