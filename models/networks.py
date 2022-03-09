import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'dcgan':
        net = DCGenerator(nz=input_nc, ngf=ngf, nc=output_nc)
    elif netG == 'oinnlitho':
        net = oinnlitho(modes1=50, modes2=50, width=16, in_channel=1, refine_channel=32, refine_kernel=3)
    elif netG == 'oinnopc':
        net = oinnopc(modes1=50, modes2=50, width=16, in_channel=1, refine_channel=32, refine_kernel=3)
    elif netG == 'oinnopc_v001':
        net = oinnopc_v001(modes1=50, modes2=50, width=16, in_channel=1, refine_channel=32, refine_kernel=3)
    elif netG == 'oinnopc_large':
        net = oinnopc_large(modes1=50, modes2=50, width=16, in_channel=1, refine_channel=32, refine_kernel=3)
    elif netG == 'oinnopc_multi':
        net == oinnopc_multi(modes1=50, modes2=50, width=16, in_channel=1, refine_channel=32, refine_kernel=3)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'dcgan':
        net = DCDiscriminator(input_nc, ndf)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss



class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        pseudo_y_norms = torch.ones(num_examples).type('torch.cuda.FloatTensor') *250000.0
        y_norms = torch.where(y_norms==0,pseudo_y_norms,y_norms)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

class DCGenerator(nn.Module):
    """DCGAN from "UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS"

    Implementations based on https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """
    def __init__(self, nz=128, ngf=64, nc=1):
        """Construct a deep convolution generator

        Parameters:
            nz (int)            -- dimension of hiddent z
            ngf(int)            -- depth of generator feature map
            nc (int)            -- dimension output channel
        """
        super(DCGenerator, self).__init__()
        self.upsample = nn.Upsample(scale_factor=8, mode='nearest')
            # Convert state size. (nc) x 2048 x 2048
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),
            # state size. (ngf*32) x 4 x 4
            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 8 x 8
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 16 x 16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 32 x 32
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 64 x 64
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 256 x 256
        )

    def forward(self, input):
        return self.model(input)

class DCDiscriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        """Construct a deep convolution discriminator

        Parameters:
            nc (int)            -- dimension input channel
            ndf(int)            -- depth of generator feature map
        """
        super(DCDiscriminator, self).__init__()
        self.model = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 16 x 16
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 8 x 8
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*32) x 4 x 4
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.model(input)
    
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x





################################################################
# fourier layer, lift input channel after fft
################################################################
class SpectralConv2dLiftChannel(nn.Module):
    def __init__(self, in_channel, width, out_channels, modes1, modes2):
        super(SpectralConv2dLiftChannel, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        lift in_channel to width after fft
        """

        self.in_channel = in_channel
        self.width = width 
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (self.width * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.width, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(self.width, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.liftchannel = nn.Linear(self.in_channel, width, dtype=torch.cfloat)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        print(x.shape)
        x_ft = torch.fft.rfft2(x).permute(0, 2, 3, 1) #N H W C
        print(x_ft.shape)
        x_lift = self.liftchannel(x_ft).permute(0, 3, 1, 2) #N C H W
        print(x_lift.shape)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_lift[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_lift[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x




class VGGBlock(nn.Module):
    def __init__(self, channels, act_func=nn.LeakyReLU(0.2, inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func_down = act_func
        self.act_func_up = nn.ReLU(True)
        # self.act_func_up = act_func
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func_down(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func_up(out)

        return out
class PoolConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        :params:
            channels-- input and out channels
        """
        super().__init__()
        self.pool_conv = nn.Conv2d(in_channels, out_channels, kernel_size=4,
                        stride=2, padding=1, bias=False
        )
    def forward(self, input):
        return self.pool_conv(input)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels,
                                    kernel_size=4, stride=2,
                                    padding=1, bias=False)

    def forward(self, input):
        return self.up_conv(input)

## DAMO baseline

class PoolConv2(nn.Module):
    def __init__(self, channels):
        """
        :params:
            channels-- input and out channels
        """
        super().__init__()
        self.pool_conv = nn.Conv2d(channels, channels, kernel_size=4,
                        stride=2, padding=1, bias=False
        )
    def forward(self, input):
        return self.pool_conv(input)

class UpConv2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(channels, channels,
                                    kernel_size=4, stride=2,
                                    padding=1, bias=False)

    def forward(self, input):
        return self.up_conv(input)

class VGGBlock2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.LeakyReLU(0.2, inplace=True)):
        super(VGGBlock2, self).__init__()
        self.act_func_down = act_func
        self.act_func_up = nn.ReLU(True)
        # self.act_func_up = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func_down(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func_up(out)

        return out
        
class NestedUNet(nn.Module):
    def __init__(self, input_nc, output_nc=1, lambda_o=1, tanh_act=True, deepsupervision=True, upp_scale=2):
        """
        :param args:
            input_channels
            deepsupervison
        """
        super(NestedUNet, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.deepsupervision = deepsupervision
        self.tanh_act = tanh_act
        self.lambda_o = lambda_o

        nb_filter = [64, 128, 256, 512, 1024]
        nb_filter = [int(x / upp_scale) for x in nb_filter]
        self.nb_filter = nb_filter
        """
        change the pooling layer to conv2d stride
        using func self.pool
        """
        # self.pool = nn.MaxPool2d(2, 2)
        # self.pool = nn.Conv2d()

        """
        change the upsampleing layer to conv2dtransposed stride
        using func self.up
        """
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock2(self.input_nc, nb_filter[0], nb_filter[0])
        self.pool0_0 = PoolConv2(nb_filter[0])
        self.conv1_0 = VGGBlock2(nb_filter[0], nb_filter[1], nb_filter[1])
        self.pool1_0 = PoolConv2(nb_filter[1])
        self.up1_0 = UpConv2(nb_filter[1])
        self.conv2_0 = VGGBlock2(nb_filter[1], nb_filter[2], nb_filter[2])
        self.pool2_0 = PoolConv2(nb_filter[2])
        self.up2_0 = UpConv2(nb_filter[2])
        self.conv3_0 = VGGBlock2(nb_filter[2], nb_filter[3], nb_filter[3])
        self.pool3_0 = PoolConv2(nb_filter[3])
        self.up3_0 = UpConv2(nb_filter[3])
        self.conv4_0 = VGGBlock2(nb_filter[3], nb_filter[4], nb_filter[4])
        self.up4_0 = UpConv2(nb_filter[4])


        self.conv0_1 = VGGBlock2(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock2(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.up1_1 = UpConv2(nb_filter[1])
        self.conv2_1 = VGGBlock2(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.up2_1 = UpConv2(nb_filter[2])
        self.conv3_1 = VGGBlock2(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.up3_1 = UpConv2(nb_filter[3])

        self.conv0_2 = VGGBlock2(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock2(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.up1_2 = UpConv2(nb_filter[1])
        self.conv2_2 = VGGBlock2(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        self.up2_2 = UpConv2(nb_filter[2])

        self.conv0_3 = VGGBlock2(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock2(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        self.up1_3 = UpConv2(nb_filter[1])

        self.conv0_4 = VGGBlock2(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], self.output_nc, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], self.output_nc, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], self.output_nc, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], self.output_nc, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], self.output_nc, kernel_size=1)

        if self.tanh_act:
            self.tanh = nn.Tanh()

    def forward(self, input):
        nb_filter = self.nb_filter
        input = nn.ConstantPad2d((12,12,12,12),0)(input)
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool0_0(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool1_0(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool2_0(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool3_0(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], 1))

        if self.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            output = (output1 + output2 + output3 + output4)/4
            output = output * self.lambda_o
            if self.tanh_act:
                return self.tanh(output)[:,:,12:-12,12:-12]
            else:
                return ((output1 + output2 + output3 + output4)/4)[:,:,12:-12,12:-12]
            # return [output1, output2, output3, output4]

        else:
            if self.tanh_act:
                output = self.final(x0_4)
                output = output * self.lambda_o
                output = self.tanh(output)[:,:,12:-12,12:-12]
            else:
                output = self.final(x0_4)
                output = output * self.lambda_o
            return output[:,:,12:-12,12:-12]

class oinnopc_multi(nn.Module):
    def __init__(self, modes1, modes2,  width, in_channel=1, refine_channel=32, refine_kernel = 3, smooth_kernel = 3, oinn_num = 4):
        super(oinnopc, self).__init__()

        # from design to mask, four concatenated 

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.refine_kernel = refine_kernel
        self.in_channel = in_channel
        self.refine_channel = refine_channel
        self.smooth_kernel = smooth_kernel   
        self.oinn_num = oinn_num
        self.layers = nn.Sequential()

        for i in range(oinn_num):
            self.layers.add_module("oinn%g"%(i+1), oinnopc(modes1, modes2,  width, in_channel=1, refine_channel=32, refine_kernel = 3, smooth_kernel = 3))

    def forward(self, x): 
        x = self.layers(x)

        return x 







class oinnopc(nn.Module): 
    def __init__(self, modes1, modes2,  width, in_channel=1, refine_channel=32, refine_kernel = 3, smooth_kernel = 3):
        super(oinnopc, self).__init__()

        # from design to mask, same as forward v2 as baseline.

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.refine_kernel = refine_kernel
        self.in_channel = in_channel
        self.refine_channel = refine_channel
        self.smooth_kernel = smooth_kernel
        #self.cemap = cemap
        self.vgg_channels = [4,8,16,16,8,4]

        #resize
        self.resize0 = nn.AvgPool2d(8)
        #fourier
        self.fno = SpectralConv2dLiftChannel(self.in_channel, self.width, self.width, self.modes1, self.modes2)

        #refine
        self.convr0 = nn.Conv2d(in_channels=self.vgg_channels[5], out_channels=self.refine_channel, kernel_size=self.refine_kernel, padding = (self.refine_kernel-1)//2)
        self.convr1 = nn.Conv2d(in_channels=self.refine_channel, out_channels=self.refine_channel//2, kernel_size=self.refine_kernel, padding = (self.refine_kernel-1)//2)
        self.convr2 = nn.Conv2d(in_channels=self.refine_channel//2, out_channels=self.refine_channel//2, kernel_size=self.refine_kernel, padding = (self.refine_kernel-1)//2)

        self.convr3 = nn.Conv2d(in_channels=self.refine_channel//2, out_channels=1, kernel_size=self.refine_kernel, padding = (self.refine_kernel-1)//2)
        self.act_fn = nn.LeakyReLU(0.1)

        #bypass unet
        self.ds0 = PoolConv(1, self.vgg_channels[0])
        self.vgg0 = VGGBlock(self.vgg_channels[0])
        self.ds1 = PoolConv(self.vgg_channels[0], self.vgg_channels[1])
        self.vgg1 = VGGBlock(self.vgg_channels[1])
        self.ds2 = PoolConv(self.vgg_channels[1], self.vgg_channels[2])
        self.vgg2 = VGGBlock(self.vgg_channels[2])   #250
        self.us3 = UpConv(self.vgg_channels[2]+self.width, self.vgg_channels[3]) #merge with fno output
        self.vgg3 = VGGBlock(self.vgg_channels[3])
        self.us4 = UpConv(self.vgg_channels[3]+self.vgg_channels[1], self.vgg_channels[4])
        self.vgg4 = VGGBlock(self.vgg_channels[4])
        self.us5 = UpConv(self.vgg_channels[4]+self.vgg_channels[0], self.vgg_channels[5])
        self.vgg5 = VGGBlock(self.vgg_channels[5])


        self.tanh = nn.Tanh()
        


    def forward(self, x):

        #fno pass
        x_fno = self.resize0(x) 
        print(x_fno.shape)
        x_fno = self.fno(x_fno)       
        x_fno = self.act_fn(x_fno) 
        #unet pass
        x_unet = self.ds0(x)
        x_unet_1000 = self.vgg0(x_unet)
        x_unet = self.ds1(x_unet_1000)
        x_unet_500 = self.vgg1(x_unet)
        x_unet = self.ds2(x_unet_500)
        x_unet_250 = self.vgg2(x_unet)

        #merge fno and unet
        x = torch.cat((x_fno, x_unet_250), 1)

        #dconv
        x = self.us3(x)
        x = torch.cat((self.vgg3(x),x_unet_500), 1)
        x = self.us4(x)
        x = torch.cat((self.vgg4(x),x_unet_1000), 1)
        x = self.us5(x)
        x = self.vgg5(x)
        #refine
        x = self.convr0(x)
        x = self.act_fn(x)
        x = self.convr1(x)
        x = self.act_fn(x)
        x = self.convr2(x)
        x = self.act_fn(x)
        x = self.convr3(x)


        return self.tanh(x)

class oinnopc_v001(nn.Module): 
    def __init__(self, modes1, modes2,  width, in_channel=1, refine_channel=32, refine_kernel = 3, smooth_kernel = 3):
        super(oinnopc_v001, self).__init__()

        # from design to mask, same as forward v2 as baseline. change output to sigmoid

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.refine_kernel = refine_kernel
        self.in_channel = in_channel
        self.refine_channel = refine_channel
        self.smooth_kernel = smooth_kernel
        #self.cemap = cemap
        self.vgg_channels = [4,8,16,16,8,4]

        #resize
        self.resize0 = nn.AvgPool2d(8)
        #fourier
        self.fno = SpectralConv2dLiftChannel(self.in_channel, self.width, self.width, self.modes1, self.modes2)

        #refine
        self.convr0 = nn.Conv2d(in_channels=self.vgg_channels[5], out_channels=self.refine_channel, kernel_size=self.refine_kernel, padding = (self.refine_kernel-1)//2)
        self.convr1 = nn.Conv2d(in_channels=self.refine_channel, out_channels=self.refine_channel//2, kernel_size=self.refine_kernel, padding = (self.refine_kernel-1)//2)
        self.convr2 = nn.Conv2d(in_channels=self.refine_channel//2, out_channels=self.refine_channel//2, kernel_size=self.refine_kernel, padding = (self.refine_kernel-1)//2)

        self.convr3 = nn.Conv2d(in_channels=self.refine_channel//2, out_channels=1, kernel_size=self.refine_kernel, padding = (self.refine_kernel-1)//2)
        self.act_fn = nn.LeakyReLU(0.1)

        #bypass unet
        self.ds0 = PoolConv(1, self.vgg_channels[0])
        self.vgg0 = VGGBlock(self.vgg_channels[0])
        self.ds1 = PoolConv(self.vgg_channels[0], self.vgg_channels[1])
        self.vgg1 = VGGBlock(self.vgg_channels[1])
        self.ds2 = PoolConv(self.vgg_channels[1], self.vgg_channels[2])
        self.vgg2 = VGGBlock(self.vgg_channels[2])   #250
        self.us3 = UpConv(self.vgg_channels[2]+self.width, self.vgg_channels[3]) #merge with fno output
        self.vgg3 = VGGBlock(self.vgg_channels[3])
        self.us4 = UpConv(self.vgg_channels[3]+self.vgg_channels[1], self.vgg_channels[4])
        self.vgg4 = VGGBlock(self.vgg_channels[4])
        self.us5 = UpConv(self.vgg_channels[4]+self.vgg_channels[0], self.vgg_channels[5])
        self.vgg5 = VGGBlock(self.vgg_channels[5])


        self.final = nn.Sigmoid()
        


    def forward(self, x):

        #fno pass
        x_fno = self.resize0(x) 

        x_fno = self.fno(x_fno)       
        x_fno = self.act_fn(x_fno) 
        #unet pass
        x_unet = self.ds0(x)
        x_unet_1000 = self.vgg0(x_unet)
        x_unet = self.ds1(x_unet_1000)
        x_unet_500 = self.vgg1(x_unet)
        x_unet = self.ds2(x_unet_500)
        x_unet_250 = self.vgg2(x_unet)

        #merge fno and unet
        x = torch.cat((x_fno, x_unet_250), 1)

        #dconv
        x = self.us3(x)
        x = torch.cat((self.vgg3(x),x_unet_500), 1)
        x = self.us4(x)
        x = torch.cat((self.vgg4(x),x_unet_1000), 1)
        x = self.us5(x)
        x = self.vgg5(x)
        #refine
        x = self.convr0(x)
        x = self.act_fn(x)
        x = self.convr1(x)
        x = self.act_fn(x)
        x = self.convr2(x)
        x = self.act_fn(x)
        x = self.convr3(x)


        return self.final(x)

class oinnlitho(nn.Module): 
    def __init__(self, modes1, modes2,  width, in_channel=1, refine_channel=32, refine_kernel = 3, smooth_kernel = 3):
        super(oinnlitho, self).__init__()

        # from design to mask, same as forward v2 as baseline.

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.refine_kernel = refine_kernel
        self.in_channel = in_channel
        self.refine_channel = refine_channel
        self.smooth_kernel = smooth_kernel
        #self.cemap = cemap
        self.vgg_channels = [4,8,16,16,8,4]

        #resize
        self.resize0 = nn.AvgPool2d(8)
        #fourier
        self.fno = SpectralConv2dLiftChannel(self.in_channel, self.width, self.width, self.modes1, self.modes2)

        #refine
        self.convr0 = nn.Conv2d(in_channels=self.vgg_channels[5], out_channels=self.refine_channel, kernel_size=self.refine_kernel, padding = (self.refine_kernel-1)//2)
        self.convr1 = nn.Conv2d(in_channels=self.refine_channel, out_channels=self.refine_channel//2, kernel_size=self.refine_kernel, padding = (self.refine_kernel-1)//2)
        self.convr2 = nn.Conv2d(in_channels=self.refine_channel//2, out_channels=self.refine_channel//2, kernel_size=self.refine_kernel, padding = (self.refine_kernel-1)//2)

        self.convr3 = nn.Conv2d(in_channels=self.refine_channel//2, out_channels=1, kernel_size=self.refine_kernel, padding = (self.refine_kernel-1)//2)
        self.act_fn = nn.LeakyReLU(0.1)

        #bypass unet
        self.ds0 = PoolConv(1, self.vgg_channels[0])
        self.vgg0 = VGGBlock(self.vgg_channels[0])
        self.ds1 = PoolConv(self.vgg_channels[0], self.vgg_channels[1])
        self.vgg1 = VGGBlock(self.vgg_channels[1])
        self.ds2 = PoolConv(self.vgg_channels[1], self.vgg_channels[2])
        self.vgg2 = VGGBlock(self.vgg_channels[2])   #250
        self.us3 = UpConv(self.vgg_channels[2]+self.width, self.vgg_channels[3]) #merge with fno output
        self.vgg3 = VGGBlock(self.vgg_channels[3])
        self.us4 = UpConv(self.vgg_channels[3]+self.vgg_channels[1], self.vgg_channels[4])
        self.vgg4 = VGGBlock(self.vgg_channels[4])
        self.us5 = UpConv(self.vgg_channels[4]+self.vgg_channels[0], self.vgg_channels[5])
        self.vgg5 = VGGBlock(self.vgg_channels[5])


        self.tanh = nn.Tanh()


    def forward(self, x):

        #fno pass
        x_fno = self.resize0(x) 
        x_fno = self.fno(x_fno)       
        x_fno = self.act_fn(x_fno) 
        #unet pass
        x_unet = self.ds0(x)
        x_unet_1000 = self.vgg0(x_unet)
        x_unet = self.ds1(x_unet_1000)
        x_unet_500 = self.vgg1(x_unet)
        x_unet = self.ds2(x_unet_500)
        x_unet_250 = self.vgg2(x_unet)

        #merge fno and unet
        x = torch.cat((x_fno, x_unet_250), 1)

        #dconv
        x = self.us3(x)
        x = torch.cat((self.vgg3(x),x_unet_500), 1)
        x = self.us4(x)
        x = torch.cat((self.vgg4(x),x_unet_1000), 1)
        x = self.us5(x)
        x = self.vgg5(x)
        #refine
        x = self.convr0(x)
        x = self.act_fn(x)
        x = self.convr1(x)
        x = self.act_fn(x)
        x = self.convr2(x)
        x = self.act_fn(x)
        x = self.convr3(x)


        return self.tanh(x)





class oinnopc_large(nn.Module): #one fno unit only
    def __init__(self, modes1, modes2,  width, in_channel=1, refine_channel=32, refine_kernel = 3, smooth_kernel = 3):
        super(oinnopc_large, self).__init__()

        '''Support larger 8kX8k tile input'''

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.refine_kernel = refine_kernel
        self.in_channel = in_channel
        self.refine_channel = refine_channel
        self.smooth_kernel = smooth_kernel
        #self.cemap = cemap
        self.vgg_channels = [4,8,16,16,8,4]

        #resize
        self.resize0 = nn.AvgPool2d(8)
        #fourier
        self.fno = SpectralConv2dLiftChannel(self.in_channel, self.width, self.width, self.modes1, self.modes2)

        #refine
        self.convr0 = nn.Conv2d(in_channels=self.vgg_channels[5], out_channels=self.refine_channel, kernel_size=self.refine_kernel, padding = (self.refine_kernel-1)//2)
        self.convr1 = nn.Conv2d(in_channels=self.refine_channel, out_channels=self.refine_channel//2, kernel_size=self.refine_kernel, padding = (self.refine_kernel-1)//2)
        self.convr2 = nn.Conv2d(in_channels=self.refine_channel//2, out_channels=self.refine_channel//2, kernel_size=self.refine_kernel, padding = (self.refine_kernel-1)//2)

        self.convr3 = nn.Conv2d(in_channels=self.refine_channel//2, out_channels=1, kernel_size=self.refine_kernel, padding = (self.refine_kernel-1)//2)
        self.act_fn = nn.LeakyReLU(0.1)

        #bypass unet
        self.ds0 = PoolConv(1, self.vgg_channels[0])
        self.vgg0 = VGGBlock(self.vgg_channels[0])
        self.ds1 = PoolConv(self.vgg_channels[0], self.vgg_channels[1])
        self.vgg1 = VGGBlock(self.vgg_channels[1])
        self.ds2 = PoolConv(self.vgg_channels[1], self.vgg_channels[2])
        self.vgg2 = VGGBlock(self.vgg_channels[2])   #250
        self.us3 = UpConv(self.vgg_channels[2]+self.width, self.vgg_channels[3]) #merge with fno output
        self.vgg3 = VGGBlock(self.vgg_channels[3])
        self.us4 = UpConv(self.vgg_channels[3]+self.vgg_channels[1], self.vgg_channels[4])
        self.vgg4 = VGGBlock(self.vgg_channels[4])
        self.us5 = UpConv(self.vgg_channels[4]+self.vgg_channels[0], self.vgg_channels[5])
        self.vgg5 = VGGBlock(self.vgg_channels[5])

        #S-G smoothing
        #self.conv_smooth = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=self.smooth_kernel, padding = (self.smooth_kernel-1)//2, bias=False) 
        #self.conv_smooth.weight = nn.Parameter(torch.tensor(data=np.expand_dims(np.expand_dims(sg.get_2D_filter(self.smooth_kernel,1,0),0),0), dtype=torch.float), requires_grad=False)
        ##self.bn0 = nn.BatchNorm2d(self.refine_channel)
        ##self.bn1 = nn.BatchNorm2d(self.refine_channel//2)
        ##self.bn2 = nn.BatchNorm2d(self.refine_channel//2)
        ##self.bn1 = nn.BatchNorm2d(self.refine_channel)
        ##self.fc1 = nn.Linear(self.width, 128)
        ##self.fc2 = nn.Linear(128, 1)
        self.tanh = nn.Tanh()


        #attentions
        #self.attention0 = attention_block(self.refine_channel)
        #self.attention1 = spatial_attention(self.refine_channel//2)
        #self.attention2 = spatial_attention(self.refine_channel//2)
    def forward(self, x):

        #fno pass


        x_fno = self.resize0(x) 
        #print(x_fno.shape)
        #fno_size = x_fno.shape[-1]
        x_fnos=torch.zeros_like(x_fno).cuda().repeat(1, self.vgg_channels[2], 1, 1)

        for i in range(7):
            for j in range(7):
                tmpfno = self.fno(x_fno[:,:,i*128:i*128+256, j*128:j*128+256])
                tmpfno = self.act_fn(tmpfno)
                #print(tmpfno.shape)
                x_fnos[:,:,i*128+64:i*128+192,j*128+64:j*128+192] = tmpfno[:,:,64:192,64:192]

        #x_fno = self.fno(x_fno)       
        #x_fno = self.act_fn(x_fno) 
        #unet pass
        x_unet = self.ds0(x)
        x_unet_1000 = self.vgg0(x_unet)
        x_unet = self.ds1(x_unet_1000)
        x_unet_500 = self.vgg1(x_unet)
        x_unet = self.ds2(x_unet_500)
        x_unet_250 = self.vgg2(x_unet)
        #print(x_unet_250.shape)
        #merge fno and unet
        x = torch.cat((x_fnos, x_unet_250), 1)
        #print(x.shape)
        #dconv
        x = self.us3(x)
        x = torch.cat((self.vgg3(x),x_unet_500), 1)
        x = self.us4(x)
        x = torch.cat((self.vgg4(x),x_unet_1000), 1)
        x = self.us5(x)
        x = self.vgg5(x)
        #refine
        x = self.convr0(x)
        x = self.act_fn(x)
        #x = self.attention0(x)
        x = self.convr1(x)
        x = self.act_fn(x)
        #x = self.attention1(x)
        x = self.convr2(x)
        x = self.act_fn(x)
        #x = self.attention2(x)
        x = self.convr3(x)
        #x = self.conv_smooth(x)

        return self.tanh(x)


