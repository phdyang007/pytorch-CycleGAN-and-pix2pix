import torch
import itertools
from .base_model import BaseModel

import tempfile
import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
import legacy
from torch_utils import custom_ops
from torch.distributions.normal import Normal

def uniform_quantize(k, gradient_clip=False):
    class qfn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            if k == 32:
                out = input
            elif k == 1:
                out = torch.sign(input)
            else:
                n = float(2 ** k - 1)
                out = torch.round(input * n) / n
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            if gradient_clip:
                grad_input.clamp_(-1, 1)
            return grad_input

    return qfn().apply

class Noise:
    def __init__(self, block_resolutions, device='cpu'):
        self.block_resolutions = block_resolutions
        self.num = (1, )  + (2, ) * (len(self.block_resolutions) - 1)
        self.device = device
        self.size = sum([(self.block_resolutions[i] ** 2) * self.num[i] for i in range(len(self.num))])
    def generate(self, batch=1, input=None):
        if input == None:
            noise = torch.randn(batch*self.size, device=self.device)
        else:
            noise = input
            assert noise.shape == (batch*self.size, )
        block_noise = {}
        idx = 0
        for num, res in zip(self.num, self.block_resolutions):
            block_noise[f'b{res}'] = []
            for _ in range(num):
                length = batch * res * res
                cur = noise[idx: idx+length].reshape(-1, 1, res, res)
                idx += length
                block_noise[f'b{res}'].append(cur)
        return noise, block_noise
    def transform_noise(self, noise, batch=1):
        block_noise = {}
        idx = 0
        for num, res in zip(self.num, self.block_resolutions):
            block_noise[f'b{res}'] = []
            for _ in range(num):
                length = batch * res * res
                cur = noise[idx: idx+length].reshape(-1, 1, res, res)
                idx += length
                block_noise[f'b{res}'].append(cur)
        return block_noise

class StyleGANModel(BaseModel):
    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.batch = opt.batch_size 
        if opt.rank_buffer_size % self.batch == 0:
            self.generate_num = opt.rank_buffer_size // self.batch
        else:
            self.generate_num = opt.rank_buffer_size // self.batch + 1
        self.upsampler = torch.nn.Upsample(scale_factor=8, mode='bicubic')
        self.seed = 0
        
    def init(self, G_kwargs, D_kwargs):
        # setup netG/netD
        common_kwargs = dict(c_dim=0, img_resolution=256, img_channels=1)
        # keep these models on cpu a copy?
        self.netG = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False) # subclass of torch.nn.Module
        self.netD = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False) # subclass of torch.nn.Module
        # load models from pkl file
        with dnnlib.util.open_url(self.opt.gan_model) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', self.netG), ('D', self.netD)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
            
    def forward(x):
        return x
    
    def optimize_parameters():
        pass

    def set_input(x):
        return
            
    def finetune(self, args):
        torch.multiprocessing.set_start_method('spawn', force=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            if args.num_gpus == 1:
                self.subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
            else:
                torch.multiprocessing.spawn(fn=self.subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)
    
    def save_model(self, i, args):
        save_filename = 'stylegan_%d.pkl' % i
        save_path = os.path.join(self.save_dir, save_filename)
        snapshot_data = dict(training_set_kwargs=dict(args.training_set_kwargs))
        for name, module in [('G', self.netG), ('D', self.netD)]:
            module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
            snapshot_data[name] = module
            del module # conserve memory
        with open(save_path, 'wb') as f:
             pickle.dump(snapshot_data, f)

    def subprocess_fn(self, rank, args, temp_dir):
        # Init torch.distributed.
        if args.num_gpus > 1:
            init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
            if os.name == 'nt':
                init_method = 'file:///' + init_file.replace('\\', '/')
                torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
            else:
                init_method = f'file://{init_file}'
                torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)
        # Init torch_utils.
        if rank != 0:
            custom_ops.verbosity = 'none'
        # Execute training loop.
        self.training_loop(rank=rank, **args)
    
    def training_loop(
        self,
        run_dir                 = '.',      # Output directory.
        training_set_kwargs     = {},       # Options for training set.
        data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
        G_kwargs                = {},       # Options for generator network.
        D_kwargs                = {},       # Options for discriminator network.
        G_opt_kwargs            = {},       # Options for generator optimizer.
        D_opt_kwargs            = {},       # Options for discriminator optimizer.
        loss_kwargs             = {},       # Options for loss function.
        random_seed             = 0,        # Global random seed.
        num_gpus                = 1,        # Number of GPUs participating in the training.
        rank                    = 0,        # Rank of the current process in [0, num_gpus[.
        batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
        batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
        G_reg_interval          = None,        # How often to perform regularization for G? None = disable lazy regularization.
        D_reg_interval          = None,       # How often to perform regularization for D? None = disable lazy regularization.
        total_kimg              = 6,    # Total length of the training, measured in thousands of real images.
        cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
        allow_tf32              = False,    # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
        abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
        progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    ):
 
        device = torch.device('cuda', rank)
        np.random.seed(random_seed * num_gpus + rank)
        torch.manual_seed(random_seed * num_gpus + rank)
        torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
        torch.backends.cudnn.allow_tf32 = allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
        conv2d_gradfix.enabled = True                       # Improves training speed.
        grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

        # Load training set.
        training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
        training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
        training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))

        # Construct networks.
        common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
        G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module

        # Resume from existing.
        misc.copy_params_and_buffers(self.netG, G, require_all=False)
        misc.copy_params_and_buffers(self.netD, D, require_all=False)  

        # Distribute across GPUs.
        ddp_modules = dict()
        for name, module in [('G_mapping', G.mapping), ('G_synthesis', G.synthesis), ('D', D)]:
            if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
                module.requires_grad_(True)
                module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
                module.requires_grad_(False)
            if name is not None:
                ddp_modules[name] = module

        # Setup training phases.
        loss = dnnlib.util.construct_class_by_name(device=device, **ddp_modules, **loss_kwargs) # subclass of training.loss.Loss
        phases = []
        for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
            if reg_interval is None:
                opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
                phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
            else: # Lazy regularization.
                mb_ratio = reg_interval / (reg_interval + 1)
                opt_kwargs = dnnlib.EasyDict(opt_kwargs)
                opt_kwargs.lr = opt_kwargs.lr * mb_ratio
                opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
                opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
                phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
                phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
        for phase in phases:
            phase.start_event = None
            phase.end_event = None
            if rank == 0:
                phase.start_event = torch.cuda.Event(enable_timing=True)
                phase.end_event = torch.cuda.Event(enable_timing=True)
                
        # Train.
        cur_nimg = 0
        cur_tick = 0
        tick_start_nimg = cur_nimg
        batch_idx = 0
        if progress_fn is not None:
            progress_fn(0, total_kimg)
        while True:

            # Fetch training data.
            with torch.autograd.profiler.record_function('data_fetch'):
                phase_real_img, phase_real_c = next(training_set_iterator)
                phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
                phase_real_c = phase_real_c.to(device).split(batch_gpu)
                all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
                all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
                all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
                all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
                all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

            # Execute training phases.
            for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
                if batch_idx % phase.interval != 0:
                    continue

                # Initialize gradient accumulation.
                if phase.start_event is not None:
                    phase.start_event.record(torch.cuda.current_stream(device))
                phase.opt.zero_grad(set_to_none=True)
                phase.module.requires_grad_(True)

                # Accumulate gradients over multiple rounds.
                for round_idx, (real_img, real_c, gen_z, gen_c) in enumerate(zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c)):
                    sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                    gain = phase.interval
                    loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, sync=sync, gain=gain)

                # Update weights.
                phase.module.requires_grad_(False)
                with torch.autograd.profiler.record_function(phase.name + '_opt'):
                    for param in phase.module.parameters():
                        if param.grad is not None:
                            misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                    phase.opt.step()
                if phase.end_event is not None:
                    phase.end_event.record(torch.cuda.current_stream(device))

            # Update state.
            cur_nimg += batch_size
            batch_idx += 1


            # Perform maintenance tasks once per tick.
            done = (cur_nimg >= total_kimg * 1000)

            # Check for abort.
            if (not done) and (abort_fn is not None) and abort_fn():
                done = True

            # Update state.
            cur_tick += 1
            tick_start_nimg = cur_nimg
            if done:
                break
        
        # Copy back weights.
        snapshot_data = dict()
        for name, module in [('G', G), ('D', D)]:
            if module is not None:
                if num_gpus > 1:
                    misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
            snapshot_data[name] = module
            del module # conserve memory
        if rank == 0:
            misc.copy_params_and_buffers(snapshot_data['G'], self.netG, require_all=False)
            misc.copy_params_and_buffers(snapshot_data['D'], self.netD, require_all=False)
            
        del G, D

    def write_out_single(self, args):
        PIL.Image.fromarray(args[3], 'L').save(f'{args[0]}/seed{args[1]:04d}_{args[2]:02d}.png')
    
    def write_out(self, args):
        for i in range(self.batch):
            self.write_out_single(args[i])

    #TODO: get batched generation + multi-gpu
    def generate_random(self, outdir, model):
        batch = self.batch
        device = torch.device('cuda', 0)
        seeds = list(range(self.seed, self.seed + self.generate_num)) 
        self.seed += self.generate_num
        results = []
        label = torch.zeros([batch, self.netG.c_dim], device=device)
        self.netG.to(device)
        if len(self.gpu_ids) > 1:
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids) 
        for i, seed in enumerate(seeds):
            z = torch.from_numpy(np.random.RandomState(seed).randn(batch, self.netG.module.z_dim)).to(device)
            img = self.netG(z, label, truncation_psi=1.0, noise_mode='const')
            img = self.upsampler(img)
            img = (img + 1) * 0.5
            model.mask = img
            model.eval()
            with torch.no_grad():
                model.legalize_mask(model.mask)
                model.forward()
                _, iou_fg = model.get_F_criterion(None)
            model.train()
            results.append(iou_fg.item())
            mask_golden = (model.real_resist[:,0,:,:] * 255).to(torch.uint8)
            img_output = (model.mask[:,0,:,:] * 255).to(torch.uint8)
            img_output = torch.cat((img_output,mask_golden), 2)
            img_output = img_output.detach().cpu().numpy()
            args = []
            for b in range(batch):
                args.append( [outdir, i, b, img_output[b,...]] )
            self.write_out(args)
        self.netG.cpu()
        return results
    
    def attack_style(self, z, G, model, device, past_model=None, lr_alpha=0.01, dist_norm=0.1, loss_type='houdini', quantize_aware=True, epochs=10, gradient_clip=True):
        batch = self.batch
        upsampler = torch.nn.Upsample(scale_factor=8, mode='bicubic')
        G.eval(), model.eval()
        label = torch.zeros([batch, G.c_dim], device=device)
        optimizer = torch.optim.Adam([z], lr=lr_alpha)
        z.requires_grad = True
        norm_dist = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
        quantizer = uniform_quantize(k=1, gradient_clip=gradient_clip)
        for i in range(epochs):
            optimizer.zero_grad()
            img = G(z, label, truncation_psi=1.0, noise_mode='const')
            img = upsampler(img)
            if quantize_aware:
                img = quantizer(img)
            else:
                img = model.legalize_mask(img, 0)
            img = (img + 1 ) * 0.5
            if loss_type == 'TOD':
                model.mask = img
                model.forward()
                real_resist = model.simulate(model.mask)
                model_loss = model.criterionLitho(model.real_mask, model.to_one_hot(real_resist))
                past_model.mask = img
                past_model.forward()
                past_model_loss = past_model.criterionLitho(past_model.real_mask, past_model.to_one_hot(real_resist))
                if dist_norm > 1e-5:
                    loss = -torch.abs(model_loss - past_model_loss).mean() - dist_norm * norm_dist.log_prob(z).mean()
                else:
                    loss = -torch.abs(model_loss - past_model_loss).mean()
            else:
                model.mask = img
                if dist_norm > 1e-5:
                    loss = model.forward_uncertainty(loss_type) - dist_norm * norm_dist.log_prob(z).mean()
                else:
                    loss = model.forward_uncertainty(loss_type)
            loss.backward()
            optimizer.step()
        return img
    
    def attack_style_loop(self, outdir, model, past_model=None): 
        batch = self.batch
        device = torch.device('cuda', 0)
        seeds = list(range(self.seed, self.seed + self.generate_num)) 
        self.seed += self.generate_num
        results = []
        label = torch.zeros([batch, self.netG.c_dim], device=device)
        self.netG.to(device)
        if len(self.gpu_ids) > 1:
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids) 
        for i, seed in enumerate(seeds):
            z = torch.from_numpy(np.random.RandomState(seed).randn(batch, self.netG.module.z_dim)).to(device)
            img = self.attack_style(z, self.netG, model, device=device, loss_type=self.opt.adv_loss_type, past_model=past_model) 
            model.mask = img
            model.eval()
            with torch.no_grad():
                model.legalize_mask(model.mask)
                model.forward()
                _, iou_fg = model.get_F_criterion(None)
            model.train()
            results.append(iou_fg.item())
            mask_golden = (model.real_resist[:,0,:,:] * 255).to(torch.uint8)
            img_output = (model.mask[:,0,:,:] * 255).to(torch.uint8)
            img_output = torch.cat((img_output,mask_golden), 2)
            img_output = img_output.detach().cpu().numpy()
            args = []
            for b in range(batch):
                args.append( [outdir, i, b, img_output[b,...]] )
            self.write_out(args)
        self.netG.cpu()
        return results
    
    def attack_noise(self, z, G, model, device, past_model=None, lr_alpha=1.0, dist_norm=0.1, quantize_aware=True, epochs=10, gradient_clip=True, loss_type='pixel'):
        batch = self.batch
        noise_module = Noise(G.synthesis.block_resolutions, device)
        noise, noise_block = noise_module.generate(batch)
        upsampler = torch.nn.Upsample(scale_factor=8, mode='bicubic')
        G.eval(), model.eval()
        label = torch.zeros([batch, G.c_dim], device=device)
        optimizer = torch.optim.Adam([noise], lr=lr_alpha)
        noise.requires_grad = True
        original = None
        quantizer = uniform_quantize(k=1, gradient_clip=gradient_clip)
        norm_dist = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
        for i in range(epochs):
            optimizer.zero_grad()
            noise_block = noise_module.transform_noise(noise)
            img = G(z, label, truncation_psi=1.0, noise_mode='random', input_noise=noise_block)
            img = upsampler(img)
            if quantize_aware:
                img = quantizer(img)
            else:
                img = model.legalize_mask(img, 0)
            img = (img + 1) * 0.5
            if loss_type == 'TOD':
                model.mask = img
                model.forward()
                real_resist = model.simulate(model.mask)
                model_loss = model.criterionLitho(model.real_mask, model.to_one_hot(real_resist))
                past_model.mask = img
                past_model.forward()
                past_model_loss = past_model.criterionLitho(past_model.real_mask, past_model.to_one_hot(real_resist))
                if dist_norm > 1e-5:
                    loss = -torch.abs(model_loss - past_model_loss).mean() - dist_norm * norm_dist.log_prob(noise).mean()
                else:
                    loss = -torch.abs(model_loss - past_model_loss).mean()
            else:
                model.mask = img
                if loss_type == 'pixel':
                    if dist_norm > 1e-5:
                        loss = -model.forward_attack(original) - dist_norm * norm_dist.log_prob(noise).mean()
                    else:
                        loss = -model.forward_attack(original)
                else:
                    if dist_norm > 1e-5:
                        loss = model.forward_uncertainty(loss_type) - dist_norm * norm_dist.log_prob(noise).mean()
                    else:
                        loss = model.forward_uncertainty(loss_type)
                if original is None:
                    original = model.real_resist.detach()
            loss.backward()
            optimizer.step()
        return img
    
    def attack_noise_loop(self, outdir, model, past_model=None): 
        batch = self.batch
        device = torch.device('cuda', 0)
        seeds = list(range(self.seed, self.seed + self.generate_num)) 
        self.seed += self.generate_num
        results = []
        label = torch.zeros([batch, self.netG.c_dim], device=device)
        self.netG.to(device)
        if len(self.gpu_ids) > 1:
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids) 
        for i, seed in enumerate(seeds):
            z = torch.from_numpy(np.random.RandomState(seed).randn(batch, self.netG.module.z_dim)).to(device)
            img = self.attack_noise(z, self.netG, model, device, loss_type=self.opt.adv_loss_type, past_model=past_model)
            model.mask = img
            model.eval()
            with torch.no_grad():
                model.legalize_mask(model.mask)
                model.forward()
                _, iou_fg = model.get_F_criterion(None)
            model.train()
            results.append(iou_fg.item())
            img_output = (model.mask[:,0,:,:] * 255).to(torch.uint8)
            mask_golden = (model.real_resist[:,0,:,:] * 255).to(torch.uint8)
            img_output = torch.cat((img_output,mask_golden), 2)
            img_output = img_output.detach().cpu().numpy()
            args = []
            for b in range(batch):
                args.append( [outdir, i, b, img_output[b,...]] )
            self.write_out(args)
        self.netG.cpu()
        return results
    
    def generate_data(self, outdir, model, method, past_model=None):
        if method == 'adv_style':
            return self.attack_style_loop(outdir, model, past_model)
        if method == 'adv_noise':
            return self.attack_noise_loop(outdir, model, past_model)
        if method == 'random':
            return self.generate_random(outdir, model)
        assert False, "{} not supported".format(method)
        
