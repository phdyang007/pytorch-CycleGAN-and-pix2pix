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


class StyleGANModel(BaseModel):
    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.generate_num = opt.rank_buffer_size 
        self.upsampler = torch.nn.Upsample(scale_factor=8, mode='bicubic')
        
    def init(self, G_kwargs, D_kwargs)
        # setup netG/netD
        common_kwargs = dict(c_dim=1, img_resolution=256, img_channels=1)
        self.netG = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        self.netD = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        # load models from pkl file
        with dnnlib.util.open_url(self.opt.stylegan_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', self.netG), ('D', self.netD)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
            
    def finetune(self, args):
        torch.multiprocessing.set_start_method('spawn')
        with tempfile.TemporaryDirectory() as temp_dir:
            if args.num_gpus == 1:
                self.subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
            else:
                torch.multiprocessing.spawn(fn=self.subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

    def subprocess_fn(rank, args, temp_dir):
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
        self.training_loop.training_loop(rank=rank, **args)
    
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

        #TODO: get batched generation + multi-gpu
        def generate_random(self, outdir, model):
            seeds = list(range(self.generate_num)) 
            results = []
            label = torch.zeros([1, self.G.c_dim], device=device)
            for i, seed in enumerate(seeds):
                z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.G.z_dim)).to(device)
                img = self.G(z, label, truncation_psi=truncation_psi, noise_mode='const')
                img = self.upsampler(img)
                img = (img + 1) * 0.5
                model.mask = img
                model.legalize_mask(model.mask)
                model.forward()
                _, iou_fg = model.get_F_criterion(None)
                results.append(iou_fg)
                mask_golden = (model.real_resist[0,0,:,:] * 255).to(torch.uint8)
                img_output = (img[0,0,:,:] * 255).to(torch.uint8)
                img_output = torch.cat((img_output,mask_golden), 1)
                PIL.Image.fromarray(img_output.detach().cpu().numpy(), 'L').save(f'{outdir}/seed{i:04d}.png')
            return results