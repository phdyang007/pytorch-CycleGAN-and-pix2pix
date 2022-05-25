import os
from data.base_dataset import BaseDataset, get_params, get_transform, get_resize_transform
from data.image_folder import make_dataset
from PIL import Image

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
    
class DataGenerator():
    def __init__(self, opt):
        self.opt = opt
        self.dataroot = opt.augroot
        self.size = opt.augsize # augment dataset by size every iteration
        self.device = torch.device('cuda')
        self.mode = opt.augmode #'rand', 'adv_style', 'adv_noise', 'none'
        self.iter = 0
        with dnnlib.util.open_url(opt.gan_model) as f:
            self.generator = legacy.load_network_pkl(f)['G_ema'].to(self.device) # type: ignore
        self.label = torch.zeros([1, self.generator.c_dim], device=device)
        self.upsampler = torch.nn.Upsample(scale_factor=8, mode='bicubic')
        if self.mode == 'adv_noise':
            self.noise_module = Noise(self.generator.synthesis.block_resolutions, self.device)    
        self.norm_dist = torch.distributions.normal.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
        self.model = None
    def set_model(model):
        self.model = model
    def generate():
        # model: downstream litho_gan model with newest weight
        outpath = self.dataroot + "_" + str(self.iter)
        os.makedirs(outpath, exist_ok=True)
        # generate deterministic seeds on iteration
        seeds = [self.iter * self.size + i for i in range(self.size)]
        for i, seed in enumerate(seeds):
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.generator.z_dim)).to(device)
            if self.mode == 'adv_noise':
                img = self.attack_noise(z, self.model)
            elif self.mode == 'adv_style':
                img = self.attack_style(z, self.model)
            else:
                img = self.upsampler(self.generator(z, self.label, truncation_psi=1.0, noise_mode='const'))
                img = (img.clamp(-1, 1) + 1) * 0.5
            img = (img[0,0,:,:] * 255).to(torch.uint8)
            PIL.Image.fromarray(img.detach().cpu().numpy(), 'L').save(f'{outpath}/seed{seed:04d}.png')
        self.iter += 1
        return outpath
        
    def attack_noise(z, model, alpha=100, lr=0.01, epochs=100):
        noise, noise_block = self.noise_module.generate()
        optimizer = torch.optim.Adam([noise], lr=lr)
        noise.requires_grad = True
        original = None
        for i in range(epochs):
            optimizer.zero_grad()
            noise_block = noise_module.transform_noise(noise)
            img = self.generator(z, self.label, truncation_psi=1.0, noise_mode='random', input_noise=noise_block)
            img = self.upsampler(img)
            img = (img.clamp(-1, 1) + 1) * 0.5
            model.real_high_res = img
            loss = -model.forward_attack(original) - alpha * self.norm_dist.log_prob(noise).mean()
            if original is None:
                original = model.real_mask_img.detach()
            loss.backward()
            optimizer.step()
        return img
    
    def attack_style(z, model, alpha=100, lr=0.01, epochs=100):
        optimizer = torch.optim.Adam([z], lr=lr)
        z.requires_grad = True
        for i in range(epochs):
            optimizer.zero_grad()
            img = self.generator(z, label, truncation_psi=1.0, noise_mode='const')
            img = self.upsampler(img)
            img = (img.clamp(-1, 1) + 1) * 0.5
            model.real_high_res = img
            loss = model.forward_uncertainty() - alpha * self.norm_dist.log_prob(z).mean()
            loss.backward()
            optimizer.step()
        return img
    
class LithoGANDataset(BaseDataset):
    """A dataset class for 3 paired image dataset.
    A: design
    B: mask
    C: resist
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B,C} 
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.high_res = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        if opt.augmode != 'none':
            self.generator = DataGenerator(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        Image.MAX_IMAGE_PIXELS = 301326592
        # read a image given a random integer index
        high_res_path = self.high_res[index]
        high_res = Image.open(high_res_path).convert('L')
        high_res_transform = get_resize_transform(self.opt, grayscale=True, convert=True, resize=False)
        low_res_transform = get_resize_transform(self.opt, grayscale=True, convert=True, resize=True)
        real_high_res = high_res_transform(high_res)
        real_low_res = low_res_transform(high_res)
        return {'real_high_res': real_high_res, 'real_low_res': real_low_res, 'image_paths':high_res_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.high_res)

    def augment(self, model):
        self.generator.set_model(model)
        outpath = self.generator.generate()
        new_data = sorted(make_dataset(outpath))
        self.high_res += new_data