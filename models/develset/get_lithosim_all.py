import os 
import sys 
sys.path.append("/home/haoyuy/scratch_code/projects/pytorch-CycleGAN-and-pix2pix/")
from models.develset.src.models.const import *
from models.develset.src.models.kernels import Kernel
from models.develset.src.metrics.metrics import Metrics
import torchvision.transforms as transforms
from models.develset.src.models.litho_layer import CUDA_LITHO
from models.develset.lithosim import *
import numpy as np 

def lithosim(target, mask):
    kernel = Kernel()
    cl = CUDA_LITHO(kernel)
    metric = []
    nominal_aerial, nominal = cl.simulateImageOpt(mask, LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
    outer_aerial, outer = cl.simulateImageOpt(mask, LITHO_KERNEL_FOCUS, MAX_DOSE)
    inner_aerial, inner = cl.simulateImageOpt(mask, LITHO_KERNEL_DEFOCUS, MIN_DOSE)
    aerials = [nominal_aerial, inner_aerial, outer_aerial]
    m = Metrics(target, aerials)
    l2, pvb = m.get_all()
    print(l2, pvb)
    return l2, pvb
    #nom_save_path = f'./test_nom_pvb_{pvb}_l2_{l2}.png'
    #torchvision.utils.save_image(m.nominal[0], nom_save_path)
    #torchvision.utils.save_image(m.target[0], './target.png')

def litho_trans(img_path):
    im = Image.open(str(img_path)).convert('L')
    #print(max(im))
    trans = transforms.Compose([transforms.ToTensor()])
    im = trans(im)
    im = im.unsqueeze(0)
    #im = torch.nn.ZeroPad2d(24)(im)
    im = im.cuda()
    print(im.shape, torch.max(im))
    return im


img_count = 100
max_iter  = 24


l2=np.zeros((img_count,max_iter))
pvb=np.zeros((img_count,max_iter))


import os
for dirname, dirnames, filenames in os.walk("/home/haoyuy/scratch_code/projects/via/v5data/viav0test/via_test"):
    for j in range(len(dirnames)):
        tgt_path = os.path.join(dirname, dirnames[j], 'targetImg.png')
        tgt_img = litho_trans(tgt_path)
        for k in range(max_iter):
            mask_path = os.path.join(dirname, dirnames[j], 'mask%g.png'%(k+1))
            mask_img = litho_trans(mask_path)
            _l2, _pvb = lithosim(tgt_img, mask_img)
            l2[j,k] = _l2
            pvb[j,k]= _pvb





np.savetxt('l2.csv',l2,delimiter=',')
np.savetxt('pvb.csv',pvb,delimiter=',')

