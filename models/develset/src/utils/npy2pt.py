'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2021-02-20 12:48:28
LastEditTime: 2021-02-20 13:00:08
Contact: cgjhaha@qq.com
Description: This script transfer the npy kernel to the torch pt kernels
'''
import torch
import numpy as np
from pathlib import Path

np_path = '/research/d2/xfyao/guojin/develset_opc/levelset_net/kernels/np'
np_path = Path(np_path)

for k in np_path.glob('*.npy'):
    print(k.stem)
    k_name = k.stem
    k_np = np.load(str(k))
    if k_np.dtype == np.complex64:
        k_t_real = torch.from_numpy(k_np.real)
        k_t_imag = torch.from_numpy(k_np.imag)
        k_t = torch.complex(k_t_real, k_t_imag)
    else:
        k_t = torch.from_numpy(k_np)
    k_t_path = k.parent.parent / 'torch' / '{}.pt'.format(k_name)
    torch.save(k_t, str(k_t_path))
