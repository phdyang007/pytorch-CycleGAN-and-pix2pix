'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2020-12-09 19:48:41
LastEditTime: 2020-12-19 18:44:56
Contact: cgjhaha@qq.com
Description: convert the txt kernel to the npy kernel
'''

'''
output optkernel 0 scale 86.943428
output optkernel 1 scale 83.156715
output optkernel 2 scale 86.943428
output optkernel 3 scale 83.156715
'''
'''
output comboOptKernels 0 scale 1.000000
output comboOptKernels 1 scale 1.000000
output comboOptKernels 2 scale 1.000000
output comboOptKernels 3 scale 1.000000
'''

import os
import numpy as np

def parse_kernel():
    txt_dir = '/Users/dekura/chen/bei/cuilt/cu-ilt/build'
    # kernel_name = 'optKernel'
    kernel_name = 'comboOptKernel'
    # kernel_name = 'optKernel_scale'
    save_path = '../kernels/{}.npy'.format(kernel_name)
    print(kernel_name)
    if kernel_name == 'optKernel':
        kernel_num = 24
        kernel_cplx = np.zeros((4, kernel_num, 35, 35), dtype=np.complex64)
    elif kernel_name == 'comboOptKernel':
        kernel_num = 24
        kernel_cplx = np.zeros((4, kernel_num, 35, 35), dtype=np.complex64)
    elif kernel_name == 'optKernel_scale':
        kernel_num = 1
        kernel_cplx = np.zeros((4, 24), dtype=np.float32)
    for t in range(4):
        for i in range(kernel_num):
            print('type {} kernel {}'.format(t,i))
            # cao haiyou scale
            real_path = os.path.join(txt_dir, '{}_nku{}_t{}_0r.txt'.format(kernel_name, i+1, t, i))
            imag_path = os.path.join(txt_dir, '{}_nku{}_t{}_0i.txt'.format(kernel_name, i+1, t, i))
            real_np = np.loadtxt(real_path, dtype=np.float32)
            imag_np = np.loadtxt(imag_path, dtype=np.float32)
            kernel_cplx[t][i].real = real_np
            kernel_cplx[t][i].imag = imag_np
            # print(real_np.shape)
    print(kernel_cplx[0][0][12])
    np.save(save_path, kernel_cplx)

def parse_scale():
    txt_dir = '/Users/dekura/chen/bei/cuilt/cu-ilt/build'
    kernel_name = 'optKernel_scale'
    save_path = '../kernels/{}.npy'.format(kernel_name)
    print(kernel_name)
    scale_array = np.zeros((4, 24), dtype=np.float32)
    for t in range(4):
        print('type {} kernel scale'.format(t))
        # cao haiyou scale
        real_path = os.path.join(txt_dir, '{}{}.txt'.format(kernel_name,t))
        real_np = np.loadtxt(real_path, dtype=np.float32)
        scale_array[t] = real_np
    print(scale_array)
    np.save(save_path, scale_array)

if __name__ == '__main__':
    parse_kernel()
    # parse_scale()

# inner_scales = np.loadtxt('./inner_scales.txt')
# np.save('inner_scales.npy', inner_scales)
# outer_scales = np.loadtxt('./outer_scales.txt')
# np.save('outer_scales.npy', outer_scales)


# TODO:
# process combokernel
# process kernel scales
# since the precompute process have been done, whether the kernel may be updated?
# updated the kernel to get better results, this may be a way