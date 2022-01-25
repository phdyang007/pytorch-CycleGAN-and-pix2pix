'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2020-12-02 19:08:15
LastEditTime: 2021-04-09 16:42:07
Contact: cgjhaha@qq.com
Description: the cuda litho simulation using pytorch and cupy
'''



import logging
import torch
import cupy as np
from src.models.kernels import Kernel
from src.models.const import *
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
from cupy._core.dlpack import toDlpack
from cupy._core.dlpack import fromDlpack
from src.utils.debug_tools import debug_tensor
class CUDA_LITHO:
    def __init__(self, kernel):
        self.kernel = kernel

    # kernel_level = num_kernel_used
    def simulateImageOpt(self, mask, kernel_type, dose):
        kernels = self.kernel.optKernels
        kernels = {
            'kernel_head': kernels['kernel_head'][kernel_type],
            'kernel_scale': kernels['kernel_scale'][kernel_type]
        }
        I, Z = self.computeImage(kernels, mask, dose)
        return I, Z

    def img2cplx(self, img, dose):
        img_cplx = torch.complex(img*dose, torch.zeros(img.shape, dtype=torch.float32).cuda())
        return img_cplx

    def tensor2cupy_cpx(self, t):
        if t.is_cuda:
            t_real = t.real
            t_imag = t.imag
            c_real = fromDlpack(to_dlpack(t_real.contiguous()))
            c_imag = fromDlpack(to_dlpack(t_imag.contiguous()))
            c_cpx = np.empty(c_real.shape, dtype=np.complex64)
            c_cpx.real = c_real
            c_cpx.imag = c_imag
            return c_cpx
        else:
            raise RuntimeError(f'tensor need to be on gpu.')

    def cupy2tensor_cpx(self, c):
        if type(c) is np.ndarray:
            c_real = c.real
            c_imag = c.imag
            t_real = from_dlpack(toDlpack(c_real))
            t_imag = from_dlpack(toDlpack(c_imag))
            t_cpx = torch.complex(t_real, t_imag).cuda()
            return t_cpx
        else:
            raise RuntimeError(f'not cupy ndarry')

    def computeImage(self, kernels, mask, dose):
        mask_cplx = self.img2cplx(mask, dose)
        mask_cplx = self.tensor2cupy_cpx(mask_cplx)
        mask_cplx = np.fft.fftshift(np.fft.fft2(mask_cplx))
        kernel_nums, kernel_w, kernel_h = kernels['kernel_head'].shape
        km_cplx = np.zeros((mask_cplx.shape[0], kernel_nums, mask_cplx.shape[2], mask_cplx.shape[3]), dtype=np.complex64)
        imxh = IMAGE_HEIGHT_WIDTH//2
        imyh = IMAGE_HEIGHT_WIDTH//2
        xoff = imxh - kernel_w//2
        yoff = imyh - kernel_h//2
        km_cplx[:, :, xoff: xoff+kernel_w, yoff: yoff+kernel_h] = (
        mask_cplx[:, :, xoff: xoff+kernel_w, yoff: yoff+kernel_h] * kernels['kernel_head'])
        km_cplx = np.fft.ifft2(km_cplx)
        real = km_cplx.real
        imag = km_cplx.imag
        sim_ri = (real*real + imag*imag)
        sim_ri = sim_ri * kernels['kernel_scale']
        I = np.sum(sim_ri, axis=1, keepdims=True)
        # cupy to torch tensor
        I = from_dlpack(toDlpack(I))
        Z = self.sigmoid(I)
        return I, Z

    def sigmoid(self, img, steepness=PHOTORISIST_SIGMOID_STEEPNESS, target_intensity=TARGET_INTENSITY):
        return 1/(1 + torch.exp(-steepness*(img - target_intensity)))

    '''
    return type complex
    # use only one kernel
    # here need to be changed
    # may be a bug: the convolvekernel part need to use only one kernel
    '''
    # def maskfft(self, mask, dose):
    def convolveKernel(self, mask, kernel_type, dose):
        kernels = self.kernel.comboKernels
        kernels = {
            'kernel_head': kernels['kernel_head'][kernel_type],
            'kernel_scale': kernels['kernel_scale'][kernel_type]
        }
        wafer = self.convolveKernelCpx(kernels, mask, dose)
        return wafer

    def convolveKernelCpx(self, kernels, mask, dose):
        if mask.dtype is torch.float32:
            mask_cplx = self.img2cplx(mask, dose)
            mask_cplx = self.tensor2cupy_cpx(mask_cplx)
            mask_cplx = np.fft.fftshift(mask_cplx)
            mask_cplx = np.fft.fftshift(np.fft.fft2(mask_cplx))
        else:
            mask = np.fft.fftshift(mask)
            mask_cplx = np.fft.fftshift(np.fft.fft2(mask))
        kernel_nums, kernel_w, kernel_h = kernels['kernel_head'].shape
        km_cplx = np.zeros((mask_cplx.shape[0], kernel_nums, mask_cplx.shape[2], mask_cplx.shape[3]), dtype=np.complex64)
        imxh = IMAGE_HEIGHT_WIDTH//2
        imyh = IMAGE_HEIGHT_WIDTH//2
        xoff = imxh - kernel_w//2
        yoff = imyh - kernel_h//2
        km_cplx[:, :, xoff: xoff+kernel_w, yoff: yoff+kernel_h] = (
        mask_cplx[:, :, xoff: xoff+kernel_w, yoff: yoff+kernel_h] * kernels['kernel_head'])
        km_cplx = np.fft.fftshift(km_cplx)
        km_cplx = np.fft.fftshift(np.fft.ifft2(km_cplx))
        return km_cplx

    def l4_loss(self, x, y):
        return torch.sum(torch.pow((x - y), 4))

    def l2_loss(self, x, y):
        return torch.sum(torch.pow((x - y), 2))

    def lexp_loss(self, x, y, exp: int = 4):
        return torch.sum(torch.pow((x - y), exp))

'''
here we calculate the ilt loss (target - nominal)^4
indeed we need the epeweight matrix
'''
class ILT_LAYER(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask, target, cl):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        nominal_aerial, nominal = cl.simulateImageOpt(mask, LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
        ilt_loss = cl.l4_loss(nominal, target)
        # epe loss
        # calculate d (Z0-Z^)^4 ==========================
        m_diff_p3 = torch.pow((nominal - target),3) #(Z-Z^)^3
        tc = m_diff_p3 * nominal * (1 - nominal) #(Z-Z^)^3 Z(1-Z)
        gl = tc * cl.convolveKernel(mask, LITHO_KERNEL_FOCUS_CT, NOMINAL_DOSE)
        dl = cl.convolveKernel(gl, LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
        gr = tc * cl.convolveKernel(mask, LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
        dr = cl.convolveKernel(gr, LITHO_KERNEL_FOCUS_CT, NOMINAL_DOSE)
        d1 = dl + dr
        ctx.save_for_backward(mask, cl.cupy2tensor_cpx(d1))
        return ilt_loss, nominal

    @staticmethod
    def backward(ctx, ilt_loss, nominal):
        mask, d1 = ctx.saved_tensors
        constant1 = 4 * PHOTORISIST_SIGMOID_STEEPNESS * MASKRELAX_SIGMOID_STEEPNESS
        gradient = (constant1 * d1.real) * mask * (1 - mask)
        return gradient, None, None


'''
the first kind of pvband loss is (inner - target)^2 + (outer - target)^2
the second kind of pvband loss is (inner - outer)^2

here we calculate the second kind of pvband loss
'''
class PVB_LAYER(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mask, target, cl):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        outer_aerial, outer = cl.simulateImageOpt(mask, LITHO_KERNEL_FOCUS, MAX_DOSE)
        inner_aerial, inner = cl.simulateImageOpt(mask, LITHO_KERNEL_DEFOCUS, MIN_DOSE)
        pvb_loss = cl.l2_loss(inner, outer)

        # calculate for backward
        #    //calculate d (inner-outer)^2 ==========================
        tc_in = inner * (1 - inner)
        gl_in = tc_in * cl.convolveKernel(mask, LITHO_KERNEL_DEFOCUS_CT, MIN_DOSE)
        dl_in = cl.convolveKernel(gl_in, LITHO_KERNEL_DEFOCUS, MIN_DOSE)
        gr_in = tc_in * cl.convolveKernel(mask, LITHO_KERNEL_DEFOCUS, MIN_DOSE)
        dr_in = cl.convolveKernel(gr_in, LITHO_KERNEL_DEFOCUS_CT, MIN_DOSE)
        d_in = dl_in + dr_in

        tc_out = outer * (1 - outer)
        gl_out = tc_out * cl.convolveKernel(mask, LITHO_KERNEL_FOCUS_CT, MAX_DOSE)
        dl_out = cl.convolveKernel(gl_out, LITHO_KERNEL_FOCUS, MAX_DOSE)
        gr_out = tc_out * cl.convolveKernel(mask, LITHO_KERNEL_FOCUS, MAX_DOSE)
        dr_out = cl.convolveKernel(gr_out, LITHO_KERNEL_FOCUS_CT, MAX_DOSE)
        d_out = dl_out + dr_out
        ctx.save_for_backward(mask, cl.cupy2tensor_cpx(d_in), cl.cupy2tensor_cpx(d_out))

        return pvb_loss, inner, outer

    def backward(ctx, pvb_loss, inner, outer):
        mask, d_in, d_out = ctx.saved_tensors
        constant_pvb = PHOTORISIST_SIGMOID_STEEPNESS * MASKRELAX_SIGMOID_STEEPNESS
        gradient = 2*(inner - outer) * ( constant_pvb * (d_in.real - d_out.real) * mask * (1-mask))
        return gradient, None, None


'''
TODO:
    modify the loss function.
    add some loss weight.
    how to reach the best epe and the pvband.
'''
class LEVELSET_LAYER(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask, target, cl, cfg):
        ilt_exp = cfg.model.ilt_exp
        ilt_weight, pvb_weight = cfg.model.loss_weights.ilt_weight, cfg.model.loss_weights.pvb_weight

        nominal_aerial, nominal = cl.simulateImageOpt(mask, LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
        outer_aerial, outer = cl.simulateImageOpt(mask, LITHO_KERNEL_FOCUS, MAX_DOSE)
        inner_aerial, inner = cl.simulateImageOpt(mask, LITHO_KERNEL_DEFOCUS, MIN_DOSE)

        levelset_loss = ilt_weight * cl.lexp_loss(nominal, target, ilt_exp) + pvb_weight * (cl.l2_loss(inner, target) + cl.l2_loss(outer, target))
        # levelset_loss = cl.l2_loss(nominal, target)

        #calculate d (Z0-Z^)^(ilt_exp)========================
        # the lexp loss gradient of (mask - target)
        m_diff_exp = torch.pow((nominal - target), ilt_exp - 1)
        tc_nominal = m_diff_exp * nominal * (1 - nominal)
        gl_nominal = tc_nominal * cl.convolveKernel(mask, LITHO_KERNEL_FOCUS_CT, NOMINAL_DOSE)
        dl_nominal = cl.convolveKernel(gl_nominal, LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
        gr_nominal = tc_nominal * cl.convolveKernel(mask, LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
        dr_nominal = cl.convolveKernel(gr_nominal, LITHO_KERNEL_FOCUS_CT, NOMINAL_DOSE)
        d1_nominal = dl_nominal + dr_nominal

        # calculate d (Z_outer - Z^)^2===================
        # (outer - target)^2
        tc_outer = (outer-target) * outer * (1 - outer)
        gl_outer = tc_outer * cl.convolveKernel(mask, LITHO_KERNEL_FOCUS_CT, MAX_DOSE)
        dl_outer = cl.convolveKernel(gl_outer, LITHO_KERNEL_FOCUS, MAX_DOSE)
        gr_outer = tc_outer * cl.convolveKernel(mask, LITHO_KERNEL_FOCUS, MAX_DOSE)
        dr_outer = cl.convolveKernel(gr_outer, LITHO_KERNEL_FOCUS_CT, MAX_DOSE)
        d1_outer = dl_outer + dr_outer

        #calculate d (Z_inner - Z^)^2===================
        # (inner - target)^2
        tc_inner = (inner-target) * inner * (1 - inner)
        gl_inner = tc_inner * cl.convolveKernel(mask, LITHO_KERNEL_FOCUS_CT, MIN_DOSE)
        dl_inner = cl.convolveKernel(gl_inner, LITHO_KERNEL_FOCUS, MIN_DOSE)
        gr_inner = tc_inner * cl.convolveKernel(mask, LITHO_KERNEL_FOCUS, MIN_DOSE)
        dr_inner = cl.convolveKernel(gr_inner, LITHO_KERNEL_FOCUS_CT, MIN_DOSE)
        d1_inner = dl_inner + dr_inner

        d1_defocuse = d1_outer + d1_inner

        ctx.save_for_backward(
            mask,
            cl.cupy2tensor_cpx(d1_nominal),
            cl.cupy2tensor_cpx(d1_defocuse)
        )
        ctx.ilt_weight = ilt_weight
        ctx.pvb_weight = pvb_weight
        ctx.ilt_exp = ilt_exp
        aerials = torch.stack((nominal_aerial, inner_aerial, outer_aerial), dim=0)
        wafers = torch.stack((nominal, inner, outer), dim=0)
        return levelset_loss, aerials, wafers

    @staticmethod
    def backward(ctx, levelset_loss, aerials, wafers):
        mask, d1_nominal, d1_defocuse = ctx.saved_tensors
        ilt_weight, pvb_weight, ilt_exp = ctx.ilt_weight, ctx.pvb_weight, ctx.ilt_exp
        gradient = PHOTORISIST_SIGMOID_STEEPNESS * (ilt_exp * ilt_weight * d1_nominal.real + pvb_weight * d1_defocuse.real)
        return gradient, None, None, None



# the traditional pvb calculation.
# class LEVELSET_LAYER(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, mask, target, cl, cfg):
#         ilt_exp = cfg.model.ilt_exp
#         ilt_weight, pvb_weight = cfg.model.loss_weights.ilt_weight, cfg.model.loss_weights.pvb_weight

#         nominal_aerial, nominal = cl.simulateImageOpt(mask, LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
#         outer_aerial, outer = cl.simulateImageOpt(mask, LITHO_KERNEL_FOCUS, MAX_DOSE)
#         inner_aerial, inner = cl.simulateImageOpt(mask, LITHO_KERNEL_DEFOCUS, MIN_DOSE)

#         # levelset_loss = ilt_weight * cl.lexp_loss(nominal, target, ilt_exp) + pvb_weight * (cl.l2_loss(inner, target) + cl.l2_loss(outer, target))
#         levelset_loss = ilt_weight * cl.lexp_loss(nominal, target, ilt_exp) + pvb_weight * (cl.l2_loss(inner, outer))
#         # levelset_loss = cl.l2_loss(nominal, target)

#         #calculate d (Z0-Z^)^(ilt_exp)========================
#         # the lexp loss gradient of (mask - target)
#         m_diff_exp = torch.pow((nominal - target), ilt_exp - 1)
#         tc_nominal = m_diff_exp * nominal * (1 - nominal)
#         gl_nominal = tc_nominal * cl.convolveKernel(mask, LITHO_KERNEL_FOCUS_CT, NOMINAL_DOSE)
#         dl_nominal = cl.convolveKernel(gl_nominal, LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
#         gr_nominal = tc_nominal * cl.convolveKernel(mask, LITHO_KERNEL_FOCUS, NOMINAL_DOSE)
#         dr_nominal = cl.convolveKernel(gr_nominal, LITHO_KERNEL_FOCUS_CT, NOMINAL_DOSE)
#         d1_nominal = dl_nominal + dr_nominal

#         # calculate d (inner - outer) ^2
#         tc_inner = inner * (1 - inner)
#         gl_inner = tc_inner * cl.convolveKernel(mask, LITHO_KERNEL_FOCUS_CT, MIN_DOSE)
#         dl_inner = cl.convolveKernel(gl_inner, LITHO_KERNEL_FOCUS, MIN_DOSE)
#         gr_inner = tc_inner * cl.convolveKernel(mask, LITHO_KERNEL_FOCUS, MIN_DOSE)
#         dr_inner = cl.convolveKernel(gr_inner, LITHO_KERNEL_FOCUS_CT, MIN_DOSE)
#         d1_inner = dl_inner + dr_inner

#         tc_outer = outer * (1 - outer)
#         gl_outer = tc_outer * cl.convolveKernel(mask, LITHO_KERNEL_FOCUS_CT, MAX_DOSE)
#         dl_outer = cl.convolveKernel(gl_outer, LITHO_KERNEL_FOCUS, MAX_DOSE)
#         gr_outer = tc_outer * cl.convolveKernel(mask, LITHO_KERNEL_FOCUS, MAX_DOSE)
#         dr_outer = cl.convolveKernel(gr_outer, LITHO_KERNEL_FOCUS_CT, MAX_DOSE)
#         d1_outer = dl_outer + dr_outer

#         d1_defocuse = 2 * (inner - outer) * (d1_inner - d1_outer)

#         ctx.save_for_backward(
#             mask,
#             cl.cupy2tensor_cpx(d1_nominal),
#             cl.cupy2tensor_cpx(d1_defocuse)
#         )
#         ctx.ilt_weight = ilt_weight
#         ctx.pvb_weight = pvb_weight
#         ctx.ilt_exp = ilt_exp
#         aerials = torch.stack((nominal_aerial, inner_aerial, outer_aerial), dim=0)
#         wafers = torch.stack((nominal, inner, outer), dim=0)
#         return levelset_loss, aerials, wafers

#     @staticmethod
#     def backward(ctx, levelset_loss, aerials, wafers):
#         mask, d1_nominal, d1_defocuse = ctx.saved_tensors
#         ilt_weight, pvb_weight, ilt_exp = ctx.ilt_weight, ctx.pvb_weight, ctx.ilt_exp
#         gradient = PHOTORISIST_SIGMOID_STEEPNESS * (ilt_exp * ilt_weight * d1_nominal.real + pvb_weight * d1_defocuse.real)
#         return gradient, None, None, None

