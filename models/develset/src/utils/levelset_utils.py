'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2020-12-25 17:52:17
LastEditTime: 2021-04-16 13:14:40
Contact: cgjhaha@qq.com
Description: the utils to calculate levelset parameters

Input:
    target
Output:
    levelset params
'''

import os
import sys
sys.path.append('/home/guojin/projects/develset_opc/levelset_net')
import time
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from src.models.const import *
from src.utils.unit_tests.test_case1 import *
import torchvision.transforms as transforms


'''
t2boudry_lines: use only cpu, because cpu is fast enough
'''
def t2boudry_lines(target, device):
    target = target.to(device)
    boundary = torch.zeros(target.shape).to(device)
    corner = torch.zeros(target.shape).to(device)
    vertical = torch.zeros(target.shape).to(device)
    horizontal = torch.zeros(target.shape).to(device)
    site_bool1 = target[:, :, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X]==1
    site_bool2 = [
        target[:, :, LITHOSIM_OFFSET-1:MASK_TILE_END_Y-1, LITHOSIM_OFFSET:MASK_TILE_END_X] == 1,
        target[:, :, LITHOSIM_OFFSET+1:MASK_TILE_END_Y+1, LITHOSIM_OFFSET:MASK_TILE_END_X] == 1,
        target[:, :, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET-1:MASK_TILE_END_X-1] == 1,
        target[:, :, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET+1:MASK_TILE_END_X+1] == 1,
        target[:, :, LITHOSIM_OFFSET-1:MASK_TILE_END_Y-1, LITHOSIM_OFFSET-1:MASK_TILE_END_X-1] ==1,
        target[:, :, LITHOSIM_OFFSET-1:MASK_TILE_END_Y-1, LITHOSIM_OFFSET+1:MASK_TILE_END_X+1] ==1,
        target[:, :, LITHOSIM_OFFSET+1:MASK_TILE_END_Y+1, LITHOSIM_OFFSET-1:MASK_TILE_END_X-1] ==1,
        target[:, :, LITHOSIM_OFFSET+1:MASK_TILE_END_Y+1, LITHOSIM_OFFSET+1:MASK_TILE_END_X+1] ==1,
    ]
    site_0 = torch.stack(site_bool2,axis = 4).all(axis=4)
    site_bool1[site_0] = 0
    boundary[:,:,LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] = site_bool1
    b_c = boundary[:,:,LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X]==1
    b_d = boundary[:, :, LITHOSIM_OFFSET-1:MASK_TILE_END_Y-1, LITHOSIM_OFFSET:MASK_TILE_END_X] == 1
    b_u = boundary[:, :, LITHOSIM_OFFSET+1:MASK_TILE_END_Y+1, LITHOSIM_OFFSET:MASK_TILE_END_X] == 1
    b_l = boundary[:, :, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET-1:MASK_TILE_END_X-1] == 1
    b_r = boundary[:, :, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET+1:MASK_TILE_END_X+1] == 1
    #vertical
    vertical[:,:,LITHOSIM_OFFSET:MASK_TILE_END_Y,LITHOSIM_OFFSET:MASK_TILE_END_X] = b_c
    vertical[:,:,LITHOSIM_OFFSET:MASK_TILE_END_Y,LITHOSIM_OFFSET:MASK_TILE_END_X][b_l & b_r]=0
    v_site = vertical.nonzero()
    if torch.cuda.is_available():
        ind_tmp_v = np.lexsort((v_site[:,2].cpu().numpy(), v_site[:,3].cpu().numpy()))
        ind_tmp_v = torch.from_numpy(ind_tmp_v).to(device)
    else:
        ind_tmp_v = np.lexsort((v_site[:,2].numpy(), v_site[:,3].numpy()))
    v_site = v_site[ind_tmp_v]
    tmp_v_start = torch.cat((torch.tensor([True]).to(device), v_site[:,2][1:]!=v_site[:,2][:-1]+1))
    tmp_v_end = torch.cat((v_site[:,2][1:]!=v_site[:,2][:-1]+1, torch.tensor([True]).to(device)))
    start_p_v = v_site[(tmp_v_start==True).nonzero()[:,0],:]
    end_p_v = v_site[(tmp_v_end==True).nonzero()[:,0],:]
    p_v = torch.stack((v_site[(tmp_v_start==True).nonzero()[:,0],:], v_site[(tmp_v_end==True).nonzero()[:,0],:]), axis = 2)
    #horizontal
    horizontal[:,:,LITHOSIM_OFFSET:MASK_TILE_END_Y,LITHOSIM_OFFSET:MASK_TILE_END_X] = b_c
    horizontal[:,:,LITHOSIM_OFFSET:MASK_TILE_END_Y,LITHOSIM_OFFSET:MASK_TILE_END_X][b_u&b_d]=0
    h_site = horizontal.nonzero()
    if torch.cuda.is_available():
        ind_tmp_h = np.lexsort((h_site[:,3].cpu().numpy(), h_site[:,2].cpu().numpy()))
        ind_tmp_h = torch.from_numpy(ind_tmp_h).to(device)
    else:
        ind_tmp_h = np.lexsort((h_site[:,3].numpy(), h_site[:,2].numpy()))
    h_site = h_site[ind_tmp_h]
    tmp_h_start = torch.cat((torch.tensor([True]).to(device), h_site[:,3][1:]!=h_site[:,3][:-1]+1))
    tmp_h_end = torch.cat((h_site[:,3][1:]!=h_site[:,3][:-1]+1, torch.tensor([True]).to(device)))
    start_p_h = h_site[(tmp_h_start==True).nonzero()[:,0],:]
    end_p_h = h_site[(tmp_h_end==True).nonzero()[:,0],:]
    p_h = torch.stack((h_site[(tmp_h_start==True).nonzero()[:,0],:], h_site[(tmp_h_end==True).nonzero()[:,0],:]), axis = 2)
    return p_v.float(), p_h.float()


'''
t2boudry_lines: use only cpu, because cpu is fast enough
'''
def t2boudry_lines_cpu(target, device):
    device = torch.device('cpu')
    target = target.to(device)
    boundary = torch.zeros(target.shape).to(device)
    corner = torch.zeros(target.shape).to(device)
    vertical = torch.zeros(target.shape).to(device)
    horizontal = torch.zeros(target.shape).to(device)
    site_bool1 = target[:, :, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X]==1
    site_bool2 = [
        target[:, :, LITHOSIM_OFFSET-1:MASK_TILE_END_Y-1, LITHOSIM_OFFSET:MASK_TILE_END_X] == 1,
        target[:, :, LITHOSIM_OFFSET+1:MASK_TILE_END_Y+1, LITHOSIM_OFFSET:MASK_TILE_END_X] == 1,
        target[:, :, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET-1:MASK_TILE_END_X-1] == 1,
        target[:, :, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET+1:MASK_TILE_END_X+1] == 1,
        target[:, :, LITHOSIM_OFFSET-1:MASK_TILE_END_Y-1, LITHOSIM_OFFSET-1:MASK_TILE_END_X-1] ==1,
        target[:, :, LITHOSIM_OFFSET-1:MASK_TILE_END_Y-1, LITHOSIM_OFFSET+1:MASK_TILE_END_X+1] ==1,
        target[:, :, LITHOSIM_OFFSET+1:MASK_TILE_END_Y+1, LITHOSIM_OFFSET-1:MASK_TILE_END_X-1] ==1,
        target[:, :, LITHOSIM_OFFSET+1:MASK_TILE_END_Y+1, LITHOSIM_OFFSET+1:MASK_TILE_END_X+1] ==1,
    ]
    site_0 = torch.stack(site_bool2,axis = 4).all(axis=4)
    site_bool1[site_0] = 0
    boundary[:,:,LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] = site_bool1
    b_c = boundary[:,:,LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X]==1
    b_d = boundary[:, :, LITHOSIM_OFFSET-1:MASK_TILE_END_Y-1, LITHOSIM_OFFSET:MASK_TILE_END_X] == 1
    b_u = boundary[:, :, LITHOSIM_OFFSET+1:MASK_TILE_END_Y+1, LITHOSIM_OFFSET:MASK_TILE_END_X] == 1
    b_l = boundary[:, :, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET-1:MASK_TILE_END_X-1] == 1
    b_r = boundary[:, :, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET+1:MASK_TILE_END_X+1] == 1
    #vertical
    vertical[:,:,LITHOSIM_OFFSET:MASK_TILE_END_Y,LITHOSIM_OFFSET:MASK_TILE_END_X] = b_c
    vertical[:,:,LITHOSIM_OFFSET:MASK_TILE_END_Y,LITHOSIM_OFFSET:MASK_TILE_END_X][b_l & b_r]=0
    v_site = vertical.nonzero()
    # if torch.cuda.is_available():
    #     ind_tmp_v = np.lexsort((v_site[:,2].cpu().numpy(), v_site[:,3].cpu().numpy()))
    #     ind_tmp_v = torch.from_numpy(ind_tmp_v).to(device)
    # else:
    ind_tmp_v = np.lexsort((v_site[:,2].numpy(), v_site[:,3].numpy()))
    v_site = v_site[ind_tmp_v]
    tmp_v_start = torch.cat((torch.tensor([True]).to(device), v_site[:,2][1:]!=v_site[:,2][:-1]+1))
    tmp_v_end = torch.cat((v_site[:,2][1:]!=v_site[:,2][:-1]+1, torch.tensor([True]).to(device)))
    start_p_v = v_site[(tmp_v_start==True).nonzero()[:,0],:]
    end_p_v = v_site[(tmp_v_end==True).nonzero()[:,0],:]
    p_v = torch.stack((v_site[(tmp_v_start==True).nonzero()[:,0],:], v_site[(tmp_v_end==True).nonzero()[:,0],:]), axis = 2)
    #horizontal
    horizontal[:,:,LITHOSIM_OFFSET:MASK_TILE_END_Y,LITHOSIM_OFFSET:MASK_TILE_END_X] = b_c
    horizontal[:,:,LITHOSIM_OFFSET:MASK_TILE_END_Y,LITHOSIM_OFFSET:MASK_TILE_END_X][b_u&b_d]=0
    h_site = horizontal.nonzero()
    # if torch.cuda.is_available():
    #     ind_tmp_h = np.lexsort((h_site[:,3].cpu().numpy(), h_site[:,2].cpu().numpy()))
    #     ind_tmp_h = torch.from_numpy(ind_tmp_h).to(device)
    # else:
    ind_tmp_h = np.lexsort((h_site[:,3].numpy(), h_site[:,2].numpy()))
    h_site = h_site[ind_tmp_h]
    tmp_h_start = torch.cat((torch.tensor([True]).to(device), h_site[:,3][1:]!=h_site[:,3][:-1]+1))
    tmp_h_end = torch.cat((h_site[:,3][1:]!=h_site[:,3][:-1]+1, torch.tensor([True]).to(device)))
    start_p_h = h_site[(tmp_h_start==True).nonzero()[:,0],:]
    end_p_h = h_site[(tmp_h_end==True).nonzero()[:,0],:]
    p_h = torch.stack((h_site[(tmp_h_start==True).nonzero()[:,0],:], h_site[(tmp_h_end==True).nonzero()[:,0],:]), axis = 2)
    return p_v.float(), p_h.float()




'''
target: torch.tensor
'''
def t2lsparams(p_v, p_h, target, device):
# def __initlevelset_lines(p_v, p_h, target):
    p_v = p_v.to(device)
    p_h = p_h.to(device)
    target = target.to(device)
    levelset = torch.zeros(target.shape).to(device)
    levelset = levelset + 2048
    x,y = torch.meshgrid(torch.arange(target.shape[2]).to(device),torch.arange(target.shape[3]).to(device))
    coord_2d = torch.stack((x,y), axis = 2) # (2048, 2048, 2)
    part_2d = coord_2d[LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X, :].flatten(start_dim=0,end_dim=1).float()  #(1280*1280,2)(y,x)

    #vertical
    distance_v = torch.zeros(part_2d.shape[0], p_v.shape[0]).to(device) #(1280*1280,N_v)
    for i in range(p_v.shape[0]):     #start_p = p_v[i,[2,3],0], end_p =p_v[i,[2,3],1]
        ind0 =  (part_2d[:,1] != p_v[i,3,0]) & ((part_2d[:,0] < p_v[i,2,0]) | (part_2d[:,0] > p_v[i,2,1]))
        distance_v[ind0,i] = torch.min(torch.sqrt((p_v[i,2,0] - part_2d[ind0,0]) **2+(p_v[i,3,0] - part_2d[ind0,1]) **2),  torch.sqrt((p_v[i,2,1] - part_2d[ind0,0]) **2+(p_v[i,3,1] - part_2d[ind0,1]) **2))   
        ind1 = (part_2d[:,1] == p_v[i,3,0]) & (part_2d[:,0] >= p_v[i,2,0]) & (part_2d[:,0] <= p_v[i,2,1])
        distance_v[ind1,i] = 0
        ind2 =  (part_2d[:,1] == p_v[i,3,0]) & ((part_2d[:,0] < p_v[i,2,0]) | (part_2d[:,0] > p_v[i,2,1]))
        distance_v[ind2,i] = torch.min( torch.abs(part_2d[ind2,0] - p_v[i,2,0])  , torch.abs(part_2d[ind2,0] - p_v[i,2,1]) )
        ind3 = (part_2d[:,1] != p_v[i,3,0]) & (part_2d[:,0] >= p_v[i,2,0]) & (part_2d[:,0] <= p_v[i,2,1])
        distance_v[ind3,i] = torch.abs(part_2d[ind3,1]-p_v[i,3,0])

    #horizontal
    distance_h = torch.zeros(part_2d.shape[0], p_h.shape[0]).to(device)#(14*14,N_h)
    for i in range(p_h.shape[0]):
        ind0 = (part_2d[:,0] != p_h[i,2,0]) & ((part_2d[:,1] < p_h[i,3,0]) | (part_2d[:,1] > p_h[i,3,1]))
        distance_h[ind0,i] = torch.min(torch.sqrt((p_h[i,2,0] - part_2d[ind0,0]) **2 + (p_h[i,3,0] - part_2d[ind0,1]) **2),  torch.sqrt((p_h[i,2,1] - part_2d[ind0,0]) **2+(p_h[i,3,1] - part_2d[ind0,1]) **2))
        ind1 = (part_2d[:,0] == p_h[i,2,0]) & (part_2d[:,1] >= p_h[i,3,0]) & (part_2d[:,1] <= p_h[i,3,1])
        distance_h[ind1,i] = 0
        ind2 = (part_2d[:,0] == p_h[i,2,0]) & ((part_2d[:,1] < p_h[i,3,0]) | (part_2d[:,1] > p_h[i,3,1]))
        distance_h[ind2,i] = torch.min( torch.abs(part_2d[ind2,1] - p_h[i,3,0]), torch.abs(part_2d[ind2,1] - p_h[i,3,1]) )
        ind3 = (part_2d[:,0] != p_h[i,2,0]) & (part_2d[:,1] >= p_h[i,3,0]) & (part_2d[:,1] <= p_h[i,3,1])
        distance_h[ind3,i] = torch.abs(part_2d[ind3,0] - p_h[i,2,0])
    #the following procedure only satisfies: batch = channel =1!
    distance = torch.min(distance_v.min(axis =1)[0], distance_h.min(axis=1)[0]).reshape(MASK_TILE_END_Y-LITHOSIM_OFFSET, MASK_TILE_END_X-LITHOSIM_OFFSET)
    levelset[0,0,LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] = distance
    ind_inner = target==1
    levelset[ind_inner] = -levelset[ind_inner]

    # use truncated signed distance function.
    levelset[torch.where(levelset > UP_TRUNCATED_D)] = UP_TRUNCATED_D
    levelset[torch.where(levelset < DOWN_TRUNCATED_D)] = DOWN_TRUNCATED_D
    return levelset


def truncated_ls(levelset):
    levelset[torch.where(levelset > UP_TRUNCATED_D)] = UP_TRUNCATED_D
    levelset[torch.where(levelset < DOWN_TRUNCATED_D)] = DOWN_TRUNCATED_D
    return levelset


def ls2npy(ls):
    ls = ls[:, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X]
    ls_npy = ls.cpu().detach().numpy()
    return ls_npy
    # ls_npz = np.savez_compressed()


def ls2gray(ls):
    a = 1 / (UP_TRUNCATED_D - DOWN_TRUNCATED_D)
    b = DOWN_TRUNCATED_D / (DOWN_TRUNCATED_D - UP_TRUNCATED_D)
    gray = a * ls + b
    return gray

def gray2ls(gray):
    a = 1 / (UP_TRUNCATED_D - DOWN_TRUNCATED_D)
    b = DOWN_TRUNCATED_D / (DOWN_TRUNCATED_D - UP_TRUNCATED_D)
    ls = (gray - b)/a
    return ls


def gradient_geo(levelset):
    # print('ls     is nan: ', torch.isnan(levelset).any())
    gradient_geo = torch.zeros(levelset.shape).cuda()
    centor = levelset[:, :, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X]
    # print('centor is nan: ', torch.isnan(centor).any())
    left = levelset[:, :, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET-1:MASK_TILE_END_X-1]
    # print('left   is nan: ',  torch.isnan(left).any())
    right = levelset[:, :, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET+1:MASK_TILE_END_X+1]
    up = levelset[:, :, LITHOSIM_OFFSET+1:MASK_TILE_END_Y+1, LITHOSIM_OFFSET:MASK_TILE_END_X]
    low = levelset[:, :, LITHOSIM_OFFSET-1:MASK_TILE_END_Y-1, LITHOSIM_OFFSET:MASK_TILE_END_X]
    grad_x = (right-left) / 2
    # print('grad_x is nan: ', torch.isnan(grad_x).any())
    grad_x[:,:,:,0] = centor[:,:,:,1] - centor[:,:,:,0]  #left boundary
    grad_x[:,:,:,-1] = centor[:,:,:,-1] - centor[:,:,:,-2]  #right boundary
    # print('grad_x boundry is nan: ', torch.isnan(grad_x).any())
    grad_y = (up - low) / 2
    grad_y[:,:,0,:] = centor[:,:,1,:] - centor[:,:,0,:]  # upper boundary
    grad_y[:,:,-1,:] = centor[:,:,-1,:] - centor[:,:,-2,:]  # lower boundary
    gradient_geo[:,:, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] = (grad_x.pow(2) + grad_y.pow(2)).sqrt()
    return gradient_geo




def gradient_geo_part(levelset):
    '''
    input: (b,c,x,y) leveset tensor
    return grad_x, grad_y: (b,c,LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X) central tensor
    return grad_xx, grad_yy, grad_xy: (b,c,LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X) central tensor
    '''
    centor = levelset[:, :, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X]
    left = levelset[:, :, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET-1:MASK_TILE_END_X-1]
    right = levelset[:, :, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET+1:MASK_TILE_END_X+1]
    up = levelset[:, :, LITHOSIM_OFFSET+1:MASK_TILE_END_Y+1, LITHOSIM_OFFSET:MASK_TILE_END_X]
    low = levelset[:, :, LITHOSIM_OFFSET-1:MASK_TILE_END_Y-1, LITHOSIM_OFFSET:MASK_TILE_END_X]
    ll = levelset[:, :, LITHOSIM_OFFSET-1:MASK_TILE_END_Y-1, LITHOSIM_OFFSET-1:MASK_TILE_END_X-1]
    lr = levelset[:,:, LITHOSIM_OFFSET-1:MASK_TILE_END_Y-1, LITHOSIM_OFFSET+1:MASK_TILE_END_X+1]
    ul = levelset[:,:, LITHOSIM_OFFSET+1:MASK_TILE_END_Y+1, LITHOSIM_OFFSET-1:MASK_TILE_END_X-1]
    ur = levelset[:,:, LITHOSIM_OFFSET+1:MASK_TILE_END_Y+1, LITHOSIM_OFFSET+1:MASK_TILE_END_X+1]
    # 1st order
    grad_x = (right-left) / 2
    grad_x[:,:,:,0] = centor[:,:,:,1] - centor[:,:,:,0]  #left boundary
    grad_x[:,:,:,-1] = centor[:,:,:,-1] - centor[:,:,:,-2]  #right boundary
    grad_y = (up - low) / 2
    grad_y[:,:,0,:] = centor[:,:,1,:] - centor[:,:,0,:]  # upper boundary
    grad_y[:,:,-1,:] = centor[:,:,-1,:] - centor[:,:,-2,:]  # lower boundary
    # 2nd order
    grad_xx = right + left - 2 * centor
    grad_xx[:,:,:,0] = centor[:,:,:,1] - centor[:,:,:,0]
    grad_xx[:,:,:,-1] = - (centor[:,:,:,-1] - centor[:,:,:,-2])
    grad_yy = up + low - 2 * centor
    grad_yy[:,:,0,:] = centor[:,:,1,:] - centor[:,:,0,:]
    grad_yy[:,:,-1,:] = -(centor[:,:,-1,:] - centor[:,:,-2,:])
    grad_xy = 0.25 * ((ur - ul) - (lr - ll))
    grad_xy[:,:,:,0] = 0.5 * (ur - up - lr + low)[:,:,:,0]
    grad_xy[:,:,:,-1] = 0.5 * (up - ul - low + ll)[:,:,:,-1]
    grad_xy[:,:,0,:] = 0.5 * (ur -ul - right + left)[:,:,0,:]
    grad_xy[:,:,-1,:] = 0.5 * (right - left - lr + ll)[:,:,-1,:]
    grad_xy[:,:,0,0] = 0.25 * centor[:,:,1,1]
    grad_xy[:,:,-1,0] = -0.25 * centor[:,:,-2,1]
    grad_xy[:,:,0,-1] = -0.25 * centor[:,:,1, -2]
    grad_xy[:,:,-1,-1] = 0.25 * centor[:,:,-2,-2]
    return grad_x, grad_y, grad_xx, grad_yy, grad_xy

def calculate_k(grad_x, grad_y, grad_xx, grad_yy, grad_xy, levelset):
    k = torch.zeros(levelset.shape)
    num = torch.mul(grad_xx, grad_y.pow(2)) - 2 * torch.mul(torch.mul(grad_y, grad_x), grad_xy) + torch.mul(grad_yy, grad_x.pow(2))
    denom = (grad_x.pow(2) + grad_y.pow(2)).pow(1.5)
    k[:,:, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X ] = torch.mul(num, denom.reciprocal())
    return k

def curvature_term(levelset):
    grad_part = gradient_geo_part(levelset)
    grad_x, grad_y, grad_xx, grad_yy, grad_xy = grad_part
    curv = torch.zeros(levelset.shape).cuda()
    num = torch.mul(grad_xx, grad_y.pow(2)) - 2 * torch.mul(torch.mul(grad_y, grad_x), grad_xy) + torch.mul(grad_yy, grad_x.pow(2))
    denom = (grad_x.pow(2) + grad_y.pow(2)).pow(1.5) + EPSILON
    curv[:,:, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X ] = torch.div(num, denom)
    return curv



if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    test1 = Image.open('/home/guojin/projects/develset_opc/levelset_net/binary_images/M1_test1/M1_test1.png').convert('L')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test1 = transform(test1)
    test1 = test1.unsqueeze(1)  #(1,2048,2048) to (1,1,2048,2048)

    start = time.time()
    # p_v,p_h = t2boudry_lines(test1, device)
    p_v,p_h = t2boudry_lines_cpu(test1, device)
    b_end = time.time()


    print('init boundary time: ',b_end-start,' s')

    LS1 = t2lsparams(p_v,p_h, test1, device)
    l_end = time.time()
    print('init levelset function time: ',l_end-b_end,' s')

    torch.save(LS1, 'LS1.pt')

    test_corner(216, 80, 'll', test1, LS1)
    test_corner(744, 776, 'ur', test1, LS1)
    test_convex(304, 140, 'lr', test1, LS1)
    test_convex(680, 708, 'ul', test1, LS1)
    test_convex(680, 148, 'ul', test1, LS1)
    test_outer(-100, -100, test1, LS1)
    print(torch.min(LS1))