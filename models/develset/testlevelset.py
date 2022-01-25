import os
import torch
import os.path
from PIL import Image
from src.models.const import *
import torchvision.transforms as transforms
import time
import numpy as np


def __binaryboundary(target):
    '''
    calculate the target image boundary,
    the boundary point value is 1, otherwise 0, the boundary map is (batchsize, channel, x, y) tensor
    '''
    boundary = torch.zeros(target.shape).cuda()
    site_bool1 = target[:,:,LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X]==1
    site_bool2 = [
        target[:, :, LITHOSIM_OFFSET-1:MASK_TILE_END_Y-1, LITHOSIM_OFFSET:MASK_TILE_END_X] == 1,
        target[:, :, LITHOSIM_OFFSET+1:MASK_TILE_END_Y+1, LITHOSIM_OFFSET:MASK_TILE_END_X] == 1,
        target[:, :, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET-1:MASK_TILE_END_X-1] == 1,
        target[:, :, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET+1:MASK_TILE_END_X+1] == 1,
        target[:, :, LITHOSIM_OFFSET-1:MASK_TILE_END_Y-1, LITHOSIM_OFFSET-1:MASK_TILE_END_X-1] == 1,
        target[:, :, LITHOSIM_OFFSET+1:MASK_TILE_END_Y+1, LITHOSIM_OFFSET-1:MASK_TILE_END_X-1] == 1,
        target[:, :, LITHOSIM_OFFSET-1:MASK_TILE_END_Y-1, LITHOSIM_OFFSET+1:MASK_TILE_END_X+1] == 1,
        target[:, :, LITHOSIM_OFFSET+1:MASK_TILE_END_Y+1, LITHOSIM_OFFSET+1:MASK_TILE_END_X+1] == 1,
    ]
    site_0 = torch.stack(site_bool2,axis = 4).all(axis=4)
    site_bool1[site_0] = 0
    boundary[:,:,LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] = site_bool1.int()
    return boundary

def __boundary_lines(target):
    '''
    
    '''
    boundary = torch.zeros(target.shape)
    corner = torch.zeros(target.shape)
    vertical = torch.zeros(target.shape)
    horizontal = torch.zeros(target.shape)
    site_bool1 = target[:,:,LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X]==1
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
    ind_tmp_v = np.lexsort((v_site[:,2].numpy(), v_site[:,3].numpy()))
    v_site = v_site[ind_tmp_v]
    tmp_v_start = torch.cat((torch.tensor([True]), v_site[:,2][1:]!=v_site[:,2][:-1]+1))
    tmp_v_end = torch.cat((v_site[:,2][1:]!=v_site[:,2][:-1]+1, torch.tensor([True])))
    start_p_v = v_site[(tmp_v_start==True).nonzero()[:,0],:]
    end_p_v = v_site[(tmp_v_end==True).nonzero()[:,0],:]
    p_v = torch.stack((v_site[(tmp_v_start==True).nonzero()[:,0],:], v_site[(tmp_v_end==True).nonzero()[:,0],:]), axis = 2)
    #horizontal
    horizontal[:,:,LITHOSIM_OFFSET:MASK_TILE_END_Y,LITHOSIM_OFFSET:MASK_TILE_END_X] = b_c
    horizontal[:,:,LITHOSIM_OFFSET:MASK_TILE_END_Y,LITHOSIM_OFFSET:MASK_TILE_END_X][b_u&b_d]=0
    h_site = horizontal.nonzero()
    ind_tmp_h = np.lexsort((h_site[:,3].numpy(), h_site[:,2].numpy()))
    h_site = h_site[ind_tmp_h]
    tmp_h_start = torch.cat((torch.tensor([True]), h_site[:,3][1:]!=h_site[:,3][:-1]+1))
    tmp_h_end = torch.cat((h_site[:,3][1:]!=h_site[:,3][:-1]+1, torch.tensor([True])))
    start_p_h = h_site[(tmp_h_start==True).nonzero()[:,0],:]
    end_p_h = h_site[(tmp_h_end==True).nonzero()[:,0],:]
    p_h = torch.stack((h_site[(tmp_h_start==True).nonzero()[:,0],:], h_site[(tmp_h_end==True).nonzero()[:,0],:]), axis = 2)
    return p_v.float(), p_h.float()

def __initlevelsetfunc(boundary, target):
    '''
    calculate levelset function, phi(x,y) = |(x,y)-boundary|_norm2.
    output is the (2048,2048) levelset function tensor
    only central part LITHO_OFFSET: MASK_TILE_SIZE is nonzero
    ***for now, only support (1,1,x,y) input target and boundary***
    '''
    x,y = torch.meshgrid(torch.arange(IMAGE_HEIGHT_WIDTH),torch.arange(IMAGE_HEIGHT_WIDTH))
    coord_2d = torch.stack((x,y),axis = 2)
    part_2d = coord_2d[LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X,:].flatten(start_dim=0,end_dim=1)    #（1280*1280,2）
    boundary_tensor = torch.nonzero(boundary)
    tmp1 = part_2d.unsqueeze(1).repeat(1,boundary_tensor.shape[0],1)   #(1280*1280, n, 2)
    tmp2 = boundary_tensor[:,2:].unsqueeze(0) #(1,n,2)
    tmp3 = tmp1-tmp2
    distance = torch.min(torch.linalg.norm(tmp3.float(), ord = 2, dim = 2 ), dim = 1)[0]
    levelset_C = distance.reshape(1,1,MASK_TILE_END_Y-LITHOSIM_OFFSET,MASK_TILE_END_X-LITHOSIM_OFFSET) #(1,1,1280,1280)
    levelset = torch.zeros(target.shape)
    levelset[0,0,LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] = levelset_C #(1,1,2048,2048)
    levelset[target==1]=-levelset[target==1]
    return levelset


def __initlevelset_lines(p_v, p_h, target):
    levelset = torch.zeros(target.shape)
    x,y = torch.meshgrid(torch.arange(target.shape[2]),torch.arange(target.shape[3]))
    coord_2d = torch.stack((x,y), axis = 2) # (2048, 2048, 2)
    part_2d = coord_2d[LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X, :].flatten(start_dim=0,end_dim=1).float()  #(1280*1280,2)(y,x)
    distance_v = torch.zeros(part_2d.shape[0], p_v.shape[0]) #(1280*1280,N_v)
    for i in range(p_v.shape[0]):     #start_p = p_v[i,[2,3],0], end_p =p_v[i,[2,3],1]
        ind0 =  (part_2d[:,1] != p_v[i,3,0]) & ((part_2d[:,0] < p_v[i,2,0]) | (part_2d[:,0] > p_v[i,2,1]))
        distance_v[ind0,i] = torch.min(torch.sqrt((p_v[i,2,0] - part_2d[ind0,0]) **2+(p_v[i,3,0] - part_2d[ind0,1]) **2),  torch.sqrt((p_v[i,2,1] - part_2d[ind0,0]) **2+(p_v[i,3,1] - part_2d[ind0,1]) **2))   
        ind1 = (part_2d[:,1] == p_v[i,3,0]) & (part_2d[:,0] >= p_v[i,2,0]) & (part_2d[:,0] <= p_v[i,2,1])
        distance_v[ind1,i] = 0
        ind2 =  (part_2d[:,1] == p_v[i,3,0]) & ((part_2d[:,0] < p_v[i,2,0]) | (part_2d[:,0] > p_v[i,2,1]))
        distance_v[ind2,i] = torch.min( torch.abs(part_2d[ind2,0] - p_v[i,2,0])  , torch.abs(part_2d[ind2,0] - p_v[i,2,1]) )
        ind3 = (part_2d[:,1] != p_v[i,3,0]) & (part_2d[:,0] >= p_v[i,2,0]) & (part_2d[:,0] <= p_v[i,2,1])
        distance_v[ind3,i] = torch.abs(part_2d[ind3,1]-p_v[i,3,0])
    distance_h = torch.zeros(part_2d.shape[0], p_h.shape[0])#(14*14,N_h)
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
    return levelset

def __gradient_geo(levelset):
    '''
    input: (b,c,x,y) leveset tensor
    '''
    gradient_geo = torch.zeros(levelset.shape).cuda()
    centor = levelset[:, :, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X]
    left = levelset[:, :, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET-1:MASK_TILE_END_X-1]
    right = levelset[:, :, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET+1:MASK_TILE_END_X+1]
    up = levelset[:, :, LITHOSIM_OFFSET+1:MASK_TILE_END_Y+1, LITHOSIM_OFFSET:MASK_TILE_END_X]
    low = levelset[:, :, LITHOSIM_OFFSET-1:MASK_TILE_END_Y-1, LITHOSIM_OFFSET:MASK_TILE_END_X]
    grad_x = (right-left) / 2
    grad_x[:,:,:,0] = centor[:,:,:,1] - centor[:,:,:,0]  #left boundary
    grad_x[:,:,:,-1] = centor[:,:,:,-1] - centor[:,:,:,-2]  #right boundary
    grad_y = (up - low) / 2
    grad_y[:,:,0,:] = centor[:,:,1,:] - centor[:,:,0,:]  # upper boundary
    grad_y[:,:,-1,:] = centor[:,:,-1,:] - centor[:,:,-2,:]  # lower boundary
    gradient_geo[:,:, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] = (grad_x.pow(2) + grad_y.pow(2)).sqrt()
    return gradient_geo


def __Epechecker(b_image, p_v, p_h, target):
    '''
    input: binary image tensor: (b, c, x, y); vertical points pair p_v: (N_v,4,2); horizontal points pair: (N_h, 4, 2), target image (b, c, x, y)
    output the total number of epe violations
    '''
    epe_inner = 0
    epe_outer = 0
    epe_map  = torch.zeros(target.shape)
    vio_map = torch.zeros(target.shape)
    for i in range(p_v.shape[0]):
        center = (p_v[i,:,0] + p_v[i,:,1]) / 2
        center = center.int().float().unsqueeze(0) #(1,4)
        if (p_v[i,2,1] - p_v[i,2,0]) <= MIN_EPE_CHECK_LENGTH:
            sample = center
            epe_map[0,0,sample[:,2].long(),sample[:,3].long()] = 1
            v_in_site, v_out_site = epecheck(b_image, sample, target,'v')
        else:
            sample_y = torch.cat((torch.arange(p_v[i,2,0] + EPE_CHECK_START_INTERVEL, center[0,2]+1, step = EPE_CHECK_INTERVEL), torch.arange(p_v[i,2,1] - EPE_CHECK_START_INTERVEL, center[0,2], step = -EPE_CHECK_INTERVEL))).unique()
            sample = p_v[i,:,0].repeat(sample_y.shape[0],1)
            sample[:,2] = sample_y
            epe_map[0,0,sample[:,2].long(),sample[:,3].long()] = 1
            v_in_site, v_out_site = epecheck(b_image, sample, target,'v')
        epe_inner = epe_inner + v_in_site.shape[0]
        epe_outer = epe_outer + v_out_site.shape[0]
        vio_map[0,0, v_in_site[:,2].long(), v_in_site[:,3].long()] = 1
        vio_map[0,0, v_out_site[:,2].long(), v_out_site[:,3].long()] = 1

    for i in range(p_h.shape[0]):
        center = (p_h[i,:,0] + p_h[i,:,1]) / 2
        center = center.int().float().unsqueeze(0)
        if (p_h[i,3,1] - p_h[i,3,0]) <= MIN_EPE_CHECK_LENGTH:
            sample = center
            epe_map[0,0,sample[:,2].long(),sample[:,3].long()] = 1
            v_in_site, v_out_site = epecheck(b_image, sample, target,'h')
        else: 
            sample_x = torch.cat(( torch.arange(p_h[i,3,0] + EPE_CHECK_START_INTERVEL, center[0,3]+1, step = EPE_CHECK_INTERVEL), torch.arange(p_h[i,3,1] - EPE_CHECK_START_INTERVEL, center[0,3], step = -EPE_CHECK_INTERVEL))).unique()
            sample = p_h[i,:,0].repeat(sample_x.shape[0], 1)
            sample[:,3] = sample_x
            epe_map[0,0,sample[:,2].long(),sample[:,3].long()] = 1
            v_in_site, v_out_site = epecheck(b_image, sample, target,'h')
        epe_inner = epe_inner + v_in_site.shape[0]
        epe_outer = epe_outer + v_out_site.shape[0]
        vio_map[0,0, v_in_site[:,2].long(), v_in_site[:,3].long()] = 1
        vio_map[0,0, v_out_site[:,2].long(), v_out_site[:,3].long()] = 1
    return epe_inner, epe_outer,vio_map

def epecheck(image, sample, target, direction):
    # vio_in = 0
    # vio_out = 0
    # epe_inner_map = torch.zeros(target.shape)
    # epe_outer_map = torch.zeros(target.shape)
    if direction == 'v':
        if ((target[0, 0, sample[0,2].long(),sample[0,3].long() + 1] == 1) & (target[0, 0, sample[0,2].long(), sample[0,3].long() - 1] == 0)): #left ,x small
            sample_inner = sample + torch.tensor([0,0,0, EPE_CONSTRAINT])
            sample_outer = sample + torch.tensor([0,0,0, -EPE_CONSTRAINT])
            epe_inner_site = sample[image[0,0, sample_inner[:,2].long(), sample_inner[:,3].long()] == 0,:]
            epe_outer_site = sample[image[0,0, sample_outer[:,2].long(), sample_outer[:,3].long()] == 1,:]
            # epe_inner_map[0,0,epe_inner_site[:,2].long(), epe_inner_site[:,3].long()] = 1
            # epe_outer_map[0,0,epe_outer_site[:,2].long(), epe_outer_site[:,3].long()] = 1

        elif ((target[0, 0, sample[0,2].long(),sample[0,3].long() + 1] == 0) & (target[0, 0, sample[0,2].long(), sample[0,3].long() - 1] == 1)): #right, x large
            sample_inner = sample + torch.tensor([0,0,0, -EPE_CONSTRAINT])
            sample_outer = sample + torch.tensor([0,0,0, EPE_CONSTRAINT])
            epe_inner_site = sample[image[0,0, sample_inner[:,2].long(), sample_inner[:,3].long()] == 0,:]
            epe_outer_site = sample[image[0,0, sample_outer[:,2].long(), sample_outer[:,3].long()] == 1,:]
            # epe_inner_map[0,0,epe_inner_site[:,2].long(), epe_inner_site[:,3].long()] = 1
            # epe_outer_map[0,0,epe_outer_site[:,2].long(), epe_outer_site[:,3].long()] = 1

    if direction == 'h':
        if((target[0, 0, sample[0,2].long() + 1, sample[0,3].long()] == 1) & (target[0, 0, sample[0,2].long() - 1, sample[0,3].long()] == 0)): #up, y small
            sample_inner = sample + torch.tensor([0,0, EPE_CONSTRAINT, 0])
            sample_outer = sample + torch.tensor([0,0, -EPE_CONSTRAINT, 0])
            epe_inner_site = sample[image[0,0, sample_inner[:,2].long(), sample_inner[:,3].long()] == 0,:]
            epe_outer_site = sample[image[0,0, sample_outer[:,2].long(), sample_outer[:,3].long()] == 1,:]
            # epe_inner_map[0,0,epe_inner_site[:,2].long(), epe_inner_site[:,3].long()] = 1
            # epe_outer_map[0,0,epe_outer_site[:,2].long(), epe_outer_site[:,3].long()] = 1

        elif (target[0, 0, sample[0,2].long() + 1, sample[0,3].long()] == 0) & (target[0, 0, sample[0,2].long()-1, sample[0,3].long()] == 1): #low, y large
            sample_inner = sample + torch.tensor([0,0, -EPE_CONSTRAINT, 0])
            sample_outer = sample + torch.tensor([0,0, EPE_CONSTRAINT, 0])
            epe_inner_site = sample[image[0,0, sample_inner[:,2].long(), sample_inner[:,3].long()] == 0,:]
            epe_outer_site = sample[image[0,0, sample_outer[:,2].long(), sample_outer[:,3].long()] == 1,:]
            # epe_inner_map[0,0,epe_inner_site[:,2].long(), epe_inner_site[:,3].long()] = 1
            # epe_outer_map[0,0,epe_outer_site[:,2].long(), epe_outer_site[:,3].long()] = 1

    # vio_in = epe_inner_site.shape[0]
    # vio_out = epe_outer_site.shape[0]
    return epe_inner_site, epe_outer_site


transform = transforms.Compose([
    transforms.ToTensor()
])
T1 = Image.open('binary_images/M1_test1/M1_test1.png').convert('L')
T1 = transform(T1).unsqueeze(1)      #(1,2048,2048) to (1,1,2048,2048)

start = time.time()
p_v,p_h = __boundary_lines(T1)
b_end = time.time()
print('init boundary time: ',b_end-start,' s')

img5 = Image.open('binary_images/M1_test1/imgNominal5.png').convert('L')   #the fifth iteration result from cpp flow, cpp result: 41 inner, 8 outer
img5 = transform(img5).unsqueeze(1)
E_in, E_out, epemap = __Epechecker(img5, p_v, p_h, T1)

LS4 = __initlevelset_lines(p_v,p_h, T1)
l_end = time.time()
print('init levelset function time: ',l_end-b_end,' s')