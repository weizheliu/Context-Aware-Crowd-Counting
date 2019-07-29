import h5py
import torch
import shutil
from skimage.transform import resize
import numpy as np

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')            

def genScalePos(imSize,ph,pw):
    h = imSize[2]
    w = imSize[3]

    dx = ph/2
    dy = pw/2

    x = range(dx,h,dx)
    y = range(dy,w,dy)

    lpos = []
    for m in x:
        for n in y:
            pos = [m,n]
            lpos.append(pos)
    return lpos

def cropMultiScale(im,N_scale):
    '''
    @brief: Crop patches with different scale at different position,
    the scale is depend on img shap and number of scales

    @ im : input img[N,C,H,W]

    @ N_scale: different scale level
    '''
    lpatch = []
    h = im.shape[2]
    w = im.shape[3]
    ph_base = int(h/(N_scale+1))
    pw_base = int(w/(N_scale+1))

    for level in range(N_scale):
        ph = int(ph_base*(level+1))
        pw = int(pw_base*(level+1))
        lpos = genScalePos(im.shape,ph,pw)
        dx = ph/2
        dy = pw/2
        for p in lpos:
            x,y = p
            sx = slice(x-dx,x+dx+1,None)
            sy = slice(y-dy,y+dy+1,None)
            crop_im = im[:,:,sx,sy,...]
            lpatch.append(crop_im)
    return lpatch



def resizeDensityPatch(patch, opt_size):
    '''
    @brief: Take a density map and resize it to the opt_size.
    @param patch: input density map.
    @param opt_size: output size.
    @return: returns resized version of the density map.
    '''
    # Get patch size
    h, w = patch.shape[0:2]

    # Total sum
    patch_sum = patch.sum()

    # Normalize values between 0 and 1. It is in order to performa a resize.
    p_max = patch.max()
    p_min = patch.min()
    # Avoid 0 division
    if patch_sum !=0:
        patch = (patch - p_min)/(p_max - p_min)

    # Resize
    patch = resize(patch, opt_size)

    # Return back to the previous scale
    patch = patch*(p_max - p_min) + p_min

    # Keep count
    res_sum = patch.sum()
    if res_sum != 0:
        return patch * (patch_sum/res_sum)

    return patch



def resizeListDens(patch_list, psize):
    for ix, patch in enumerate(patch_list):
        # Keep count
        patch_list[ix] = resizeDensityPatch(patch, psize)

    return patch_list

def combinePatchList(density_list,im,N_scale):
    '''
    @density_list: a list of density map of each patch
    @im: original img
    @N_scale: number of different scales (exclue the whole image)
    '''
    h = im.shape[2]/8
    w = im.shape[3]/8
    dens_map = np.zeros( (h, w), dtype = np.float32 )
    count_map = np.zeros( (h, w), dtype = np.int32 )

    ph_base = int(h/(N_scale+1))
    pw_base = int(w/(N_scale+1))

    dens_index = 0

    for level in range(N_scale):
        ph = int(ph_base*(level+1))
        pw = int(pw_base*(level+1))
        lpos = genScalePos((im.shape[0]/8,im.shape[1]/8,im.shape[2]/8,im.shape[3]/8),ph,pw)
        dx = ph/2
        dy = pw/2
        for p in lpos:
            x,y = p
            sx = slice(x-dx,x+dx+1,None)
            sy = slice(y-dy,y+dy+1,None)
            pred = density_list[dens_index]
            pred[pred<0.0] = 0.0
            dens_map[sx,sy] += pred
            count_map[sx,sy] += 1
            dens_index = dens_index+1

     # Remove Zeros
    count_map[ count_map == 0 ] = 1

    # Average density map
    dens_map = dens_map / count_map
    return dens_map
