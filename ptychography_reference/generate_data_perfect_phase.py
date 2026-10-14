import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread
import cv2
from submodules.phaseplate import PP

from submodules.function import *
from submodules.gs import GS
from submodules.bases import hadamard, triangular

import torch

print(torch.cuda.is_available())

## IMPORT AND CREATE A PHASE PLATE FROM A BINARY MASK

def pp_img_preprocess(name):
    img = imread(name)[...,0].astype('uint16')
    img += img[::-1,::-1]
    img[img<=255] = 0
    img[img>255] = 255
    lx,ly = img.shape
    r = 3
    img1 = np.zeros((lx*r,ly*r))
    img1[lx*(r//2):lx*(r//2+1),ly*(r//2):ly*(r//2+1)]=img
    return img1

def obj_phase_preprocess(name):
    obj_0 = imread(name)[:256,:256,0]
    obj_0 = np.pad(obj_0,((256,256),(256,256)))[::-1]
    obj_0 = np.rot90(obj_0,2)
    obj_0 = normalize(obj_0,minmax=(0,1))
    obj_0 = np.exp(1j*obj_0*np.pi)
    return obj_0
def obj_amp_preprocess(name):
    obj_0 = imread(name)[-256:,-256:,0]
    obj_0 = np.pad(obj_0,((256,256),(256,256)))[::-1]
    obj_0 = np.rot90(obj_0,2)
    obj_0 = normalize(obj_0,minmax=(0.5,1))
    return obj_0

def fft2d_t(arr):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(torch.tensor(arr))))/(arr.shape[-1]*arr.shape[-2])**0.5
def ifft2d_t(arr):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(torch.tensor(arr))))*(arr.shape[-1]*arr.shape[-2])**0.5
def prop(arr, direction):
    if direction == 1:
        return fft2d_t(arr)
    elif direction == -1:
        return ifft2d_t(arr)

def fresnel_t(arr, df):
    KX,KY,k_par = init_kspace(1.37e-8,768)
    K2 = KX**2 + KY**2
    return ifft2d_t(fft2d_t(arr)*torch.exp(1j*K2*df*0.5/k_par))
def ifresnel_t(arr, df):
    KX,KY,k_par = init_kspace(1.37e-8,768)
    K2 = KX**2 + KY**2
    return ifft2d_t(fft2d_t(arr)*torch.exp(-1j*K2*df*0.5/k_par))

def prop_nf(arr, direction):
    df = DF
    if direction == 1:
        return fresnel_t(arr, df)
    elif direction == -1:
        return ifresnel_t(arr, df)

def init_kspace(pixel: float, npointsx: int = 2048, npointsy: int = None , wavelength: float = 1.97e-12) -> tuple:
    k_par = 2*np.pi/wavelength
    if npointsy == None :
        npointsy = npointsx
        
    kxmax = k_par*pixel*npointsx/2
    kymax = k_par*pixel*npointsy/2
    kx = np.linspace(-kxmax,kxmax, npointsx,endpoint=False)
    ky = np.linspace(-kymax,kymax, npointsy,endpoint=False)
    KX, KY = torch.meshgrid(torch.from_numpy(kx),torch.from_numpy(ky))
    return KX, KY, k_par
    
def phase_L2(cls):
    obj_d = np.angle(cls.obj[cls.idx]/cls.obj_0[cls.idx])
    obj_d -= obj_d.mean()
    return (obj_d**2).sum()
def amp_L2(cls):
    obj_d = np.abs(cls.obj[cls.idx]-cls.obj_0[cls.idx])
    return (obj_d**2).sum()
def ortho(cls):
    return (
        torch.abs((cls.iw.conj()[:,None,cls.idx] * cls.prop(cls.ew, -1)[:,cls.idx] / cls.obj[cls.idx]).sum((-1))) \
        - torch.eye(cls.ew.shape[0])
        ).mean()

DF = 5

ab_dict = {} ##ADD MICROSCOPE ABERRATIONS TO TEST WHEN DO I GO OUTSIDE OF THE TOLERABLE AB

pp_type = 'cartesian'
pp1 = PP(pp_img_preprocess("phaseplates/"+pp_type+".png"))
np.random.seed(0)

phase = 'random' ## 'random' 'flat'  'both' 'hadamard'

if phase == 'random':
    phase_matrix = np.exp(1j*np.random.rand(pp1.n_px,pp1.n_px)*2*np.pi)                        ## PURE RANDOM PHASE
elif phase == 'hadamard':
    phase_matrix = hadamard(pp1.px_img.shape[0])                                               ## PURE HADAMARD BASIS
elif phase == 'both':   
    phase_matrix = hadamard(pp1.px_img.shape[0]) * np.exp(1j*np.random.rand(pp1.n_px)*2*np.pi) ## HADAMARD + RANDOM
elif phase == 'flat':   
    phase_matrix = np.ones((36,36))                                                            ## AMPLITUDE ONLY

pp_type = 'cartesian'
pp1 = PP(pp_img_preprocess("phaseplates/"+pp_type+".png"))
np.random.seed(0)
#phase_matrix = hadamard(pp1.px_img.shape[0]) * np.exp(1j*np.random.rand(pp1.n_px)*2*np.pi)
iw_set = np.zeros((phase_matrix.shape[0], pp1.raw.shape[0],pp1.raw.shape[1]), 'complex64')
for i in range(phase_matrix.shape[0]):
    for ii, idx in enumerate(pp1.px_idx):
        np.add.at(iw_set[i], (idx[:,0],idx[:,1]), phase_matrix[i,ii])
iw_set /= ((abs(iw_set[0])**2).sum())**0.5
iw_set = torch.tensor(iw_set)
obj_0 = obj_phase_preprocess("images/lena.jpg")*obj_amp_preprocess("images/cameraman.jpg")
obj_1 = np.conj(obj_0)[::-1,::-1]
# ew_set = np.abs(fft2d_t(iw_set*obj_0))**2
ew_set = np.abs(fresnel_t(iw_set*obj_0, DF))**2
# ew_set2 = np.abs(fft2d_t(iw_set*obj_1))**2


type = 1
ite_max = 10
step_size = 1
b_phobj = False


dose_lst = [0,1e8,1e7,1e5]
n_config = [1,4]
metrics = [phase_L2,amp_L2,ortho]
obj_arr = np.zeros((len(dose_lst), iw_set.shape[0], iw_set.shape[1], iw_set.shape[2]),'complex64')
score_arr = np.zeros((len(dose_lst), len(n_config), len(metrics), ite_max+1))
for i_d, dose in enumerate(dose_lst):
    for i, n_c in enumerate(n_config):
        d = dose / n_c

        gs = GS(
            np.array(iw_set[:n_c]),
            np.array(ew_set[:n_c]),
            prop_nf, metrics, obj_0
            ) #prop_nf
        gs.run(type, 0, step_size, d, b_phobj, phase_matrix[:n_c])

        gs.obj = torch.clone(gs.obj_0)
        _ew = gs.prop(gs.iw * gs.obj,1)
        gs.ew = torch.exp(1j*torch.angle(_ew)) * torch.abs(gs.ew) 
        _obj = (gs.prop(gs.ew,-1) / gs.iw).mean(0)
        gs._update_obj(_obj, step_size, b_phobj)
        gs._metric()

        print(gs.score[-1,-1])
        obj_arr[i_d, i][gs.idx] = np.array(gs.obj)[gs.idx]
        score_arr[i_d, i] = np.copy(gs.score)

savename = pp_type+'_type'+str(type)+'_itemax'+str(ite_max)+'_stepsize'+str(step_size)
np.save('./results/'+savename+'_perf_fresnel_obj_8.npy', obj_arr)
np.save('./results/'+savename+'_perf_fresnel_score_8.npy', score_arr)


print('done')







