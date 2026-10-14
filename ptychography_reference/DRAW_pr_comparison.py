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
from mpl_toolkits.axes_grid1 import make_axes_locatable

pad = 256
obj_size = 768

def obj_phase_preprocess(name):
    obj_0 = imread(name)[...,0]
    obj_0 = np.pad(obj_0,((pad,pad),(pad,pad)))[::-1]
    obj_0 = np.rot90(obj_0,2)
    obj_0 = normalize(obj_0,minmax=(0,1))
    obj_0 = np.exp(1j*obj_0*np.pi)
    return obj_0
def obj_amp_preprocess(name):
    obj_0 = imread(name)[...,0]
    obj_0 = np.pad(obj_0,((pad,pad),(pad,pad)))[::-1]
    obj_0 = np.rot90(obj_0,2)
    obj_0 = normalize(obj_0,minmax=(0.5,1))
    return obj_0
def colarbar(ax,im,ticks,name=None):
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes("left", size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.set_ticks_position('left')
    #ax_cb.set_yticks(ticks=ticks,fontsize=15)
    if name is not None:
        ax_cb.set_xlabel(name,fontsize=15)
        ax_cb.xaxis.set_label_position('top') 

fig_main = plt.figure(layout='constrained', figsize=(20, 5))
subfigs = fig_main.subfigures(1, 5, wspace=0.1, width_ratios=[1,1,1,1,1])

############################ GT ############################

obj_0 = obj_phase_preprocess("images/smile.jpg")*obj_amp_preprocess("images/cameraman.jpg")
o_phase = np.angle(obj_0) - np.angle(obj_0).min() - np.pi/2
o_amp   = np.abs(obj_0)
fig = subfigs[0]

size_constant = 5
axes = fig.subplots(2,1)
im0=axes[0].imshow(o_phase[pad:2*pad,pad:2*pad], vmin=-np.pi/2, vmax=np.pi/2)
im1=axes[1].imshow(o_amp[pad:2*pad,pad:2*pad], vmin=0, vmax=1.5)
axes[0].set_title('Phase',fontsize=15)
axes[1].set_title('Amp.',fontsize=15)
axes[0].set_axis_off()
axes[1].set_axis_off()
# axes[2].set_axis_off()
colarbar(axes[0],im0,np.linspace(-1.5,1.5,6),'rad')
colarbar(axes[1],im1,np.linspace(.5,1,6))
fig.suptitle('Ground Truth', fontsize=15)

############################ no phase ############################

name = './results/segmented_flat_type1_itemax50_stepsize1'
obj = np.load(name+'_fresnel_obj.npy')
dose_lst = [0,1e7,1e5]
fig = subfigs[1]

axes = fig.subplots(len(dose_lst), 2)
for ii in range(len(dose_lst)):
    id = ii
    o = np.angle(obj[id,0,pad:2*pad,pad:2*pad])
    # o-=o.mean()
    axes[ii,0].imshow(o, vmin=-np.pi/2, vmax=np.pi/2)
    axes[ii,0].set_xticks([])
    axes[ii,0].set_yticks([])
    axes[ii,1].imshow(np.abs(obj[id,-1,pad:2*pad,pad:2*pad]), vmin=0, vmax=1.5)
    axes[ii,1].set_xticks([])
    axes[ii,1].set_yticks([])
    if ii == 0:
        axes[ii,0].set_ylabel('Dose = Inf.',fontsize=15)
    else:
        axes[ii,0].set_ylabel("{:.0e}".format(dose_lst[ii]),fontsize=15)
axes[0,0].set_title('Phase',fontsize=15)
axes[0,1].set_title('Amp.',fontsize=15)
fig.suptitle('Amp. Modulation, NF', fontsize=15)

############################ random phase ############################

name = './results/segmented_both_type1_itemax50_stepsize1'
obj = np.load(name+'_fresnel_obj_2.npy')
dose_lst = [0,1e7,1e5]
fig = subfigs[2]

axes = fig.subplots(len(dose_lst), 2)
for ii in range(len(dose_lst)):
    id = ii
    o = np.angle(obj[id,0,256:512,256:512])
    # o-=o.mean()
    axes[ii,0].imshow(o, vmin=-np.pi/2, vmax=np.pi/2)
    axes[ii,0].set_xticks([])
    axes[ii,0].set_yticks([])
    axes[ii,1].imshow(np.abs(obj[id,-1,256:512,256:512]), vmin=0, vmax=1.5)
    axes[ii,1].set_xticks([])
    axes[ii,1].set_yticks([])
axes[0,0].set_title('Phase',fontsize=15)
axes[0,1].set_title('Amp.',fontsize=15)
fig.suptitle('Amp. & 2 Phase, NF', fontsize=15)

############################ hadamard phase config ############################

name = './results/segmented_flat_type1_itemax50_stepsize1'
obj = np.load(name+'_farf_obj.npy')
dose_lst = [0,1e7,1e5]
fig = subfigs[3]

axes = fig.subplots(len(dose_lst), 2)
for ii in range(len(dose_lst)):
    id = ii
    o = np.angle(obj[id,0,256:512,256:512])
    # o-=o.mean()
    axes[ii,0].imshow(o, vmin=-np.pi/2, vmax=np.pi/2)
    axes[ii,0].set_xticks([])
    axes[ii,0].set_yticks([])
    axes[ii,1].imshow(np.abs(obj[id,-1,256:512,256:512]), vmin=0, vmax=1.5)
    axes[ii,1].set_xticks([])
    axes[ii,1].set_yticks([])
axes[0,0].set_title('Phase',fontsize=15)
axes[0,1].set_title('Amp.',fontsize=15)
fig.suptitle('Amp. Modulation, FF', fontsize=15)


############################ random + hadamard phase config ############################

name = './results/segmented_both_type1_itemax50_stepsize1'
obj = np.load(name+'_farf_obj_2.npy')
dose_lst = [0,1e7,1e5]
fig = subfigs[4]

axes = fig.subplots(len(dose_lst), 2)
for ii in range(len(dose_lst)):
    id = ii
    o = np.angle(obj[id,0,256:512,256:512])
    # o-=o.mean()
    axes[ii,0].imshow(o, vmin=-np.pi/2, vmax=np.pi/2)
    axes[ii,0].set_xticks([])
    axes[ii,0].set_yticks([])
    axes[ii,1].imshow(np.abs(obj[id,-1,256:512,256:512]), vmin=0, vmax=1.5)
    axes[ii,1].set_xticks([])
    axes[ii,1].set_yticks([])
axes[0,0].set_title('Phase',fontsize=15)
axes[0,1].set_title('Amp.',fontsize=15)
fig.suptitle('Amp. & 2 Phase, FF', fontsize=15)

plt.savefig('./results/FIG_segmented_fresnel_50ite_FFvNF.png', dpi=300) #FIG_cartesian_fresnel_50ite_compare_4_ogd

#pltplot = -np.angle(obj_phase_preprocess("images/smile.jpg")[::-1])

print('done')