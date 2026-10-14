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
def colarbar(ax,im,ticks,name=None):
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes("left", size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.set_ticks_position('left')
    ax_cb.set_yticks(ticks)
    ax_cb.tick_params(axis='y', labelsize=15)
    if name is not None:
        ax_cb.set_xlabel(name,fontsize=15)
        ax_cb.xaxis.set_label_position('top') 

fig_main = plt.figure(layout='constrained', figsize=(11, 5))
subfigs = fig_main.subfigures(1, 3, wspace=0.01, width_ratios=[0.6,0.4,1])

obj_0 = obj_phase_preprocess("images/cameraman.jpg")*obj_amp_preprocess("images/waves.jpg")
o_phase = (np.angle(obj_0) - np.angle(obj_0).min() - np.pi/2)[256:512,256:512]
o_conj_phase = o_phase[::-1,::-1]*-1
o_amp = np.abs(obj_0)[256:512,256:512]

fig = subfigs[0]
size_constant = 5
axes = fig.subplots(2,1)
im0=axes[0].imshow(o_phase, vmin=-np.pi/2, vmax=np.pi/2)
im1=axes[1].imshow(o_amp, vmin=0, vmax=1.5)
axes[0].set_title('Phase',fontsize=15)
axes[1].set_title('Amp.',fontsize=15)
axes[0].set_axis_off()
axes[1].set_axis_off()
# axes[2].set_axis_off()
colarbar(axes[0],im0,np.linspace(-1.5,1.5,6),'rad')
colarbar(axes[1],im1,np.linspace(0,1,6))
fig.suptitle('Groud Truth', fontsize=15)

fig = subfigs[1]
size_constant = 5
axes = fig.subplots(2,1)
im0=axes[0].imshow(o_conj_phase, vmin=-np.pi/2, vmax=np.pi/2)
im1=axes[1].imshow(o_amp[::-1,::-1], vmin=0, vmax=1.5)
axes[0].set_title('Phase',fontsize=15)
axes[1].set_title('Amp.',fontsize=15)
axes[0].set_axis_off()
axes[1].set_axis_off()
# axes[2].set_axis_off()
fig.suptitle('Complex Conj.', fontsize=15)

fig = subfigs[2]
size_constant = 5
ax = fig.subplots(1,1)
ax.imshow(abs(fft2d(obj_0[300:-300,300:-300]))**0.5)
ax.set_axis_off()
fig.suptitle('Diffraction', fontsize=15)

plt.savefig('./results/FIG_pr_twin.png', dpi=300)




print('done')

