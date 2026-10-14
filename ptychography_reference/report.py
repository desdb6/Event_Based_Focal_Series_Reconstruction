import numpy as np
import matplotlib.pyplot as plt
from submodules.function import *

name = './results/12_type1_itemax100_stepsize1'
name = './results/12_typenophase'
obj = np.load(name+'_fresnel_obj_12.npy')
score = np.load(name+'_fresnel_score_12.npy')

name = './results/12_type1_itemax100_stepsize1'
obj4  = np.load(name+'_fresnel_obj_12.npy')
score4 = np.load(name+'_fresnel_score_12.npy')

dose_lst = [0,1e8,1e7,1e6,1e5]
n_config = [1,4,12]
# n_config = [1]
n_px = 12
n_px = [1,4,12]

fig, axes = plt.subplots(len(dose_lst), len(n_config), figsize = [2.5*len(n_config),2.5*len(dose_lst)])
for id in range(len(dose_lst)):
    for i, n_c in enumerate(n_config):
        o = np.angle(obj[id,i,256:512,256:512])
        o-=o.mean()
        axes[id,i].imshow(o, vmin=-np.pi, vmax=np.pi)
        axes[id,i].set_axis_off()
fig.tight_layout(pad=0)
plt.savefig(name+'_fresnel_obj_ph.png', dpi=600)

fig, axes = plt.subplots(len(dose_lst), len(n_config), figsize = [2.5*len(n_config),2.5*len(dose_lst)])
for id in range(len(dose_lst)):
    for i, n_c in enumerate(n_config):
        axes[id,i].imshow(np.abs(obj[id,i,256:512,256:512]), vmin=0, vmax=1)
        axes[id,i].set_axis_off()
fig.tight_layout(pad=0)
plt.savefig(name+'_fresnel_obj_amp.png', dpi=600)

# metrics = ['phase_L2', 'amplitude_L2', 'orthogonality']
# fig, axes = plt.subplots(len(dose_lst), len(metrics), figsize = [2.5*len(metrics),2.5*len(dose_lst)])
# for id in range(len(dose_lst)):
#     for im, m in enumerate(metrics):
#         for ic, c in enumerate(n_config):
#             axes[id,im].plot(score4[id,ic,im,1:],label = 'n='+str(c))
        
#         # axes[id,i].set_axis_off()
# axes[-1,-1].legend()
# fig.tight_layout(pad=0)
# plt.savefig(name+'_score14.png', dpi=600)

print('done')
