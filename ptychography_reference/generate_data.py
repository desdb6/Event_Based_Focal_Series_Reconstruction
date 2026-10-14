import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread
import cv2
from submodules.phaseplate import PP
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LogLocator, NullFormatter, Formatter
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from submodules.function import *
from submodules.gs import GS
from submodules.bases import hadamard, triangular
from scipy.special import erf
import gc

import torch
import time


print(torch.cuda.is_available())
torch.set_default_device('cuda')

#################################
##### SIMULATION PARAMETERS #####
#################################

cameraman = True

array_size = 64 + 64 #64 + 64  # 64 (image) + 2*32 (padding)
os = 32
image_path = "images/cameraman64.jpg"
phase_plate_path = "phaseplates/"
phase_plate_type = '12'
n_config = [2, 4, 16, 32, 64]
DF = 10 # Defocus for Near Field reconstruction
ite_max = 200
step_size = 1
b_phobj = True
fraction = 0.05
type = 1 # Per IW/EW pair update 0, batch update (avg all IW/EW) 1

RPI_range = 2*np.pi #Provide a maximum phase shift for the RPI, this applies (-RPI_range//2,RPI_range//2) shift
RPI_pxsize = 1 # Must be an integer, and power of 2 if the array_size is power of 2

dose_lst = np.logspace(np.log10(1e2)/max(n_config),np.log10(1e7),10)*max(n_config)

t0 = time.time()

#######################################
### Defining a few useful Functions ###
#######################################

def pp_img_preprocess(name,r=1):
    img = imread(name)[...,0].astype('uint16')
    img += img[::1,::1]
    img[img<=255] = 0
    img[img>255] = 255
    lx,ly = img.shape
    img1 = np.zeros((int(lx*r),int(ly*r)))
    img1[lx*(r//2):lx*(r//2+1),ly*(r//2):ly*(r//2+1)]=img
    return img1

def obj_phase_preprocess(name,wpoa_range = 2*np.pi,size=array_size):
    obj_0 = imread(name)[:,:,0]
    obj_0 = np.rot90(obj_0,2)
    obj_0 = normalize(obj_0,minmax=(-wpoa_range,wpoa_range))
    obj_0-= obj_0.mean()
    obj_0 = np.exp(1j*obj_0)
    pad   = (array_size - len(obj_0[0]))//2
    obj_0 = np.pad(obj_0,((pad,pad),(pad,pad)))[::-1]
    return obj_0
def obj_amp_preprocess(name,obj_min=0.95,size=array_size):
    obj_0 = imread(name)[:,:,0]
    pad   = (array_size - len(obj_0[0]))//2 
    obj_0 = np.pad(obj_0,((pad,pad),(pad,pad)))[::-1]
    obj_0 = np.rot90(obj_0,2)
    obj_0 = normalize(obj_0,minmax=(obj_min,1))
    return obj_0

def fft2d_t(arr):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(arr)))/(arr.shape[-1]*arr.shape[-2])
def ifft2d_t(arr):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(arr)))*(arr.shape[-1]*arr.shape[-2])
def prop(arr, direction):
    if direction == 1:
        return fft2d_t(arr)
    elif direction == -1:
        return ifft2d_t(arr)

def fresnel_t(arr, df):
    KX,KY,k_par = init_kspace(1e-3,array_size)
    K2 = KX**2 + KY**2

    return ifft2d_t(fft2d_t(arr)*torch.exp(1j*K2*df*0.5/k_par))
def ifresnel_t(arr, df):
    KX,KY,k_par = init_kspace(1e-3,array_size)
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
    KX, KY = torch.meshgrid(torch.from_numpy(kx).to('cuda'),torch.from_numpy(ky).to('cuda'),indexing='ij')
    return KX, KY, k_par

def Zernike_estimator(object, dose,mask_radius,os=32):
    #object is scaled to amplitude of 1 in each pixel
    #object is not padded
    n = object.size #figure out the size of the object eg n=64*64

    np.random.seed(0)
    obj_pad=np.pad(object,(os,os)) #pad the object with oversampling on each side
    array_size = np.size(obj_pad[0]) #size of 1 dimension of padded object
    oversample= array_size/object[0].size
    C=1/oversample


    zernike_mask = np.ones_like(obj_pad).astype('complex64') #set up a zernike phase plate

    if mask_radius!=0:
        zernike_mask[array_size//2-mask_radius:array_size//2+mask_radius,
                      array_size//2-mask_radius:array_size//2+mask_radius] *= np.exp(1j)
    else:
        zernike_mask[array_size//2,array_size//2] *= 1j #put zernike shift in center of padded bfp

    im_wave = ifft2d(fft2d(obj_pad)*zernike_mask) #image wave
    Iz  = np.abs(im_wave)**2   #image intensity

    Iz  = Iz/Iz.sum() #normalise to 1 electron in whole padded image

    if dose != 0:
        Izn = n*np.random.poisson(Iz*dose)/dose #this is now an object with 1 electron per pixel on average in the central patch and only if the zernike phase is off
    else:
        Izn = n*np.copy(Iz) #this is object with just 1 electron in it in total

    Izn_ob = Izn[os:-os,os:-os]

    z_phase = (Izn_ob- 1+2*C**2-2*C**4)/(2*C**2)
    z_phase-= z_phase.mean()

    z_error = z_phase - np.angle(object) 

    mask_pad  = np.ones_like(obj_pad,dtype=bool)
    mask_pad[os:-os,os:-os] = 0
    border    = Izn[mask_pad==1]
    center    = Izn[mask_pad==0]

    return z_phase, np.sqrt((z_error**2).mean()),(border.sum()/Izn.sum(),center.sum()/Izn.sum()), Izn

def phase_L2(cls):
    angle_obj = torch.angle(cls.obj[cls.idx]) - torch.angle(cls.obj[cls.idx]).mean()
    obj_d = angle_obj - torch.angle(cls.obj_0[cls.idx])
    obj_d-= obj_d.mean()
    return torch.sqrt((obj_d**2).mean())
def amp_L2(cls):
    obj_d = torch.abs(cls.obj[cls.idx]-cls.obj_0[cls.idx])
    return (obj_d**2).sum()
def ortho(cls):
    return (torch.abs(
        (cls.iw.conj()[:,None,cls.idx] * cls.prop(cls.ew, -1)[:,cls.idx] / cls.obj[cls.idx]).sum(-1) \
        - cls.I
        )).mean()

def phi(x):
    #normal distribution
    return (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)
def psi(x):
    #cumulative normal distribution
    return 0.5*(1+erf(x/np.sqrt(2)))

def vartruncnormal(lb,ub,sigma):
    #calculate the variance of a truncated normal distribution
    #lb is lower bound
    #ub is upper bound
    #assumes mean to be zero and ub>0 and lb<0
    sigma=np.array(sigma) #in case sigma would be a list
    beta=ub/sigma
    alpha=lb/sigma
    var=sigma**2*(1-(beta*phi(beta)-alpha*phi(alpha))/(psi(beta)-psi(alpha))-((phi(beta)-phi(alpha))/(psi(beta)-psi(alpha)))**2)
    return var

def N_RPI(N, n, padding=0,ill_range=np.pi):
    """
    Generate an (N+2*padding)x(N+2*padding) array with a checkerboard pattern of nxn sections filled with random numbers and zero padding.
    
    :param N: Size of the array (NxN)
    :param n: Size of each section (nxn)
    :param padding: Padding around the checkerboard pattern
    :return: (N+2*padding)x(N+2*padding) numpy array
    """
    # Ensure N is a multiple of n
    if N % n != 0:
        raise ValueError("N must be a multiple of n")
    
    # Calculate the number of sections along one dimension
    num_sections = N // n
    
    # Initialize the result array with zeros including padding
    total_size = N + 2 * padding
    result = np.zeros((total_size, total_size))
    
    for i in range(num_sections):
        for j in range(num_sections):
            # Generate a random number for each section
            random_value = np.random.uniform(-ill_range/2,ill_range/2)
            # Fill the nxn section with the random number
            result[padding + i*n:padding + (i+1)*n, padding + j*n:padding + (j+1)*n] = random_value
            
    return np.exp(1j*result)


######################
######################
#### BEGIN SCRIPT ####
######################
######################

pp1 = PP(pp_img_preprocess(f"{phase_plate_path}{phase_plate_type}.png", r=1))

np.random.seed(0)

phase = 'random' ## 'random' 'hadamard' 'both' 'flat'

if phase == 'random':
    phase_matrix = np.exp(np.random.rand(pp1.px_img.shape[0],pp1.px_img.shape[0])*np.pi*1j)  ## PURE RANDOM PHASE
elif phase == 'hadamard':
    phase_matrix = np.exp(hadamard(pp1.px_img.shape[0])*1j)                                  ## PURE HADAMARD BASIS
elif phase == 'both':
    phase_matrix = np.exp(hadamard(pp1.px_img.shape[0])*np.random.rand(pp1.n_px)*np.pi*1j)   ## HADAMARD + RANDOM
elif phase == 'flat':
    phase_matrix = np.exp(-1*np.ones((pp1.px_img.shape[0],pp1.px_img.shape[0]))*1j)          ## AMPLITUDE ONLY

iw_set = np.ones((max(n_config),array_size,array_size), 'complex64') ## (Num. Configs, array size, array size), type

# for i in range(phase_matrix.shape[0]):
#     for ii, idx in enumerate(pp1.px_idx):
#         np.add.at(iw_set[i], (idx[:,0],idx[:,1]), phase_matrix[i,ii]) ## I pass the iw_set as the set of KNOWN AMPLITUDES AND PHASES

iw_set /= ((abs(iw_set[0])**2).sum())**0.5 ## THIS ARE THE PARTIALLY KNOWN PHASE PLATE CONFIGURATIONS, THE SQRT(INT) OF THE REFERENCE IMAGE SHOULD BE HERE SOMEHOW
#iw_set = torch.from_numpy(iw_set).to('cuda')

ones  = np.ones_like(obj_amp_preprocess(image_path))
zernike = []
zerrors = []
Izns    = []


#######################################
######## Scaling of the WPOA  #########
####### Definition of the Obj #########
#######################################

#obj_0 = obj_phase_preprocess("images/smile.jpg",fraction = 0.05)*obj_amp_preprocess("images/cameraman.jpg") ## THIS IS UNKNOWN FOR MY EXPERIMENT

if cameraman:

    obj_0 = obj_phase_preprocess(image_path,wpoa_range=2*np.pi*fraction,size=array_size)*ones 
else:

    obj_0 = np.random.rand(int(array_size-2*os),int(array_size-2*os))-0.5
    obj_0 = normalize(obj_0,minmax=(-2*np.pi*fraction,2*np.pi*fraction))
    obj_0-= obj_0.mean()
    obj_0 = np.exp(1j*obj_0)
    pad   = (array_size - len(obj_0[0]))//2
    obj_0 = np.pad(obj_0,((pad,pad),(pad,pad)))

    obj_0 = ones*obj_0

# plt.figure()
# plt.imshow(np.abs(obj_0))
# plt.show()

for dose in dose_lst:
    z_img, z_err, counts_,izn = Zernike_estimator(obj_0[os:-os,os:-os],dose,mask_radius=0,os=os)
    zernike.append(z_img)
    zerrors.append(z_err)
    Izns.append(izn)


############################################
### SCANNING FOR AMPLITUDE MODULATED RPI ###
############################################

max_shift = 0
steps     = 1
list_x    = np.linspace(-max_shift//2,max_shift//2,num=steps,endpoint=True)
list_y    = np.linspace(-max_shift//2,max_shift//2,num=steps,endpoint=True)

############################################

metrics   = [phase_L2,amp_L2,ortho]
obj_arr   = np.zeros((len(dose_lst), len(n_config), iw_set.shape[1], iw_set.shape[2]),'complex64')
score_arr = np.zeros((len(dose_lst), len(n_config), len(metrics), ite_max+1))

#phase_hadamard = hadamard(array_size**2)

#############################################
### Random Phase Illumination block (RPI) ###
#############################################

RPI = np.ones_like(iw_set).astype('complex64')

for i in range(max(n_config)):
    RPI[i] *= N_RPI(array_size//2,RPI_pxsize,padding=os)
    #_hadamard = np.reshape(phase_hadamard[-i,:], (array_size//2,array_size//2))
    #RPI[i] *= np.exp(_hadamard*np.pi*1j)

RPI[:,:(array_size-2*os)//2,:]      = 0+0j ## Make sure that the illumination is 0 outside of the sample area if you use zero pad
RPI[:,2*os+(array_size-2*os)//2:,:] = 0+0j
RPI[:,:,:(array_size-2*os)//2]      = 0+0j
RPI[:,:,2*os+(array_size-2*os)//2:] = 0+0j

c_obj = np.copy(obj_0)
obj_0 = torch.from_numpy(obj_0).to('cuda')

##############################
### Gerchberg-Saxton Block ###
##############################

##############################
### Propagators:           ###
### fft2d_t => Farfield    ###
### prop_nf => Fresnel     ###
##############################

for i_d, dose in enumerate(dose_lst):

    iw_set = RPI

    for i, n_c in enumerate(n_config):

        d = dose / n_c

        for x_i in list_x:
            for y_i in list_y:
                print("Dose per Config: "+str(d)+" , # Config: "+str(n_c))
                _iw_set = np.roll(iw_set[:n_c], (int(y_i), int(x_i)), axis=(1,2))
                _iw_set = torch.from_numpy(_iw_set).to('cuda')
                ew_set  = torch.abs(fft2d_t(_iw_set*obj_0))**2 
                #ew_set  = torch.abs(fresnel_t(_iw_set*obj_0,DF))**2 ##fresnel_t #Choose where to propagate the wave (FF vs NF)

                gs = GS(
                        np.array(_iw_set.to('cpu')),
                        np.array(ew_set.to('cpu')),
                        prop, metrics, np.array(obj_0.to('cpu'))
                        ) 
                gs.run(type, ite_max, step_size, d, b_phobj, phase_matrix[:])

                print("Phase Error: "+str(gs.score[0,-1]))

                obj = gs.obj.to('cpu')
                idx = gs.idx.to('cpu')

                #obj_arr[i_d, i][idx] = np.mean([np.array(obj)[idx], obj_arr[i_d, i][idx]],axis=0) ##ADD THIS LINE WITH AMPLITUDE MODULATION
                obj_arr[i_d, i][idx] = np.array(obj)[idx]
                score_arr[i_d, i]    = np.copy(gs.score)
        
        #iw_set[n_c:] *= np.exp(-1j*np.angle(np.array(obj))*np.pi*1.85)

################################################################################################################

################################################################################################################
######################################## DEFINING THE PLOTTING FUNCTION ########################################
################################################################################################################

################################################################################################################

def plot_image_grid(images, nrows, ncols, shared_colorbar=False, symmetric_colorbar=False, titles=None, xlabel=None, ylabel=None, show_ticks=True,auto_range=True,crange=[-np.pi,np.pi], **kwargs):
    """
    Plots a grid of 2D images with optional shared colorbar, symmetric colorbar, axis labels, and titles.
    
    Parameters:
    - images: List of 2D numpy arrays to plot.
    - nrows: Number of rows in the grid.
    - ncols: Number of columns in the grid.
    - shared_colorbar: Boolean indicating whether to use a shared colorbar for all images.
    - symmetric_colorbar: Boolean indicating whether the colorbar should be symmetric around zero.
    - titles: List of titles for each subplot. Should be the same length as images.
    - xlabel: Label for the x-axis. If provided, applies to all subplots.
    - ylabel: List of labels for the y-axis. Should be the same length as images.
    - show_ticks: Boolean indicating whether to show ticks and labels on axes.
    - kwargs: Additional keyword arguments to pass to `imshow`.
    
    Note:
    - The number of images must be equal to nrows * ncols.
    - If shared_colorbar is True, the images should be on the same scale.
    """
    fig = plt.figure(figsize=(ncols * 5, nrows * 5), constrained_layout=True)  # Use constrained_layout
    
    # Create a GridSpec with space for colorbar
    gs = GridSpec(nrows + 1, ncols, height_ratios=[1] * nrows + [0.05], figure=fig, wspace=0.02, hspace=0.02)
    
    # Determine color range
    if auto_range:
        if symmetric_colorbar:
            max_abs_value = max(abs(img).max() for img in images)
            vmin, vmax = -max_abs_value, max_abs_value
        else:
            vmin = kwargs.get('vmin', min(img.min() for img in images))
            vmax = kwargs.get('vmax', max(img.max() for img in images))
    else:
        vmin,vmax = crange[0], crange[1]
    
    axs = []
    for i, img in enumerate(images):
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(img, vmin=vmin, vmax=vmax, aspect='equal', **kwargs)
        
        # Set title if provided
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=22)
        
        # Set y-axis label if provided
        if ylabel and i < len(ylabel):
            ax.set_ylabel(ylabel[i],fontsize=18)
        
        # Set common x-axis label if provided
        if xlabel:
            ax.set_xlabel(xlabel)
        
        # Handle tick marks
        if not show_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            ax.tick_params(axis='both', which='both', length=0)
        
        axs.append(ax)
    
    if shared_colorbar:
        # Add a horizontal colorbar spanning all columns
        cbar_ax = fig.add_subplot(gs[nrows, :])  # Use the last row for the colorbar
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', fraction=0.05, pad=0.02)  # Thinner colorbar with appropriate fraction
        cbar.ax.tick_params(labelsize=16)

    plt.show()

ew_set = np.array(ew_set.to('cpu'))

dose_lbl = ['{:.0e}'.format(float(x)) for x in dose_lst]

elapsed_time = (time.time() - t0)/60
print("Done, "+ str(ite_max)+ " iterations. Time Elapsed: "+str(round(elapsed_time,2))+ " minutes.")

plt.figure(2)
plt.title('GS Convergence',fontsize=22)

plt.semilogy(score_arr[5,0,0,:], 'y^-',label='Dose: ' + str(dose_lbl[5])+ ' $N_{configs}$: ' + str(n_config[0]))
plt.semilogy(score_arr[6,0,0,:], 'yo-',label='Dose: ' + str(dose_lbl[6])+ ' $N_{configs}$: ' + str(n_config[0]))
plt.semilogy(score_arr[7,0,0,:], 'yp-',label='Dose: ' + str(dose_lbl[7])+ ' $N_{configs}$: ' + str(n_config[0]))
plt.semilogy(score_arr[8,0,0,:], 'y*-',label='Dose: ' + str(dose_lbl[8])+ ' $N_{configs}$: ' + str(n_config[0]))

plt.semilogy(score_arr[5,1,0,:], 'c^-',label='Dose: ' + str(dose_lbl[5])+ ' $N_{configs}$: ' + str(n_config[1]))
plt.semilogy(score_arr[6,1,0,:], 'co-',label='Dose: ' + str(dose_lbl[6])+ ' $N_{configs}$: ' + str(n_config[1]))
plt.semilogy(score_arr[7,1,0,:], 'cp-',label='Dose: ' + str(dose_lbl[7])+ ' $N_{configs}$: ' + str(n_config[1]))
plt.semilogy(score_arr[8,1,0,:], 'c*-',label='Dose: ' + str(dose_lbl[8])+ ' $N_{configs}$: ' + str(n_config[1]))

plt.semilogy(score_arr[5,2,0,:], 'r^-',label='Dose: ' + str(dose_lbl[5])+ ' $N_{configs}$: ' + str(n_config[2]))
plt.semilogy(score_arr[6,2,0,:], 'ro-',label='Dose: ' + str(dose_lbl[6])+ ' $N_{configs}$: ' + str(n_config[2]))
plt.semilogy(score_arr[7,2,0,:], 'rp-',label='Dose: ' + str(dose_lbl[7])+ ' $N_{configs}$: ' + str(n_config[2]))
plt.semilogy(score_arr[8,2,0,:], 'r*-',label='Dose: ' + str(dose_lbl[8])+ ' $N_{configs}$: ' + str(n_config[2]))

plt.semilogy(score_arr[5,3,0,:], 'g^-',label='Dose: ' + str(dose_lbl[5])+ ' $N_{configs}$: ' + str(n_config[3]))
plt.semilogy(score_arr[6,3,0,:], 'go-',label='Dose: ' + str(dose_lbl[6])+ ' $N_{configs}$: ' + str(n_config[3]))
plt.semilogy(score_arr[7,3,0,:], 'gp-',label='Dose: ' + str(dose_lbl[7])+ ' $N_{configs}$: ' + str(n_config[3]))
plt.semilogy(score_arr[8,3,0,:], 'g*-',label='Dose: ' + str(dose_lbl[8])+ ' $N_{configs}$: ' + str(n_config[3]))

plt.semilogy(score_arr[5,4,0,:], 'b^-',label='Dose: ' + str(dose_lbl[5])+ ' $N_{configs}$: ' + str(n_config[4]))
plt.semilogy(score_arr[6,4,0,:], 'bo-',label='Dose: ' + str(dose_lbl[6])+ ' $N_{configs}$: ' + str(n_config[4]))
plt.semilogy(score_arr[7,4,0,:], 'bp-',label='Dose: ' + str(dose_lbl[7])+ ' $N_{configs}$: ' + str(n_config[4]))
plt.semilogy(score_arr[8,4,0,:], 'b*-',label='Dose: ' + str(dose_lbl[8])+ ' $N_{configs}$: ' + str(n_config[4]))

plt.xlabel('Iteration',fontsize=18)
plt.ylabel('RMS Error',fontsize=18)

# Number of columns in the legend
n_cols = 5

# Get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

# Create the legend with the specified number of columns and position it inside the plot
plt.legend(handles=handles, labels=labels, ncol=n_cols, loc='upper right',
           bbox_to_anchor=(1, 1),  # Position legend inside the plot area
           borderaxespad=0.1,  # Adjust the padding between the legend and plot edge
           frameon=True,  # Optionally add a border around the legend
           fontsize='small')  # Adjust font size if needed

plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

err_pr = [np.sqrt(2)/(2*np.sqrt(d_/((array_size- 2*os)**2-1))) for d_ in dose_lst]

def error_adjustment(cramer_vec,nRPI=1):
    #apply truncation correction
    lb=-np.pi
    ub=np.pi
    sigma_GS_trunc=np.sqrt(vartruncnormal(lb,ub,np.array(cramer_vec)*np.sqrt(nRPI))/nRPI)

    return sigma_GS_trunc

correction=4*np.sqrt(5/8)
n= array_size -2*os
sigma_zernike=1/(2*np.sqrt(dose_lst/(n*n-1)))
sigma_zernike_corrected=correction/(2*np.sqrt(dose_lst/(n*n-1)))
sigma_zernike_trunc=np.sqrt(vartruncnormal(-1,np.pi,sigma_zernike_corrected)) #conv zernike is lower truncated to -1
sigma_zernike_corrected_trunc=np.sqrt(vartruncnormal(-10/8,np.pi,sigma_zernike_corrected)) #2D zernike is truncated to -10/8 at bottom


err_pr_RPI2 = error_adjustment(err_pr,nRPI=2)
err_pr_RPI16= error_adjustment(err_pr,nRPI=16)
err_pr_RPI64= error_adjustment(err_pr,nRPI=64)

plt.figure(3)
plt.title('Mean RMS phase error vs dose (2D)',fontsize=22)
plt.loglog(dose_lst,score_arr[:,0,0,-1] ,'r*', label ='GS RPI '+str(n_config[0]),markersize=7)
plt.loglog(dose_lst,score_arr[:,2,0,-1] ,'ro', label ='GS RPI '+str(n_config[2]),markersize=7)
plt.loglog(dose_lst,score_arr[:,4,0,-1] ,'r^', label ='GS RPI '+str(n_config[4]),markersize=7)
plt.loglog(dose_lst,zerrors             ,'bv', label ='Zernike',markersize=7)
plt.loglog(dose_lst,err_pr              ,'k-.',label ='CRLB',markersize=7)

plt.loglog(dose_lst,err_pr_RPI2 ,'k-*',label ='CRLB RPI 2' ,markersize=5)
plt.loglog(dose_lst,err_pr_RPI16,'k-o',label ='CRLB RPI 16',markersize=5)
plt.loglog(dose_lst,err_pr_RPI64,'k-^',label ='CRLB RPI 64',markersize=5)

plt.loglog(dose_lst,sigma_zernike_corrected_trunc ,'b:',label ='Corrected Zernike Prediction',markersize=7)

plt.ylabel('RMS phase error',fontsize=18)
plt.xlabel('Electron budget',fontsize=18)

# Custom formatter to display ticks in 10^n format
class PowerFormatter(Formatter):
    def __init__(self, base=10, **kwargs):
        self.base = base
        super().__init__(**kwargs)

    def __call__(self, x, pos=0):
        if x <= 0:
            return ""
        if x == 1:
            return "1"
        # Determine the power of the base
        power = int(np.log10(x))
        if self.base ** power == x:
            return f"$10^{{{power}}}$"
        else:
            return ""

# Enable minor ticks
plt.minorticks_on()

# Setting major tick locator for x-axis
plt.gca().xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))

# Setting minor tick locator for x-axis
plt.gca().xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.1, 10, 1), numticks=100))

# Set major tick formatter to display as 10^n
plt.gca().xaxis.set_major_formatter(PowerFormatter(base=10))

# Optional: Remove minor tick labels to avoid clutter
plt.gca().xaxis.set_minor_formatter(NullFormatter())

# Customize tick parameters
plt.tick_params(axis='y', which='major', length=5, width=1.5, direction='out', labelsize=12)
plt.tick_params(axis='y', which='minor', length=3, width=1, direction='out')
plt.tick_params(axis='x', which='major', length=5, width=1.5, direction='out', labelsize=12)
plt.tick_params(axis='x', which='minor', length=3, width=1, direction='out')

plt.legend(fontsize=16)

gsrpi_    = [np.angle(obj_)[os:-os,os:-os]-np.angle(obj_)[os:-os,os:-os].mean() for obj_ in obj_arr[:,-3]]

zernike_res = [z_ph - np.angle(c_obj)[os:-os,os:-os] for z_ph in zernike]
rpi_res     = [g_ph - np.angle(c_obj)[os:-os,os:-os] for g_ph in gsrpi_]
gt_powspec  = np.abs(fft2d(np.angle(c_obj)[os:-os,os:-os]))**2


plt.figure(4)
plt.title('Zernike Intensity with $N_e$ = '+str(dose_lbl[5]),fontsize=22)
plt.imshow(Izns[5])
plt.axis('off')
# Add a textbox to the top right corner
plt.text(0.98,0.98, 'Reference counts = 3/8', 
         fontsize=20, ha='right', va='top', transform=plt.gca().transAxes,color='white',
         fontweight='bold',bbox=dict(facecolor='none',edgecolor='none'))

plt.text(0.73, 0.73, 'Object counts = 5/8',
         fontsize=20, ha='right', va='top', transform=plt.gca().transAxes,
         fontweight='bold',bbox=dict(facecolor='none',edgecolor='none'))


# plots = [zernike[1],np.angle(obj_arr[1,-3])[os:-os,os:-os]-np.angle(obj_arr[1,-3])[os:-os,os:-os].mean(),
#          zernike[3],np.angle(obj_arr[3,-3])[os:-os,os:-os]-np.angle(obj_arr[3,-3])[os:-os,os:-os].mean(),
#          zernike[5],np.angle(obj_arr[5,-3])[os:-os,os:-os]-np.angle(obj_arr[5,-3])[os:-os,os:-os].mean(),
#          zernike[7],np.angle(obj_arr[7,-3])[os:-os,os:-os]-np.angle(obj_arr[7,-3])[os:-os,os:-os].mean()]
# y_lbl = ['$N_e$='+str(dose_lbl[1]),"",
#          '$N_e$='+str(dose_lbl[3]),"",
#          '$N_e$='+str(dose_lbl[5]),"",
#          '$N_e$='+str(dose_lbl[7]),""]
plots = [zernike[1],
         zernike[3],
         zernike[5],
         zernike[7],
         np.angle(obj_arr[1,-3])[os:-os,os:-os]-np.angle(obj_arr[1,-3])[os:-os,os:-os].mean(),
         np.angle(obj_arr[3,-3])[os:-os,os:-os]-np.angle(obj_arr[3,-3])[os:-os,os:-os].mean(),
         np.angle(obj_arr[5,-3])[os:-os,os:-os]-np.angle(obj_arr[5,-3])[os:-os,os:-os].mean(),
         np.angle(obj_arr[7,-3])[os:-os,os:-os]-np.angle(obj_arr[7,-3])[os:-os,os:-os].mean()]
t_lbl = ['$N_e$='+str(dose_lbl[1]),
         '$N_e$='+str(dose_lbl[3]),
         '$N_e$='+str(dose_lbl[5]),
         '$N_e$='+str(dose_lbl[7]),"","","",""]
y_lbl = ['Zernike','','','','RPI 16','','','']

plot_image_grid(plots,2,4,shared_colorbar=True, symmetric_colorbar=True, 
                titles=t_lbl,ylabel=y_lbl,
                show_ticks=False,auto_range=False,crange=[-0.5,0.5],cmap='viridis')


data = [zernike_res[4],rpi_res[4],zernike_res[5],rpi_res[5]]
data_lbl = ['Zernike',"RPI","",""]
dose_lbl = ['$N_e$='+str(dose_lbl[4]),"",
            '$N_e$='+str(dose_lbl[5]),""]


plot_image_grid(data ,2,2,shared_colorbar=True, symmetric_colorbar=True,
                titles=['Zernike','RPI 16','',''],ylabel=dose_lbl,show_ticks=False,cmap='viridis')

ni = 3

plots = [np.angle(obj_arr[ni,0])[os:-os,os:-os]-np.angle(obj_arr[ni,0])[os:-os,os:-os].mean(),
         np.angle(obj_arr[ni,1])[os:-os,os:-os]-np.angle(obj_arr[ni,1])[os:-os,os:-os].mean(),
         np.angle(obj_arr[ni,2])[os:-os,os:-os]-np.angle(obj_arr[ni,2])[os:-os,os:-os].mean(),
         np.angle(obj_arr[ni,3])[os:-os,os:-os]-np.angle(obj_arr[ni,3])[os:-os,os:-os].mean()]

plot_image_grid(plots ,2,2,shared_colorbar=True, symmetric_colorbar=True,
                titles=['RPI 2','RPI 4','RPI 16','RPI 32'],show_ticks=False,cmap='viridis')

# data2 = [np.divide(gt_powspec, np.abs(fft2d(arr))**2, out=np.zeros_like(gt_powspec), where=(arr)!=0) for arr in data]

# ###############################
# ### Save Figures or Objects ###
# ###############################

# #savename = pp_type+'_'+phase+'_type'+str(type)+'_itemax'+str(ite_max)+'_stepsize'+str(step_size)
# #np.save('./results/'+savename+'_fresnel_obj_2.npy', obj_arr)
# #np.save('./results/'+savename+'_fresnel_score_2.npy', score_arr)