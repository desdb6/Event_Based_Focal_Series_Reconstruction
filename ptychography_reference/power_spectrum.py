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
from PIL import Image
import colorsys
import torch
import time
import scipy.ndimage


print(torch.cuda.is_available())
torch.set_default_device('cuda')

#################################
##### SIMULATION PARAMETERS #####
#################################

array_size = 64 + 448 #64 + 64  # 64 (image) + 2*32 (padding)
os = 224

phase_plate_path = "phaseplates/"
phase_plate_type = 'Seg64'

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

def rescale_array(input_png, new_x, new_y,save_as_png=False,filename='PPX.png'):
    # Open the image and convert to grayscale (if you want a 2D array)
    image = Image.open(input_png).convert('L')  # 'L' mode is for grayscale
    array = np.array(image)

    # Calculate zoom factors for each dimension
    zoom_x = new_x / array.shape[0]
    zoom_y = new_y / array.shape[1]
    
    # Rescale the array using the zoom factors
    rescaled_array = scipy.ndimage.zoom(array, (zoom_x, zoom_y), order=0)  # order=1 for bilinear interpolation
    
    # Convert the rescaled array to binary using the threshold
    binary_array = (rescaled_array > 128).astype(np.uint8) * 255 

    # If save_as_png is True, save the array as a PNG image
    if save_as_png:
        plt.imsave(filename, rescaled_array, cmap='gray')  # Save using grayscale colormap
        print(f"Array saved as {filename}")

    return rescaled_array

def fft2d_t(arr):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(arr)))/(arr.shape[-1]*arr.shape[-2])
def ifft2d_t(arr):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(arr)))*(arr.shape[-1]*arr.shape[-2])

def fresnel_t(arr, df):
    KX,KY,k_par = init_kspace(1e-3,array_size)
    K2 = KX**2 + KY**2
    return ifft2d_t(fft2d_t(arr)*torch.exp(1j*K2*df*0.5/k_par))
def ifresnel_t(arr, df):
    KX,KY,k_par = init_kspace(1e-3,array_size)
    K2 = KX**2 + KY**2
    return ifft2d_t(fft2d_t(arr)*torch.exp(-1j*K2*df*0.5/k_par))

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

def plot_complex_array(complex_array):
    # Get the modulus (magnitude) and angle (phase) of the complex array
    modulus = np.abs(complex_array)
    angle = np.angle(complex_array)
    
    # Normalize modulus and angle for plotting
    modulus_normalized = modulus / np.max(modulus)  # Normalize modulus to range [0, 1]
    angle_normalized = (angle + np.pi) / (2 * np.pi)  # Normalize angle to range [0, 1]
    
    # Create an empty RGB image
    height, width = complex_array.shape
    rgb_image = np.zeros((height, width, 3))
    
    # Convert HSV to RGB for each pixel
    for i in range(height):
        for j in range(width):
            hue = angle_normalized[i, j]  # Hue corresponds to phase
            value = modulus_normalized[i, j]  # Value corresponds to modulus (intensity)
            saturation = 1.0  # Full saturation for clear color mapping
            rgb_image[i, j] = colorsys.hsv_to_rgb(hue, saturation, value)
    
    # Plot the resulting RGB image
    plt.imshow(rgb_image, origin='lower')
    plt.axis('off')  # Turn off axis for better visualization
    
    plt.show()

def radial_integrated_intensity(arrays, labels=None,styles=None, center=None, num_bins=100, crop_=75, save_as_png=False, output_filename="radial_intensity.png"):
    """
    Compute and plot the radially integrated intensity for multiple 2D arrays.
    
    Parameters:
        arrays (list of 2D numpy arrays): List of input intensity arrays.
        labels (list of str): Labels for each array in the plot. Defaults to None.
        center (tuple): Coordinates of the center (x_center, y_center). If None, defaults to the center of the array.
        num_bins (int): Number of radial bins.
        save_as_png (bool): If True, save the plot as a PNG file.
        output_filename (str): Filename for saving the PNG plot.
        
    Returns:
        radial_distances (list of 1D numpy arrays): The radial distances (bin centers) for each array.
        radial_intensities (list of 1D numpy arrays): The radially integrated intensity for each array.
    """
    if labels is None:
        labels = [f"Array {i+1}" for i in range(len(arrays))]  # Default labels if not provided

    if styles is None:
        styles = ['-' for _ in range(len(arrays))]  # Default to solid lines if no styles are provided

    radial_distances = []
    radial_intensities = []

    # Set up the plot
    plt.figure()
    
    for array, label, style in zip(arrays, labels, styles):
        # Get array dimensions
        height, width = array.shape

        # Determine the center of the array if not provided
        if center is None:
            center = (width // 2, height // 2)

        x_center, y_center = center

        # Create arrays of x and y coordinates
        y, x = np.indices((height, width))

        # Calculate radial distance from the center for each point
        r = np.sqrt((x - x_center)**2 + (y - y_center)**2)

        # Flatten the array and the corresponding radial distances
        r = r.ravel()
        intensity = array.ravel()

        # Bin the radial distances
        max_radius = np.max(r)
        bin_edges = np.linspace(0, max_radius, num_bins + 1)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # Compute the radial integrated intensity by averaging intensity values in each radial bin
        radial_intensity = np.zeros(num_bins)
        for i in range(num_bins):
            in_bin = (r >= bin_edges[i]) & (r < bin_edges[i+1])
            radial_intensity[i] = np.mean(intensity[in_bin]) if np.sum(in_bin) > 0 else 0

        # Plot the result with the given style
        plt.plot(bin_centers[:crop_], radial_intensity[:crop_], style, label=label)

        # Store results
        radial_distances.append(bin_centers)
        radial_intensities.append(radial_intensity)
    
    # Customize the plot
    plt.xlabel("Spatial Frequency (pixels -> 1/nm)")
    plt.ylabel("Radially Integrated Intensity")
    plt.title("Power Spectrum")
    plt.legend()  # Show the legend with array labels
    
    # Save the plot if requested
    if save_as_png:
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
        print(f"Radial intensity plot saved as {output_filename}")

    return radial_distances, radial_intensities

def P_matrix(phase):

    np.random.seed(0)

    if phase == 'random':
        phase_matrix = np.exp(np.random.rand(pp1.px_img.shape[0],pp1.px_img.shape[0])*np.pi*1j)  ## PURE RANDOM PHASE
    elif phase == 'hadamard':
        phase_matrix = np.exp(hadamard(pp1.px_img.shape[0])*1j)                                  ## PURE HADAMARD BASIS
    elif phase == 'both':
        phase_matrix = np.exp(hadamard(pp1.px_img.shape[0])*np.random.rand(pp1.n_px)*np.pi*1j)   ## HADAMARD + RANDOM
    elif phase == 'flat':
        phase_matrix = np.exp(-1*np.ones((pp1.px_img.shape[0],pp1.px_img.shape[0]))*1j)          ## AMPLITUDE ONLY

    return phase_matrix

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

def Circle(shape, radius, center=None):
    """
    Create a boolean circular mask for a 2D array.
    
    Parameters:
        shape (tuple): Shape of the output mask (height, width).
        radius (float): Radius of the circle.
        center (tuple): Coordinates of the center (x_center, y_center). If None, defaults to the center of the array.
    
    Returns:
        mask (2D numpy array): Boolean circular mask with True inside the circle and False outside.
    """
    height, width = shape

    if center is None:
        center = (width // 2, height // 2)
    
    x_center, y_center = center

    # Create a 2D grid of x and y coordinates
    y, x = np.indices((height, width))

    # Calculate the radial distance from the center for each point
    r = np.sqrt((x - x_center)**2 + (y - y_center)**2)

    # Create the boolean circular mask
    mask = r <= radius

    return mask


#################################################
#################################################
#################################################
######## BEGIN SCRIPT, DEFINE PhasePlate ########
#################################################
#################################################
#################################################

#_ = rescale_array(f"{phase_plate_path}{phase_plate_type}.png",64,64,save_as_png=True,filename='Seg64.png')

pow_specs = []
labels = []
styles = ['r*-', 'bo-','g^-','yv-','r-']

pp1 = PP(pp_img_preprocess(f"{phase_plate_path}{phase_plate_type}.png", r=1))

iw = np.zeros((64,64), 'complex64')

phase_matrix = P_matrix('random')
for ii, idx in enumerate(pp1.px_idx):
    np.add.at(iw, (idx[:,0],idx[:,1]), phase_matrix[47,ii]) ## I pass the iw_set as the set of KNOWN AMPLITUDES AND PHASES

_iw = np.pad(iw,os)
_iwc = np.copy(_iw)
probe = np.abs(fft2d(_iw)**2)
#probe/= np.sum(probe)

pow_spec = abs(fft2d(probe))

pow_specs.append(pow_spec)
labels.append('random')

phase_matrix = P_matrix('hadamard')
for ii, idx in enumerate(pp1.px_idx):
    np.add.at(iw, (idx[:,0],idx[:,1]), phase_matrix[47,ii]) ## I pass the iw_set as the set of KNOWN AMPLITUDES AND PHASES

_iw = np.pad(iw,os)
probe = np.abs(fft2d(_iw)**2)
#probe/= np.sum(probe)

pow_spec = abs(fft2d(probe))

pow_specs.append(pow_spec)
labels.append('Hadamard')

phase_matrix = P_matrix('flat')
for ii, idx in enumerate(pp1.px_idx):
    np.add.at(iw, (idx[:,0],idx[:,1]), phase_matrix[47,ii]) ## I pass the iw_set as the set of KNOWN AMPLITUDES AND PHASES

_iw = np.pad(iw,os)
probe = np.abs(fft2d(_iw)**2)
#probe/= np.sum(probe)

pow_spec = abs(fft2d(probe))

pow_specs.append(pow_spec)
labels.append('flat')

rpi = N_RPI(64, 1, padding=os,ill_range=np.pi)*Circle((array_size,array_size),32)

probe = np.abs(fft2d(rpi)**2)
#probe/= np.sum(probe)

pow_spec = abs(fft2d(probe))

pow_specs.append(pow_spec)
labels.append('Round + Speckle')

round = Circle((array_size,array_size),32)

probe = np.abs(fft2d(round)**2)
#probe/= np.sum(probe)

pow_spec = abs(fft2d(probe))

pow_specs.append(pow_spec)
labels.append('Round Aperture')

plt.figure(1)
plt.subplot(221)
plt.imshow(np.angle(_iwc)[os:-os,os:-os])
plt.subplot(222)
plt.imshow(np.angle(_iw)[os:-os,os:-os])
plt.subplot(223)
plt.imshow(np.angle(rpi)[os:-os,os:-os])
plt.subplot(224)
plt.imshow(round[os:-os,os:-os])

radial_integrated_intensity(pow_specs, labels=labels,styles=styles,crop_=100, center=None,num_bins=512,save_as_png=False)

plt.show()