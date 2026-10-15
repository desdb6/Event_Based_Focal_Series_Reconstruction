"""
Script to generate specimen image for linear focal series reconstruction.

Author: Des De Borger
Email: des.deborger@student.uantwerpen.be
Last modified: 14/10/2026
"""

import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from skimage.transform import resize

from utils import *

def load_grayscale(path: str, crop=None) -> np.ndarray:
    """Load a grayscale or RBG image into a np.array."""
    img = iio.imread(path)
    if img.ndim == 3: # RGB image, average the color values
        img = img[..., :3].mean(axis=-1)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D or RGB image, got shape {img.shape}")
    if crop is not None:
        img = img[crop]
    return img.astype(np.float64)

class Specimen():
    """Specimen class to generate object wavefunctions from image"""
    def __init__(self,
        phase_source: str = None,
        phase_strength: float = np.pi,
        amp_source: str = None,
        amp_strength: float = 1,
        pixel_size: float = 1, # in nm
        pad: int = 256,
        crop: int = None,
        resize_img: int = None,
        ):
        if phase_source is None and amp_source is None:
            raise ValueError("Provide at least one of phase_image or amp_image.")

        self.phase_source = phase_source
        self.phase_strength = phase_strength
        self.amp_source = amp_source
        self.amp_strength = amp_strength
        self.pixel_size = pixel_size
        self.pad = pad
        self.crop = crop
        self.resize_img = resize_img

        self.phase_image()
        self.amp_image()
        self.owf = self.phase * self.amp

        self.shape = self.owf.shape
        self.frequency_grid()
        self.real_space_grid()
        

    def phase_image(self):
        """Generate a phase image."""
        if self.phase_source is not None: # If no phase source is given, make it 0 everywhere
            img = load_grayscale(self.phase_source, self.crop)
            img = normalize(img,minmax=(0,self.phase_strength))
        else:
            amp = load_grayscale(self.amp_source, self.crop)
            img = np.zeros_like(amp)
        if self.resize_img is not None: # Resize so phase and amplitude are the same size
            img = resize(img, (self.resize_img, self.resize_img), anti_aliasing=True)
        img = np.pad(img,((self.pad,self.pad),(self.pad,self.pad)), constant_values=0)[::-1]
        img = np.rot90(img,2)
        self.phase = np.exp(1j*img)

    def amp_image(self):
        """Generate an amplitude image."""
        if self.amp_source is not None: # If no amplitude source is given, make it 1 everywhere
            img = load_grayscale(self.amp_source, self.crop)
            img = normalize(img,minmax=(1-self.amp_strength,1))
        else:
            phase = load_grayscale(self.phase_source, self.crop)
            img = np.ones_like(phase)
        if self.resize_img is not None: # Resize so phase and amplitude are the same size
            img = resize(img, (self.resize_img, self.resize_img), anti_aliasing=True)
        img = np.pad(img,((self.pad,self.pad),(self.pad,self.pad)), constant_values=1)[::-1]
        img = np.rot90(img,2)
        self.amp = img

    def real_space_grid(self):
        """Generate real-space coordinate grids."""
        ny, nx = self.shape
        self.y = (np.arange(ny) - ny // 2) * self.pixel_size
        self.x = (np.arange(nx) - nx // 2) * self.pixel_size
        self.Y, self.X = np.meshgrid(self.y, self.x, indexing="ij")

    def frequency_grid(self):
        """Generate reciprocal-space frequency grids."""
        self.ky = np.fft.fftshift(np.fft.fftfreq(self.shape[0], d=self.pixel_size))
        self.kx = np.fft.fftshift(np.fft.fftfreq(self.shape[1], d=self.pixel_size))
        self.K_Y, self.K_X = np.meshgrid(self.ky, self.kx, indexing="ij")
        self.k_grid = np.hypot(self.K_X, self.K_Y)

if __name__ ==  "__main__":
    spec = Specimen(phase_source='images/lena_1.jpg', phase_strength=0.01, resize_img=512, pixel_size=2)
    ax1 = plot_2d(spec.owf, mode="phase")
    ax2 = plot_2d(spec.owf, mode="abs")
    ax3 = plot_2d(spec.k_grid)
    plt.show()