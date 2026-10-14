"""
Script to simulate optics for linear focal series reconstruction.

Author: Des De Borger
Email: des.deborger@student.uantwerpen.be
Last modified: 14/10/2026
"""

import numpy as np
import torch

from utils import *

ACC_V = 300

def electron_wavelength(voltage_kv):
    """Relativistic electron wavelength in nm for an accelerating voltage in kV.
 
    lambda = h / sqrt(2 m0 e V (1 + e V / (2 m0 c^2)))
 
    Examples: 80 kV -> 0.00418 nm, 200 kV -> 0.002508 nm, 300 kV -> 0.001969 nm
    """
    from scipy.constants import h, m_e, e, c
 
    V = np.asarray(voltage_kv, dtype=float) * 1e3          # volts
    lam = h / np.sqrt(2 * m_e * e * V * (1 + e * V / (2 * m_e * c**2)))
    return lam * 1e9   

def fft2d(arr):
    """2D Fourier transform."""
    dims = (-2, -1)
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(arr, dim=dims), dim=dims), dim=dims)

def ifft2d(arr):
    """2D Inverse Fourier transform."""
    dims = (-2, -1)
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(arr, dim=dims), dim=dims), dim=dims)

def phase_error(k, wavelength, defocus, cs):
    """Calculate the coherent transfer function."""
    return torch.pi * wavelength * k**2 * (cs * wavelength**2 * k**2 / 2 + defocus)

def ctf_complex(k, wavelength, defocus, cs):
    return torch.exp(1j*phase_error(k_x, k_y, wavelength, defocus, cs))

if __name__ == "__main__":
    wavelength = electron_wavelength(ACC_V)

    k_values = torch.tensor(np.linspace(0, 10, 1000))
    ctf_values = torch.sin(phase_error(k_values, 0, wavelength, defocus=100, cs=1e3))
    ax1 = plot_1d(ctf_values, k_values, title="Sin(chi)")

    ctf_values_complex = ctf_complex(k_values, 0, wavelength, defocus=100, cs=1e3)
    ax2 = plot_1d(torch.real(ctf_values_complex), k_values, title="Real part of the complex coherent phase transfer fucntion.")
    ax3 = plot_1d(torch.imag(ctf_values_complex), k_values, title="Imaginary part of the complex coherent phase transfer fucntion.")

    plt.show()