"""
Script to simulate optics for linear focal series reconstruction.

Author: Des De Borger
Email: des.deborger@student.uantwerpen.be
Last modified: 14/10/2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, m_e, e, c

from utils import plot_1d

ACC_V = 300  # kV


def electron_wavelength(voltage_kv):
    """Relativistic electron wavelength in nm for an accelerating voltage in kV.
    Examples: 80 kV -> 0.00418 nm, 200 kV -> 0.002508 nm, 300 kV -> 0.001969 nm
    """
    V = np.asarray(voltage_kv, dtype=float) * 1e3  # volts
    lam = h / np.sqrt(2 * m_e * e * V * (1 + e * V / (2 * m_e * c**2)))
    return lam * 1e9


def fft2d(arr):
    """Centered 2D Fourier transform."""
    axes = (-2, -1)
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arr, axes=axes), axes=axes), axes=axes)


def ifft2d(arr):
    """Centered 2D inverse Fourier transform."""
    axes = (-2, -1)
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(arr, axes=axes), axes=axes), axes=axes)


def phase_error(k: float, wavelength: float, defocus: float, cs: float) -> np.float64:
    """Aberration function chi(k)."""
    return np.pi * wavelength * k**2 * (cs * wavelength**2 * k**2 / 2 + defocus)


def ctf_complex(k: float, wavelength: float, defocus: float, cs: float) -> np.complex128:
    """Complex coherent transfer function exp(i*chi)."""
    return np.exp(-1j * phase_error(k, wavelength, defocus, cs))


def spatial_coherence_envelope(k_grid: np.ndarray, wavelength: float, defocus: float, cs: float, alpha_mrad: float) -> np.ndarray:
    """Spatial coherence envelope for a semi-convergence angle in mrad."""
    alpha = alpha_mrad * 1e-3  # mrad -> rad
    grad = cs * wavelength**3 * k_grid**3 + defocus * wavelength * k_grid
    return np.exp(-(np.pi * alpha / wavelength) ** 2 * grad ** 2)


def aperture_grid(k_grid: np.ndarray, wavelength: float, alpha_max: float) -> np.ndarray:
    """Generate an aperture mask to cut off maximal k values."""
    return (k_grid < alpha_max/(wavelength * 1000)).astype(float)


def forward_linear(owf: np.ndarray, k_grid: np.ndarray, wavelength: float, defocus: float, cs: float, alpha_max: float = None, sc_alpha: float = None) -> np.ndarray:
    """Generate the detected linear intensity in the limit of a small phase and amplitude modulation."""
    x0 = 2 * fft2d(np.real(owf - 1))
    y0 = 2 * fft2d(np.imag(owf - 1))
    chi = phase_error(k_grid, wavelength, defocus, cs)

    I_g = x0 * np.cos(chi) + y0 * np.sin(chi)
    if alpha_max is not None:
        I_g = I_g * aperture_grid(k_grid, wavelength, alpha_max)
    if sc_alpha is not None:
        I_g = I_g * spatial_coherence_envelope(k_grid, wavelength, defocus, cs, sc_alpha)

    return ifft2d(I_g).real + 1


def forward_nonlinear(owf: np.ndarray, k_grid: np.ndarray, wavelength: float, defocus: float, cs: float, alpha_max: float = None) -> np.ndarray:
    """Generate the detected intensity."""
    H0 = ctf_complex(k_grid, wavelength, defocus, cs)
    if alpha_max is not None:
        H0 = H0 * aperture_grid(k_grid, wavelength, alpha_max)
    return np.abs(ifft2d(H0 * fft2d(owf))) ** 2


def focal_series_nonlinear(defoci: np.ndarray, owf: np.ndarray, k_grid: np.ndarray, wavelength: float, cs: float, alpha_max: float = None) -> np.ndarray:
    """Generate a defocus series of images."""
    return np.stack([
        forward_nonlinear(owf, k_grid, wavelength, df, cs, alpha_max)
        for df in defoci
    ])


def poisson_noise(defocus_series: np.ndarray, dose_per_pixel: int, seed: int = None) -> np.ndarray:
    """Apply poisson noise to a defocus series."""
    rng = np.random.default_rng(seed)
    counts = np.clip(defocus_series, 0, None) * dose_per_pixel
    noisy = rng.poisson(counts) / dose_per_pixel
    return noisy


if __name__ == "__main__":
    wavelength = electron_wavelength(ACC_V)

    k_values = np.linspace(0, 10, 1000)
    ctf_values = np.sin(phase_error(k_values, wavelength, defocus=100, cs=1e6))
    plot_1d(ctf_values, k_values, title="Sin(chi)", xlabel="k (1/nm)")

    ctf_c = ctf_complex(k_values, wavelength, defocus=100, cs=1e6)
    plot_1d(ctf_c.real, k_values, title="Real part of the complex CTF", xlabel="k (1/nm)")
    plot_1d(ctf_c.imag, k_values, title="Imaginary part of the complex CTF", xlabel="k (1/nm)")

    plt.show()